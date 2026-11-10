import Lean
import Remora.Prop
import Remora.Intrinsics

/-! Static-shape Remora lowering to standard tensor/linalg MLIR.

Records become parallel primitive arrays. Scalar maps and builtin associative
reductions become linalg operations. A lifted array-cell function is lowered in
its principal frame, so its internal maps/reductions remain visible to fusion.
Other folds retain their evaluation order in scf loops. External MLIR verifies
this translation; there is no Lean proof of compiler correctness here.
-/
namespace Remora.MLIR
structure Ty where
  base : String
  dims : List Int
  deriving BEq, Inhabited
instance : ToString Ty where
  toString t := if t.dims.isEmpty then t.base else
    "tensor<" ++ String.join (t.dims.map fun n => s!"{n}x") ++ t.base ++ ">"
structure Value where
  name : String
  ty : Ty
  deriving Inhabited
structure Builder where
  next : Nat := 0
  lines : List String := []
  env : List (VName × List Value) := []
  functions : List String := []
  globals : List (VName × List Ty) := []
  definitions : List (VName × (List Pat × Exp)) := []
  -- Additional prefix while lowering a rank-lifted array-cell helper in place.
  frame : List Int := []
abbrev M := StateT Builder (Except String)
private def fresh : M String := do
  let n := (← get).next
  modify fun s => {s with next := n+1}
  return s!"%v{n}"
private def line (s : String) : M Unit := modify fun b => {b with lines := b.lines ++ [s]}
private def emit (ty : Ty) (rhs : String) : M Value := do
  let name ← fresh
  line s!"{name} = {rhs}"
  return ⟨name,ty⟩
private def shape (s : Shape VName) : Except String (List Int) := do
  let ns ← shapeToInts (dimToInt fun v => .error s!"unresolved dimension {v}")
    (fun v => .error s!"unresolved shape {v}") s
  if ns.any (· < 0) then throw "negative dimension"
  return ns
private def typeOf (t : ArrayType VName) : Except String Ty := do
  let base ← match t.atom with
    | .int => pure "i64" | .float => pure "f32" | .bool => pure "i1"
    | _ => throw "only Int, Float, Bool and record values are supported"
  return ⟨base, ← shape t.shape⟩
private def symbol (v : VName) := s!"@remora_fn_{v.varTag}"
private def bind (v : VName) (x : List Value) : M Unit :=
  modify fun s => {s with env := (v,x)::s.env}
private def affine (rank : Nat) (axes : List Nat) :=
  "affine_map<(" ++ String.intercalate "," ((List.range rank).map fun i => s!"d{i}") ++
  ")->(" ++ String.intercalate "," (axes.map fun i => s!"d{i}") ++ ")>"
private def signature (xs : List Value) := String.intercalate ", " (xs.map fun x => toString x.ty)
private def names (xs : List Value) := String.intercalate ", " (xs.map (·.name))
private def scalarOp (name : VName) (xs : List Value) (out : Ty) : M Value := do
  let ops := [("+","arith.addi"),("-","arith.subi"),("*","arith.muli"),
    ("f.+","arith.addf"),("f.-","arith.subf"),("f.*","arith.mulf")]
  if name.varTag < intrinsicTypes.length then
    let some op := ops.lookup name.varName | throw s!"unsupported intrinsic {name.varName}"
    unless xs.length == 2 do throw "arithmetic requires two operands"
    emit out s!"{op} {names xs} : {out}"
  else
    emit out s!"func.call {symbol name}({names xs}) : ({signature xs}) -> {out}"
private def mapped (out : Ty) (xs : List Value) (f : List Value → M Value) : M Value := do
  if out.dims.isEmpty then return ← f xs
  let inputs := xs.filter fun x => !x.ty.dims.isEmpty
  for x in inputs do
    unless x.ty.dims == out.dims.take x.ty.dims.length do throw "incompatible prefix frames"
  let dest ← emit out s!"tensor.empty() : {out}"
  let result ← fresh
  let maps := (inputs.map fun x => affine out.dims.length (List.range x.ty.dims.length)) ++
    [affine out.dims.length (List.range out.dims.length)]
  let iters := String.intercalate ", " (out.dims.map fun _ => "\"parallel\"")
  line <| s!"{result} = linalg.generic \{indexing_maps = [{String.intercalate ", " maps}], " ++
    s!"iterator_types = [{iters}]} ins({names inputs} : {signature inputs}) " ++
    s!"outs({dest.name} : {out}) \{"
  let args ← inputs.mapM fun x => return Value.mk (← fresh) ⟨x.ty.base,[]⟩
  let unused ← fresh
  line ("^bb0(" ++ String.intercalate ", " ((args.map fun x => s!"{x.name}: {x.ty}") ++
    [s!"{unused}: {out.base}"]) ++ "):")
  let mut remaining := args
  let mut operands := []
  for x in xs do
    if x.ty.dims.isEmpty then operands := operands ++ [x]
    else
      let a::rest := remaining | throw "missing region operand"
      operands := operands ++ [a]
      remaining := rest
  let value ← f operands
  line s!"linalg.yield {value.name} : {out.base}"
  line s!"} -> {out}"
  return ⟨result,out⟩
private def sum (x : Value) (out : Ty) : M Value := do
  let tensorOut := "tensor<" ++ String.join (out.dims.map fun n => s!"{n}x") ++ "i64>"
  if x.ty.dims.isEmpty then return x
  let z ← fresh
  line s!"{z} = arith.constant dense<0> : {tensorOut}"
  let result ← fresh
  let rank := x.ty.dims.length
  let maps := [affine rank (List.range rank), affine rank (List.range out.dims.length)]
  let iters := (List.range rank).map fun i =>
    if i < out.dims.length then "\"parallel\"" else "\"reduction\""
  line <| s!"{result} = linalg.generic \{indexing_maps = [{String.intercalate ", " maps}], " ++
    s!"iterator_types = [{String.intercalate ", " iters}]} " ++
    s!"ins({x.name} : {x.ty}) outs({z} : {tensorOut}) \{"
  let a ← fresh
  let b ← fresh
  line s!"^bb0({a}: i64, {b}: i64):"
  let v ← emit ⟨"i64",[]⟩ s!"arith.addi {a}, {b} : i64"
  line s!"linalg.yield {v.name} : i64"
  line s!"} -> {tensorOut}"
  if out.dims.isEmpty then emit out s!"tensor.extract {result}[] : {tensorOut}"
  else return ⟨result,out⟩

private partial def types (t : ArrayType VName) : Except String (List Ty) := do
  match t.atom with
  | .record fs =>
    let ds ← shape t.shape
    return (← fs.toList.mapM fun (_,t) => do
      return (← types t).map fun t => {t with dims := ds ++ t.dims}).flatten
  | _ => return [← typeOf t]
private def one (xs : List α) : M α :=
  match xs with | [x] => pure x | _ => throw "expected one primitive component"
private def emitResults (ts : List Ty) (rhs : String) : M (List Value) := do
  let vs ← ts.mapM fun t => return Value.mk (← fresh) t
  line s!"{names vs} = {rhs}"
  return vs
private def returnTypes (ts : List Ty) :=
  if ts.length == 1 then toString ts.head! else
    "(" ++ String.intercalate ", " (ts.map toString) ++ ")"
private def indexConstant (n : Int) : M Value := emit ⟨"index",[]⟩ s!"arith.constant {n} : index"
private def asIndex (v : Value) : M Value :=
  if v.ty.base == "index" then pure v else
    emit ⟨"index",[]⟩ s!"arith.index_cast {v.name} : {v.ty} to index"
private def extract (x : Value) (indices : List Value) : M Value :=
  if indices.isEmpty then pure x else
    emit ⟨x.ty.base,[]⟩ s!"tensor.extract {x.name}[{names indices}] : {x.ty}"
-- Index-driven array construction stays in linalg, so downstream tensor compilers
-- can schedule it along with arithmetic maps and reductions.
private def generate (t : Ty) (f : List Value → M Value) : M Value := do
  if t.dims.isEmpty then return ← f []
  let dest ← emit t s!"tensor.empty() : {t}"
  let name ← fresh
  let iters := String.intercalate ", " (t.dims.map fun _ => "\"parallel\"")
  line <| s!"{name} = linalg.generic \{indexing_maps = [{affine t.dims.length (List.range t.dims.length)}], " ++
    s!"iterator_types = [{iters}]} outs({dest.name} : {t}) \{"
  let unused ← fresh
  line s!"^bb0({unused}: {t.base}):"
  let indices ← (List.range t.dims.length).mapM fun i =>
    emit ⟨"index",[]⟩ s!"linalg.index {i} : index"
  let value ← f indices
  line s!"linalg.yield {value.name} : {t.base}"
  line s!"} -> {t}"
  return ⟨name,t⟩
private def reshape (x : Value) (t : Ty) : M Value := do
  if x.ty == t then return x
  if t.dims.isEmpty then
    let zs ← x.ty.dims.mapM fun _ => indexConstant 0
    return ← extract x zs
  let x ← if x.ty.dims.isEmpty then
    emit ⟨x.ty.base,[1]⟩ s!"tensor.from_elements {x.name} : tensor<1x{x.ty.base}>"
    else pure x
  let st := s!"tensor<{t.dims.length}xi64>"
  let dims ← fresh
  let sizes := String.intercalate ", " (t.dims.map toString)
  line s!"{dims} = arith.constant dense<[{sizes}]> : {st}"
  emit t s!"tensor.reshape {x.name}({dims}) : ({x.ty}, {st}) -> {t}"
private def slice (x : Value) (i : Value) : M Value := do
  let t := {x.ty with dims := x.ty.dims.drop 1}
  if t.dims.isEmpty then return ← extract x [i]
  let offsets := String.intercalate ", " (i.name :: t.dims.map fun _ => "0")
  let sizes := String.intercalate ", " ("1" :: t.dims.map toString)
  let strides := String.intercalate ", " (x.ty.dims.map fun _ => "1")
  emit t s!"tensor.extract_slice {x.name}[{offsets}] [{sizes}] [{strides}] : {x.ty} to {t}"
private def insert (dest x : Value) (i : Value) : M Value := do
  if x.ty.dims.isEmpty then
    emit dest.ty s!"tensor.insert {x.name} into {dest.name}[{i.name}] : {dest.ty}"
  else
    let offsets := String.intercalate ", " (i.name :: x.ty.dims.map fun _ => "0")
    let sizes := String.intercalate ", " ("1" :: x.ty.dims.map toString)
    let strides := String.intercalate ", " (dest.ty.dims.map fun _ => "1")
    emit dest.ty s!"tensor.insert_slice {x.name} into {dest.name}[{offsets}] [{sizes}] [{strides}] : {x.ty} into {dest.ty}"
private def loop (n start : Int) (initial : List Value)
    (f : Value → List Value → M (List Value)) : M (List Value) := do
  let lo ← indexConstant start
  let hi ← indexConstant n
  let step ← indexConstant 1
  let i ← fresh
  let results ← initial.mapM fun x => return Value.mk (← fresh) x.ty
  let args ← initial.mapM fun x => return Value.mk (← fresh) x.ty
  let bindings := String.intercalate ", " ((args.zip initial).map fun (a,x) => s!"{a.name} = {x.name}")
  let ts := String.intercalate ", " (initial.map fun x => toString x.ty)
  line s!"{names results} = scf.for {i} = {lo.name} to {hi.name} step {step.name} iter_args({bindings}) -> ({ts}) \{"
  let values ← f ⟨i,⟨"index",[]⟩⟩ args
  line s!"scf.yield {names values} : {signature values}"
  line "}"
  return results
private partial def mapCells (frame : List Int) (ranks : List Nat)
    (xs : List (List Value)) (ts : List Ty)
    (f : List (List Value) → List Ty → M (List Value)) : M (List Value) := do
  match frame with
  | [] => f xs ts
  | d::ds =>
    let dest ← ts.mapM fun t => emit t s!"tensor.empty() : {t}"
    loop d 0 dest fun i acc => do
      let inputs ← (ranks.zip xs).mapM fun (rank,vs) =>
        if rank == 0 then pure vs else vs.mapM fun x => slice x i
      let inner := ts.map fun t => {t with dims := t.dims.drop 1}
      let out ← mapCells ds (ranks.map (· - 1)) inputs inner f
      (acc.zip out).mapM fun (dest,x) => insert dest x i
-- Explicit input maps let fusion see through broadcasts and permutations.
private def mappedAt (out : Ty) (xs : List Value) (axes : List (List Nat))
    (f : List Value → M Value) : M Value := do
  let dest ← emit out s!"tensor.empty() : {out}"
  let result ← fresh
  let maps := axes.map (affine out.dims.length) ++ [affine out.dims.length (List.range out.dims.length)]
  let iters := String.intercalate ", " (out.dims.map fun _ => "\"parallel\"")
  line <| s!"{result} = linalg.generic \{indexing_maps = [{String.intercalate ", " maps}], " ++
    s!"iterator_types = [{iters}]} ins({names xs} : {signature xs}) outs({dest.name} : {out}) \{"
  let args ← xs.mapM fun x => return Value.mk (← fresh) ⟨x.ty.base,[]⟩
  let unused ← fresh
  line ("^bb0(" ++ String.intercalate ", " ((args.map fun x => s!"{x.name}: {x.ty}") ++ [s!"{unused}: {out.base}"]) ++ "):")
  let value ← f args
  line s!"linalg.yield {value.name} : {out.base}"
  line s!"} -> {out}"
  return ⟨result,out⟩
private def broadcast (x : Value) (t : Ty) (sourceFrame targetFrame : Nat) : M Value := do
  if x.ty == t then return x
  if x.ty.dims.isEmpty then return ← generate t fun _ => pure x
  let axes := List.range sourceFrame ++ (List.range (x.ty.dims.length-sourceFrame)).map (· + targetFrame)
  mappedAt t [x] [axes] fun vs => one vs
private def principal (frames : List (List Int)) : M (List Int) := do
  let frame := frames.foldl (fun a b => if a.length < b.length then b else a) []
  unless frames.all (fun f => f == frame.take f.length) do throw s!"incompatible prefix frames {frames}"
  return frame
private def appArgs : Exp → Exp × List Exp × List (ArrayType VName)
  | .app f x .. =>
    let (root,xs,ps) := appArgs f
    let ps := match (arrayTypeOf f).atom with | .arrow p _ => ps ++ [p] | _ => ps
    (root,xs ++ [x],ps)
  | .iApp f .. | .tApp f .. => appArgs f
  | e => (e,[],[])
mutual
private partial def applyName (name : VName) (xs : List (List Value))
    (ts : List Ty) : M (List Value) := do
  let flat := xs.flatten
  if name.varTag >= intrinsicTypes.length then
    return ← emitResults ts s!"func.call {symbol name}({names flat}) : ({signature flat}) -> {returnTypes ts}"
  let out ← one ts
  let x : Nat → M Value := fun n => do one (← match xs[n]? with | some x => pure x | none => throw "missing argument")
  if name.varName == "iota/static" then
    return [← generate out fun indices => do
      let mut acc ← indexConstant 0
      for (i,d) in indices.zip out.dims do
        let dim ← indexConstant d
        acc ← emit ⟨"index",[]⟩ s!"arith.muli {acc.name}, {dim.name} : index"
        acc ← emit ⟨"index",[]⟩ s!"arith.addi {acc.name}, {i.name} : index"
      emit ⟨"i64",[]⟩ s!"arith.index_cast {acc.name} : index to i64"]
  if ["reshape","flatten"].contains name.varName then return [← reshape (← x 0) out]
  if name.varName == "transpose2d" then
    let a ← x 0
    return [← mappedAt out [a] [[1,0]] fun vs => one vs]
  if name.varName == "append" then
    let a ← x 0
    let b ← x 1
    return [← emit out s!"tensor.concat dim(0) {a.name}, {b.name} : ({a.ty}, {b.ty}) -> {out}"]
  if name.varName == "index" then
    return [← slice (← x 0) (← asIndex (← x 1))]
  if name.varName == "index2d" then
    let a ← x 0
    let idx ← x 1
    let i ← indexConstant 0
    let j ← indexConstant 1
    return [← slice (← slice a (← asIndex (← extract idx [i]))) (← asIndex (← extract idx [j]))]
  if name.varName == "length" then
    let a ← x 0
    return [← emit out s!"arith.constant {a.ty.dims.headD 0} : i64"]
  if name.varName == "head" then return [← slice (← x 0) (← indexConstant 0)]
  if name.varName == "trace" || name.varName == "trace-file" then
    throw "runtime tracing is unsupported; return the value explicitly"
  if let some op := ([ ("f./","arith.divf"),("f.max","arith.maximumf"),
      ("f.min","arith.minimumf"),("f.^","math.powf"),("/","arith.divsi"),
      ("mod","arith.remsi") ] : List (String × String)).lookup name.varName then
    return [← emit out s!"{op} {names flat} : {out}"]
  if let some op := ([ ("f.sqrt","math.sqrt"),("f.exp","math.exp"),
      ("f.ln","math.log"),("f.tanh","math.tanh") ] : List (String × String)).lookup name.varName then
    return [← emit out s!"{op} {names flat} : {out}"]
  if let some predicate := ([("f.==","oeq"),("f.!=","une"),("f.<","olt"),
      ("f.<=","ole"),("f.>","ogt"),("f.>=","oge")] : List (String × String)).lookup name.varName then
    return [← emit out s!"arith.cmpf {predicate}, {names flat} : {(← x 0).ty}"]
  if let some predicate := ([("==","eq"),("!=","ne"),("<","slt"),
      ("<=","sle"),(">","sgt"),(">=","sge")] : List (String × String)).lookup name.varName then
    return [← emit out s!"arith.cmpi {predicate}, {names flat} : {(← x 0).ty}"]
  if name.varName == "bool->f" then
    return [← emit out s!"arith.uitofp {(← x 0).name} : i1 to f32"]
  if name.varName == "bool->i" then
    return [← emit out s!"arith.extui {(← x 0).name} : i1 to i64"]
  if name.varName == "i->f" then
    let a ← x 0
    return [← emit out s!"arith.sitofp {a.name} : i64 to f32"]
  return [← scalarOp name flat out]
private partial def reduction (name : VName) (args : List Exp)
    (ps : List (ArrayType VName)) (oldFrame : List Int) (ts : List Ty) : M (List Value) := do
  let some op := args.head? | throw "missing reduction operator"
  let (root,captured,capturePs) := appArgs op
  let .var operator .. := root | throw "reduction requires a known operator"
  let captured ← captured.mapM expression
  let captureFrames ← (captured.zip capturePs).mapM fun (vs,p) => do
    let v ← one vs
    let t ← liftM (types p)
    return v.ty.dims.take (v.ty.dims.length-t.head!.dims.length)
  let dataArgs := args.drop 1
  let vals ← dataArgs.mapM expression
  let frames ← (vals.zip (ps.drop 1)).mapM fun (vs,p) => do
    let v ← one vs
    return v.ty.dims.take (v.ty.dims.length - (← liftM (shape p.shape)).length)
  let frame ← principal (captureFrames ++ frames)
  let ranks := frames.map List.length
  let ts := ts.map fun t => {t with dims := frame ++ t.dims.drop oldFrame.length}
  -- A builtin associative operation (or its eta expansion) becomes one
  -- structured reduction across the array's leading cell axis.
  let mut reducer := operator
  if let some (params,body) := (← get).definitions.lookup operator then
    let (root,operands,_) := appArgs body
    if let .var v .. := root then
      if operands.length == params.length && (operands.zip params).all (fun (a,p) =>
          match a with | .var v .. => v == patVar p | _ => false) then
        reducer := v
  if reducer.varTag < intrinsicTypes.length && captured.isEmpty &&
      ["f.+","f.*","f.max","f.min","+","*"].contains reducer.varName &&
      ["reduce","reduce/zero"].contains name.varName then
    let input ← one vals.getLast!
    let inputFrame := ranks.getLast!
    let cell := input.ty.dims.drop inputFrame
    let n := cell.headD 0
    if name.varName == "reduce" && n == 0 then throw "empty nonzero reduction"
    let inputTy := {input.ty with dims := frame ++ cell}
    let input ← broadcast input inputTy inputFrame frame.length
    let out ← one ts
    let tensorOut := "tensor<" ++ String.join (out.dims.map fun n => s!"{n}x") ++ out.base ++ ">"
    let initial ← if name.varName == "reduce/zero" then do
      let v ← one vals.head!
      broadcast v out ranks.head! frame.length
      else
        let identity := if ["f.*","*"].contains reducer.varName then
            (if out.base == "f32" then "1.0" else "1")
          else if reducer.varName == "f.max" then "0xFF800000"
          else if reducer.varName == "f.min" then "0x7F800000"
          else if out.base == "f32" then "0.0" else "0"
        let v ← emit ⟨out.base,[]⟩ s!"arith.constant {identity} : {out.base}"
        broadcast v out 0 out.dims.length
    let init ← if out.dims.isEmpty then do
        let v ← fresh
        line s!"{v} = tensor.from_elements {initial.name} : {tensorOut}"
        pure v
      else pure initial.name
    let rank := input.ty.dims.length
    let axes := (List.range rank).filter (· != frame.length)
    let maps := [affine rank (List.range rank),affine rank axes]
    let iters := (List.range rank).map fun i => if i == frame.length then "\"reduction\"" else "\"parallel\""
    let result ← fresh
    line <| s!"{result} = linalg.generic \{indexing_maps = [{String.intercalate ", " maps}], " ++
      s!"iterator_types = [{String.intercalate ", " iters}]} ins({input.name} : {input.ty}) outs({init} : {tensorOut}) \{"
    let a ← fresh
    let b ← fresh
    line s!"^bb0({a}: {out.base}, {b}: {out.base}):"
    let v ← one (← applyName reducer [[⟨b,⟨out.base,[]⟩⟩],[⟨a,⟨out.base,[]⟩⟩]] [⟨out.base,[]⟩])
    line s!"linalg.yield {v.name} : {out.base}"
    line s!"} -> {tensorOut}"
    if out.dims.isEmpty then return [← emit out s!"tensor.extract {result}[] : {tensorOut}"]
    return [⟨result,out⟩]
  let allRanks := captureFrames.map List.length ++ ranks
  return ← mapCells frame allRanks (captured ++ vals) ts fun operands ts => do
    let captured := operands.take captured.length
    let values := operands.drop captured.length
    let input ← one values.getLast!
    let n := input.ty.dims.headD 0
    let initial ← if name.varName == "reduce" then
      if n == 0 then throw "empty nonzero reduction" else pure [← slice input (← indexConstant 0)]
      else pure values.head!
    loop n (if name.varName == "reduce" then 1 else 0) initial fun i acc => do
      let idx ← if name.varName == "fold-right" then do
        let last ← indexConstant (n-1)
        emit ⟨"index",[]⟩ s!"arith.subi {last.name}, {i.name} : index"
        else pure i
      let value ← slice input idx
      let operands := if name.varName == "fold-right" then [[value],acc] else [acc,[value]]
      applyName operator (captured ++ operands) ts
private partial def application (e : Exp) (ts : List Ty) : M (List Value) := do
  let (root,args,ps) := appArgs e
  let .var name .. := root | throw "application requires a known function"
  let oldFrame ← match e with | .app _ _ i _ => liftM (shape i.value.2) | _ => pure []
  if ["reduce","reduce/zero","fold","fold-right"].contains name.varName && name.varTag < intrinsicTypes.length then
    return ← reduction name args ps oldFrame ts
  let values ← args.mapM expression
  let frames ← (values.zip ps).mapM fun (vs,p) => do
    let v ← match vs with | v::_ => pure v | _ => throw "empty record argument"
    -- Record leaves also contain the field's shape, which is part of the cell.
    let t ← liftM (types p)
    let cellRank := (t.head!).dims.length
    return v.ty.dims.take (v.ty.dims.length-cellRank)
  let frame ← principal frames
  let ts := ts.map fun t => {t with dims := frame ++ t.dims.drop oldFrame.length}
  if name.varName == "sum" && name.varTag < intrinsicTypes.length then
    return [← sum (← one values.flatten) (← one ts)]
  let ranks := frames.map List.length
  if name.varTag < intrinsicTypes.length then
    if ["reshape","flatten"].contains name.varName then
      return [← reshape (← one values.flatten) (← one ts)]
    if name.varName == "transpose2d" then
      let x ← one values.flatten
      let r := frame.length
      let axes := List.range r ++ [r+1,r] ++ (List.range (x.ty.dims.length-r-2)).map (· + r+2)
      return [← mappedAt (← one ts) [x] [axes] fun vs => one vs]
    if name.varName == "append" then
      let xs ← ((values.zip ranks).zip ps).mapM fun ((vs,r),p) => do
        let x ← one vs
        let cell ← liftM (shape p.shape)
        broadcast x {x.ty with dims := frame ++ cell} r frame.length
      let a := xs[0]!
      let b := xs[1]!
      let out ← one ts
      return [← emit out s!"tensor.concat dim({frame.length}) {a.name}, {b.name} : ({a.ty}, {b.ty}) -> {out}"]
    if ["index","index2d","head"].contains name.varName then
      let x ← one values.head!
      let out ← one ts
      return [← generate out fun indices => do
        let mut coordinates := indices.take ranks.head!
        if name.varName == "head" then coordinates := coordinates ++ [← indexConstant 0]
        else
          let idx ← one values[1]!
          let loc := indices.take ranks[1]!
          if name.varName == "index" then
            coordinates := coordinates ++ [← asIndex (← extract idx loc)]
          else
            for j in [0,1] do
              coordinates := coordinates ++ [← asIndex (← extract idx (loc ++ [← indexConstant j]))]
        extract x (coordinates ++ indices.drop frame.length)]
    if name.varName == "length" then
      let x ← one values.flatten
      let out ← one ts
      let v ← emit ⟨"i64",[]⟩ s!"arith.constant {x.ty.dims[frame.length]!} : i64"
      return [← broadcast v out 0 out.dims.length]
  let scalarCells := ps.all fun p => isEmptyShape p.shape && !(match p.atom with | .record .. => true | _ => false)
  if scalarCells && ts.length == 1 && ts.head!.dims.length == frame.length && !frame.isEmpty then
    let xs ← values.mapM one
    let out ← one ts
    return [← mapped out xs fun xs => do one (← applyName name (xs.map fun x => [x]) [⟨out.base,[]⟩])]
  if name.varTag >= intrinsicTypes.length && !frame.isEmpty then
    let some (params,body) := (← get).definitions.lookup name | throw s!"unknown function {name}"
    let old := (← get).env
    let oldFrame := (← get).frame
    modify fun s => {s with frame := frame}
    for ((p,vs),rank) in (params.zip values).zip ranks do
      let vs ← vs.mapM fun v => broadcast v {v.ty with dims := frame ++ v.ty.dims.drop rank} rank frame.length
      bind (patVar p) vs
    let result ← expression body
    modify fun s => {s with env := old, frame := oldFrame}
    return result
  mapCells frame ranks values ts (applyName name)
-- An invariant expression inside a lifted helper still belongs to the helper's
-- outer frame. Broadcasting it here prevents local frames (e.g. an iota inside
-- replicate) from being confused with that extra prefix.
private partial def expression (e : Exp) : M (List Value) := do
  let values ← expressionRaw e
  let expected ← liftM (types (arrayTypeOf e))
  let frame := (← get).frame
  (values.zip expected).mapM fun (v,t) =>
    broadcast v {t with dims := frame ++ t.dims} (v.ty.dims.length-t.dims.length) frame.length
private partial def expressionRaw (e : Exp) : M (List Value) := do
  try
    let ts ← liftM (types (arrayTypeOf e))
    match e with
    | .var v .. =>
      match (← get).env.lookup v with
      | some x => pure x
      | none =>
        let some ts := (← get).globals.lookup v | throw s!"unknown value {v}"
        emitResults ts s!"func.call {symbol v}() : () -> {returnTypes ts}"
    | .array _ atoms .. =>
      let out ← one ts
      let vals ← atoms.toList.mapM fun a => match a with
        | .base (.boolVal b) .. => pure (if b then "1" else "0")
        | .base (.intVal n) .. => pure (toString n)
        | .base (.floatVal n) .. => pure (toString n)
        | _ => throw "unsupported array element"
      if out.dims.isEmpty then return [← emit out s!"arith.constant {vals.head!} : {out}"]
      let vs ← vals.mapM fun n => emit ⟨out.base,[]⟩ s!"arith.constant {n} : {out.base}"
      let flat : Ty := ⟨out.base,[Int.ofNat vs.length]⟩
      return [← reshape (← emit flat s!"tensor.from_elements {names vs} : {flat}") out]
    | .emptyArray .. | .emptyFrame .. =>
      ts.mapM fun t => emit t s!"arith.constant dense<> : {t}"
    | .frame ds es .. =>
      unless ds.length == 1 do throw "multi-axis computed frames are unsupported"
      let rows ← es.toList.mapM expression
      let frames ← (rows.zip es.toList).mapM fun (vs,e) => do
        let actual ← one vs
        let expected ← liftM (types (arrayTypeOf e))
        return actual.ty.dims.take (actual.ty.dims.length-expected.head!.dims.length)
      let frame ← principal frames
      if frame.isEmpty then
        return ← ts.zipIdx.mapM fun (t,j) => do
          let mut dest ← emit t s!"tensor.empty() : {t}"
          for (row,i) in rows.zipIdx do
            dest ← insert dest row[j]! (← indexConstant i)
          return dest
      ts.zipIdx.mapM fun (t,j) => do
        let out := {t with dims := frame ++ t.dims}
        let xs ← (rows.zip frames).mapM fun (vs,f) => do
          let x := vs[j]!
          broadcast x {x.ty with dims := frame ++ x.ty.dims.drop f.length} f.length frame.length
        if out.dims.isEmpty then throw "frame must have an axis"
        let axes := (List.range out.dims.length).filter (· != frame.length)
        mappedAt out xs (xs.map fun _ => axes) fun args => do
          let i ← emit ⟨"index",[]⟩ s!"linalg.index {frame.length} : index"
          let mut v := args.getLast!
          for (a,n) in args.dropLast.zipIdx do
            let k ← indexConstant n
            let c ← emit ⟨"i1",[]⟩ s!"arith.cmpi eq, {i.name}, {k.name} : index"
            v ← emit a.ty s!"arith.select {c.name}, {a.name}, {v.name} : {a.ty}"
          return v
    | .letE bs body .. =>
      for b in bs.toList do binding b
      expression body
    | .structE fs info _ =>
      let targetFrame := (← get).frame ++ (← liftM (shape info.value.2))
      let mut result := []
      for (_,cell,e) in fs.toList do
        let vs ← expression e
        let s ← liftM (shape (arrayTypeOf e).shape)
        let c ← liftM (shape cell)
        let frameRank := (← get).frame.length + s.length - c.length
        for v in vs do
          let t := {v.ty with dims := targetFrame ++ v.ty.dims.drop frameRank}
          result := result ++ [← broadcast v t frameRank targetFrame.length]
      return result
    | .fieldProj e field .. =>
      let .record fs := (arrayTypeOf e).atom | throw "projection from non-record"
      let values ← expression e
      let mut offset := 0
      for (f,t) in fs.toList do
        let count := (← liftM (types t)).length
        if f == field then return values.drop offset |>.take count
        offset := offset + count
      throw s!"unknown field {field}"
    | .app .. | .iApp .. | .tApp .. => application e ts
    | _ => throw "unsupported expression or residual polymorphism"
  catch msg => throw s!"MLIR {posOf e}: {msg}"
private partial def binding : Bind → M Unit
  | .val v _ e _ => do bind v (← expression e)
  | .typeE .. | .ispace .. => pure ()
  | _ => throw "residual function binding"
end
private def function (name : String) (ps : List Pat) (body : Exp) (exported := false) : M Unit := do
  let old := (← get).env
  let oldLines := (← get).lines
  modify fun s => {s with env := [], lines := []}
  let mut params := []
  for p in ps do
    let ts ← liftM (types (patType p))
    let xs ← ts.mapM fun t => return Value.mk (← fresh) t
    bind (patVar p) xs
    params := params ++ xs
  let result ← expression body
  let lines := (← get).lines
  let header := String.intercalate ", " (params.map fun x => s!"{x.name}: {x.ty}")
  let ts := returnTypes (result.map (·.ty))
  let visibility := if exported then "" else "private "
  let text := s!"func.func {visibility}{name}({header}) -> {ts} \{\n" ++
    String.intercalate "\n" lines ++ s!"\nreturn {names result} : {signature result}\n}"
  modify fun s => {s with env := old, lines := oldLines, functions := s.functions ++ [text]}
/-- Emit ordinary static-shape Remora, preserving array cells and flattening records. -/
def compile (p : Prog) : Except String String := do
  unless p.progDecs.any (fun d => match d with | .entry .. => true | _ => false) do
    throw "MLIR: program has no entry point"
  let globals ← p.progDecs.filterMap (fun d => match d with
    | .defE (.val v _ e _) => some (v,arrayTypeOf e) | _ => none)
    |>.mapM fun (v,t) => do return (v,← types t)
  let action : M Unit := p.progDecs.forM fun d => match d with
    | .defE (.funE v ps _ e ..) => function (symbol v) ps.toList e
    | .entry v ps _ e .. => function ("@" ++ (Lean.Json.str v.varName).compress) ps e true
    | .defE (.val v _ e _) => function (symbol v) [] e
    | .defE (.typeE ..) | .defE (.ispace ..) => pure ()
    | _ => throw "MLIR: residual polymorphic declaration"
  let definitions := p.progDecs.filterMap fun d => match d with
    | .defE (.funE v ps _ e ..) => some (v,(ps.toList,e)) | _ => none
  let (_,s) ← action.run {globals,definitions}
  return "module {\n" ++ String.intercalate "\n" s.functions ++ "\n}\n"
end Remora.MLIR
