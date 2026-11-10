import Lean
import Remora.Prop
import Remora.Intrinsics

/-! A deliberately small static-shape tensor/linalg emitter. External MLIR verifies the IR. -/
namespace Remora.MLIR
structure Ty where
  base : String
  dims : List Int
  deriving BEq
instance : ToString Ty where
  toString t := if t.dims.isEmpty then t.base else
    "tensor<" ++ String.join (t.dims.map fun n => s!"{n}x") ++ t.base ++ ">"
structure Value where
  name : String
  ty : Ty
structure Builder where
  next : Nat := 0
  lines : List String := []
  env : List (VName × Value) := []
  functions : List String := []
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
    | .int => pure "i64" | .float => pure "f32"
    | _ => throw "only Int and Float values are supported"
  return ⟨base, ← shape t.shape⟩
private def ty (e : Exp) : M Ty := liftM (typeOf (arrayTypeOf e))
private def symbol (v : VName) := s!"@remora_fn_{v.varTag}"
private def lookup (v : VName) : M Value := do
  match (← get).env.lookup v with
  | some x => return x
  | none => throw s!"unknown value {v}"
private def bind (v : VName) (x : Value) : M Unit :=
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
private def appArgs : Exp → Exp × List Exp
  | .app f x .. => let (f,xs) := appArgs f; (f,xs ++ [x])
  | .iApp f .. | .tApp f .. => appArgs f
  | e => (e,[])
mutual
private partial def expression (e : Exp) : M Value := do
  try
    let out ← ty e
    match e with
    | .var v .. => lookup v
    | .array _ atoms .. =>
      let vals ← atoms.toList.mapM fun a => match a with
        | .base (.intVal n) .. => pure (toString n)
        | .base (.floatVal n) .. => pure (toString n)
        | _ => throw "unsupported array element"
      if out.dims.isEmpty then emit out s!"arith.constant {vals.head!} : {out}"
      else
        let vs ← vals.mapM fun n => emit ⟨out.base,[]⟩ s!"arith.constant {n} : {out.base}"
        let flat : Ty := ⟨out.base,[Int.ofNat vs.length]⟩
        let x ← emit flat s!"tensor.from_elements {names vs} : {flat}"
        if flat == out then return x
        else
          let shapeTy := s!"tensor<{out.dims.length}xi64>"
          let dims ← fresh
          let sizes := String.intercalate ", " (out.dims.map toString)
          line s!"{dims} = arith.constant dense<[{sizes}]> : {shapeTy}"
          emit out s!"tensor.reshape {x.name}({dims}) : ({flat}, {shapeTy}) -> {out}"
    | .emptyArray .. | .emptyFrame .. => emit out s!"arith.constant dense<> : {out}"
    | .frame ds es .. =>
      unless ds.length == 1 do throw "multi-axis computed frames are unsupported"
      let rows ← es.toList.mapM expression
      let dest ← emit out s!"tensor.empty() : {out}"
      let mut dest := dest
      for (x,i) in rows.zipIdx do
        if x.ty.dims.isEmpty then
          let idx ← emit ⟨"index",[]⟩ s!"arith.constant {i} : index"
          dest ← emit out s!"tensor.insert {x.name} into {dest.name}[{idx.name}] : {out}"
        else
          let offsets := String.intercalate ", " (toString i :: x.ty.dims.map fun _ => "0")
          let sizes := String.intercalate ", " ("1" :: x.ty.dims.map toString)
          let strides := String.intercalate ", " (out.dims.map fun _ => "1")
          dest ← emit out <| s!"tensor.insert_slice {x.name} into {dest.name}" ++
            s!"[{offsets}] [{sizes}] [{strides}] : {x.ty} into {out}"
      return dest
    | .letE bs body .. =>
      for b in bs.toList do binding b
      expression body
    | .app .. =>
      let (f,args) := appArgs e
      let .var name .. := f | throw "application requires a known named function"
      let xs ← args.mapM expression
      if name.varName == "sum" && name.varTag < intrinsicTypes.length then
        let [x] := xs | throw "sum requires one operand"
        sum x out
      else mapped out xs fun vs => scalarOp name vs ⟨out.base,[]⟩
    | _ => throw "unsupported expression or residual polymorphism"
  catch msg => throw s!"MLIR {posOf e}: {msg}"
private partial def binding : Bind → M Unit
  | .val v _ e _ => do bind v (← expression e)
  | .typeE .. | .ispace .. => pure ()
  | _ => throw "MLIR: residual function binding"
end
private def function (name : String) (ps : List Pat) (body : Exp) : M Unit := do
  let old := (← get).env
  let oldLines := (← get).lines
  modify fun s => {s with env := [], lines := []}
  let params ← ps.mapM fun p => do
    let t ← liftM (typeOf (patType p))
    let x := Value.mk (← fresh) t
    bind (patVar p) x
    return x
  let result ← expression body
  let lines := (← get).lines
  let header := String.intercalate ", " (params.map fun x => s!"{x.name}: {x.ty}")
  let text := s!"func.func {name}({header}) -> {result.ty} \{\n" ++
    String.intercalate "\n" lines ++ s!"\nreturn {result.name} : {result.ty}\n}"
  modify fun s => {s with env := old, lines := oldLines, functions := s.functions ++ [text]}
/-- Emit the supported static, first-order subset as a standard MLIR module. -/
def compile (p : Prog) : Except String String := do
  unless p.progDecs.any (fun d => match d with | .entry .. => true | _ => false) do
    throw "MLIR: program has no entry point"
  let action : M Unit := p.progDecs.forM fun d => match d with
    | .defE (.funE v ps _ e ..) => do
      for p in ps.toList do
        unless (← liftM (typeOf (patType p))).dims.isEmpty do
          throw "MLIR: user functions require scalar parameters"
      unless (← ty e).dims.isEmpty do throw "MLIR: user functions require scalar results"
      function (symbol v) ps.toList e
    | .entry v ps _ e .. => function ("@" ++ (Lean.Json.str v.varName).compress) ps e
    | .defE b => binding b
  let (_,s) ← action.run {}
  unless s.lines.isEmpty do throw "MLIR: top-level value bindings are not supported"
  return "module {\n" ++ String.intercalate "\n" s.functions ++ "\n}\n"
end Remora.MLIR
