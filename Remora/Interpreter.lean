import Remora.Interpreter.Intrinsics
namespace Remora.Interpreter
abbrev EvalM := ExceptT String IO
structure Env where
  values : List (VName × Val) := []
  indices : List (VName × Index) := []
structure Config where
  readExpression : String → IO (Except String Exp) := fun _ => pure (.error "read-file: no frontend configured")
private def liftE (e : Except String α) : EvalM α := ExceptT.mk (pure e)
private def patName : Pat → VName | .id v _ _ _ => v
private def expType : Exp → ArrayType VName
  | .var _ t _ | .array _ _ t _ | .emptyArray _ _ t _ | .frame _ _ t _
  | .emptyFrame _ _ t _ | .tApp _ _ t _ | .iApp _ _ t _ | .unbox _ _ _ _ t _
  | .letE _ _ t _ | .fieldProj _ _ t _ => t.value
  | .app _ _ t _ | .structE _ t _ => t.value.1
private def findIndex (env : Env) (v : VName) : Except String Index :=
  match env.indices.find? (fun p => p.1 == v) with
  | some (_,i) => pure i
  | none => throw s!"unknown index: {v}"
private def intDim (env : Env) (d : Dim VName) : EvalM Int := do
  let n ← liftE (dimToInt (fun v => do
    match ← findIndex env v with
    | .inl n => pure n
    | _ => throw "expected dimension") d)
  if n < 0 then throw "negative dimension"
  pure n
private def intShape (env : Env) (s : Shape VName) : EvalM (List Int) :=
  shapeToInts (intDim env) (fun v => do
    match ← liftE (findIndex env v) with
    | .inr s => pure s
    | _ => throw "expected shape") s
private def intIndex (env : Env) : ISpace VName → EvalM Index
  | .dim d => return .inl (← intDim env d)
  | .shape s => return .inr (← intShape env s)
private def getDim : Index → Except String Int
  | .inl n => pure n
  | _ => throw "expected dimension argument"
private def getShape : Index → Except String (List Int)
  | .inr s => pure s
  | _ => throw "expected shape argument"
private def lookupVal (env : Env) (v : VName) : EvalM Val :=
  match env.values.find? (fun p => p.1 == v) with
  | some (_,x) => pure x
  | none =>
    if intrinsicTypes.any (fun p => p.1 == v.varName) then pure (builtin v.varName)
    else throw s!"unknown variable: {v}"
private def nth (xs : List α) (n : Nat) : EvalM α :=
  match xs[n]? with
  | some x => pure x
  | none => throw "index out of bounds"
private def asArrayM (v : Val) : EvalM (List Int × List Val) := liftE (asArray v)

mutual
partial def eval (cfg : Config) (env : Env) : Exp → EvalM Val
  | .var v _ _ => lookupVal env v
  | .array s atoms _ _ => return .array s (← atoms.toList.mapM (evalAtom cfg env))
  | .frame s es _ _ => return .array s (← es.toList.mapM (eval cfg env))
  | .emptyArray s _ _ _ | .emptyFrame s _ _ _ => pure (.array s [])
  | .app f x t _ => do
    let pf ← intShape env t.value.2
    let fn ← eval cfg env f
    let arg ← eval cfg env x
    let .arrow param _ := (expType f).atom | throw "application of a non-function type"
    let cell ← intShape env param.shape
    let (fs,vs) ← asArrayM fn
    let (ars,avs) ← asArrayM arg
    let af ← liftE (suffixSubtract ars cell)
    let frameCount := product pf
    let fcount := product fs
    let acount := product af
    if frameCount == 0 then return .array pf []
    if fcount == 0 || acount == 0 then throw "cannot extend an empty frame"
    let functions := vs.flatMap (fun f => List.replicate (frameCount/fcount) f)
    let arguments := (split acount (product cell) avs).flatMap
      (fun xs => List.replicate (frameCount/acount) xs)
    let out ← (functions.zip arguments).mapM (fun (f,xs) => apply cfg env f (.array cell xs))
    liftE (collapse (.array pf out))
  | .tApp e _ _ _ => do applyType cfg env (← eval cfg env e)
  | .iApp e i _ _ => do applyIndex cfg env (← eval cfg env e) (← intIndex env i)
  | .unbox p v box body _ _ => do
    let (s,vs) ← asArrayM (← eval cfg env box)
    let out ← vs.mapM fun x => do
      match x with
      | .box [i] v' => eval cfg ⟨(v,v')::env.values, (unISpaceParam p,i)::env.indices⟩ body
      | _ => throw "unbox: expected a box"
    liftE (collapse (.array s out))
  | .letE bs body _ _ => do
    let env ← bs.toList.foldlM (bind cfg) env
    eval cfg env body
  | .structE fs t _ => do
    let frame ← intShape env t.value.2
    let count := product frame
    let fields ← fs.toList.mapM fun (name,cell,e) => do
      let c ← intShape env cell
      let (s,vs) ← asArrayM (← eval cfg env e)
      let outer ← liftE (suffixSubtract s c)
      let n := product outer
      if count == 0 then return (name,[])
      if n == 0 then throw "struct: cannot extend empty field"
      let cells := (split n (product c) vs).flatMap
        (fun xs => List.replicate (count/n) (.array c xs))
      pure (name,cells)
    let records ← (List.range count).mapM fun i => do
      let row ← fields.mapM fun (name,vs) => return (name,← nth vs i)
      pure (.record row)
    pure (.array frame records)
  | .fieldProj e name _ _ => do
    let (s,vs) ← asArrayM (← eval cfg env e)
    let fields ← vs.mapM fun v => do
      let .record fs := v | throw "field projection: expected a record"
      match fs.find? (fun p => p.1 == name) with
      | some (_,x) => pure x
      | none => throw s!"unknown field: {name}"
    liftE (collapse (.array s fields))

partial def evalAtom (cfg : Config) (env : Env) : Atom → EvalM Val
  | .base b _ _ => pure (.base b)
  | .lambda p e _ _ => pure (.termFun [p] e env.values env.indices)
  | .tLambda p e _ _ => pure (.typeFun [p] e env.values env.indices)
  | .iLambda p e _ _ => pure (.indexFun [p] e env.values env.indices)
  | .box i e _ _ _ => return .box [← intIndex env i] (← eval cfg env e)

partial def bind (cfg : Config) (env : Env) : Bind → EvalM Env
  | .val v _ e _ => return {env with values := (v,← eval cfg env e)::env.values}
  | .funE v ps _ e _ _ => pure {env with values :=
      (v,.termFun ps.toList e env.values env.indices)::env.values}
  | .tFun v ps _ e _ _ => pure {env with values :=
      (v,.typeFun ps.toList e env.values env.indices)::env.values}
  | .iFun v ps _ e _ _ => pure {env with values :=
      (v,.indexFun ps.toList e env.values env.indices)::env.values}
  | .typeE .. | .ispace .. => pure env

partial def apply (cfg : Config) (env : Env) (f arg : Val) : EvalM Val := do
  match ← liftE (asScalar f) with
  | .var v => apply cfg env (← lookupVal env v) arg
  | .termFun (p::ps) body vs is =>
    let vs := (patName p,arg)::vs
    if ps.isEmpty then eval cfg ⟨vs,is⟩ body else pure (.termFun ps body vs is)
  | .builtin name 0 0 is args =>
    let args := args ++ [arg]
    if args.length == (intrinsicArity name).2.2 then runBuiltin cfg env name is args
    else pure (.builtin name 0 0 is args)
  | _ => throw "cannot apply non-function value"

partial def applyType (cfg : Config) (env : Env) (f : Val) : EvalM Val := do
  match f with
  | .array s vs => return .array s (← vs.mapM (applyType cfg env))
  | .var v => applyType cfg env (← lookupVal env v)
  | .typeFun (_::ps) body vs is =>
    if ps.isEmpty then eval cfg ⟨vs,is⟩ body else pure (.typeFun ps body vs is)
  | .builtin n (t+1) i is args =>
    if t == 0 && i == 0 && (intrinsicArity n).2.2 == args.length then
      runBuiltin cfg env n is args
    else pure (.builtin n t i is args)
  | _ => throw "type application: expected a type function"

partial def applyIndex (cfg : Config) (env : Env) (f : Val) (idx : Index) : EvalM Val := do
  match f with
  | .array s vs => return .array s (← vs.mapM (fun v => applyIndex cfg env v idx))
  | .indexFun (p::ps) body vs is =>
    let is := (unISpaceParam p,idx)::is
    if ps.isEmpty then eval cfg ⟨vs,is⟩ body else pure (.indexFun ps body vs is)
  | .builtin n 0 (i+1) is args =>
    let is := is ++ [idx]
    if i == 0 && (intrinsicArity n).2.2 == args.length then runBuiltin cfg env n is args
    else pure (.builtin n 0 i is args)
  | _ => throw "index application: expected an index function"

partial def runBuiltin (cfg : Config) (env : Env) (name : String)
    (indices : List Index) (args : List Val) : EvalM Val := do
  if let some op := UnOp.all.find? (fun op => op.name == name) then
    return ← liftE (evalUnary op (← nth args 0))
  if let some op := BinOp.all.find? (fun op => op.name == name) then
    return ← liftE (evalBinary op (← nth args 0) (← nth args 1))
  let arg := fun i => nth args i
  let shp : Nat → EvalM (List Int) := fun i => do liftE (getShape (← nth indices i))
  let dim : Nat → EvalM Int := fun i => do liftE (getDim (← nth indices i))
  match name with
  | "head" =>
    let (s,vs) ← asArrayM (← arg 0)
    let s := s.drop 1
    pure (.array s (vs.take (product s)))
  | "tail" =>
    let (s,vs) ← asArrayM (← arg 0)
    let d::ds := s | throw "tail: empty shape"
    pure (.array ((d-1)::ds) (vs.drop (product ds)))
  | "length" =>
    let (s,_) ← asArrayM (← arg 0)
    let d::_ := s | throw "length: empty shape"
    pure (scalarVal (.intVal (Int64.ofInt d)))
  | "append" =>
    let (_,xs) ← asArrayM (← arg 0)
    let (_,ys) ← asArrayM (← arg 1)
    pure (.array (((← dim 0)+(← dim 1))::(← shp 2)) (xs++ys))
  | "reverse" =>
    let (s,vs) ← asArrayM (← arg 0)
    let cell ← shp 1
    let frame ← liftE (suffixSubtract s cell)
    pure (.array s ((split (product frame) (product cell) vs).reverse.flatten))
  | "reduce" | "reduce/zero" | "fold" | "fold-right" =>
    let cell ← shp 1
    let zeroMode := name != "reduce"
    let (_,vs) ← asArrayM (← arg (if zeroMode then 2 else 1))
    let (_,arr) ← asArrayM (← arg (if zeroMode then 2 else 1))
    let dim0 ← dim 0
    let cells := split (if product cell == 0 then (dim0+
      (if name == "reduce" || name == "fold" || name == "fold-right" then 1 else 0)).toNat
      else arr.length / product cell) (product cell) vs
    let cells := cells.map (Val.array cell)
    let op ← arg 0
    let combine := fun a b => do apply cfg env (← apply cfg env op a) b
    if zeroMode then
      let zero ← arg 1
      if name == "fold-right" then
        cells.reverse.foldlM (fun acc x => combine x acc) zero
      else cells.foldlM combine zero
    else
      let c::cs := cells | throw "reduce: empty array"
      cs.foldlM combine c
  | "sum" =>
    let (_,vs) ← asArrayM (← arg 0)
    let ns ← vs.mapM (fun x => liftE (asInt x))
    pure (scalarVal (.intVal (ns.foldl (·+·) 0)))
  | "flatten" =>
    let (s,vs) ← asArrayM (← arg 0)
    let m::n::ss := s | throw "flatten: expected rank at least two"
    pure (.array ((m*n)::ss) vs)
  | "reshape" =>
    let old ← shp 0
    let new ← shp 1
    if product old != product new then throw "reshape: element counts differ"
    let (_,vs) ← asArrayM (← arg 0)
    pure (.array new vs)
  | "iota" =>
    let (_,vs) ← asArrayM (← arg 0)
    let shape ← vs.mapM (fun v => return (← liftE (asInt v)).toInt)
    if shape.any (· < 0) then throw "iota: negative dimension"
    pure (.box [.inr shape] (.array shape ((List.range (product shape)).map
      (fun n => .base (.intVal (Int64.ofNat n))))))
  | "iota/static" =>
    let shape ← shp 0
    pure (.array shape ((List.range (product shape)).map
      (fun n => .base (.intVal (Int64.ofNat n)))))
  | "transpose2d" =>
    let (s,vs) ← asArrayM (← arg 0)
    let [m,n] := s | throw "transpose2d: expected rank two"
    let rows ← (List.range n.toNat).mapM fun j =>
      (List.range m.toNat).mapM (fun i => nth vs (i*n.toNat+j))
    pure (.array [n,m] rows.flatten)
  | "index" =>
    let (_,vs) ← asArrayM (← arg 0)
    let idx ← liftE (asInt (← arg 1))
    if idx < 0 then throw "index out of bounds"
    nth vs idx.toInt.toNat
  | "index2d" =>
    let (s,vs) ← asArrayM (← arg 0)
    let [_,n] := s | throw "index2d: expected rank two"
    let (_,ij) ← asArrayM (← arg 1)
    let [i,j] := ij | throw "index2d: expected a pair"
    let i ← liftE (asInt i)
    let j ← liftE (asInt j)
    if i < 0 || j < 0 then throw "index out of bounds"
    nth vs (i.toInt.toNat*n.toNat+j.toInt.toNat)
  | "trace" =>
    let msg ← arg 0
    IO.eprintln (pretty msg)
    arg 1
  | "trace-file" =>
    let path ← liftE (valToString (← arg 0))
    let msg ← arg 1
    let _ ← (IO.FS.withFile path .append (fun h => h.putStrLn (pretty msg)) : IO Unit)
    arg 2
  | "read-file" =>
    let path ← liftE (valToString (← arg 0))
    let input ← IO.FS.readFile path
    let e ← ExceptT.mk (cfg.readExpression input)
    eval cfg {} e
  | "read-file-f32bin" =>
    let path ← liftE (valToString (← arg 0))
    let shape ← shp 1
    let bytes ← IO.FS.readBinFile path
    let count := product shape
    if bytes.size < count*4 then throw "read-file-f32bin: truncated file"
    let vs := (List.range count).map fun n =>
      let word := (List.range 4).foldl (fun w k =>
        w ||| (UInt32.ofNat bytes[n*4+k]!.toNat <<< UInt32.ofNat (8*k))) 0
      Val.base (.floatVal (Float32.ofBits word))
    pure (.array shape vs)
  | "reify-dim" => pure (scalarVal (.intVal (Int64.ofInt (← dim 0))))
  | "reify-shape" =>
    let ds ← shp 0
    pure (.box [.inl (Int.ofNat ds.length)] (.array [Int.ofNat ds.length]
      (ds.map (fun n => .base (.intVal (Int64.ofInt n))))))
  | "undefined" => throw "undefined"
  | _ => throw s!"unknown intrinsic: {name}"
end

def interpretExp (e : Exp) (cfg : Config := {}) : IO (Except String Val) :=
  (eval cfg {} e).run

private def interpretM (entry : String) (args : List Val) (prog : Prog)
    (cfg : Config) : EvalM Val := do
  let mut env : Env := {}
  for decl in prog.progDecs do
    match decl with
    | .defE b => env ← bind cfg env b
    | .entry v ps _ body _ _ =>
      let fn := Val.termFun ps body env.values env.indices
      env := {env with values := (v,fn)::env.values}
  let entries := prog.progDecs.filterMap fun d => match d with
    | .entry v ps _ body _ _ => if v.varName == entry then some (v,ps,body) else none
    | _ => none
  let [(v,ps,body)] := entries | throw s!"expected exactly one {entry} entry point"
  if ps.length != args.length then throw s!"entry {entry}: wrong number of arguments"
  if ps.isEmpty then eval cfg env body
  else
    let fn ← lookupVal env v
    args.foldlM (apply cfg env) fn

def interpret (entry : String) (args : List Val) (prog : Prog)
    (cfg : Config := {}) : IO (Except String Val) := (interpretM entry args prog cfg).run
end Remora.Interpreter
