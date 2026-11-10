import Remora.ISpace
namespace Remora.Symbolic
open Remora
inductive ShapeRel where
  | eq | prefix
  deriving BEq
private structure SMTEnv where
  names : List (String × Bool) := []
  constraints : List String := []
private def symName (v : String) (shape : Bool) : StateM SMTEnv String := do
  let key := (v,shape)
  let env ← get
  let i := env.names.idxOf key
  if i < env.names.length then return s!"v{i}"
  set {env with names := env.names ++ [key]}
  let n := s!"v{env.names.length}"
  modify fun e => {e with constraints := e.constraints ++ [if shape then
    s!"(forall ((i Int)) (=> (and (<= 0 i) (< i (seq.len {n}))) (>= (seq.nth {n} i) 0)))"
    else s!"(>= {n} 0)"]}
  return n
private def app (op : String) (xs : List String) := s!"({op} {String.intercalate " " xs})"
private partial def symDim [ToString v] : Dim v → StateM SMTEnv String
  | .var v => symName (toString v) false
  | .lit n => pure (if n < 0 then s!"(- {-n})" else toString n)
  | .add ds => if ds.isEmpty then pure "0" else return app "+" (← ds.mapM symDim)
  | .mul ds => if ds.isEmpty then pure "1" else return app "*" (← ds.mapM symDim)
  | .sub ds => if ds.isEmpty then pure "0" else return app "-" (← ds.mapM symDim)
private partial def symShape [ToString v] : Shape v → StateM SMTEnv String
  | .var v => symName (toString v) true
  | .dim d => do
    let d ← symDim d
    modify fun e => {e with constraints := e.constraints ++ [s!"(>= {d} 0)"]}
    return s!"(seq.unit {d})"
  | .concat ss => do
    let xs ← ss.mapM symShape
    match xs with
    | [] => return "(as seq.empty (Seq Int))"
    | [x] => return x
    | xs => return app "seq.++" xs
/-- Emit a validity query: unsatisfiability proves the requested relation. -/
def query [ToString v] (rel : ShapeRel) (s t : Shape v) : String := Id.run do
  let ((s,t), env) := (do let a ← symShape s; let b ← symShape t; pure (a,b)).run {}
  let decls := env.names.zipIdx |>.map fun ((_,shape),i) =>
    s!"(declare-fun v{i} () {if shape then "(Seq Int)" else "Int"})"
  let relation := if rel == .eq then s!"(= {s} {t})" else s!"(seq.prefixof {s} {t})"
  return String.intercalate "\n" (["(set-logic ALL)"] ++ decls ++
    env.constraints.map (fun c => s!"(assert {c})") ++ [s!"(assert (not {relation}))", "(check-sat)"])
def askShapes [Ord v] [BEq v] [ToString v] (rel : ShapeRel) (s t : Shape v) : IO (Except String Bool) := do
  if normShape s == normShape t then return .ok true
  let concrete (s : Shape v) := shapeToInts (dimToInt (fun _ => (none : Option Int))) (fun _ => none) s
  if let (some ds, some es) := (concrete s, concrete t) then
    return .ok (if rel == .eq then ds == es else ds.isPrefixOf es)
  try
    let r ← IO.Process.output {cmd := "z3", args := #["-in", "-smt2"]} (some (query rel s t))
    if r.exitCode != 0 then return .error s!"z3: {r.stderr}{r.stdout}"
    match r.stdout.trimAscii.toString with
    | "unsat" => return .ok true
    | "sat" | "unknown" => return .ok false
    | s => return .error s!"z3: unexpected response {s}"
  catch e => return .error s!"z3: {e}"
def equal [Ord v] [BEq v] [ToString v] (s t : Shape v) := askShapes .eq s t
def isPrefix [Ord v] [BEq v] [ToString v] (s t : Shape v) := askShapes .prefix s t
def maximumShape [Ord v] [BEq v] [ToString v] (ss : List (Shape v)) : IO (Except String (Option (Shape v))) := do
  let mut result : Shape v := .concat []
  for s in ss do
    match ← isPrefix s result with
    | .error e => return .error e
    | .ok true => pure ()
    | .ok false =>
      match ← isPrefix result s with
      | .error e => return .error e
      | .ok true => result := s
      | .ok false => return .ok none
  return .ok (some result)
end Remora.Symbolic
