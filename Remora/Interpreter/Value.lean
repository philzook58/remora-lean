import Remora.Syntax
namespace Remora.Interpreter
abbrev Index := Sum Int (List Int)
/-- Function values contain syntax and a captured environment, rather than negative callbacks. -/
inductive Val where
  | var : VName → Val
  | base : Base → Val
  | array : List Int → List Val → Val
  | box : List Index → Val → Val
  | termFun : List Pat → Exp → List (VName × Val) → List (VName × Index) → Val
  | typeFun : List (TypeParam VName) → Exp → List (VName × Val) → List (VName × Index) → Val
  | indexFun : List (ISpaceParam VName) → Exp → List (VName × Val) → List (VName × Index) → Val
  | builtin : String → Nat → Nat → List Index → List Val → Val
  | record : List (String × Val) → Val
  deriving Inhabited

def product (ns : List Int) : Nat := (ns.foldl (· * ·) 1).toNat

def split (n size : Nat) (xs : List α) : List (List α) :=
  match n with
  | 0 => []
  | n+1 => xs.take size :: split n size (xs.drop size)

partial def collapse : Val → Except String Val
  | .array s vs => do
    let vs ← vs.mapM collapse
    match vs with
    | .array t _ :: rest =>
      let rows ← rest.mapM fun v => match v with
        | .array t' xs => if t == t' then pure xs else throw "collapse: irregular array"
        | _ => throw "collapse: irregular array"
      let .array _ first := vs.head! | throw "collapse: invalid array"
      pure (.array (s ++ t) (first ++ rows.flatten))
    | _ => pure (.array s vs)
  | v => pure v

def valShapeOf : Val → List Int
  | .array s _ => s
  | _ => []

def arrayifyVal : Val → Val
  | v@(.array ..) => v
  | v => .array [] [v]

def asArray (v : Val) : Except String (List Int × List Val) := do
  match ← collapse v with
  | .array s vs => pure (s,vs)
  | v => pure ([],[v])

partial def asBase : Val → Except String Base
  | .base b => pure b
  | .array [] [v] => asBase v
  | _ => throw "expected a base value"

def asInt (v : Val) : Except String Int64 := do
  match ← asBase v with
  | .intVal n => pure n
  | _ => throw "expected an integer"

def asFloat (v : Val) : Except String Float32 := do
  match ← asBase v with
  | .floatVal x => pure x
  | _ => throw "expected a float"

def asBool (v : Val) : Except String Bool := do
  match ← asBase v with
  | .boolVal b => pure b
  | _ => throw "expected a boolean"

def asScalar (v : Val) : Except String Val := do
  match (← asArray v).2 with
  | [x] => pure x
  | _ => throw "expected a scalar"

def suffixSubtract (s t : List Int) : Except String (List Int) :=
  if t.length ≤ s.length && s.drop (s.length-t.length) == t then
    pure (s.take (s.length-t.length))
  else throw "shape is not a suffix"

partial def equal : Val → Val → Bool
  | .array [] [v], w => equal v w
  | v, .array [] [w] => equal v w
  | .base a, .base b => a == b
  | .var a, .var b => a == b
  | .array s xs, .array t ys => s == t && xs.length == ys.length &&
      (xs.zip ys).all (fun (x,y) => equal x y)
  | .box is v, .box js w => is == js && equal v w
  | .record fs, .record gs => fs.length == gs.length &&
      (fs.zip gs).all (fun ((f,v),(g,w)) => f == g && equal v w)
  | _, _ => false
instance : BEq Val := ⟨equal⟩

partial def pretty : Val → String
  | .base (.boolVal b) => if b then "#t" else "#f"
  | .base (.intVal n) => toString n
  | .base (.floatVal x) => toString x
  | .var v => toString v
  | .array [] [v] => pretty v
  | .array [_] vs => "[" ++ String.intercalate " " (vs.map pretty) ++ "]"
  | .array (d::ds) vs => "[" ++ String.intercalate " "
      ((split d.toNat (product ds) vs).map fun xs => pretty (.array ds xs)) ++ "]"
  | .array [] _ => "[]"
  | .box is v => "(box " ++ String.intercalate " " (is.map fun i => match i with
      | .inl n => toString n
      | .inr ns => "[" ++ String.intercalate " " (ns.map toString) ++ "]") ++ " " ++ pretty v ++ ")"
  | .termFun .. => "#<fun>"
  | .typeFun .. => "#<tfun>"
  | .indexFun .. => "#<ifun>"
  | .builtin .. => "#<intrinsic>"
  | .record fs => "{" ++ String.intercalate ", " (fs.map fun (f,v) => f ++ " = " ++ pretty v) ++ "}"
instance : ToString Val := ⟨pretty⟩

def valToString (v : Val) : Except String String := do
  let (_,vs) ← asArray v
  let cs ← vs.mapM fun x => do
    let n ← asInt x
    pure (Char.ofNat n.toInt.toNat)
  pure (String.ofList cs)
end Remora.Interpreter
