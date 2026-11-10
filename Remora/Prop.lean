import Remora.Substitute
import Remora.Symbolic

namespace Remora

def arrayTypeOf : Exp → ArrayType VName
  | .var _ t _ | .array _ _ t _ | .emptyArray _ _ t _ | .frame _ _ t _
  | .emptyFrame _ _ t _ | .tApp _ _ t _ | .iApp _ _ t _ | .unbox _ _ _ _ t _
  | .letE _ _ t _ | .fieldProj _ _ t _ => t.value
  | .app _ _ t _ | .structE _ t _ => t.value.1

def scalarTypeOf : Atom → AtomType VName
  | .base _ t _ | .lambda _ _ t _ | .tLambda _ _ t _ | .iLambda _ _ t _
  | .box _ _ _ t _ => t.value

def patType : Pat → ArrayType VName | .id _ _ t _ => t.value

def baseTypeOf : Base → AtomType v
  | .boolVal _ => .bool
  | .intVal _ => .int
  | .floatVal _ => .float

def asArrayType : Ty v → ArrayType v
  | .atom t => mkScalarArrayType t
  | .array t => t

def arrayOf (t : Ty v) (s : Shape v) : ArrayType v :=
  match t with
  | .atom a => ⟨a, s⟩
  | .array t => ⟨t.atom, .concat [s, t.shape]⟩

partial def unfoldArrow (t : ArrayType v) : List (ArrayType v) × ArrayType v :=
  match t.atom with
  | .arrow p r =>
    match t.shape with
    | .concat [] => let (ps, r) := unfoldArrow r; (p :: ps, r)
    | _ => ([], t)
  | _ => ([], t)

namespace Properties
abbrev M := ExceptT String IO

mutual
  partial def equivalentAtom (a b : AtomType VName) : M Bool := do
    match a, b with
    | .arrow p r, .arrow q s =>
      return (← equivalentArray p q) && (← equivalentArray r s)
    | .forallE (.atom p) r, .forallE (.atom q) s =>
      equivalentArray (Substitute.arrayType (Substitute.atomVar p (.var q)) r) s
    | .pi p r, .pi q s | .sigma p r, .sigma q s =>
      let subst ← match p, q with
        | .dim p, .dim q => pure (Substitute.ispaceVar p (.dim (.var q)))
        | .shape p, .shape q => pure (Substitute.ispaceVar p (.shape (.var q)))
        | _, _ => return false
      equivalentArray (Substitute.arrayType subst r) s
    | .record fs, .record gs =>
      if fs.length != gs.length then return false
      for (f, t) in fs.toList do
        let some u := gs.toList.lookup f | return false
        unless ← equivalentArray t u do return false
      return true
    | _, _ => return a == b
  partial def equivalentArray (a b : ArrayType VName) : M Bool := do
    unless ← equivalentAtom a.atom b.atom do return false
    ExceptT.mk (Symbolic.equal a.shape b.shape)
end

/-- Remove a cell suffix from an argument shape, using symbolic equality at each step. -/
partial def subtractShape (s t : Shape VName) : M (Option (Shape VName)) := do
  let s := normShape s
  let t := normShape t
  if ← ExceptT.mk (Symbolic.equal s t) then return some (.concat [])
  match s, t with
  | _, .concat [] => return some s
  | .concat [], _ => return none
  | .concat ss, .concat ts =>
    match ss.reverse, ts.reverse with
    | x :: xs, y :: ys =>
      if ← ExceptT.mk (Symbolic.equal x y) then
        subtractShape (.concat xs.reverse) (.concat ys.reverse)
      else return none
    | _, _ => return none
  | .concat ss, _ =>
    match ss.reverse with
    | x :: xs =>
      if ← ExceptT.mk (Symbolic.equal x t) then return some (.concat xs.reverse)
      else return none
    | _ => return none
  | _, _ => return none

end Properties
end Remora
