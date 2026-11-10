import Std
namespace Remora
abbrev Tag := Nat
structure VName where
  varName : String
  varTag : Tag
  deriving Repr, Inhabited, Ord
instance : BEq VName where
  beq a b := a.varTag == b.varTag
instance : ToString VName where
  toString v := s!"{v.varName}_{v.varTag}"
def initTag : Tag := 0
def nextTag (t : Tag) : Tag := t + 1
class MonadVName (m : Type → Type) where
  getVarTag : m Tag
  putVarTag : Tag → m Unit
def newTag [Monad m] [MonadVName m] : m Tag := do
  let t ← MonadVName.getVarTag
  MonadVName.putVarTag (t + 1)
  pure t
def newVName [Monad m] [MonadVName m] (s : String) : m VName := do
  pure ⟨s, ← newTag⟩
end Remora
