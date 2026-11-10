import Remora.ISpace
import Remora.VName

/-! Syntax translated from Remora's `Syntax.hs`. -/
namespace Remora

structure SourcePos where
  fileName : String := "<no location>"
  line : Nat := 1
  column : Nat := 1
  offset : Nat := 0
  deriving Repr, BEq, Ord, Inhabited

instance : ToString SourcePos := ⟨fun p => s!"{p.fileName}:{p.line}:{p.column}"⟩

structure NEList (α : Type) where
  head : α
  tail : List α := []
  deriving Repr, BEq, Ord, Inhabited

namespace NEList
def toList (xs : NEList α) : List α := xs.head :: xs.tail
def ofList? : List α → Option (NEList α)
  | [] => none
  | x :: xs => some ⟨x, xs⟩
def singleton (x : α) : NEList α := ⟨x, []⟩
def map (f : α → β) (xs : NEList α) : NEList β := ⟨f xs.head, xs.tail.map f⟩
def mapM [Monad m] (f : α → m β) (xs : NEList α) : m (NEList β) := do
  return ⟨← f xs.head, ← xs.tail.mapM f⟩
def length (xs : NEList α) : Nat := xs.tail.length + 1
def append (xs ys : NEList α) : NEList α := ⟨xs.head, xs.tail ++ ys.toList⟩
instance : Functor NEList where
  map := map
end NEList

inductive NoInfo (α : Type) where
  | mk
  deriving Repr, BEq, Inhabited
structure Info (α : Type) where
  value : α
  deriving Repr, BEq, Ord, Inhabited

inductive TypeParamExp (v : Type) where
  | atom : v → TypeParamExp v
  | array : v → TypeParamExp v
  deriving Repr, BEq, Ord, Inhabited
inductive TypeParam (v : Type) where
  | atom : v → TypeParam v
  deriving Repr, BEq, Ord, Inhabited

def unTypeParam : TypeParam v → v | .atom x => x

def unTypeParamExp : TypeParamExp v → v | .atom x | .array x => x

inductive TypeExp (v : Type) where
  | atomVar : v → SourcePos → TypeExp v
  | arrayVar : v → SourcePos → TypeExp v
  | bool : SourcePos → TypeExp v
  | int : SourcePos → TypeExp v
  | float : SourcePos → TypeExp v
  | array : TypeExp v → Shape v → SourcePos → TypeExp v
  | arrow : TypeExp v → TypeExp v → SourcePos → TypeExp v
  | forallE : NEList (TypeParamExp v) → TypeExp v → SourcePos → TypeExp v
  | pi : NEList (ISpaceParam v) → TypeExp v → SourcePos → TypeExp v
  | sigma : NEList (ISpaceParam v) → TypeExp v → SourcePos → TypeExp v
  | record : NEList (String × TypeExp v) → SourcePos → TypeExp v
  deriving Repr, BEq, Inhabited

mutual
inductive AtomType (v : Type) where
  | var : v → AtomType v
  | bool : AtomType v
  | int : AtomType v
  | float : AtomType v
  | arrow : ArrayType v → ArrayType v → AtomType v
  | forallE : TypeParam v → ArrayType v → AtomType v
  | pi : ISpaceParam v → ArrayType v → AtomType v
  | sigma : ISpaceParam v → ArrayType v → AtomType v
  | record : NEList (String × ArrayType v) → AtomType v
  deriving Repr, BEq, Inhabited
inductive ArrayType (v : Type) where
  | mk : AtomType v → Shape v → ArrayType v
  deriving Repr, BEq, Inhabited
end

def ArrayType.atom : ArrayType v → AtomType v | .mk t _ => t
def ArrayType.shape : ArrayType v → Shape v | .mk _ s => s
inductive Ty (v : Type) where
  | atom : AtomType v → Ty v
  | array : ArrayType v → Ty v
  deriving Repr, BEq, Inhabited

def fromAtomType : Ty v → Option (AtomType v) | .atom t => some t | _ => none
def fromArrayType : Ty v → Option (ArrayType v) | .array t => some t | _ => none
def mkScalarArrayType (t : AtomType v) : ArrayType v := .mk t (.concat [])

def nestedType (con : α → ArrayType v → AtomType v) (xs : NEList α)
    (r : ArrayType v) : AtomType v :=
  (xs.toList.foldr (fun x t => mkScalarArrayType (con x t)) r).atom

def arrowType {v : Type} := @nestedType (ArrayType v) v AtomType.arrow
def forallType {v : Type} := @nestedType (TypeParam v) v AtomType.forallE
def piType {v : Type} := @nestedType (ISpaceParam v) v AtomType.pi
def sigmaType {v : Type} := @nestedType (ISpaceParam v) v AtomType.sigma

def elemType : Ty v → AtomType v | .atom t => t | .array t => t.atom

inductive Base where
  | boolVal : Bool → Base
  | intVal : Int64 → Base
  | floatVal : Float32 → Base
  deriving Repr, BEq, Inhabited

inductive PatBase (te f : Type → Type) (v : Type) where
  | id : v → te v → f (ArrayType v) → SourcePos → PatBase te f v

def patVar : PatBase te f v → v | .id v _ _ _ => v
def unpackPat : PatBase te f v → v × te v | .id v t _ _ => (v, t)

mutual
inductive AtomBase (te tp f : Type → Type) (v : Type) where
  | base : Base → f (AtomType v) → SourcePos → AtomBase te tp f v
  | lambda : PatBase te f v → ExpBase te tp f v → f (AtomType v) → SourcePos → AtomBase te tp f v
  | tLambda : tp v → ExpBase te tp f v → f (AtomType v) → SourcePos → AtomBase te tp f v
  | iLambda : ISpaceParam v → ExpBase te tp f v → f (AtomType v) → SourcePos → AtomBase te tp f v
  | box : ISpace v → ExpBase te tp f v → Option (te v) → f (AtomType v) → SourcePos → AtomBase te tp f v
inductive BindBase (te tp f : Type → Type) (v : Type) where
  | val : v → Option (te v) → ExpBase te tp f v → SourcePos → BindBase te tp f v
  | typeE : tp v → te v → f (Ty v) → SourcePos → BindBase te tp f v
  | ispace : ISpaceParam v → ISpace v → SourcePos → BindBase te tp f v
  | funE : v → NEList (PatBase te f v) → Option (te v) → ExpBase te tp f v → f (AtomType v) → SourcePos → BindBase te tp f v
  | tFun : v → NEList (tp v) → Option (te v) → ExpBase te tp f v → f (AtomType v) → SourcePos → BindBase te tp f v
  | iFun : v → NEList (ISpaceParam v) → Option (te v) → ExpBase te tp f v → f (AtomType v) → SourcePos → BindBase te tp f v
inductive ExpBase (te tp f : Type → Type) (v : Type) where
  | var : v → f (ArrayType v) → SourcePos → ExpBase te tp f v
  | array : List Int → NEList (AtomBase te tp f v) → f (ArrayType v) → SourcePos → ExpBase te tp f v
  | emptyArray : List Int → te v → f (ArrayType v) → SourcePos → ExpBase te tp f v
  | frame : List Int → NEList (ExpBase te tp f v) → f (ArrayType v) → SourcePos → ExpBase te tp f v
  | emptyFrame : List Int → te v → f (ArrayType v) → SourcePos → ExpBase te tp f v
  | app : ExpBase te tp f v → ExpBase te tp f v → f (ArrayType v × Shape v) → SourcePos → ExpBase te tp f v
  | tApp : ExpBase te tp f v → te v → f (ArrayType v) → SourcePos → ExpBase te tp f v
  | iApp : ExpBase te tp f v → ISpace v → f (ArrayType v) → SourcePos → ExpBase te tp f v
  | unbox : ISpaceParam v → v → ExpBase te tp f v → ExpBase te tp f v → f (ArrayType v) → SourcePos → ExpBase te tp f v
  | letE : NEList (BindBase te tp f v) → ExpBase te tp f v → f (ArrayType v) → SourcePos → ExpBase te tp f v
  | structE : NEList (String × Shape v × ExpBase te tp f v) → f (ArrayType v × Shape v) → SourcePos → ExpBase te tp f v
  | fieldProj : ExpBase te tp f v → String → f (ArrayType v) → SourcePos → ExpBase te tp f v
end

inductive DeclBase (te tp f : Type → Type) (v : Type) where
  | defE : BindBase te tp f v → DeclBase te tp f v
  | entry : v → List (PatBase te f v) → Option (te v) → ExpBase te tp f v → f (AtomType v) → SourcePos → DeclBase te tp f v
structure ProgBase (te tp f : Type → Type) (v : Type) where
  progDecs : List (DeclBase te tp f v)
structure Import where
  fileName : System.FilePath
  pos : SourcePos
  deriving Repr, BEq, Inhabited

abbrev UncheckedPat := PatBase TypeExp NoInfo String
abbrev UncheckedAtom := AtomBase TypeExp TypeParamExp NoInfo String
abbrev UncheckedBind := BindBase TypeExp TypeParamExp NoInfo String
abbrev UncheckedExp := ExpBase TypeExp TypeParamExp NoInfo String
abbrev UncheckedDecl := DeclBase TypeExp TypeParamExp NoInfo String
abbrev UncheckedProg := ProgBase TypeExp TypeParamExp NoInfo String
abbrev UniquePat := PatBase TypeExp NoInfo VName
abbrev UniqueAtom := AtomBase TypeExp TypeParam NoInfo VName
abbrev UniqueBind := BindBase TypeExp TypeParam NoInfo VName
abbrev UniqueExp := ExpBase TypeExp TypeParam NoInfo VName
abbrev UniqueDecl := DeclBase TypeExp TypeParam NoInfo VName
abbrev UniqueProg := ProgBase TypeExp TypeParam NoInfo VName
abbrev Pat := PatBase Ty Info VName
abbrev Atom := AtomBase Ty TypeParam Info VName
abbrev Bind := BindBase Ty TypeParam Info VName
abbrev Exp := ExpBase Ty TypeParam Info VName
abbrev Decl := DeclBase Ty TypeParam Info VName
abbrev Prog := ProgBase Ty TypeParam Info VName

class HasSrcPos (α : Type) where
  posOf : α → SourcePos
export HasSrcPos (posOf)
instance : HasSrcPos (PatBase te f v) := ⟨fun | .id _ _ _ p => p⟩
instance : HasSrcPos (AtomBase te tp f v) := ⟨fun
  | .base _ _ p | .lambda _ _ _ p | .tLambda _ _ _ p | .iLambda _ _ _ p | .box _ _ _ _ p => p⟩
instance : HasSrcPos (ExpBase te tp f v) := ⟨fun
  | .var _ _ p | .array _ _ _ p | .emptyArray _ _ _ p | .frame _ _ _ p
  | .emptyFrame _ _ _ p | .app _ _ _ p | .tApp _ _ _ p | .iApp _ _ _ p
  | .unbox _ _ _ _ _ p | .letE _ _ _ p | .structE _ _ p | .fieldProj _ _ _ p => p⟩
instance : HasSrcPos (BindBase te tp f v) := ⟨fun
  | .val _ _ _ p | .typeE _ _ _ p | .ispace _ _ p | .funE _ _ _ _ _ p
  | .tFun _ _ _ _ _ p | .iFun _ _ _ _ _ p => p⟩
instance : HasSrcPos (TypeExp v) := ⟨fun
  | .atomVar _ p | .arrayVar _ p | .bool p | .int p | .float p | .array _ _ p
  | .arrow _ _ p | .forallE _ _ p | .pi _ _ p | .sigma _ _ p | .record _ p => p⟩

def bindName : BindBase te tp f v → Option v
  | .val v _ _ _ | .funE v _ _ _ _ _ | .tFun v _ _ _ _ _ | .iFun v _ _ _ _ _ => some v
  | _ => none

def declName : DeclBase te tp f v → Option v
  | .defE b => bindName b
  | .entry v _ _ _ _ _ => some v

def mkScalar (a : AtomBase te tp NoInfo v) : ExpBase te tp NoInfo v :=
  .array [] (.singleton a) .mk (posOf a)
def asScalar : ExpBase te tp f v → Option (AtomBase te tp f v)
  | .array [] ⟨a, []⟩ _ _ => some a
  | _ => none
def arrayElems : ExpBase te tp f v → Option (NEList (AtomBase te tp f v))
  | .array _ as _ _ => some as
  | _ => none
def frameElems : ExpBase te tp f v → Option (NEList (ExpBase te tp f v))
  | .frame _ es _ _ => some es
  | _ => none
partial def unfoldApp (e : ExpBase te tp f v) : ExpBase te tp f v × List (ExpBase te tp f v) :=
  go e []
where
  go (e : ExpBase te tp f v) (args : List (ExpBase te tp f v)) := match e with
    | .app fn arg _ _ => go fn (arg :: args)
    | _ => (e, args)

partial def flattenExp : ExpBase te tp f v → ExpBase te tp f v
  | .frame shape es t pos =>
    let es' := es.map flattenExp
    match es'.head with
    | .frame inner _ _ _ =>
      match es'.toList.mapM frameElems with
      | some (xs :: xss) => .frame (shape ++ inner) (xss.foldl NEList.append xs) t pos
      | _ => .frame shape es' t pos
    | .array inner _ _ _ =>
      match es'.toList.mapM arrayElems with
      | some (xs :: xss) => .array (shape ++ inner) (xss.foldl NEList.append xs) t pos
      | _ => .frame shape es' t pos
    | _ => .frame shape es' t pos
  | e => e

def arrayifyType : Ty v → Ty v
  | .atom t => .array (mkScalarArrayType t)
  | t => t

def arrayTypeView (t : Ty v) (f : AtomType v × Shape v → α) : α :=
  match t with
  | .array t => f (t.atom, t.shape)
  | .atom t => f (t, .concat [])

end Remora
