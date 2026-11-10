import Remora.Syntax

namespace Remora.Substitute

/-- Capture-free substitutions; the preceding uniquification pass makes binders distinct. -/
structure Subst where
  names : VName → VName := id
  terms : VName → Option Exp := fun _ => none
  atoms : VName → Option (AtomType VName) := fun _ => none
  dims : VName → Option (Dim VName) := fun _ => none
  shapes : VName → Option (Shape VName) := fun _ => none

partial def dim (s : Subst) : Dim VName → Dim VName
  | .var v => (s.dims v).getD (.var (s.names v))
  | .lit n => .lit n
  | .add ds => .add (ds.map (dim s))
  | .mul ds => .mul (ds.map (dim s))
  | .sub ds => .sub (ds.map (dim s))

partial def shape (s : Subst) : Shape VName → Shape VName
  | .var v => (s.shapes v).getD (.var (s.names v))
  | .dim d => .dim (dim s d)
  | .concat ss => .concat (ss.map (shape s))

def ispace (s : Subst) : ISpace VName → ISpace VName
  | .dim d => .dim (dim s d)
  | .shape sh => .shape (shape s sh)

def renameTParam (s : Subst) : TypeParam VName → TypeParam VName
  | .atom v => .atom (s.names v)
def renameIParam (s : Subst) : ISpaceParam VName → ISpaceParam VName
  | .dim v => .dim (s.names v)
  | .shape v => .shape (s.names v)

mutual
  partial def atomType (s : Subst) : AtomType VName → AtomType VName
    | .var v => (s.atoms v).getD (.var (s.names v))
    | .bool => .bool
    | .int => .int
    | .float => .float
    | .arrow a b => .arrow (arrayType s a) (arrayType s b)
    | .forallE p t => .forallE (renameTParam s p) (arrayType s t)
    | .pi p t => .pi (renameIParam s p) (arrayType s t)
    | .sigma p t => .sigma (renameIParam s p) (arrayType s t)
    | .record fs => .record ⟨(fs.head.1, arrayType s fs.head.2),
        fs.tail.map fun (f, t) => (f, arrayType s t)⟩
  partial def arrayType (s : Subst) (t : ArrayType VName) : ArrayType VName :=
    ⟨atomType s t.atom, shape s t.shape⟩
end

def ty (s : Subst) : Ty VName → Ty VName
  | .atom t => .atom (atomType s t)
  | .array t => .array (arrayType s t)

def atomVar (v : VName) (t : AtomType VName) : Subst :=
  { atoms := fun w => if w == v then some t else none }

def ispaceVar (v : VName) : ISpace VName → Subst
  | .dim d => { dims := fun w => if w == v then some d else none }
  | .shape s => { shapes := fun w => if w == v then some s else none }

def pat (s : Subst) : Pat → Pat
  | .id v t a p => .id (s.names v) (ty s t) ⟨arrayType s a.value⟩ p

mutual
  partial def exp (s : Subst) : Exp → Exp
    | .var v t p => (s.terms v).getD (.var (s.names v) ⟨arrayType s t.value⟩ p)
    | .array ns xs t p => .array ns (xs.map (atom s)) ⟨arrayType s t.value⟩ p
    | .emptyArray ns te t p => .emptyArray ns (ty s te) ⟨arrayType s t.value⟩ p
    | .frame ns xs t p => .frame ns (xs.map (exp s)) ⟨arrayType s t.value⟩ p
    | .emptyFrame ns te t p => .emptyFrame ns (ty s te) ⟨arrayType s t.value⟩ p
    | .app f a t p => .app (exp s f) (exp s a)
        ⟨(arrayType s t.value.1, shape s t.value.2)⟩ p
    | .tApp f te t p => .tApp (exp s f) (ty s te) ⟨arrayType s t.value⟩ p
    | .iApp f i t p => .iApp (exp s f) (ispace s i) ⟨arrayType s t.value⟩ p
    | .unbox ip v b e t p => .unbox (renameIParam s ip) (s.names v)
        (exp s b) (exp s e) ⟨arrayType s t.value⟩ p
    | .letE bs e t p => .letE (bs.map (bind s)) (exp s e) ⟨arrayType s t.value⟩ p
    | .structE fs t p => .structE (fs.map fun (f, sh, e) => (f, shape s sh, exp s e))
        ⟨(arrayType s t.value.1, shape s t.value.2)⟩ p
    | .fieldProj e f t p => .fieldProj (exp s e) f ⟨arrayType s t.value⟩ p
  partial def atom (s : Subst) : Atom → Atom
    | .base b t p => .base b ⟨atomType s t.value⟩ p
    | .lambda q e t p => .lambda (pat s q) (exp s e) ⟨atomType s t.value⟩ p
    | .tLambda q e t p => .tLambda (renameTParam s q) (exp s e) ⟨atomType s t.value⟩ p
    | .iLambda q e t p => .iLambda (renameIParam s q) (exp s e) ⟨atomType s t.value⟩ p
    | .box i e te t p => .box (ispace s i) (exp s e) (te.map (ty s))
        ⟨atomType s t.value⟩ p
  partial def bind (s : Subst) : Bind → Bind
    | .val v te e p => .val (s.names v) (te.map (ty s)) (exp s e) p
    | .typeE q te t p => .typeE (renameTParam s q) (ty s te) ⟨ty s t.value⟩ p
    | .ispace q i p => .ispace (renameIParam s q) (ispace s i) p
    | .funE v qs te e t p => .funE (s.names v) (qs.map (pat s)) (te.map (ty s))
        (exp s e) ⟨atomType s t.value⟩ p
    | .tFun v qs te e t p => .tFun (s.names v) (qs.map (renameTParam s)) (te.map (ty s))
        (exp s e) ⟨atomType s t.value⟩ p
    | .iFun v qs te e t p => .iFun (s.names v) (qs.map (renameIParam s)) (te.map (ty s))
        (exp s e) ⟨atomType s t.value⟩ p
end

def decl (s : Subst) : Decl → Decl
  | .defE b => .defE (bind s b)
  | .entry v ps te e t p => .entry (s.names v) (ps.map (pat s)) (te.map (ty s))
      (exp s e) ⟨atomType s t.value⟩ p

def termVar (v : VName) (e : Exp) : Subst :=
  { terms := fun w => if w == v then some e else none }

end Remora.Substitute
