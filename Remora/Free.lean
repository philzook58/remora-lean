import Remora.Prop
namespace Remora.Free
structure FreeVars where
  terms : List (VName × ArrayType VName) := []
  atoms : List VName := []
  dims : List VName := []
  shapes : List VName := []
  deriving Inhabited

def union (a b : FreeVars) : FreeVars :=
  ⟨(a.terms ++ b.terms).eraseDups, (a.atoms ++ b.atoms).eraseDups,
    (a.dims ++ b.dims).eraseDups, (a.shapes ++ b.shapes).eraseDups⟩
def unions (xs : List FreeVars) : FreeVars := xs.foldl union {}
def withoutTerm (v : VName) (s : FreeVars) : FreeVars :=
  {s with terms := s.terms.filter (fun x => x.1 != v)}
def withoutType (v : VName) (s : FreeVars) : FreeVars := {s with atoms := s.atoms.erase v}
def withoutIndex (p : ISpaceParam VName) (s : FreeVars) : FreeVars :=
  match p with
  | .dim v => {s with dims := s.dims.erase v}
  | .shape v => {s with shapes := s.shapes.erase v}
partial def dim : Dim VName → FreeVars
  | .var v => {dims := [v]}
  | .lit _ => {}
  | .add ds | .mul ds | .sub ds => unions (ds.map dim)
partial def shape : Shape VName → FreeVars
  | .var v => {shapes := [v]}
  | .dim d => dim d
  | .concat ss => unions (ss.map shape)
def ispace : ISpace VName → FreeVars
  | .dim d => dim d
  | .shape s => shape s
mutual
  partial def atomType : AtomType VName → FreeVars
    | .var v => {atoms := [v]}
    | .bool | .int | .float => {}
    | .arrow a b => union (arrayType a) (arrayType b)
    | .forallE (.atom v) t => withoutType v (arrayType t)
    | .pi p t | .sigma p t => withoutIndex p (arrayType t)
    | .record fs => unions (fs.toList.map fun (_, t) => arrayType t)
  partial def arrayType (t : ArrayType VName) : FreeVars := union (atomType t.atom) (shape t.shape)
end
def ty : Ty VName → FreeVars
  | .atom t => atomType t
  | .array t => arrayType t
mutual
  partial def exp : Exp → FreeVars
    | .var v t _ => union {terms := [(v,t.value)]} (arrayType t.value)
    | .array _ xs _ _ => unions (xs.toList.map atom)
    | .emptyArray _ te _ _ | .emptyFrame _ te _ _ => ty te
    | .frame _ xs _ _ => unions (xs.toList.map exp)
    | .app f a _ _ => union (exp f) (exp a)
    | .tApp f te _ _ => union (exp f) (ty te)
    | .iApp f i _ _ => union (exp f) (ispace i)
    | .unbox ip v b e _ _ => union (exp b) (withoutIndex ip (withoutTerm v (exp e)))
    | .letE bs e _ _ => bs.toList.foldr scopedBind (exp e)
    | .structE fs _ _ => unions (fs.toList.map fun (_, s, e) => union (shape s) (exp e))
    | .fieldProj e _ _ _ => exp e
  partial def atom : Atom → FreeVars
    | .base _ _ _ => {}
    | .lambda q e _ _ => union (arrayType (patType q)) (withoutTerm (patVar q) (exp e))
    | .tLambda (.atom v) e _ _ => withoutType v (exp e)
    | .iLambda ip e _ _ => withoutIndex ip (exp e)
    | .box i e te _ _ => unions [ispace i, exp e, (te.map ty).getD {}]
  partial def scopedBind (b : Bind) (rest : FreeVars) : FreeVars :=
    match b with
    | .val v te e _ => unions [exp e, (te.map ty).getD {}, withoutTerm v rest]
    | .funE v ps te e _ _ => unions [(te.map ty).getD {},
        ps.toList.foldr (fun p s => union (arrayType (patType p))
          (withoutTerm (patVar p) s)) (exp e), withoutTerm v rest]
    | .tFun v ps te e _ _ => unions [(te.map ty).getD {},
        ps.toList.foldr (fun p s => withoutType (unTypeParam p) s) (exp e), withoutTerm v rest]
    | .iFun v ps te e _ _ => unions [(te.map ty).getD {},
        ps.toList.foldr withoutIndex (exp e), withoutTerm v rest]
    | .typeE (.atom v) te _ _ => union (ty te) (withoutType v rest)
    | .ispace ip i _ => union (ispace i) (withoutIndex ip rest)
end
end Remora.Free
