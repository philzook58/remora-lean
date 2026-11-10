import Remora.Free
import Remora.Intrinsics
namespace Remora.LambdaLift
structure St where
  next : Nat := 1000000
  functions : List Decl := []
  globals : List VName := []
abbrev M := StateM St
private def asLambda : Exp → Option (Pat × Exp)
  | .array [] ⟨.lambda q b _ _,[]⟩ _ _ => some (q,b)
  | _ => none
private partial def unpack (ps : List Pat) (b : Exp) : List Pat × Exp :=
  match asLambda b with
  | some (q,b) => unpack (ps ++ [q]) b
  | none => (ps,b)
private def mkApp (f x : Exp) : Exp :=
  match (arrayTypeOf f).atom with
  | .arrow _ ret => .app f x ⟨ret,.concat []⟩ (posOf f)
  | _ => f
mutual
partial def exp (e : Exp) : M Exp := do
  match e with
  | .array [] ⟨.lambda q b _t p,[]⟩ _ _ =>
    let (ps,b) := unpack [q] b
    let b ← exp b
    let globals := (← get).globals
    let captures := ((ps.foldl (fun fv q => Free.withoutTerm (patVar q) fv) (Free.exp b)).terms).filter
      (fun (v,_) => v.varTag >= intrinsicTypes.length && !globals.contains v)
    let cps := captures.map fun (v,t) => PatBase.id v (.array t) ⟨t⟩ p
    let allps := cps ++ ps
    let ftype := (allps.foldr (fun q r => mkScalarArrayType (.arrow (patType q) r)) (arrayTypeOf b))
    let n := (← get).next
    modify fun s => {s with next := n+1}
    let name := VName.mk "lifted" n
    modify fun s => {s with globals := name :: s.globals}
    match NEList.ofList? allps with
    | none => return e
    | some qs => modify fun s => {s with functions := s.functions ++ [.defE (.funE name qs none b ⟨ftype.atom⟩ p)]}
    let f := ExpBase.var name ⟨ftype⟩ p
    return captures.foldl (fun f (v,t) => mkApp f (.var v ⟨t⟩ p)) f
  | .array ns xs t p => return .array ns (← xs.mapM atom) t p
  | .frame ns xs t p => return .frame ns (← xs.mapM exp) t p
  | .app f x t p => return .app (← exp f) (← exp x) t p
  | .tApp f x t p => return .tApp (← exp f) x t p
  | .iApp f x t p => return .iApp (← exp f) x t p
  | .letE bs e t p => liftBindings bs.toList e t p
  | .unbox q v b e t p => return .unbox q v (← exp b) (← exp e) t p
  | .structE fs t p => return .structE (← fs.mapM fun (f,s,e) => do return (f,s,← exp e)) t p
  | .fieldProj e f t p => return .fieldProj (← exp e) f t p
  | e => return e
partial def liftBindings (bs : List Bind) (body : Exp)
    (t : Info (ArrayType VName)) (p : SourcePos) : M Exp := do
  match bs with
  | [] => exp body
  | .funE v ps _ e _ bp :: bs =>
    -- Lift once and substitute the closure reference, rather than its function body.
    let closure := ps.toList.foldr (fun q e =>
      let ty := AtomType.arrow (patType q) (arrayTypeOf e)
      ExpBase.array [] ⟨.lambda q e ⟨ty⟩ bp,[]⟩ ⟨mkScalarArrayType ty⟩ bp) e
    let replacement ← exp closure
    let subst := Substitute.termVar v replacement
    liftBindings (bs.map (Substitute.bind subst)) (Substitute.exp subst body) t p
  | b :: bs =>
    let b ← bind b
    return .letE ⟨b,[]⟩ (← liftBindings bs body t p) t p
partial def atom : Atom → M Atom
  | .lambda q b t p => return .lambda q (← exp b) t p
  | .tLambda q b t p => return .tLambda q (← exp b) t p
  | .iLambda q b t p => return .iLambda q (← exp b) t p
  | .box i b a t p => return .box i (← exp b) a t p
  | a => pure a
partial def bind : Bind → M Bind
  | .val v a e p => return .val v a (← exp e) p
  | .funE v qs a e t p => return .funE v qs a (← exp e) t p
  | .tFun v qs a e t p => return .tFun v qs a (← exp e) t p
  | .iFun v qs a e t p => return .iFun v qs a (← exp e) t p
  | b => pure b
end
def lambdaLift (p : Prog) : Prog := Id.run do
  let action : M (List Decl) := p.progDecs.mapM fun d => do
    match d with
    | .defE b => return .defE (← bind b)
    | .entry v ps a e t p => return .entry v ps a (← exp e) t p
  let globals := p.progDecs.filterMap fun d => match d with
    | .defE (.funE v ..) => some v
    | .entry v .. => some v
    | _ => none
  let (ds,s) := action.run {globals}
  return ⟨s.functions ++ ds⟩
def lambdaLiftExp (e : Exp) : Prog :=
  lambdaLift ⟨[.entry ⟨"main",999999⟩ [] none e ⟨(arrayTypeOf e).atom⟩ (posOf e)]⟩
end Remora.LambdaLift
