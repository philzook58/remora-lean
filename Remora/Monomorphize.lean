import Remora.Prop
import Remora.Intrinsics
namespace Remora.Monomorphize
private def scalar (a : Atom) : Exp :=
  .array [] ⟨a,[]⟩ ⟨mkScalarArrayType (scalarTypeOf a)⟩ (posOf a)
private def asAtom : Exp → Option Atom
  | .array [] ⟨a,[]⟩ _ _ => some a
  | _ => none
private def bindValue : Bind → Option (VName × Exp)
  | .val v _ e _ => some (v,e)
  | .funE v ps _ e _t p => some (v,ps.toList.foldr (fun q e => scalar (.lambda q e ⟨.arrow (patType q) (arrayTypeOf e)⟩ p)) e)
  | .tFun v ps _ e _ p => some (v,ps.toList.foldr (fun q e => scalar (.tLambda q e ⟨.forallE q (arrayTypeOf e)⟩ p)) e)
  | .iFun v ps _ e _ p => some (v,ps.toList.foldr (fun q e => scalar (.iLambda q e ⟨.pi q (arrayTypeOf e)⟩ p)) e)
  | _ => none
private def expandable (e : Exp) : Bool := match (arrayTypeOf e).atom with
  | .arrow .. | .forallE .. | .pi .. => true
  | _ => false
mutual
partial def specialize (env : List (VName × Exp)) (e : Exp) : Exp :=
  match e with
  | .var v _ _ => (env.lookup v).getD e
  | .array ns xs t p => .array ns (xs.map (atom env)) t p
  | .frame ns xs t p => .frame ns (xs.map (specialize env)) t p
  | .app f x t p =>
    let f := specialize env f
    let x := specialize env x
    match f with
    | .letE bs b _ bp => .letE bs (specialize env (.app b x t p)) ⟨t.value.1⟩ bp
    | _ => match asAtom f with
    | some (.lambda q b _ _) =>
      if isEmptyShape t.value.2 then
        let b := specialize env b
        .letE ⟨.val (patVar q) none x p,[]⟩ b ⟨arrayTypeOf b⟩ p
      else .app f x t p
    | _ => .app f x t p
  | .tApp f t a p =>
    let f := specialize env f
    match asAtom f with
    | some (.tLambda (.atom v) b _ _) => specialize env (Substitute.exp (Substitute.atomVar v (elemType t)) b)
    | _ => .tApp f t a p
  | .iApp f i a p =>
    let f := specialize env f
    match f,i with
    | .var v _ _, .dim d =>
      if v.varName == "reify-dim" && v.varTag < intrinsicTypes.length then
        match dimToInt (fun _ => (none : Option Int)) d with
        | some n => scalar (.base (.intVal (Int64.ofInt n)) ⟨.int⟩ p)
        | none => .iApp f i a p
      else iApply env f i a p
    | _,_ => iApply env f i a p
  | .letE bs b t p => bindings env bs.toList b t p
  | .unbox q v b e t p => .unbox q v (specialize env b) (specialize env e) t p
  | .structE fs t p => .structE (fs.map fun (f,s,e) => (f,s,specialize env e)) t p
  | .fieldProj e f t p => .fieldProj (specialize env e) f t p
  | e => e
partial def iApply (env : List (VName × Exp)) (f : Exp) (i : ISpace VName) (a : Info (ArrayType VName)) (p : SourcePos) : Exp :=

    match asAtom f with
    | some (.iLambda q b _ _) => specialize env (Substitute.exp (Substitute.ispaceVar (unISpaceParam q) i) b)
    | _ => .iApp f i a p
partial def atom (env : List (VName × Exp)) : Atom → Atom
  | .lambda q e t p => .lambda q (specialize env e) t p
  | .tLambda q e t p => .tLambda q (specialize env e) t p
  | .iLambda q e t p => .iLambda q (specialize env e) t p
  | .box i e a t p => .box i (specialize env e) a t p
  | a => a
partial def bindings (env : List (VName × Exp)) (bs : List Bind) (body : Exp)
    (t : Info (ArrayType VName)) (p : SourcePos) : Exp :=
  match bs with
  | [] => specialize env body
  -- Ordinary functions stay shared; only polymorphic definitions need specialization.
  | .funE v ps mt e ft bp :: bs =>
    let b := BindBase.funE v ps mt (specialize env e) ft bp
    .letE ⟨b,[]⟩ (bindings env bs body t p) t p
  | b::bs =>
    match bindValue b with
    | some (v,e) =>
      let e := specialize env e
      if expandable e then bindings ((v,e)::env) bs body t p
      else .letE ⟨.val v none e (posOf b),[]⟩ (bindings env bs body t p) t p
    | none => bindings env bs body t p
end

def monomorphizeExp (e : Exp) := specialize [] e
def monomorphize (p : Prog) : Prog := Id.run do
  let mut env := []
  let mut ds := []
  for d in p.progDecs do
    match d with
    -- Preserve calls rather than copying a named function's body at every use.
    | .defE (.funE v ps mt body t pos) =>
      ds := ds ++ [.defE (.funE v ps mt (specialize env body) t pos)]
    | .defE b =>
      if let some (v,e) := bindValue b then
        let e := specialize env e
        if expandable e then env := (v,e)::env
        else ds := ds ++ [.defE (.val v none e (posOf b))]
    | .entry v ps mt e t pos => ds := ds ++ [.entry v ps mt (specialize env e) t pos]
  return ⟨ds⟩
end Remora.Monomorphize
