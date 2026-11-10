import Remora.Monomorphize
namespace Remora.Defunctionalize
/-- Eta-expand a function returning a partial application, as in the upstream proto pass. -/
def etaExpandBind (b : Bind) : Bind := Id.run do
  let .funE f ps a body t p := b | return b
  let (missing,_) := unfoldArrow (arrayTypeOf body)
  if missing.isEmpty then return b
  let mut body := body
  let mut qs := ps.toList
  for (ty,i) in missing.zipIdx do
    let v := VName.mk "eta" (2000000 + f.varTag * 1000 + i)
    let q := PatBase.id v (.array ty) ⟨ty⟩ p
    qs := qs ++ [q]
    let .arrow _ ret := (arrayTypeOf body).atom | return b
    body := .app body (.var v ⟨ty⟩ p) ⟨ret,.concat []⟩ p
  let some ps := NEList.ofList? qs | return b
  return .funE f ps a body t p

def defunctionalize (p : Prog) : Prog :=
  ⟨p.progDecs.map fun d => match d with
    | .defE b => .defE (etaExpandBind b)
    | d => d⟩
def defunctionalizeExp (e : Exp) : Exp := Monomorphize.monomorphizeExp e
end Remora.Defunctionalize
