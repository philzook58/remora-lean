import Remora.Syntax

namespace Remora.Uniquify

structure Context where
  terms : List (String × VName) := []
  atoms : List (String × VName) := []
  arrays : List (String × (VName × VName)) := []
  dims : List (String × VName) := []
  shapes : List (String × VName) := []
  next : Nat := 0

abbrev M := ReaderT Context (StateT Nat (Except String))

def fresh (name : String) : M VName := do
  let tag ← get
  set (tag + 1)
  return ⟨name, tag⟩

def fetch (kind : String) (xs : List (String × VName)) (v : String) : M VName :=
  match xs.lookup v with
  | some x => pure x
  | none => throw s!"Unknown {kind} variable: {v}"

def withTerm (v : String) (k : VName → M α) : M α := do
  let w ← fresh v
  withReader (fun e => { e with terms := (v, w) :: e.terms }) (k w)

def withIParam (p : ISpaceParam String) (k : ISpaceParam VName → M α) : M α := do
  match p with
  | .dim v =>
    let w ← fresh v
    withReader (fun e => { e with dims := (v, w) :: e.dims }) (k (.dim w))
  | .shape v =>
    let w ← fresh v
    withReader (fun e => { e with shapes := (v, w) :: e.shapes }) (k (.shape w))

def withTParam (p : TypeParamExp String) (k : TypeParam VName → List VName → M α) : M α := do
  match p with
  | .atom v =>
    let w ← fresh v
    withReader (fun e => { e with atoms := (v, w) :: e.atoms }) (k (.atom w) [])
  | .array v =>
    let _ ← fresh v
    let a ← fresh v
    let s ← fresh v
    withReader (fun e => { e with arrays := (v, (a, s)) :: e.arrays }) (k (.atom a) [s])

partial def dim : Dim String → M (Dim VName)
  | .var v => return .var (← fetch "dimension" (← read).dims v)
  | .lit n => pure (.lit n)
  | .add ds => return .add (← ds.mapM dim)
  | .mul ds => return .mul (← ds.mapM dim)
  | .sub ds => return .sub (← ds.mapM dim)

partial def shape : Shape String → M (Shape VName)
  | .var v => return .var (← fetch "shape" (← read).shapes v)
  | .dim d => return .dim (← dim d)
  | .concat ss => return .concat (← ss.mapM shape)

def ispace : ISpace String → M (ISpace VName)
  | .dim d => return .dim (← dim d)
  | .shape s => return .shape (← shape s)

partial def bindMany (bind : a → (b → M c) → M c) (xs : List a)
    (k : List b → M c) : M c :=
  match xs with
  | [] => k []
  | x :: rest => bind x fun y => bindMany bind rest fun ys => k (y :: ys)

def neMapM (f : a → M b) (xs : NEList a) : M (NEList b) := do
  return ⟨← f xs.head, ← xs.tail.mapM f⟩

partial def typeExp : TypeExp String → M (TypeExp VName)
  | .atomVar v p => return .atomVar (← fetch "type" (← read).atoms v) p
  | .arrayVar v p => do
    match (← read).arrays.lookup v with
    | some (a, s) => return .array (.atomVar a p) (.var s) p
    | none => throw s!"Unknown array type variable: {v}"
  | .bool p => pure (.bool p)
  | .int p => pure (.int p)
  | .float p => pure (.float p)
  | .array t s p => return .array (← typeExp t) (← shape s) p
  | .arrow t u p => return .arrow (← typeExp t) (← typeExp u) p
  | .forallE ps t p =>
    bindMany (fun x k => withTParam x fun a ss => k (a, ss)) ps.toList fun qs => do
      let body ← typeExp t
      return qs.foldr (fun (q, ss) b =>
        let b := ss.foldr (fun s b => .pi ⟨.shape s, []⟩ b p) b
        match q with | .atom a => .forallE ⟨.atom a, []⟩ b p) body
  | .pi ps t p => bindMany withIParam ps.toList fun qs => do
    let b ← typeExp t
    return qs.foldr (fun q b => .pi ⟨q, []⟩ b p) b
  | .sigma ps t p => bindMany withIParam ps.toList fun qs => do
    let b ← typeExp t
    return qs.foldr (fun q b => .sigma ⟨q, []⟩ b p) b
  | .record fs p => return .record (← neMapM (fun (f, t) => return (f, ← typeExp t)) fs) p

mutual
  partial def arrayType (t : ArrayType String) : M (ArrayType VName) := do
    return ⟨← atomType t.atom, ← shape t.shape⟩
  partial def atomType : AtomType String → M (AtomType VName)
    | .var v => return .var (← fetch "type" (← read).atoms v)
    | .bool => pure .bool
    | .int => pure .int
    | .float => pure .float
    | .arrow t u => return .arrow (← arrayType t) (← arrayType u)
    | .forallE (.atom v) t => withTParam (.atom v) fun p _ => return .forallE p (← arrayType t)
    | .pi p t => withIParam p fun p => return .pi p (← arrayType t)
    | .sigma p t => withIParam p fun p => return .sigma p (← arrayType t)
    | .record fs => return .record (← neMapM (fun (f, t) => return (f, ← arrayType t)) fs)
end

partial def withPat (p : UncheckedPat) (k : UniquePat → M a) : M a := do
  match p with
  | .id v t _ pos =>
    let t ← typeExp t
    withTerm v fun w => k (.id w t ⟨⟩ pos)

mutual
  partial def exp : UncheckedExp → M UniqueExp
    | .var v _ p => return .var (← fetch "term" (← read).terms v) ⟨⟩ p
    | .array ns xs _ p => return .array ns (← neMapM atom xs) ⟨⟩ p
    | .emptyArray ns t _ p => return .emptyArray ns (← typeExp t) ⟨⟩ p
    | .frame ns xs _ p => return .frame ns (← neMapM exp xs) ⟨⟩ p
    | .emptyFrame ns t _ p => return .emptyFrame ns (← typeExp t) ⟨⟩ p
    | .app f a _ p => return .app (← exp f) (← exp a) ⟨⟩ p
    | .tApp f t _ p => do
      let f ← exp f
      let t ← typeExp t
      return match t with
        | .array a s _ => .iApp (.tApp f a ⟨⟩ p) (.shape s) ⟨⟩ p
        | t => .tApp f t ⟨⟩ p
    | .iApp f i _ p => return .iApp (← exp f) (← ispace i) ⟨⟩ p
    | .unbox ip v box body _ p => withIParam ip fun ip => do
      let box ← exp box
      withTerm v fun v => return .unbox ip v box (← exp body) ⟨⟩ p
    | .letE bs body _ p => bindMany withBind bs.toList fun xs => do
      let body ← exp body
      match xs.flatten with
      | b :: rest => return .letE ⟨b, rest⟩ body ⟨⟩ p
      | [] => return body
    | .structE fs _ p =>
      return .structE (← neMapM (fun (f, s, e) => return (f, ← shape s, ← exp e)) fs) ⟨⟩ p
    | .fieldProj e f _ p => return .fieldProj (← exp e) f ⟨⟩ p
  partial def atom : UncheckedAtom → M UniqueAtom
    | .base b _ p => pure (.base b ⟨⟩ p)
    | .lambda param body _ p => withPat param fun q => return .lambda q (← exp body) ⟨⟩ p
    | .tLambda tp body _ p => withTParam tp fun q ss => do
      let body ← exp body
      let body := ss.foldr (fun s b => mkScalar (.iLambda (.shape s) b ⟨⟩ p)) body
      return .tLambda q body ⟨⟩ p
    | .iLambda ip body _ p => withIParam ip fun q => return .iLambda q (← exp body) ⟨⟩ p
    | .box i body t _ p => return .box (← ispace i) (← exp body) (← t.mapM typeExp) ⟨⟩ p
  partial def withBind (b : UncheckedBind) (k : List UniqueBind → M a) : M a := do
    match b with
    | .val v t e p =>
      let e ← exp e
      let t ← t.mapM typeExp
      withTerm v fun w => k [.val w t e p]
    | .funE f ps t body _ p =>
      let t ← t.mapM typeExp
      let (qs, body) ← bindMany withPat ps.toList fun qs => return (qs, ← exp body)
      withTerm f fun w => match qs with
        | q :: rest => k [.funE w ⟨q, rest⟩ t body ⟨⟩ p]
        | [] => k [.val w t body p]
    | .tFun f ps t body _ p =>
      let (qs, body, t) ← bindMany (fun x k => withTParam x fun a ss => k (a, ss)) ps.toList
        fun qs => return (qs, ← exp body, ← t.mapM typeExp)
      match qs with
      | [] => throw "Type function has no parameters"
      | (q, ss) :: rest =>
        let wrapBody := fun (q, ss) b =>
          let b := ss.foldr (fun s b => mkScalar (.iLambda (.shape s) b ⟨⟩ p)) b
          mkScalar (.tLambda q b ⟨⟩ p)
        let body := ss.foldr (fun s b => mkScalar (.iLambda (.shape s) b ⟨⟩ p))
          (rest.foldr wrapBody body)
        let wrapType := fun (q, ss) b =>
          let b := ss.foldr (fun s b => TypeExp.pi ⟨.shape s, []⟩ b p) b
          TypeExp.forallE ⟨.atom (unTypeParam q), []⟩ b p
        let t := t.map fun t => ss.foldr (fun s b => TypeExp.pi ⟨.shape s, []⟩ b p)
          (rest.foldr wrapType t)
        withTerm f fun w => k [.tFun w ⟨q, []⟩ t body ⟨⟩ p]
    | .iFun f ps t body _ p =>
      let (qs, body, t) ← bindMany withIParam ps.toList fun qs =>
        return (qs, ← exp body, ← t.mapM typeExp)
      withTerm f fun w => match qs with
        | q :: rest => k [.iFun w ⟨q, rest⟩ t body ⟨⟩ p]
        | [] => k [.val w t body p]
    | .typeE tp t _ p =>
      let t ← typeExp t
      withTParam tp fun q ss =>
        match ss, t with
        | [s], .array a sh _ => k [.typeE q a ⟨⟩ p, .ispace (.shape s) (.shape sh) p]
        | [], _ => k [.typeE q t ⟨⟩ p]
        | _, _ => throw "Array type alias must denote an array type"
    | .ispace ip i p =>
      let i ← ispace i
      withIParam ip fun q => k [.ispace q i p]
end

partial def decls (xs : List UncheckedDecl) : M (List UniqueDecl) := do
  match xs with
  | [] => return []
  | .defE b :: rest => withBind b fun bs => return bs.map DeclBase.defE ++ (← decls rest)
  | .entry f ps t body _ p :: rest =>
    let t ← t.mapM typeExp
    let (qs, body) ← bindMany withPat ps fun qs => return (qs, ← exp body)
    withTerm f fun w => return .entry w qs t body ⟨⟩ p :: (← decls rest)

def prepare (builtins : List (String × ArrayType String)) :
    Except String (Context × List (VName × ArrayType VName)) := do
  let terms := builtins.zipIdx.map fun ((s, _), i) => (s, VName.mk s i)
  let context : Context := { terms, next := terms.length }
  let action : M (List (VName × ArrayType VName)) := builtins.mapM fun (s, t) => do
    let v ← fetch "intrinsic" (← read).terms s
    return (v, ← arrayType t)
  let (types, next) ← (action context).run context.next
  return ({ context with next }, types)

def uniquify (context : Context) (p : UncheckedProg) : Except String UniqueProg := do
  let (ds, _) ← ((decls p.progDecs) context).run context.next
  return ⟨ds⟩

def uniquifyExp (context : Context) (e : UncheckedExp) : Except String UniqueExp := do
  return (← ((exp e) context).run context.next).1

end Remora.Uniquify
