import Remora.Prop

namespace Remora.TypeCheck

structure Env where
  terms : List (VName × ArrayType VName) := []
  types : List (VName × Ty VName) := []
  dims : List (VName × Dim VName) := []
  shapes : List (VName × Shape VName) := []

abbrev M := ReaderT Env (ExceptT String IO)

def error (pos : SourcePos) (message : String) : M α := throw s!"{pos}: {message}"
def localTerm (v : VName) (t : ArrayType VName) (m : M α) : M α :=
  withReader (fun e => { e with terms := (v, t) :: e.terms }) m

def neMapM (f : a → M b) (xs : NEList a) : M (NEList b) :=
  xs.mapM f

def equivalent (a b : ArrayType VName) : M Bool := fun _ => Properties.equivalentArray a b

def subtractShape (a b : Shape VName) : M (Option (Shape VName)) :=
  fun _ => Properties.subtractShape a b

def maximumShape (ss : List (Shape VName)) : M (Option (Shape VName)) :=
  fun _ => ExceptT.mk (Symbolic.maximumShape ss)

partial def checkDim : Dim VName → M (Dim VName)
  | .var v => return ((← read).dims.lookup v).getD (.var v)
  | .lit n => pure (.lit n)
  | .add ds => return normDim (.add (← ds.mapM checkDim))
  | .mul ds => return normDim (.mul (← ds.mapM checkDim))
  | .sub ds => return normDim (.sub (← ds.mapM checkDim))
partial def checkShape : Shape VName → M (Shape VName)
  | .var v => return ((← read).shapes.lookup v).getD (.var v)
  | .dim d => return .dim (← checkDim d)
  | .concat ss => return normShape (.concat (← ss.mapM checkShape))
def checkISpace : ISpace VName → M (ISpace VName)
  | .dim d => return .dim (← checkDim d)
  | .shape s => return .shape (← checkShape s)

partial def convert (t : TypeExp VName) : M (Ty VName) := do
  match t with
  | .atomVar v _ => return ((← read).types.lookup v).getD (.atom (.var v))
  | .arrayVar _ p => error p "Array type parameter was not expanded"
  | .bool _ => return .atom .bool
  | .int _ => return .atom .int
  | .float _ => return .atom .float
  | .array t s p =>
    let t ← convert t
    let .atom a := t | error p "Expected an atom-kinded array element type"
    return .array ⟨a, ← checkShape s⟩
  | .arrow a b _ => return .atom (.arrow (asArrayType (← convert a)) (asArrayType (← convert b)))
  | .forallE ps t _ =>
    let t ← convert t
    return .atom ((ps.toList.foldr (fun p r => mkScalarArrayType (.forallE
      (.atom (unTypeParamExp p)) r)) (asArrayType t)).atom)
  | .pi ps t _ =>
    let t ← convert t
    return .atom ((ps.toList.foldr (fun p r => mkScalarArrayType (.pi p r))
      (asArrayType t)).atom)
  | .sigma ps t _ =>
    let t ← convert t
    return .atom ((ps.toList.foldr (fun p r => mkScalarArrayType (.sigma p r))
      (asArrayType t)).atom)
  | .record fs _ =>
    return .atom (.record (← neMapM (fun (f, t) => return (f, asArrayType (← convert t))) fs))

def annot (t : ArrayType VName) (a : Option (TypeExp VName)) (p : SourcePos) :
    M (Option (Ty VName)) := do
  let a ← a.mapM convert
  if let some u := a then
    unless ← equivalent t (asArrayType u) do error p "Type does not match the annotated type"
  return a

partial def withPats (ps : List UniquePat) (k : List Pat → M a) : M a := do
  match ps with
  | [] => k []
  | .id v te _ p :: rest =>
    let t ← convert te
    let q := PatBase.id v t ⟨asArrayType t⟩ p
    localTerm v (asArrayType t) (withPats rest fun qs => k (q :: qs))

mutual
  partial def checkExp' : UniqueExp → M Exp
    | .var v _ p => do
      let some t := (← read).terms.lookup v | error p s!"Unknown variable {v}"
      return .var v ⟨t⟩ p
    | .array ns xs _ p => do
      let xs ← neMapM checkAtom xs
      let t := scalarTypeOf xs.head
      for x in xs.tail do
        unless ← equivalent (mkScalarArrayType t) (mkScalarArrayType (scalarTypeOf x)) do
          error p "Atoms in array have different types"
      unless ns.foldl (· * ·) 1 == Int.ofNat xs.length do
        error p "Array shape does not match number of elements"
      return .array ns xs ⟨⟨t, intsToShape ns⟩⟩ p
    | .emptyArray ns te _ p => do
      unless ns.foldl (· * ·) 1 == 0 do error p "Empty array has a non-empty shape"
      let .atom t ← convert te | error p "Non-atom kinded annotation on empty array"
      return .emptyArray ns (.atom t) ⟨⟨t, intsToShape ns⟩⟩ p
    | .frame ns xs _ p => do
      let xs ← neMapM checkExp' xs
      let t := arrayTypeOf xs.head
      for x in xs.tail do
        unless ← equivalent t (arrayTypeOf x) do error p "Expressions in frame have different types"
      unless ns.foldl (· * ·) 1 == Int.ofNat xs.length do
        error p "Frame shape does not match number of elements"
      return .frame ns xs ⟨arrayOf (.array t) (intsToShape ns)⟩ p
    | .emptyFrame ns te _ p => do
      unless ns.foldl (· * ·) 1 == 0 do error p "Empty frame has a non-empty shape"
      let t ← convert te
      return .emptyFrame ns t ⟨arrayOf t (intsToShape ns)⟩ p
    | .app f a _ p => do
      let f ← checkExp' f
      let a ← checkExp' a
      let ft := arrayTypeOf f
      let .arrow pt ret := ft.atom | error p "Expected an array of functions in application"
      unless ← equivalent (mkScalarArrayType pt.atom) (mkScalarArrayType (arrayTypeOf a).atom) do
        error p "Ill-typed application: parameter and argument element types differ"
      let some aframe ← subtractShape (arrayTypeOf a).shape pt.shape |
        error p "Ill-shaped application: argument has no matching cell shape"
      let some principal ← maximumShape [ft.shape, aframe] |
        error p "Ill-shaped application: frames have no common prefix"
      return .app f a ⟨(arrayOf (.array ret) principal, principal)⟩ p
    | .tApp f te _ p => do
      let f ← checkExp' f
      let ft := arrayTypeOf f
      let .forallE (.atom v) r := ft.atom | error p "Expected a forall expression in type application"
      let .atom a ← convert te | error p "Ill-kinded type application"
      let r := Substitute.arrayType (Substitute.atomVar v a) r
      return .tApp f (.atom a) ⟨arrayOf (.array r) ft.shape⟩ p
    | .iApp f i _ p => do
      let f ← checkExp' f
      let ft := arrayTypeOf f
      let .pi ip r := ft.atom | error p "Expected a pi expression in index application"
      let i ← checkISpace i
      let (v, i) ← coerceIndex ip i p
      let r := Substitute.arrayType (Substitute.ispaceVar v i) r
      return .iApp f i ⟨arrayOf (.array r) ft.shape⟩ p
    | .unbox ip v box body _ p => do
      let box ← checkExp' box
      let bt := arrayTypeOf box
      let .sigma q r := bt.atom | error p "Expected an existentially typed expression in unbox"
      let i := match ip with | .dim v => ISpace.dim (.var v) | .shape v => .shape (.var v)
      let (qv, i) ← coerceIndex q i p
      let t := Substitute.arrayType (Substitute.ispaceVar qv i) r
      let body ← localTerm v t (checkExp' body)
      return .unbox ip v box body ⟨arrayOf (.array (arrayTypeOf body)) bt.shape⟩ p
    | .letE bs body _ p => withBinds bs.toList fun bs => do
      let body ← checkExp' body
      match bs with
      | b :: rest => return .letE ⟨b, rest⟩ body ⟨arrayTypeOf body⟩ p
      | [] => return body
    | .structE fs _ p => do
      let fs ← neMapM (fun (f, s, e) => return (f, ← checkShape s, ← checkExp' e)) fs
      let mut frames := []
      for (_, s, e) in fs.toList do
        let some fr ← subtractShape (arrayTypeOf e).shape s |
          error p "Record field shape cannot be separated into frame and cell"
        frames := frames ++ [fr]
      let some principal ← maximumShape frames | error p "Record fields have incompatible frames"
      let fields := fs.map fun (f, s, e) => (f, ArrayType.mk (arrayTypeOf e).atom s)
      return .structE fs ⟨(⟨.record fields, principal⟩, principal)⟩ p
    | .fieldProj e f _ p => do
      let e ← checkExp' e
      let et := arrayTypeOf e
      let .record fs := et.atom | error p "Expected a record in field projection"
      let some t := fs.toList.lookup f | error p s!"No record field {f}"
      return .fieldProj e f ⟨arrayOf (.array t) et.shape⟩ p
  partial def coerceIndex (p : ISpaceParam VName) (i : ISpace VName) (pos : SourcePos) :
      M (VName × ISpace VName) := do
    match p, i with
    | .dim v, .dim d => return (v, .dim d)
    | .shape v, .dim d => return (v, .shape (.dim d))
    | .shape v, .shape s => return (v, .shape s)
    | .dim v, .shape s =>
      match coerceToDim s with
      | some d => return (v, .dim d)
      | none => error pos "Ill-sorted index application"
  partial def checkAtom : UniqueAtom → M Atom
    | .base b _ p => return .base b ⟨baseTypeOf b⟩ p
    | .lambda param body _ p => withPats [param] fun ps => do
      let body ← checkExp' body
      match ps with
      | [param] => return .lambda param body ⟨.arrow (patType param) (arrayTypeOf body)⟩ p
      | _ => error p "Invalid lambda parameter"
    | .tLambda tp body _ p => do
      let body ← checkExp' body
      return .tLambda tp body ⟨.forallE tp (arrayTypeOf body)⟩ p
    | .iLambda ip body _ p => do
      let body ← checkExp' body
      return .iLambda ip body ⟨.pi ip (arrayTypeOf body)⟩ p
    | .box _i _body none _ p => error p "Unannotated box"
    | .box i body (some te) _ p => do
      let .atom (.sigma q r) ← convert te | error p "Non-existential box type"
      checkBox i body q r p
  partial def checkBox (i : ISpace VName) (body : UniqueExp) (q : ISpaceParam VName)
      (r : ArrayType VName) (p : SourcePos) : M Atom := do
    let i ← checkISpace i
    let (v, i) ← coerceIndex q i p
    let expected := Substitute.arrayType (Substitute.ispaceVar v i) r
    let body ← (match expected.atom, asScalar body with
      | .sigma q r, some (.box i b none _ _) => do
        let a ← checkBox i b q r p
        return ExpBase.array [] ⟨a, []⟩ ⟨mkScalarArrayType (scalarTypeOf a)⟩ p
      | _, _ => checkExp' body : M Exp)
    unless ← equivalent expected (arrayTypeOf body) do error p "Wrong box type"
    let t := AtomType.sigma q r
    return .box i body (some (.atom t)) ⟨t⟩ p
  partial def withBinds (bs : List UniqueBind) (k : List Bind → M a) : M a := do
    match bs with
    | [] => k []
    | b :: rest => withBind b fun b => withBinds rest fun bs => k (b :: bs)
  partial def withBind (b : UniqueBind) (k : Bind → M a) : M a := do
    match b with
    | .val v mt e p =>
      let e ← checkExp' e
      let t := arrayTypeOf e
      let mt ← annot t mt p
      localTerm v t (k (.val v mt e p))
    | .funE f ps mt body _ p =>
      let (ps, body, mt) ← withPats ps.toList fun ps => do
        let body ← checkExp' body
        let mt ← annot (arrayTypeOf body) mt p
        return (ps, body, mt)
      match ps with
      | q :: qs =>
        let t := arrowType ⟨patType q, qs.map patType⟩ (arrayTypeOf body)
        localTerm f (mkScalarArrayType t) (k (.funE f ⟨q, qs⟩ mt body ⟨t⟩ p))
      | [] => error p "Function has no parameters"
    | .tFun f ps mt body _ p =>
      let body ← checkExp' body
      let mt ← annot (arrayTypeOf body) mt p
      let t := forallType ps (arrayTypeOf body)
      localTerm f (mkScalarArrayType t) (k (.tFun f ps mt body ⟨t⟩ p))
    | .iFun f ps mt body _ p =>
      let body ← checkExp' body
      let mt ← annot (arrayTypeOf body) mt p
      let t := piType ps (arrayTypeOf body)
      localTerm f (mkScalarArrayType t) (k (.iFun f ps mt body ⟨t⟩ p))
    | .typeE (.atom v) te _ p =>
      let t ← convert te
      withReader (fun e => { e with types := (v, t) :: e.types })
        (k (.typeE (.atom v) t ⟨t⟩ p))
    | .ispace ip i p =>
      let i ← checkISpace i
      let (v, i) ← coerceIndex ip i p
      let e ← read
      let env := match i with
        | .dim d => { e with dims := (v, d) :: e.dims }
        | .shape s => { e with shapes := (v, s) :: e.shapes }
      withReader (fun _ => env) (k (.ispace ip i p))
end

partial def decls (ds : List UniqueDecl) : M (List Decl) := do
  match ds with
  | [] => return []
  | .defE b :: ds => withBind b fun b => return .defE b :: (← decls ds)
  | .entry f ps mt body _ p :: ds =>
    let (ps, body, mt) ← withPats ps fun ps => do
      let body ← checkExp' body
      let mt ← annot (arrayTypeOf body) mt p
      return (ps, body, mt)
    let t := (ps.foldr (fun p r => mkScalarArrayType (.arrow (patType p) r))
      (arrayTypeOf body)).atom
    localTerm f (mkScalarArrayType t) do
      return .entry f ps mt body ⟨t⟩ p :: (← decls ds)

def check (builtins : List (VName × ArrayType VName)) (p : UniqueProg) :
    IO (Except String Prog) :=
  ((do return ProgBase.mk (← decls p.progDecs)) {terms := builtins}).run

def checkExp (builtins : List (VName × ArrayType VName)) (e : UniqueExp) :
    IO (Except String Exp) := ((checkExp' e) {terms := builtins}).run

end Remora.TypeCheck
