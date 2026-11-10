import Remora.Syntax
import Lean
import Std.Internal.Parsec

/-! The source grammar from `Parser.hs`, using Std Parsec for tokenization and
source locations. Delimiter trees separate lexical parsing from AST construction. -/
namespace Remora.Parser
open Std.Internal.Parsec
open Std.Internal.Parsec.String

inductive SExp where
  | atom : String → SourcePos → SExp
  | string : String → SourcePos → SExp
  | node : Char → List SExp → SourcePos → SExp
  deriving Repr, Inhabited

def SExp.pos : SExp → SourcePos
  | .atom _ p | .string _ p | .node _ _ p => p

partial def SExp.render : SExp → String
  | .atom s _ => s
  | .string s _ => (Lean.Json.str s).compress
  | .node c xs _ =>
    String.singleton c ++ String.intercalate " " (xs.map SExp.render) ++
      String.singleton (if c == '(' then ')' else if c == '[' then ']' else '}')

abbrev P := Std.Internal.Parsec.String.Parser

private def sourcePos (file : System.FilePath) : P SourcePos := fun it =>
  let chars := (it.1.sliceTo it.2).copy.toList
  let (line, col) := chars.foldl (fun (l,c) ch =>
    if ch == '\n' then (l+1,1) else if ch == '\t' then (l, c + 8 - (c-1)%8) else (l,c+1)) (1,1)
  .success it ⟨file.toString, line, col, it.2.offset.byteIdx⟩

private partial def spaces : P Unit := do
  match ← peek? with
  | some c =>
    if c.isWhitespace then
      discard any
      spaces
    else if c == ';' then
      discard any
      while (← peek?).isSome && (← peek?) != some '\n' do discard any
      spaces
    else pure ()
  | none => pure ()

private def isDelimiter (c : Char) : Bool :=
  c.isWhitespace || "()[]{}\";'`,#|\\".contains c

private partial def quotedChars (acc : String := "") : P String := do
  let c ← any
  if c == '"' then
    match Lean.Json.parse ("\"" ++ acc ++ "\"") with
    | .ok (.str s) => return s
    | _ => fail "invalid string escape"
  else if c == '\\' then
    let d ← any
    quotedChars (acc.push c |>.push d)
  else quotedChars (acc.push c)

private partial def sexp (file : System.FilePath) : P SExp := do
  spaces
  let pos ← sourcePos file
  let c ← any
  if c == '"' then return .string (← quotedChars) pos
  else if c == '(' || c == '[' || c == '{' then
    let closing := if c == '(' then ')' else if c == '[' then ']' else '}'
    let mut xs := []
    spaces
    while (← peek?) != some closing do
      if (← peek?).isNone then fail s!"expected '{closing}'"
      xs := (← sexp file) :: xs
      spaces
    discard any
    return .node c xs.reverse pos
  else if c == '#' then
    let d ← any
    unless d == 't' || d == 'f' do fail "expected #t or #f"
    return .atom (String.ofList [c,d]) pos
  else if isDelimiter c || ")] }".contains c then fail s!"unexpected '{c}'"
  else
    let mut s := String.singleton c
    while (← peek?).any (fun c => !isDelimiter c && !")]}".contains c) do
      s := s.push (← any)
    return .atom s pos

def tokenize (file : System.FilePath) (source : String) : Except String (List SExp) :=
  let p : P (List SExp) := do
    spaces
    let mut xs := []
    while !(← isEof) do
      xs := (← sexp file) :: xs
      spaces
    eof
    return xs.reverse
  match p.run source with
  | .ok x => .ok x
  | .error e => .error s!"{file}: {e}"

private def bad (s : SExp) (expected : String) : Except String α :=
  .error s!"{s.pos}: expected {expected}, got {s.render}"
private def nonempty (s : SExp) (xs : List α) : Except String (NEList α) :=
  match NEList.ofList? xs with
  | some xs => .ok xs
  | none => bad s "a nonempty list"

private def keywords := ["array","frame","t-app","i-app","unbox","box","dims","fn","λ",
  "t-fn","tλ","i-fn","iλ","A","->","→","Forall","∀","Pi","Π","Sigma","Σ","let",
  "type","ispace","fun","t-fun","i-fun","val","def","entry","import","field"]
private def identifier (s : SExp) : Except String String :=
  match s with
  | .atom name _ =>
    if name.isEmpty || keywords.contains name || name.toList.head!.isDigit || name.startsWith "@"
    then bad s "an identifier" else .ok name
  | _ => bad s "an identifier"

private def list (s : SExp) : Except String (List SExp) :=
  match s with | .node '(' xs _ => .ok xs | _ => bad s "a parenthesized list"

private def typeParam (s : SExp) : Except String (TypeParamExp String) := do
  match s with
  | .atom name p =>
    if name.startsWith "&" then return .atom (← identifier (.atom (name.drop 1 |>.toString) p))
    else if name.startsWith "*" then return .array (← identifier (.atom (name.drop 1 |>.toString) p))
    else bad s "a type parameter (&name or *name)"
  | _ => bad s "a type parameter"

private def ispaceParam (s : SExp) : Except String (ISpaceParam String) := do
  match s with
  | .atom name p =>
    if name.startsWith "$" then return .dim (← identifier (.atom (name.drop 1 |>.toString) p))
    else if name.startsWith "@" then return .shape (← identifier (.atom (name.drop 1 |>.toString) p))
    else bad s "an index parameter ($name or @name)"
  | _ => bad s "an index parameter"

private partial def dim (s : SExp) : Except String (Dim String) := do
  match s with
  | .atom name p =>
    if name.startsWith "$" then return .var (← identifier (.atom (name.drop 1 |>.toString) p))
    else match name.toNat? with
      | some n => return .lit n
      | none => bad s "a dimension"
  | .node '(' (.atom op _ :: ds) _ =>
    let ds ← ds.mapM dim
    match op with
    | "+" => return .add ds
    | "*" => return .mul ds
    | "-" => return .sub ds
    | _ => bad s "a dimension operator (+, *, -)"
  | _ => bad s "a dimension"

mutual
private partial def shape (s : SExp) : Except String (Shape String) := do
  match s with
  | .atom name p =>
    if name.startsWith "@" then return .var (← identifier (.atom (name.drop 1 |>.toString) p))
    else bad s "a shape"
  | .node '[' xs _ => return .concat (← xs.mapM fun x => ispaceToShape <$> ispace x)
  | .node '(' (.atom "dims" _ :: xs) _ => return .concat (← xs.mapM fun x => Shape.dim <$> dim x)
  | .node '(' (.atom "++" _ :: xs) _ => return .concat (← xs.mapM shape)
  | _ => bad s "a shape"
private partial def ispace (s : SExp) : Except String (ISpace String) :=
  match dim s with
  | .ok d => .ok (.dim d)
  | .error _ => ISpace.shape <$> shape s
end

private partial def typeExp (s : SExp) : Except String (TypeExp String) := do
  let pos := s.pos
  match s with
  | .atom "Bool" _ => return .bool pos
  | .atom "Int" _ => return .int pos
  | .atom "Float" _ => return .float pos
  | .atom name p =>
    match ← typeParam (.atom name p) with
    | .atom name => return .atomVar name pos
    | .array name => return .arrayVar name pos
  | .node '[' (t :: xs) _ =>
    return .array (← typeExp t) (.concat (← xs.mapM fun x => ispaceToShape <$> ispace x)) pos
  | .node '{' fs _ =>
    let fs ← fs.mapM fun field => do
      match field with
      | .node '(' [name,t] _ => return (← identifier name, ← typeExp t)
      | _ => bad field "a field type (name Type)"
    return .record (← nonempty s fs) pos
  | .node '(' [.atom "A" _, t, i] _ =>
    return .array (← typeExp t) (ispaceToShape (← ispace i)) pos
  | .node '(' [.atom op _, args, ret] _ =>
    if op == "->" || op == "→" then
      let args ← match typeExp args with
        | .ok t => pure [t]
        | .error _ => do (← list args).mapM typeExp
      let ret ← typeExp ret
      return args.foldr (fun arg t => .arrow arg t pos) ret
    else if op == "Forall" || op == "∀" then
      return .forallE (← nonempty args (← (← list args).mapM typeParam)) (← typeExp ret) pos
    else if op == "Pi" || op == "Π" then
      return .pi (← nonempty args (← (← list args).mapM ispaceParam)) (← typeExp ret) pos
    else if op == "Sigma" || op == "Σ" then
      return .sigma (← nonempty args (← (← list args).mapM ispaceParam)) (← typeExp ret) pos
    else bad s "a type constructor"
  | _ => bad s "a type"

private def pat (s : SExp) : Except String UncheckedPat := do
  match s with
  | .node '(' [name, t] pos => return .id (← identifier name) (← typeExp t) .mk pos
  | _ => bad s "a parameter (name Type)"

private def base (s : SExp) : Except String Base := do
  match s with
  | .atom "#t" _ => return .boolVal true
  | .atom "#f" _ => return .boolVal false
  | .atom n _ =>
    if let some i := n.toInt? then return .intVal (Int64.ofInt i)
    let n := if n.startsWith "+" then n.drop 1 |>.toString else n
    match Lean.Json.parse n with
    | .ok (.num f) => return .floatVal f.toFloat.toFloat32
    | _ => bad s "a number or boolean"
  | _ => bad s "a number or boolean"

private def shapeLit (s : SExp) : Except String (List Int) := do
  match s with
  | .node '[' xs _ => xs.mapM fun x => do
      match x with
      | .atom n _ => match n.toNat? with
        | some n => pure (Int.ofNat n)
        | none => bad x "a nonnegative dimension"
      | _ => bad x "a nonnegative dimension"
  | _ => bad s "a literal shape [dimensions]"

mutual
private partial def atom (s : SExp) : Except String UncheckedAtom := do
  let pos := s.pos
  match base s with
  | .ok b => return .base b .mk pos
  | .error _ => pure ()
  match s with
  | .node '(' [.atom kw _, params, body] _ =>
    if kw == "fn" || kw == "λ" then
      let ps ← nonempty params (← (← list params).mapM pat)
      let body ← exp body
      return .lambda ps.head (ps.tail.foldr (fun p b => mkScalar (.lambda p b .mk pos)) body) .mk pos
    else if kw == "t-fn" || kw == "tλ" then
      let ps ← nonempty params (← (← list params).mapM typeParam)
      let body ← exp body
      return .tLambda ps.head (ps.tail.foldr (fun p b => mkScalar (.tLambda p b .mk pos)) body) .mk pos
    else if kw == "i-fn" || kw == "iλ" then
      let ps ← nonempty params (← (← list params).mapM ispaceParam)
      let body ← exp body
      return .iLambda ps.head (ps.tail.foldr (fun p b => mkScalar (.iLambda p b .mk pos)) body) .mk pos
    else bad s "a lambda"
  | .node '(' [.atom "box" _, indices, body, t] _ =>
    let ds ← nonempty indices (← (← list indices).mapM ispace)
    let body ← exp body
    return .box ds.head (ds.tail.foldr (fun d b => mkScalar (.box d b none .mk pos)) body)
      (some (← typeExp t)) .mk pos
  | _ => bad s "an atom"

private partial def exp (s : SExp) : Except String UncheckedExp := do
  let pos := s.pos
  match atom s with
  | .ok a => return mkScalar a
  | .error _ => pure ()
  match s with
  | .atom _ _ => return .var (← identifier s) .mk pos
  | .string text _ =>
    let chars := text.toList
    match chars with
    | [] => return .emptyArray [0] (.int pos) .mk pos
    | c :: cs =>
      return .array [chars.length] ⟨.base (.intVal (Int64.ofNat c.toNat)) .mk pos,
        cs.map fun c => .base (.intVal (Int64.ofNat c.toNat)) .mk pos⟩ .mk pos
  | .node '[' xs _ =>
    let es ← nonempty s (← xs.mapM exp)
    return flattenExp (.frame [es.length] es .mk pos)
  | .node '{' fields _ =>
    let fs ← fields.mapM fun field => do
      match field with
      | .node '(' [name, sh, body] _ => return (← identifier name, ← shape sh, ← exp body)
      | _ => bad field "a record field (name shape expression)"
    return .structE (← nonempty s fs) .mk pos
  | .node '(' (.atom "array" _ :: sh :: xs) _ =>
    let dims ← shapeLit sh
    match xs with
    | [t] =>
      match typeExp t with
      | .ok t => return .emptyArray dims t .mk pos
      | .error _ => return .array dims (← nonempty s (← xs.mapM atom)) .mk pos
    | _ => return .array dims (← nonempty s (← xs.mapM atom)) .mk pos
  | .node '(' (.atom "frame" _ :: sh :: xs) _ =>
    let dims ← shapeLit sh
    match xs with
    | [t] =>
      match typeExp t with
      | .ok t => return .emptyFrame dims t .mk pos
      | .error _ => return .frame dims (← nonempty s (← xs.mapM exp)) .mk pos
    | _ => return .frame dims (← nonempty s (← xs.mapM exp)) .mk pos
  | .node '(' [.atom "field" _, e, name] _ =>
    return .fieldProj (← exp e) (← identifier name) .mk pos
  | .node '(' [.atom "let" _, binds, body] _ =>
    return .letE (← nonempty binds (← (← list binds).mapM bind)) (← exp body) .mk pos
  | .node '(' [.atom "unbox" _, .node '(' xs _, body] _ =>
    if xs.length < 3 then bad s "an unbox binding (indices name expression)"
    else
      let ip ← nonempty s (← (xs.take (xs.length-2)).mapM ispaceParam)
      let name ← identifier xs[xs.length-2]!
      let inner ← exp xs[xs.length-1]!
      let outer ← exp body
      let rec nest (ip : NEList (ISpaceParam String)) (box : UncheckedExp) : UncheckedExp :=
        match ip.tail with
        | [] => .unbox ip.head name box outer .mk pos
        | p :: ps => .unbox ip.head "#box" box (nest ⟨p,ps⟩ (.var "#box" .mk pos)) .mk pos
      return nest ip inner
  | .node '(' (.atom "t-app" _ :: f :: args) _ =>
    let args ← nonempty s args
    let f ← exp f
    let args ← args.toList.mapM typeExp
    return args.foldl (fun f t => .tApp f t .mk pos) f
  | .node '(' (.atom "i-app" _ :: f :: args) _ =>
    let args ← nonempty s args
    let f ← exp f
    let args ← args.toList.mapM ispace
    return args.foldl (fun f i => .iApp f i .mk pos) f
  | .node '(' (.atom head p :: rest) _ =>
    if head.startsWith "@" then
      let (f, rest) ← if head == "@" then
          match rest with
          | f :: rest => pure (f, rest)
          | [] => bad s "a function after @"
        else pure (.atom (head.drop 1 |>.toString) p, rest)
      let f ← exp f
      match rest with
      | ts :: indices :: args =>
        let f ← match ts with
          | .atom "_" _ => pure f
          | _ => do
            let ts ← (← list ts).mapM typeExp
            pure (ts.foldl (fun f t => .tApp f t .mk pos) f)
        let f ← match indices with
          | .atom "_" _ => pure f
          | _ => do
            let indices ← (← list indices).mapM ispace
            pure (indices.foldl (fun f i => .iApp f i .mk pos) f)
        return (← args.mapM exp).foldl (fun f arg => .app f arg .mk pos) f
      | _ => bad s "type and index argument groups after @function"
    else
      let f ← exp (.atom head p)
      let args ← nonempty s rest
      return (← args.toList.mapM exp).foldl (fun f arg => .app f arg .mk pos) f
  | .node '(' (f :: args) _ =>
    let f ← exp f
    let args ← nonempty s args
    return (← args.toList.mapM exp).foldl (fun f arg => .app f arg .mk pos) f
  | _ => bad s "an expression"

private partial def bind (s : SExp) : Except String UncheckedBind := do
  let pos := s.pos
  match s with
  | .node '(' [.atom "val" _, name, body] _ =>
    match name with
    | .node '(' [name, .atom ":" _, t] _ =>
      return .val (← identifier name) (some (← typeExp t)) (← exp body) pos
    | _ => return .val (← identifier name) none (← exp body) pos
  | .node '(' [.atom "type" _, name, t] _ =>
    return .typeE (← typeParam name) (← typeExp t) .mk pos
  | .node '(' [.atom "ispace" _, name, i] _ =>
    return .ispace (← ispaceParam name) (← ispace i) pos
  | .node '(' [.atom kw _, sig, body] _ =>
    let xs ← list sig
    match xs with
    | .atom name p :: params =>
      let (params, annot) ← annotation params
      let body ← exp body
      if kw == "fun" && name.startsWith "@" then
        let (name, params) ← if name == "@" then
            match params with
            | n :: rest => pure (← identifier n, rest)
            | _ => bad s "a function name"
          else pure (← identifier (.atom (name.drop 1 |>.toString) p), params)
        match params with
        | ts :: indices :: ps =>
          let ps ← ps.mapM pat
          let base := match NEList.ofList? ps with
            | none => BindBase.val name annot body pos
            | some ps => .funE name ps annot body .mk pos
          let ib ← match indices with
            | .atom "_" _ => pure base
            | _ => do
              let ips ← (← list indices).mapM ispaceParam
              match NEList.ofList? ips with
              | none => pure base
              | some ips => pure (.iFun name ips none (.letE (.singleton base) (.var name .mk pos) .mk pos) .mk pos)
          match ts with
          | .atom "_" _ => return ib
          | _ =>
            let tps ← (← list ts).mapM typeParam
            match NEList.ofList? tps with
            | none => return ib
            | some tps => return .tFun name tps none (.letE (.singleton ib) (.var name .mk pos) .mk pos) .mk pos
        | _ => bad s "type and index parameter groups"
      else
        let name ← identifier (.atom name p)
        if kw == "fun" then
          match NEList.ofList? (← params.mapM pat) with
          | none => return .val name annot body pos
          | some ps => return .funE name ps annot body .mk pos
        else if kw == "t-fun" then
          match params with
          | [ps] =>
            match NEList.ofList? (← (← list ps).mapM typeParam) with
            | none => return .val name annot body pos
            | some ps => return .tFun name ps annot body .mk pos
          | _ => bad s "a type parameter group"
        else if kw == "i-fun" then
          match params with
          | [ps] =>
            match NEList.ofList? (← (← list ps).mapM ispaceParam) with
            | none => return .val name annot body pos
            | some ps => return .iFun name ps annot body .mk pos
          | _ => bad s "an index parameter group"
        else bad s "a binding keyword"
    | _ => bad s "a function signature"
  | _ => bad s "a binding"
where
  annotation (xs : List SExp) : Except String (List SExp × Option (TypeExp String)) := do
    if xs.length >= 2 then
      match xs[xs.length-2]! with
      | .atom ":" _ => return (xs.take (xs.length-2), some (← typeExp xs[xs.length-1]!))
      | _ => return (xs, none)
    else return (xs, none)
end

private def decl (s : SExp) : Except String UncheckedDecl := do
  match s with
  | .node '(' [.atom "def" _, b] _ => return .defE (← bind b)
  | .node '(' [.atom "entry" _, sig, body] pos =>
    match ← list sig with
    | name :: params =>
      let (params, t) ← bind.annotation params
      return .entry (← identifier name) (← params.mapM pat) t (← exp body) .mk pos
    | _ => bad sig "an entry point name"
  | _ => bad s "a declaration"

def parseExp (fileName : System.FilePath) (source : String) : Except String UncheckedExp := do
  match ← tokenize fileName source with
  | [s] => exp s
  | _ => .error s!"{fileName}: expected one expression"

def parse (fileName : System.FilePath) (source : String) : Except String (List Import × UncheckedProg) := do
  let xs ← tokenize fileName source
  let mut imports := []
  let mut decls := []
  let mut declarationsStarted := false
  for s in xs do
    match s with
    | .node '(' [.atom "import" _, .string file _] pos =>
      if declarationsStarted then bad s "imports before declarations"
      else imports := imports ++ [⟨file, pos⟩]
    | _ =>
      declarationsStarted := true
      decls := decls ++ [← decl s]
  return (imports, ⟨decls⟩)

end Remora.Parser
