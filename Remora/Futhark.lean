import Remora.Syntax
import Remora.Intrinsics
namespace Remora.Futhark
open Remora
structure IRType where
  base : String
  shape : List Int := []
  deriving Inhabited
instance : ToString IRType where
  toString t := String.join (t.shape.map (fun n => s!"[{n}i64]")) ++ t.base
structure SubExp where
  text : String
  type : IRType
  deriving Inhabited
structure Builder where
  next : Nat := 100000
  statements : List String := []
  bindings : List (VName × List SubExp) := []
  functions : List String := []
abbrev M := StateT Builder (Except String)
private def fresh (name := "r") : M String := do
  let s ← get
  set {s with next := s.next + 1}
  return s!"{name}_{s.next}"
private def emit (types : List IRType) (rhs : String) : M (List SubExp) := do
  let vs ← types.mapM fun t => return SubExp.mk (← fresh) t
  let pat := String.intercalate ", " (vs.map fun v => s!"{v.text} : {v.type}")
  modify fun s => {s with statements := s.statements ++ [s!"let \{{pat}} = {rhs}"]}
  return vs
private def one (xs : List α) : M α := match xs with
  | [x] => pure x
  | _ => throw "Futhark: expected one primitive value"
private def shape (s : Shape VName) : Except String (List Int) :=
  shapeToInts (dimToInt (fun v => .error s!"Futhark: unresolved dimension {v}"))
    (fun v => .error s!"Futhark: unresolved shape {v}") s
partial def compileArrayTypes (t : ArrayType VName) : Except String (List IRType) := do
  let ds ← shape t.shape
  match t.atom with
  | .int => pure [⟨"i64",ds⟩]
  | .float => pure [⟨"f32",ds⟩]
  | .bool => pure [⟨"bool",ds⟩]
  | .record fs => return (← fs.toList.mapM fun (_,ft) => do
      return (← compileArrayTypes ft).map fun t => {t with shape := ds ++ t.shape}).flatten
  | _ => throw "Futhark: only monomorphic first-order values can be lowered"
private def types (t : ArrayType VName) : M (List IRType) :=
  match compileArrayTypes t with | .ok ts => pure ts | .error e => throw e
private def lookup (v : VName) : M (List SubExp) := do
  match (← get).bindings.lookup v with
  | some xs => pure xs
  | none => throw s!"Futhark: unknown value {v}"
private def bind (v : VName) (xs : List SubExp) : M Unit :=
  modify fun s => {s with bindings := (v,xs)::s.bindings}
private def expType : Exp → ArrayType VName
  | .app _ _ i _ | .structE _ i _ => i.value.1
  | .var _ i _ | .array _ _ i _ | .emptyArray _ _ i _ | .frame _ _ i _
  | .emptyFrame _ _ i _ | .tApp _ _ i _ | .iApp _ _ i _ | .unbox _ _ _ _ i _
  | .letE _ _ i _ | .fieldProj _ _ i _ => i.value
private def literal : Base → SubExp
  | .intVal n => ⟨s!"{n}i64",⟨"i64",[]⟩⟩
  | .floatVal n => ⟨s!"{n}f32",⟨"f32",[]⟩⟩
  | .boolVal b => ⟨if b then "true" else "false",⟨"bool",[]⟩⟩
private def binName (n : String) : Option String :=
  ([ ("+","add64"),("-","sub64"),("*","mul64"),("/","squot64"),("mod","srem64"),
     ("^","pow64"),("max","smax64"),("min","smin64"),("bit-and","and64"),
     ("bit-or","or64"),("bit-xor","xor64"),("shl","shl64"),("shr","ashr64"),
     ("f.+","fadd32"),("f.-","fsub32"),("f.*","fmul32"),("f./","fdiv32"),
     ("f.^","fpow32"),("f.max","fmax32"),("f.min","fmin32"),("==","eq_i64"),
     ("<","slt64"),("<=","sle64"),("f.==","eq_f32"),("f.<","flt32"),
     ("f.<=","fle32"),("and","logand"),("or","logor"),("bool.==","eq_bool")
   ] : List (String×String)).lookup n
private def tuple (xs : List SubExp) := "{" ++ String.intercalate ", " (xs.map (·.text)) ++ "}"
private def retTypes (ts : List IRType) := "{" ++ String.intercalate ", " (ts.map toString) ++ "}"
private def userCall (name : VName) (xs : List SubExp) (ts : List IRType) : M (List SubExp) := do
  let pals := String.intercalate ", " (List.range xs.length |>.map toString)
  let rals := String.intercalate ", " (List.range ts.length |>.map toString)
  let rt := "{" ++ String.intercalate ", " (ts.map fun t => toString t ++ s!"#([{pals}], [{rals}])") ++ "}"
  emit ts s!"apply {name}({String.intercalate ", " (xs.map (·.text))}) : {rt}"

private def body (ss : List String) (xs : List SubExp) :=
  "{\n" ++ String.intercalate "\n" ss ++ "\nin " ++ tuple xs ++ "\n}"
private def collect (m : M α) : M (α × List String) := do
  let old := (← get).statements
  modify fun s => {s with statements := []}
  let x ← m
  let ss := (← get).statements
  modify fun s => {s with statements := old}
  return (x,ss)
private def unfoldApp : Exp → Exp × List Exp
  | .app f x _ _ => let (f,xs) := unfoldApp f; (f,xs ++ [x])
  | .tApp f _ _ _ | .iApp f _ _ _ => unfoldApp f
  | e => (e,[])
private partial def argFrameRanks (e : Exp) : Except String (List Nat) := do
  match e with
  | .app f x _ _ =>
    let .arrow pt _ := (expType f).atom | throw "Futhark: application has no function type"
    let xshape ← shape (expType x).shape
    let pshape ← shape pt.shape
    return (← argFrameRanks f) ++ [xshape.length - pshape.length]
  | .tApp f _ _ _ | .iApp f _ _ _ => argFrameRanks f
  | _ => return []
private partial def nestArray (ds : List Int) (xs : List SubExp) (t : IRType) : M SubExp := do
  match ds with
  | [] => one xs
  | d::rest =>
    let size := (rest.foldl (·*·) 1).toNat
    let inner := {t with shape := t.shape.drop 1}
    let rows ← (List.range d.toNat).mapM fun i => nestArray rest (xs.drop (i*size) |>.take size) inner
    one (← emit [t] ("[" ++ String.intercalate ", " (rows.map (·.text)) ++ s!"] : []{inner}"))
mutual
partial def compileExpM (e : Exp) : M (List SubExp) := do
  match e with
  | .var v _ _ => lookup v
  | .array ds atoms _ _ =>
    let xs ← atoms.toList.mapM fun a => match a with
      | .base b _ _ => pure (literal b)
      | _ => throw "Futhark: function literals require lambda lifting"
    let t ← one (← types (expType e))
    return [← nestArray ds xs t]
  | .frame ds es _ _ =>
    let rows ← es.toList.mapM compileExpM
    let ts ← types (expType e)
    ts.zipIdx.mapM fun (t,j) => nestArray ds (rows.filterMap (·[j]?)) t
  | .emptyArray _ _ _ _ | .emptyFrame _ _ _ _ =>
    (← types (expType e)).mapM fun t => do
      let zero := if t.base == "bool" then "false" else if t.base == "f32" then "0f32" else "0i64"
      one (← emit [t] s!"replicate([{String.intercalate "][" (t.shape.map fun n => s!"{n}i64")}], {zero})")
  | .letE bs b _ _ =>
    for b in bs.toList do compileBind b
    compileExpM b
  | .structE fs i _ =>
    let frame ← match shape i.value.2 with | .ok s => pure s | .error e => throw e
    return (← fs.toList.mapM fun (_,cell,e) => do
      let ds ← match shape (expType e).shape with | .ok s => pure s | .error e => throw e
      let cell ← match shape cell with | .ok s => pure s | .error e => throw e
      let prefixDims := frame.take (frame.length - (ds.length-cell.length))
      let vs ← compileExpM e
      if prefixDims.isEmpty then return vs
      vs.mapM fun x => do
        let t := {x.type with shape := prefixDims ++ x.type.shape}
        let sh := "[" ++ String.intercalate "][" (prefixDims.map fun n => s!"{n}i64") ++ "]"
        one (← emit [t] s!"replicate({sh}, {x.text})")).flatten
  | .fieldProj x field _ _ =>
    let .record fs := (expType x).atom | throw "Futhark: projection from non-record"
    let mut offset := 0
    for (f,t) in fs.toList do
      let n := (← types t).length
      if f == field then return (← compileExpM x).drop offset |>.take n
      offset := offset + n
    throw s!"Futhark: missing field {field}"
  | .app _ _ i _ =>
    let (f,args) := unfoldApp e
    let .var name _ _ := f | throw "Futhark: application requires a named function"
    let ts ← types i.value.1
    if name.varName == "sum" then
      let sc := mkScalarArrayType (AtomType.int : AtomType VName)
      let fn := ExpBase.var (VName.mk "+" 0) ⟨mkScalarArrayType (.arrow sc (mkScalarArrayType (.arrow sc sc)))⟩ {}
      let zero := ExpBase.array [] ⟨.base (.intVal 0) ⟨.int⟩ {},[]⟩ ⟨sc⟩ {}
      return ← compileFold "reduce/zero" ([fn,zero] ++ args) ts
    if ["reduce","reduce/zero","fold","fold-right"].contains name.varName then
      return ← compileFold name.varName args ts
    let values ← args.mapM compileExpM
    let frame ← match shape i.value.2 with | .ok s => pure s | .error e => throw e
    let ranks ← match argFrameRanks e with | .ok rs => pure rs | .error e => throw e
    mapNest frame name ranks values ts
  | .tApp f _ _ _ | .iApp f _ _ _ =>
    let (root,args) := unfoldApp e
    match root,args with
    | .var name _ _,[] => applyName name [] (← types (expType e))
    | _,_ => compileExpM f
  | .unbox .. => throw "Futhark: existential values are unsupported upstream"
partial def mapNest (frame : List Int) (name : VName) (ranks : List Nat)
    (values : List (List SubExp)) (ts : List IRType) : M (List SubExp) := do
  match frame with
  | [] => applyName name values ts
  | d::ds =>
    let mut inputs : List SubExp := []
    let mut params : List SubExp := []
    let mut nextValues := []
    for (rank,vs) in ranks.zip values do
      if rank > 0 && !vs.isEmpty then
        let ps ← vs.mapM fun v => do
          let p ← fresh "x"
          pure (SubExp.mk p {v.type with shape := v.type.shape.drop 1})
        inputs := inputs ++ vs
        params := params ++ ps
        nextValues := nextValues ++ [ps]
      else nextValues := nextValues ++ [vs]
    let innerTs := ts.map fun t => {t with shape := t.shape.drop 1}
    let (result,ss) ← collect (mapNest ds name (ranks.map (· - 1)) nextValues innerTs)
    let ps := String.intercalate ", " (params.map fun p => s!"{p.text} : {p.type}")
    let lam := s!"\\ \{{ps}} : {retTypes innerTs} -> {String.intercalate "\n" ss}\nin {tuple result}"
    emit ts s!"map({d}i64, {tuple inputs}, {lam})"
partial def compileFold (kind : String) (args : List Exp) (ts : List IRType) : M (List SubExp) := do
  let some op := args.head? | throw "Futhark: missing reduction operator"
  let (f,applied) := unfoldApp op
  let .var name _ _ := f | throw "Futhark: reduction operator must be named"
  let applied ← applied.mapM compileExpM
  let some input := args.getLast? | throw "Futhark: missing reduction array"
  let x ← one (← compileExpM input)
  let t ← one ts
  let n := x.type.shape.headD 0
  let initial ← if kind == "reduce" then one (← emit [t] s!"{x.text}[0i64]")
    else if let some zero := args[1]? then one (← compileExpM zero)
    else throw "Futhark: missing fold initial value"
  let i ← fresh "i"
  let acc ← fresh "acc"
  let start : Int := if kind == "reduce" then 1 else 0
  let (result,ss) ← collect do
    let idx ← if kind == "fold-right" then
      one (← emit [⟨"i64",[]⟩] s!"sub64({n-1}i64, {i})")
      else if start == 1 then one (← emit [⟨"i64",[]⟩] s!"add64({i}, 1i64)")
      else pure (SubExp.mk i ⟨"i64",[]⟩)
    let elem ← one (← emit [t] s!"{x.text}[{idx.text}]")
    let av := SubExp.mk acc t
    applyName name (applied ++ (if kind == "fold-right" then [[elem],[av]] else [[av],[elem]])) ts
  emit ts s!"loop \{{acc} : {t}} = \{{initial.text}} for {i} : i64 < {n-start}i64 do {body ss result}"
partial def applyName (name : VName) (values : List (List SubExp)) (ts : List IRType) : M (List SubExp) := do
  let xs := values.flatten
  if name.varTag >= intrinsicTypes.length then return ← userCall name xs ts
  if [">",">=","f.>","f.>="].contains name.varName then
    let n := if name.varName == ">" then "<" else if name.varName == ">=" then "<="
      else if name.varName == "f.>" then "f.<" else "f.<="
    return ← applyName {name with varName := n} values.reverse ts
  if ["!=","f.!=","bool.!="].contains name.varName then
    let n := if name.varName == "!=" then "==" else if name.varName == "f.!=" then "f.==" else "bool.=="
    let x ← one (← applyName {name with varName := n} values ts)
    return ← emit ts s!"neg_bool {x.text}"
  if ["not","bit-not"].contains name.varName then
    let x ← one xs
    return ← emit ts s!"{if name.varName == "not" then "neg_bool" else "complement64"} {x.text}"
  if let some (op,fromT,toT) := ([
      ("i->f",("sitofp","i64","f32")),("i->bool",("itob","i64","bool")),
      ("bool->i",("btoi","bool","i64")),("truncate",("fptosi","f32","i64"))
    ] : List (String × String × String × String)).lookup name.varName then
    let x ← one xs
    return ← emit ts s!"{op} {fromT} {x.text} to {toT}"
  if name.varName == "bool->f" then
    let x ← one (← applyName {name with varName := "bool->i"} values [⟨"i64",[]⟩])
    return ← applyName {name with varName := "i->f"} [[x]] ts
  if ["round","ceiling","floor"].contains name.varName then
    let x ← one xs
    let op := if name.varName == "round" then "round32" else if name.varName == "ceiling" then "ceil32" else "floor32"
    let x ← one (← emit [⟨"f32",[]⟩] s!"apply {op}({x.text}) : \{f32}")
    return ← emit ts s!"fptosi f32 {x.text} to i64"
  if let some op := ([ ("sqrt","sqrt32"),("f.sqrt","sqrt32"),("f.ln","log32"),
      ("f.tanh","tanh32"),("f.exp","exp32") ] : List (String×String)).lookup name.varName then
    return ← emit ts s!"apply {op}({String.intercalate ", " (xs.map (·.text))}) : {retTypes ts}"
  if name.varName == "popc" then
    let x ← one xs
    let x ← one (← emit [⟨"i32",[]⟩] s!"apply popc64({x.text}) : \{i32}")
    return ← emit ts s!"sext i32 {x.text} to i64"
  if let some op := binName name.varName then
    return ← emit ts s!"{op}({String.intercalate ", " (xs.map (·.text))})"
  if name.varName == "iota/static" then
    let t ← one ts
    let n := t.shape.foldl (·*·) 1
    let flat ← one (← emit [⟨"i64",[n]⟩] s!"iota64({n}i64, 0i64, 1i64)")
    if t.shape.length == 1 then return [flat]
    let ds := "[" ++ String.intercalate "][" (t.shape.map fun d => s!"{d}i64") ++ "]"
    return ← emit ts s!"reshape({flat.text}, (0::1=>{ds}; {ds}))"
  if name.varName == "length" then
    let x ← one xs
    return [⟨s!"{x.type.shape.headD 0}i64",⟨"i64",[]⟩⟩]
  if name.varName == "trace" || name.varName == "trace-file" then
    return values.getLast?.getD []
  if ["reshape","flatten"].contains name.varName then
    let x ← one xs
    let t ← one ts
    return ← emit ts s!"reshape({x.text}, (0::{x.type.shape.length}=>[{String.intercalate "][" (t.shape.map fun d => s!"{d}i64")}]; [{String.intercalate "][" (t.shape.map fun d => s!"{d}i64") }]))"
  if name.varName == "transpose2d" then
    let x ← one xs
    return ← emit ts s!"rearrange({x.text}, (1, 0))"
  if name.varName == "index" then
    let some x := xs[0]? | throw "index: missing array"
    let some i := xs[1]? | throw "index: missing index"
    return ← emit ts s!"{x.text}[{i.text}]"
  if name.varName == "append" then
    let some vs := values[0]? | throw "append: missing array"
    let some ws := values[1]? | throw "append: missing array"
    if vs.length != ws.length || vs.length != ts.length then throw "append: mismatched record fields"
    return ← (vs.zip ws |>.zip ts).mapM fun ((x,y),t) => do
      one (← emit [t] s!"concat@0({t.shape.headD 0}i64, {x.text}, {y.text})")
  if ["head","tail","reverse"].contains name.varName then
    let x ← one xs
    let n := x.type.shape.headD 0
    let slice := if name.varName == "head" then "0i64" else if name.varName == "tail"
      then s!"1i64 :+ {n-1}i64 * 1i64" else s!"{n-1}i64 :+ {n}i64 * -1i64"
    let trailing := String.join (x.type.shape.drop 1 |>.map fun n => s!", 0i64 :+ {n}i64 * 1i64")
    return ← emit ts s!"{x.text}[{slice}{trailing}]"
  throw s!"Futhark: unsupported intrinsic {name.varName}"
partial def compileBind : Bind → M Unit
  | .val v _ e _ => do bind v (← compileExpM e)
  | .typeE .. | .ispace .. => pure ()
  | .funE name ps _ b _ _ => compileFunction (toString name) none ps.toList b
  | _ => throw "Futhark: polymorphic binding remains after monomorphization"
partial def compileFunction (name : String) (entry : Option String) (ps : List Pat) (b : Exp) : M Unit := do
  let old := (← get).bindings
  let mut params : List SubExp := []
  for .id v _ i _ in ps do
    let ts ← types i.value
    let vs ← ts.mapM fun t => return SubExp.mk (← fresh v.varName) t
    bind v vs
    params := params ++ vs
  let (xs,ss) ← collect (compileExpM b)
  let ts := xs.map (·.type)
  let aliases := String.intercalate ", " (List.range params.length |>.map toString)
  let rals := String.intercalate ", " (List.range ts.length |>.map toString)
  let returns := "{" ++ String.intercalate ", " (ts.map fun t => toString t ++ s!"#([{aliases}], [{rals}])") ++ "}"
  let externalType (t : IRType) := String.join (t.shape.map (fun _ => "[]")) ++ t.base
  let entryReturn := "{" ++ String.intercalate ", " (ts.map externalType) ++ "}"
  let entryParams := String.intercalate ", " ((ps.zip params).map fun (p,v) => s!"{(patVar p).varName}: {externalType v.type}")
  let header := match entry with
    | none => "fun"
    | some n => s!"entry(\"{n}\", \{{entryParams}}, {entryReturn})"
  let pars := String.intercalate ", " (params.map fun p => s!"{p.text} : {p.type}")
  let funText := s!"{header}\n{name} ({pars}) : {returns} = {body ss xs}"
  modify fun s => {s with bindings := old, functions := s.functions ++ [funText]}
end

def compile (p : Prog) : Except String String := do
  if !(p.progDecs.any fun d => match d with | .entry .. => true | _ => false) then
    throw "compile: program has no entry point"
  let (_,s) ← (p.progDecs.forM fun d => match d with
    | .defE b => compileBind b
    | .entry v ps _ b _ _ => compileFunction ("entry_" ++ v.varName) (some v.varName) ps b).run {}
  return s!"name_source \{{s.next}}\ntypes \{}\n" ++ String.intercalate "\n\n" (s.statements ++ s.functions)
def compileExp (e : Exp) : Except String String := do
  let (_,s) ← (compileFunction "entry_main" (some "main") [] e).run {}
  return s!"name_source \{{s.next}}\ntypes \{}\n" ++ String.intercalate "\n\n" s.functions
end Remora.Futhark
