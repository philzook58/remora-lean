import Remora.Syntax
import Lean

/-! Structural AST JSON with the tags and field names of the Haskell serializer. -/
namespace Remora
open Lean
private def obj (tag : String) (fields : List (String × Json) := []) : Json :=
  Json.mkObj (("tag", toJson tag) :: fields)
private def jsList (f : α → Json) (xs : List α) : Json := .arr (xs.map f).toArray

instance [ToJson α] : ToJson (NEList α) := ⟨fun xs => toJson xs.toList⟩
instance : ToJson VName := ⟨fun v => obj "VName" [("varName",toJson v.varName),
  ("varTag",obj "Tag" [("getTag",toJson v.varTag)])]⟩

partial def dimToJson [ToJson v] : Dim v → Json
  | .var v => obj "DimVar" [("name",toJson v)]
  | .lit n => obj "DimN" [("val",toJson n)]
  | .add ds => obj "Add" [("dims",jsList dimToJson ds)]
  | .mul ds => obj "Mul" [("dims",jsList dimToJson ds)]
  | .sub ds => obj "Sub" [("dims",jsList dimToJson ds)]
instance [ToJson v] : ToJson (Dim v) := ⟨dimToJson⟩
partial def shapeToJson [ToJson v] : Shape v → Json
  | .var v => obj "ShapeVar" [("name",toJson v)]
  | .dim d => obj "ShapeDim" [("dim",toJson d)]
  | .concat ss => obj "Concat" [("shapes",jsList shapeToJson ss)]
instance [ToJson v] : ToJson (Shape v) := ⟨shapeToJson⟩
instance [ToJson v] : ToJson (ISpace v) := ⟨fun
  | .dim d => obj "Dim" [("dim",toJson d)]
  | .shape s => obj "Shape" [("shape",toJson s)]⟩
instance [ToJson v] : ToJson (ISpaceParam v) := ⟨fun
  | .dim v => obj "DimParam" [("name",toJson v)]
  | .shape v => obj "ShapeParam" [("name",toJson v)]⟩
instance [ToJson v] : ToJson (TypeParamExp v) := ⟨fun
  | .atom v => obj "TEAtomTypeParam" [("name",toJson v)]
  | .array v => obj "TEArrayTypeParam" [("name",toJson v)]⟩
instance [ToJson v] : ToJson (TypeParam v) := ⟨fun
  | .atom v => obj "AtomTypeParam" [("name",toJson v)]⟩

partial def typeExpToJson [ToJson v] : TypeExp v → Json
  | .atomVar v _ => obj "TEAtomVar" [("name",toJson v)]
  | .arrayVar v _ => obj "TEArrayVar" [("name",toJson v)]
  | .bool _ => obj "TEBool"
  | .int _ => obj "TEInt"
  | .float _ => obj "TEFloat"
  | .array t s _ => obj "TEArray" [("elem",typeExpToJson t),("shape",toJson s)]
  | .arrow a b _ => obj "TEArrow" [("in",typeExpToJson a),("out",typeExpToJson b)]
  | .forallE ps t _ => obj "TEForall" [("params",toJson ps),("body",typeExpToJson t)]
  | .pi ps t _ => obj "TEPi" [("params",toJson ps),("body",typeExpToJson t)]
  | .sigma ps t _ => obj "TESigma" [("params",toJson ps),("body",typeExpToJson t)]
  | .record fs _ => obj "TERecord" [("fields", jsList
      (fun (name,t) => .arr #[toJson name,typeExpToJson t]) fs.toList)]
instance [ToJson v] : ToJson (TypeExp v) := ⟨typeExpToJson⟩

mutual
partial def atomTypeToJson [ToJson v] : AtomType v → Json
  | .var v => obj "AtomTypeVar" [("name",toJson v)]
  | .bool => obj "Bool"
  | .int => obj "Int"
  | .float => obj "Float"
  | .arrow a b => obj "(:->)" [("in",arrayTypeToJson a),("out",arrayTypeToJson b)]
  | .forallE p t => obj "Forall" [("param",toJson p),("body",arrayTypeToJson t)]
  | .pi p t => obj "Pi" [("param",toJson p),("body",arrayTypeToJson t)]
  | .sigma p t => obj "Sigma" [("param",toJson p),("body",arrayTypeToJson t)]
  | .record fs => obj "Record" [("fields",jsList
      (fun (name,t) => .arr #[toJson name,arrayTypeToJson t]) fs.toList)]
partial def arrayTypeToJson [ToJson v] (t : ArrayType v) : Json :=
  obj "A" [("arrayTypeAtom",atomTypeToJson t.atom),("arrayTypeShape",toJson t.shape)]
end
instance [ToJson v] : ToJson (AtomType v) := ⟨atomTypeToJson⟩
instance [ToJson v] : ToJson (ArrayType v) := ⟨arrayTypeToJson⟩
instance [ToJson v] : ToJson (Ty v) := ⟨fun
  | .atom t => obj "AtomType" [("type",toJson t)]
  | .array t => obj "ArrayType" [("type",toJson t)]⟩
instance : ToJson Base := ⟨fun
  | .boolVal b => obj "BoolVal" [("lit",toJson b)]
  | .intVal n => obj "IntVal" [("lit",toJson n.toInt)]
  | .floatVal x => obj "FloatVal" [("lit",toJson x.toFloat)]⟩

class JsonInfo (f : Type → Type) where
  jsonInfo {α : Type} [ToJson α] : f α → Json
export JsonInfo (jsonInfo)
instance : JsonInfo NoInfo := ⟨fun _ => .null⟩
instance : JsonInfo Info := ⟨fun x => toJson x.value⟩
instance : ToJson (NoInfo α) := ⟨fun _ => .null⟩
instance [ToJson α] : ToJson (Info α) := ⟨fun x => toJson x.value⟩

instance [ToJson v] [ToJson (te v)] [JsonInfo f] : ToJson (PatBase te f v) := ⟨fun
  | .id v t i _ => obj "PatId" [("var",toJson v),("type",toJson t),("info",jsonInfo i)]⟩

mutual
partial def atomToJson [ToJson v] [ToJson (te v)] [ToJson (tp v)] [JsonInfo f] :
    AtomBase te tp f v → Json
  | .base b i _ => obj "Base" [("lit",toJson b),("info",jsonInfo i)]
  | .lambda p b i _ => obj "Lambda" [("param",toJson p),("body",expToJson b),("info",jsonInfo i)]
  | .tLambda p b i _ => obj "TLambda" [("param",toJson p),("body",expToJson b),("info",jsonInfo i)]
  | .iLambda p b i _ => obj "ILambda" [("param",toJson p),("body",expToJson b),("info",jsonInfo i)]
  | .box d b t i _ => obj "Box" [("ispace",toJson d),("array",expToJson b),("type",toJson t),("info",jsonInfo i)]
partial def bindToJson [ToJson v] [ToJson (te v)] [ToJson (tp v)] [JsonInfo f] :
    BindBase te tp f v → Json
  | .val v t e _ => obj "BindVal" [("var",toJson v),("type",toJson t),("expr",expToJson e)]
  | .typeE p t i _ => obj "BindType" [("var",toJson p),("type",toJson t),("info",jsonInfo i)]
  | .ispace p d _ => obj "BindISpace" [("var",toJson p),("ispace",toJson d)]
  | .funE v ps t e i _ => obj "BindFun" [("var",toJson v),("params",toJson ps),("type",toJson t),("expr",expToJson e),("info",jsonInfo i)]
  | .tFun v ps t e i _ => obj "BindTFun" [("var",toJson v),("params",toJson ps),("type",toJson t),("expr",expToJson e),("info",jsonInfo i)]
  | .iFun v ps t e i _ => obj "BindIFun" [("var",toJson v),("params",toJson ps),("type",toJson t),("expr",expToJson e),("info",jsonInfo i)]
partial def expToJson [ToJson v] [ToJson (te v)] [ToJson (tp v)] [JsonInfo f] :
    ExpBase te tp f v → Json
  | .var v i _ => obj "Var" [("name",toJson v),("info",jsonInfo i)]
  | .array ds xs i _ => obj "Array" [("dims",toJson ds),("atoms",jsList atomToJson xs.toList),("info",jsonInfo i)]
  | .emptyArray ds t i _ => obj "EmptyArray" [("dims",toJson ds),("type",toJson t),("info",jsonInfo i)]
  | .frame ds xs i _ => obj "Frame" [("dims",toJson ds),("exprs",jsList expToJson xs.toList),("info",jsonInfo i)]
  | .emptyFrame ds t i _ => obj "EmptyFrame" [("dims",toJson ds),("type",toJson t),("info",jsonInfo i)]
  | .app f a i _ => obj "App" [("fun",expToJson f),("arg",expToJson a),("info",jsonInfo i)]
  | .tApp f a i _ => obj "TApp" [("fun",expToJson f),("arg",toJson a),("info",jsonInfo i)]
  | .iApp f a i _ => obj "IApp" [("fun",expToJson f),("arg",toJson a),("info",jsonInfo i)]
  | .unbox p v b e i _ => obj "Unbox" [("ispace",toJson p),("var",toJson v),("target",expToJson b),("body",expToJson e),("info",jsonInfo i)]
  | .letE bs e i _ => obj "Let" [("binds",jsList bindToJson bs.toList),("body",expToJson e),("info",jsonInfo i)]
  | .structE fs i _ => obj "Struct" [("fields",jsList
      (fun (name,s,e) => .arr #[toJson name,toJson s,expToJson e]) fs.toList),("info",jsonInfo i)]
  | .fieldProj e f i _ => obj "FieldProj" [("expr",expToJson e),("field",toJson f),("info",jsonInfo i)]
end
instance [ToJson v] [ToJson (te v)] [ToJson (tp v)] [JsonInfo f] : ToJson (AtomBase te tp f v) := ⟨atomToJson⟩
instance [ToJson v] [ToJson (te v)] [ToJson (tp v)] [JsonInfo f] : ToJson (BindBase te tp f v) := ⟨bindToJson⟩
instance [ToJson v] [ToJson (te v)] [ToJson (tp v)] [JsonInfo f] : ToJson (ExpBase te tp f v) := ⟨expToJson⟩
instance [ToJson v] [ToJson (te v)] [ToJson (tp v)] [JsonInfo f] : ToJson (DeclBase te tp f v) := ⟨fun
  | .defE b => obj "Def" [("bind",toJson b)]
  | .entry v ps t e i _ => obj "Entry" [("var",toJson v),("params",toJson ps),("type",toJson t),("expr",toJson e),("info",jsonInfo i)]⟩
instance [ToJson v] [ToJson (te v)] [ToJson (tp v)] [JsonInfo f] : ToJson (ProgBase te tp f v) := ⟨fun p =>
  obj "Prog" [("progDecs",toJson p.progDecs)]⟩
instance : ToJson Import := ⟨fun i => obj "Import" [("file",toJson i.fileName.toString)]⟩
end Remora
