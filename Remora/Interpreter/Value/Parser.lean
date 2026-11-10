import Remora.Interpreter.Value
import Remora.Parser
import Lean
namespace Remora.Interpreter.Value.Parser
open Remora.Parser

private def parseNumber (written : String) : Except String Base := do
  if written == "#t" || written == "true" then return .boolVal true
  if written == "#f" || written == "false" then return .boolVal false
  let n := ["i64","i32","f32","f64"].foldl
    (fun s suffix => if s.endsWith suffix then (s.dropEnd suffix.length).toString else s) written
  if !written.endsWith "f32" && !written.endsWith "f64" then
    if let some i := n.toInt? then return .intVal (Int64.ofInt i)
  let n := if n.startsWith "+" then (n.drop 1).toString else n
  match Lean.Json.parse n with
  | .ok (.num x) => pure (.floatVal x.toFloat.toFloat32)
  | _ => throw s!"invalid value: {written}"

private def parseIndex : SExp → Except String Index
  | .atom n _ => do
    let some n := n.toInt? | throw "expected dimension"
    if n < 0 then throw "negative dimension"
    pure (.inl n)
  | .node '[' xs _ => return .inr (← xs.mapM fun x => do
      match x with
      | .atom n _ =>
        let some n := n.toInt? | throw "expected dimension"
        if n < 0 then throw "negative dimension"
        pure n
      | _ => throw "expected dimension")
  | _ => throw "expected dimension or shape"

partial def fromSExp : SExp → Except String Val
  | .atom n _ => return arrayifyVal (.base (← parseNumber n))
  | .node '[' xs _ => do
    let vs ← xs.mapM fromSExp
    match vs with
    | v::rest => if rest.any (fun x => valShapeOf x != valShapeOf v) then
        throw "irregular array"
    | _ => pure ()
    collapse (.array [Int.ofNat vs.length] vs)
  | .node '(' (.atom "array" _ :: shape :: xs) _ => do
    let .inr ns ← parseIndex shape | throw "array: expected shape"
    let vs ← xs.mapM fromSExp
    if product ns != vs.length then throw "array: shape does not match element count"
    collapse (.array ns vs)
  | .node '(' (.atom "box" _ :: xs) _ => do
    let some inner := xs.getLast? | throw "box: missing value"
    let indices ← (xs.take (xs.length-1)).mapM parseIndex
    if indices.isEmpty then throw "box: missing indices"
    return .box indices (← fromSExp inner)
  | x => throw s!"invalid value: {x.render}"

def parseValues (file : System.FilePath) (source : String) : Except String (List Val) := do
  -- Futhark uses comma-separated arrays; Remora uses whitespace-separated arrays.
  let source := source.map (fun c => if c == ',' then ' ' else c)
  let xs ← Remora.Parser.tokenize file source
  xs.mapM fromSExp

def parseVal (file : System.FilePath) (source : String) : Except String Val := do
  match ← parseValues file source with
  | [v] => pure v
  | _ => throw "expected exactly one value"

private def byteAt (b : ByteArray) (i : Nat) : Except String UInt8 :=
  match b[i]? with
  | some x => pure x
  | none => throw "truncated Futhark binary input"
private def littleEndian (b : ByteArray) (offset count : Nat) : Except String UInt64 := do
  (List.range count).foldlM (fun acc k => do
    pure (acc ||| (UInt64.ofNat (← byteAt b (offset+k)).toNat <<< UInt64.ofNat (8*k)))) 0

/-- Futhark's version-2 binary values, as used by the upstream input fixtures. -/
def parseBinaryValues (b : ByteArray) : Except String (List Val) := do
  let mut pos := 0
  let mut result := []
  while pos < b.size do
    while pos < b.size && [9,10,13,32].contains (b[pos]!).toNat do pos := pos+1
    if pos == b.size then break
    unless (← byteAt b pos) == 98 do throw "expected Futhark binary value header"
    unless (← byteAt b (pos+1)) == 2 do throw "unsupported Futhark binary version"
    let rank := (← byteAt b (pos+2)).toNat
    let tag := String.ofList (← (List.range 4).mapM (fun k => do
      pure (Char.ofNat (← byteAt b (pos+3+k)).toNat)))
    pos := pos+7
    let mut shape := []
    for _ in [:rank] do
      let d ← littleEndian b pos 8
      shape := shape ++ [Int.ofNat d.toNat]
      pos := pos+8
    let size := if tag == " f32" then 4 else if tag == " i64" then 8
      else if tag == "bool" then 1 else 0
    if size == 0 then throw s!"unsupported Futhark binary type: {tag}"
    let mut vs := []
    for _ in [:product shape] do
      let bits ← littleEndian b pos size
      let value := if tag == " f32" then Base.floatVal (Float32.ofBits bits.toUInt32)
        else if tag == " i64" then Base.intVal (Int64.ofInt bits.toNat)
        else Base.boolVal (bits != 0)
      vs := .base value :: vs
      pos := pos+size
    result := .array shape vs.reverse :: result
  pure result.reverse

def readValues (file : System.FilePath) : IO (Except String (List Val)) := do
  try
    let b ← IO.FS.readBinFile file
    if b[0]? == some 98 then return parseBinaryValues b
    match String.fromUTF8? b with
    | some text => pure (parseValues file text)
    | none => pure (.error "invalid UTF-8 value input")
  catch e => pure (.error e.toString)
end Remora.Interpreter.Value.Parser
