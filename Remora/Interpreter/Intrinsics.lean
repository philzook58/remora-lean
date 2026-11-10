import Remora.Interpreter.Value
import Remora.Intrinsics
namespace Remora.Interpreter
/-- Counts of type, index and term arguments, in their source calling order. -/
def intrinsicArity (name : String) : Nat × Nat × Nat :=
  if UnOp.all.any (fun op => op.name == name) then (0,0,1)
  else if BinOp.all.any (fun op => op.name == name) then (0,0,2)
  else match name with
  | "head" | "tail" | "length" | "reverse" | "transpose2d" => (1,2,1)
  | "append" => (1,3,2)
  | "reduce" => (1,2,2)
  | "reduce/zero" => (1,2,3)
  | "fold" | "fold-right" => (2,3,3)
  | "sum" | "iota" => (0,1,1)
  | "flatten" => (1,3,1)
  | "reshape" | "read-file" => (1,2,1)
  | "iota/static" | "reify-dim" | "reify-shape" => (0,1,0)
  | "undefined" => (1,1,0)
  | "index" => (1,1,2)
  | "index2d" => (1,2,2)
  | "trace" => (2,2,2)
  | "trace-file" => (2,3,3)
  | "read-file-f32bin" => (0,2,1)
  | _ => (0,0,0)

def builtin (name : String) : Val :=
  let (t,i,_) := intrinsicArity name
  .builtin name t i [] []

def scalarVal (b : Base) : Val := .array [] [.base b]

def evalUnary (op : UnOp) (x : Val) : Except String Val := do
  let result ← match op with
  | .bitNot => pure (.intVal (~~~ (← asInt x)))
  | .popc => do
    let n ← asInt x
    pure (.intVal (Int64.ofNat ((List.range 64).countP (fun k => n.toUInt64.toNat.testBit k))))
  | .not => pure (.boolVal (!(← asBool x)))
  | .intToFloat => pure (.floatVal (← asInt x).toFloat32)
  | .intToBool => pure (.boolVal ((← asInt x) != 0))
  | .boolToInt => pure (.intVal (if ← asBool x then 1 else 0))
  | .boolToFloat => pure (.floatVal (if ← asBool x then 1 else 0))
  | .truncate => pure (.intVal (← asFloat x).toInt64)
  | .round => pure (.intVal (← asFloat x).round.toInt64)
  | .ceiling => pure (.intVal (← asFloat x).ceil.toInt64)
  | .floor => pure (.intVal (← asFloat x).floor.toInt64)
  | .sqrt | .fSqrt => pure (.floatVal (← asFloat x).sqrt)
  | .fLn => pure (.floatVal (← asFloat x).log)
  | .fTanh => pure (.floatVal (← asFloat x).tanh)
  | .fExp => pure (.floatVal (← asFloat x).exp)
  pure (scalarVal result)

def evalBinary (op : BinOp) (x y : Val) : Except String Val := do
  let intOp := fun (f : Int64 → Int64 → Int64) => do
    pure (scalarVal (.intVal (f (← asInt x) (← asInt y))))
  let intCmp := fun (f : Int64 → Int64 → Bool) => do
    pure (scalarVal (.boolVal (f (← asInt x) (← asInt y))))
  let floatOp := fun (f : Float32 → Float32 → Float32) => do
    pure (scalarVal (.floatVal (f (← asFloat x) (← asFloat y))))
  let floatCmp := fun (f : Float32 → Float32 → Bool) => do
    pure (scalarVal (.boolVal (f (← asFloat x) (← asFloat y))))
  let boolOp := fun (f : Bool → Bool → Bool) => do
    pure (scalarVal (.boolVal (f (← asBool x) (← asBool y))))
  match op with
  | .add => intOp (·+·)
  | .sub => intOp (·-·)
  | .mul => intOp (·*·)
  | .div => if (← asInt y) == 0 then throw "division by zero" else intOp (·/·)
  | .mod => if (← asInt y) == 0 then throw "modulo by zero" else intOp (·%·)
  | .pow => intOp (fun a b => a ^ b.toInt.toNat)
  | .max => intOp max
  | .min => intOp min
  | .bitAnd => intOp (· &&& ·)
  | .bitOr => intOp (· ||| ·)
  | .bitXor => intOp (· ^^^ ·)
  | .shl => intOp (· <<< ·)
  | .shr => intOp (· >>> ·)
  | .eq => intCmp (· == ·)
  | .neq => intCmp (· != ·)
  | .lt => intCmp (fun a b => a < b)
  | .gt => intCmp (fun a b => a > b)
  | .le => intCmp (fun a b => a ≤ b)
  | .ge => intCmp (fun a b => a ≥ b)
  | .fAdd => floatOp (·+·)
  | .fSub => floatOp (·-·)
  | .fMul => floatOp (·*·)
  | .fDiv => floatOp (·/·)
  | .fPow => floatOp Float32.pow
  | .fMax => floatOp max
  | .fMin => floatOp min
  | .fEq => floatCmp (· == ·)
  | .fNeq => floatCmp (· != ·)
  | .fLt => floatCmp (fun a b => a < b)
  | .fGt => floatCmp (fun a b => a > b)
  | .fLe => floatCmp (fun a b => a ≤ b)
  | .fGe => floatCmp (fun a b => a ≥ b)
  | .and => boolOp (· && ·)
  | .or => boolOp (· || ·)
  | .bEq => boolOp (· == ·)
  | .bNeq => boolOp (· != ·)
end Remora.Interpreter
