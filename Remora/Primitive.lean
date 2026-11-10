import Std
namespace Remora

inductive UnOp where
  | sqrt
  | fSqrt
  | bitNot
  | popc
  | not
  | intToFloat
  | intToBool
  | boolToInt
  | boolToFloat
  | truncate
  | round
  | ceiling
  | floor
  | fLn
  | fTanh
  | fExp
  deriving Repr, BEq, Ord, Inhabited

def UnOp.name : UnOp → String
  | .sqrt => "sqrt"
  | .fSqrt => "f.sqrt"
  | .bitNot => "bit-not"
  | .popc => "popc"
  | .not => "not"
  | .intToFloat => "i->f"
  | .intToBool => "i->bool"
  | .boolToInt => "bool->i"
  | .boolToFloat => "bool->f"
  | .truncate => "truncate"
  | .round => "round"
  | .ceiling => "ceiling"
  | .floor => "floor"
  | .fLn => "f.ln"
  | .fTanh => "f.tanh"
  | .fExp => "f.exp"

def UnOp.all : List UnOp := [.sqrt, .fSqrt, .bitNot, .popc, .not, .intToFloat, .intToBool, .boolToInt, .boolToFloat, .truncate, .round, .ceiling, .floor, .fLn, .fTanh, .fExp]

inductive BinOp where
  | add
  | sub
  | mul
  | div
  | pow
  | mod
  | max
  | min
  | bitAnd
  | bitOr
  | bitXor
  | shl
  | shr
  | fAdd
  | fSub
  | fMul
  | fDiv
  | fPow
  | fMax
  | fMin
  | eq
  | neq
  | lt
  | gt
  | le
  | ge
  | fEq
  | fNeq
  | fLt
  | fGt
  | fLe
  | fGe
  | and
  | or
  | bEq
  | bNeq
  deriving Repr, BEq, Ord, Inhabited

def BinOp.name : BinOp → String
  | .add => "+"
  | .sub => "-"
  | .mul => "*"
  | .div => "/"
  | .pow => "^"
  | .mod => "mod"
  | .max => "max"
  | .min => "min"
  | .bitAnd => "bit-and"
  | .bitOr => "bit-or"
  | .bitXor => "bit-xor"
  | .shl => "shl"
  | .shr => "shr"
  | .fAdd => "f.+"
  | .fSub => "f.-"
  | .fMul => "f.*"
  | .fDiv => "f./"
  | .fPow => "f.^"
  | .fMax => "f.max"
  | .fMin => "f.min"
  | .eq => "=="
  | .neq => "!="
  | .lt => "<"
  | .gt => ">"
  | .le => "<="
  | .ge => ">="
  | .fEq => "f.=="
  | .fNeq => "f.!="
  | .fLt => "f.<"
  | .fGt => "f.>"
  | .fLe => "f.<="
  | .fGe => "f.>="
  | .and => "and"
  | .or => "or"
  | .bEq => "bool.=="
  | .bNeq => "bool.!="

def BinOp.all : List BinOp := [.add, .sub, .mul, .div, .pow, .mod, .max, .min, .bitAnd, .bitOr, .bitXor, .shl, .shr, .fAdd, .fSub, .fMul, .fDiv, .fPow, .fMax, .fMin, .eq, .neq, .lt, .gt, .le, .ge, .fEq, .fNeq, .fLt, .fGt, .fLe, .fGe, .and, .or, .bEq, .bNeq]

end Remora
