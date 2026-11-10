import Remora.Syntax
import Remora.Primitive
namespace Remora
namespace Intrinsics
private def scalar (a : AtomType String) : ArrayType String := ⟨a, .concat []⟩
private def arr (a : AtomType String) (s : Shape String) : ArrayType String := ⟨a,s⟩
private def arrow (ps : List (ArrayType String)) (r : ArrayType String) : ArrayType String :=
  ps.foldr (fun p r => scalar (.arrow p r)) r
private def allT (ts : List String) (r : ArrayType String) : ArrayType String :=
  ts.foldr (fun t r => scalar (.forallE (.atom t) r)) r
private def piT (ps : List (ISpaceParam String)) (r : ArrayType String) : ArrayType String :=
  ps.foldr (fun p r => scalar (.pi p r)) r
private def dv (n : String) : Shape String := .dim (.var n)
private def sv (n : String) : Shape String := .var n
private def dims (ss : List (Shape String)) : Shape String := .concat ss
private def a (t : String) (s : Shape String) : ArrayType String := arr (.var t) s
private def i : ArrayType String := scalar .int
private def f : ArrayType String := scalar .float
private def b : ArrayType String := scalar .bool
private def unaryType (op : UnOp) : ArrayType String :=
  match op with
  | .bitNot | .popc => arrow [i] i
  | .not => arrow [b] b
  | .intToFloat => arrow [i] f
  | .intToBool => arrow [i] b
  | .boolToInt => arrow [b] i
  | .boolToFloat => arrow [b] f
  | .truncate | .round | .ceiling | .floor => arrow [f] i
  | _ => arrow [f] f
private def binaryType (op : BinOp) : ArrayType String :=
  match op with
  | .fAdd | .fSub | .fMul | .fDiv | .fPow | .fMax | .fMin => arrow [f,f] f
  | .fEq | .fNeq | .fLt | .fGt | .fLe | .fGe => arrow [f,f] b
  | .eq | .neq | .lt | .gt | .le | .ge => arrow [i,i] b
  | .and | .or | .bEq | .bNeq => arrow [b,b] b
  | _ => arrow [i,i] i

def intrinsicTypes : List (String × ArrayType String) := Id.run do
  let s := sv "s"
  let ds := dims [dv "d",s]
  let more := dims [.dim (.add [.lit 1,.var "d"]),s]
  let cell := a "t" s
  let quantified := fun body => allT ["t"] (piT [.dim "d",.shape "s"] body)
  let foldTy := fun right => allT ["t1","t2"] <| piT [.dim "d",.shape "s1",.shape "s2"] <|
    let e := a "t1" (sv "s1")
    let acc := a "t2" (sv "s2")
    arrow [arrow (if right then [e,acc] else [acc,e]) acc,acc,
      a "t1" (dims [.dim (.add [.lit 1,.var "d"]),sv "s1"])] acc
  return UnOp.all.map (fun op => (op.name,unaryType op)) ++
    BinOp.all.map (fun op => (op.name,binaryType op)) ++ [
    ("head",quantified (arrow [a "t" more] cell)),
    ("tail",quantified (arrow [a "t" more] (a "t" ds))),
    ("length",quantified (arrow [a "t" ds] i)),
    ("append",allT ["t"] <| piT [.dim "m",.dim "n",.shape "s"] <|
      arrow [a "t" (dims [dv "m",s]),a "t" (dims [dv "n",s])]
        (a "t" (dims [.dim (.add [.var "m",.var "n"]),s]))),
    ("reverse",quantified (arrow [a "t" ds] (a "t" ds))),
    ("reduce",quantified (arrow [arrow [cell,cell] cell,a "t" more] cell)),
    ("reduce/zero",quantified (arrow [arrow [cell,cell] cell,cell,a "t" ds] cell)),
    ("sum",piT [.shape "s"] (arrow [arr .int s] i)),
    ("flatten",allT ["t"] <| piT [.dim "m",.dim "n",.shape "s"] <|
      arrow [a "t" (dims [dv "m",dv "n",s])]
        (a "t" (dims [.dim (.mul [.var "m",.var "n"]),s]))),
    ("reshape",allT ["t"] <| piT [.shape "s1",.shape "s2"] <|
      arrow [a "t" (sv "s1")] (a "t" (sv "s2"))),
    ("iota",piT [.dim "d"] (arrow [arr .int (dv "d")]
      (scalar (.sigma (.shape "s") (arr .int s))))),
    ("iota/static",piT [.shape "s"] (arr .int s)),
    ("transpose2d",allT ["t"] <| piT [.dim "m",.dim "n"] <|
      arrow [a "t" (dims [dv "m",dv "n"])] (a "t" (dims [dv "n",dv "m"]))),
    ("undefined",allT ["t"] (piT [.shape "s"] cell)),
    ("index",allT ["t"] (piT [.dim "m"] (arrow [a "t" (dv "m"),i] (scalar (.var "t"))))),
    ("index2d",allT ["t"] (piT [.dim "m",.dim "n"]
      (arrow [a "t" (dims [dv "m",dv "n"]),arr .int (.dim (.lit 2))] (scalar (.var "t"))))),
    ("fold",foldTy false),("fold-right",foldTy true),
    ("trace",allT ["t","r"] <| piT [.shape "s",.shape "q"] <|
      arrow [cell,a "r" (sv "q")] (a "r" (sv "q"))),
    ("trace-file",allT ["t","r"] <| piT [.dim "d",.shape "s",.shape "q"] <|
      arrow [arr .int (dv "d"),cell,a "r" (sv "q")] (a "r" (sv "q"))),
    ("read-file",quantified (arrow [arr .int (dv "d")] cell)),
    ("read-file-f32bin",piT [.dim "d",.shape "s"] (arrow [arr .int (dv "d")] (arr .float s))),
    ("reify-dim",piT [.dim "d"] i),
    ("reify-shape",piT [.shape "s"] (scalar (.sigma (.dim "r") (arr .int (dv "r"))))) ]
end Intrinsics
abbrev intrinsicTypes := Intrinsics.intrinsicTypes
end Remora
