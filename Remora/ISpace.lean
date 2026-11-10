import Std
namespace Remora
inductive Dim (v : Type) where
  | var : v → Dim v
  | lit : Int → Dim v
  | add : List (Dim v) → Dim v
  | mul : List (Dim v) → Dim v
  | sub : List (Dim v) → Dim v
  deriving Repr, BEq, Ord, Inhabited
inductive Shape (v : Type) where
  | var : v → Shape v
  | dim : Dim v → Shape v
  | concat : List (Shape v) → Shape v
  deriving Repr, BEq, Ord, Inhabited
inductive ISpace (v : Type) where
  | dim : Dim v → ISpace v
  | shape : Shape v → ISpace v
  deriving Repr, BEq, Ord, Inhabited
inductive ISpaceParam (v : Type) where
  | dim : v → ISpaceParam v
  | shape : v → ISpaceParam v
  deriving Repr, BEq, Ord, Inhabited
instance : Append (Dim v) := ⟨fun a b => .add [a,b]⟩
instance : Append (Shape v) := ⟨fun a b => .concat [a,b]⟩
def ispaceToShape : ISpace v → Shape v
  | .dim d => .dim d
  | .shape s => s
def mapISpace (f : Dim v → a) (g : Shape v → a) : ISpace v → a
  | .dim d => f d
  | .shape s => g s
def fromDim : ISpace v → Option (Dim v)
  | .dim d => some d
  | _ => none
def fromShape : ISpace v → Option (Shape v)
  | .shape s => some s
  | _ => none
def isDim (s : ISpace v) : Bool := (fromDim s).isSome
def isShape (s : ISpace v) : Bool := (fromShape s).isSome
def unISpaceParam : ISpaceParam v → v
  | .dim v | .shape v => v
def coerceToDim : Shape v → Option (Dim v)
  | .var v => some (.var v)
  | .dim d => some d
  | _ => none
partial def normDim [Ord v] : Dim v → Dim v
  | .lit n => .lit n
  | .var v => .var v
  | .sub ds => .sub (ds.map normDim)
  | .add ds => normOp true ds
  | .mul ds => normOp false ds
where
  normOp [Ord v] (adding : Bool) (ds : List (Dim v)) : Dim v := Id.run do
    let terms := ds.flatMap fun d => match normDim d with
      | .add xs => if adding then xs else [.add xs]
      | .mul xs => if adding then [.mul xs] else xs
      | d => [d]
    let (c, vs) := terms.foldr (fun d (c,vs) => match d with
      | .lit n => (if adding then n+c else n*c, vs)
      | d => (c,d::vs)) (if adding then 0 else 1, [])
    if vs.isEmpty then return .lit c
    let vs := vs.mergeSort (fun a b => compare a b != .gt)
    return if adding then .add (.lit c :: vs) else .mul (.lit c :: vs)
partial def normShape [Ord v] : Shape v → Shape v
  | .var v => .var v
  | .dim d => .dim (normDim d)
  | .concat ss =>
    let xs := ss.flatMap fun s => match normShape s with
      | .concat xs => xs
      | s => [s]
    match xs with
    | [s] => s
    | _ => .concat xs
def peelShape [Ord v] (s : Shape v) : Shape v :=
  match normShape s with
  | .concat (_::ss) => .concat ss
  | _ => .concat []
def intsToShape (ns : List Int) : Shape v := .concat (ns.map (Shape.dim ∘ Dim.lit))
partial def dimToInt [Monad m] (resolve : v → m Int) : Dim v → m Int
  | .var v => resolve v
  | .lit n => pure n
  | .add ds => return (← ds.mapM (dimToInt resolve)).foldl (· + ·) 0
  | .mul ds => return (← ds.mapM (dimToInt resolve)).foldl (· * ·) 1
  | .sub [] => pure 0
  | .sub [d] => return -(← dimToInt resolve d)
  | .sub (d::ds) => return (← dimToInt resolve d) - (← ds.mapM (dimToInt resolve)).foldl (· + ·) 0
partial def shapeToInts [Monad m] (dim : Dim v → m Int) (resolve : v → m (List Int)) : Shape v → m (List Int)
  | .var v => resolve v
  | .dim d => return [← dim d]
  | .concat ss => return (← ss.mapM (shapeToInts dim resolve)).flatten
def isEmptyShape [Ord v] (s : Shape v) : Bool :=
  match normShape s with
  | .concat [] => true
  | _ => false
partial def unfoldShape : Shape v → List (Shape v)
  | .concat ss => ss.flatMap unfoldShape
  | s => [s]
end Remora
