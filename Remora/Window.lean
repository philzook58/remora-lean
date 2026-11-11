namespace Remora.MLIR

/-- Scalar samples of an array, expressed in terms of its coordinates. -/
inductive Sample where
  | axis (i : Nat)
  | parameter (i : Nat)
  | int (n : Int)
  | float (literal : String)
  | add (a b : Sample)
  | sub (a b : Sample)
  | mul (a b : Sample)
  | select (index : Sample) (key : Int) (yes no : Sample)
  deriving Inhabited

def Sample.at (coords : List Sample) : Sample → Sample
  | .axis i => coords[i]?.getD (.axis i)
  | .add a b => .add (a.at coords) (b.at coords)
  | .sub a b => .sub (a.at coords) (b.at coords)
  | .mul a b => .mul (a.at coords) (b.at coords)
  | .select i k a b =>
    match i.at coords with
    | .int n => if n == k then a.at coords else b.at coords
    | i => .select i k (a.at coords) (b.at coords)
  | x => x

def Sample.axes (rank : Nat) : List Sample := (List.range rank).map .axis

def Sample.affine (rank : Nat) : Sample → Option (Int × List Int)
  | .axis i => if i < rank then
      some (0, (List.range rank).map fun j => if i == j then 1 else 0)
    else none
  | .int n => some (n, List.replicate rank 0)
  | .add a b => do
    let (c, xs) ← a.affine rank
    let (d, ys) ← b.affine rank
    return (c+d, List.zipWith (· + ·) xs ys)
  | .sub a b => do
    let (c, xs) ← a.affine rank
    let (d, ys) ← b.affine rank
    return (c-d, List.zipWith (· - ·) xs ys)
  | .mul a b => do
    let (c, xs) ← a.affine rank
    let (d, ys) ← b.affine rank
    if xs.all (· == 0) then return (c*d, ys.map (c * ·))
    if ys.all (· == 0) then return (c*d, xs.map (d * ·))
    none
  | _ => none

structure SpatialWindow where
  kernel : List Int
  strides : List Int
  dilations : List Int

/-- Check a CHW window over an already padded source. -/
def spatialWindow (src windows out : List Int) (axes : List Nat)
    (coordinates : List Sample) (allowSwappedKernel := false) : Option SpatialWindow := do
  let [c,h,w] := src | none
  let [c',oh,ow] := out | none
  let [ch,y,x] ← coordinates.mapM (Sample.affine windows.length) | none
  let [kh,kw] := axes | none
  let [channel,row,col] := (List.range windows.length).filter (!axes.contains ·) | none
  let (0, cs) := ch | none
  if c <= 0 || c != c' || cs != ((List.range windows.length).map fun i => if i == channel then 1 else 0)
      || windows[channel]? != some c || windows[row]? != some oh
      || windows[col]? != some ow then none else do
    let check (coord : Int × List Int) (spatial k : Nat) (size output : Int) := do
      let (0, coeffs) := coord | none
      let stride ← coeffs[spatial]?
      let dilation ← coeffs[k]?
      let kernel ← windows[k]?
      if min (min stride dilation) (min kernel output) <= 0
          || ((output-1)*stride + (kernel-1)*dilation >= size)
          || (output*stride + (kernel-1)*dilation < size)
          || coeffs != ((List.range windows.length).map fun i =>
            if i == spatial then stride else if i == k then dilation else 0)
        then none
        else some (kernel, stride, dilation)
    let matchAxes (a b : Nat) := do
      let (ky, sy, dy) ← check y row a h oh
      let (kx, sx, dx) ← check x col b w ow
      return ⟨[ky,kx], [sy,sx], [dy,dx]⟩
    matchAxes kh kw <|> (if allowSwappedKernel then matchAxes kw kh else none)

end Remora.MLIR
