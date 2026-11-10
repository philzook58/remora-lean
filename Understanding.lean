import Remora

/-!
# Understanding Remora

Read this file from top to bottom in the Lean editor, or run:

    lake env lean Understanding.lean

The first half is a small, pure model of tensors and cell lifting. The second half
runs the real parser, typechecker, interpreter, and compiler. The `#guard_msgs`
blocks check the displayed results; when experimenting, change the expected
comment too, or temporarily remove the `#guard_msgs in` line.

This is a starting model, with no claim of compiler correctness. The implementation
stores arrays as shapes plus flat values and uses machine integers/floats. Here we
use mathematical `Int` so the shape and application rules are easy to see.
See `docs/understanding.md` for the corresponding code map and pass walkthrough.
-/

namespace Understanding

/-! ## 1. Scalars, vectors, and matrices -/

/-- A tensor is a scalar, or a finite family of tensors of the remaining shape. -/
abbrev Tensor (α : Type) : List Nat → Type
  | [] => α
  | n :: s => Fin n → Tensor α s

-- These are definitional equalities. The scalar's shape is [], and rank is shape length.
example : Tensor Int [] = Int := rfl
example : Tensor Int [2, 3] = (Fin 2 → Fin 3 → Int) := rfl

/-- Enumerate tensor elements in row-major order for display. -/
def elements : (s : List Nat) → Tensor α s → List α
  | [], x => [x]
  | n :: s, x => (List.ofFn fun i : Fin n ↦ elements s (x i)).flatten

def vector : Tensor Int [2] := fun i ↦ Int.ofNat i.val + 1
def matrix : Tensor Int [2, 3] := fun i j ↦ 10 * (3 * Int.ofNat i.val + Int.ofNat j.val + 1)

/-- info: [1, 2] -/
#guard_msgs in
#eval elements [2] vector

/-- info: [10, 20, 30, 40, 50, 60] -/
#guard_msgs in
#eval elements [2, 3] matrix

/-! ## 2. Cells, frames, and lifting

A function declares its *cell* shape. Extra leading dimensions form its *frame*:

    input shape = frame ++ input cell shape
    output shape = frame ++ output cell shape

A scalar function has cell shape []; applying it to a vector maps over that vector.
A function consuming a length-three vector has cell shape [3]; applying it to our
matrix maps over the two rows. The caller's shape and the declared parameter shape
together determine this split.
-/

/-- Apply a cell function independently at every position of a fixed frame. -/
def lift (cell result : List Nat) (f : Tensor α cell → Tensor β result) :
    (frame : List Nat) → Tensor α (frame ++ cell) → Tensor β (frame ++ result)
  | [], x => f x
  | _ :: frame, x => fun i ↦ lift cell result f frame (x i)

/-- info: [11, 12] -/
#guard_msgs in
#eval elements [2] (lift [] [] (fun x : Int ↦ x + 10) [2] vector)

def sumRow (row : Tensor Int [3]) : Int := row 0 + row 1 + row 2

/-- info: [60, 150] -/
#guard_msgs in
#eval elements [2] (lift [3] [] sumRow [2] matrix)

-- A useful rewrite law for this model: two lifts over the same frame compose.
-- The proof is independent of arithmetic or storage representation.
theorem lift_comp (f : Tensor α s → Tensor β t) (g : Tensor β t → Tensor γ r)
    (frame : List Nat) : ∀ x : Tensor α (frame ++ s),
    lift t r g frame (lift s t f frame x) = lift s r (g ∘ f) frame x := by
  induction frame with
  | nil => intro x; rfl
  | cons n frame ih =>
    intro x
    funext i
    exact ih (x i)

-- For multiple arguments, Remora also extends shorter PREFIX frames.
-- Here [2] is a prefix of [2,3]: vector i is repeated across all three columns.
-- A [3] frame would not agree with [2,3]. This differs from Torch's usual
-- right-aligned broadcasting; this example would need a [2,1] Torch operand.
def prefixSum : Tensor Int [2, 3] := fun i j ↦ vector i + matrix i j

/-- info: [11, 21, 31, 42, 52, 62] -/
#guard_msgs in
#eval elements [2, 3] prefixSum

/-! ## 3. The quantifiers and boxes

Remora's `Forall` binds types; `Pi` binds dimensions or shapes. A shallow Lean model
uses binders over `Type`, `Nat`, and `List Nat`, respectively. Remora's type binders
also distinguish atom and array kinds. Our example illustrates the binders without
encoding that kind system.
-/

def identity (α : Type) (shape : List Nat) (x : Tensor α shape) : Tensor α shape := x

-- Both quantifiers are Lean dependent function binders, over different domains.
example : (∀ α : Type, ∀ shape : List Nat, Tensor α shape → Tensor α shape) := identity

/-- info: [1, 2] -/
#guard_msgs in
#eval elements [2] (identity Int [2] vector)

-- A box packages a hidden shape together with a value having that shape.
-- A dimension-only package would instead have type `Σ n : Nat, Tensor Int [n]`.
def boxed : Σ shape : List Nat, Tensor Int shape := ⟨[2], vector⟩

/-- info: [1, 2] -/
#guard_msgs in
#eval elements boxed.1 boxed.2

/-! ## 4. Ask the real interpreter

The pure `lift` above covers one fixed cell function. The real application rule also
handles arrays of functions, finds the principal frame, and extends shorter frames.
Read the `.app` branches in `Remora/TypeCheck.lean` and `Remora/Interpreter.lean`.
These helpers work with self-contained strings; use the CLI for source imports.
-/

private def require (r : Except String α) : IO α :=
  match r with
  | .ok x => pure x
  | .error e => throw (IO.userError e)

/-- Parse, check, and interpret a self-contained Remora expression. -/
def eval (source : String) : IO String := do
  let e ← require (Remora.Parser.parseExp "<understanding>" source)
  return toString (← require (← Remora.Pipeline.interpretExp e))

/-- info: "3" -/
#guard_msgs in
#eval eval "(+ 1 2)"

/-- info: "[11 12]" -/
#guard_msgs in
#eval eval "((fn ((x Int)) (+ x 10)) [1 2])"

/-- info: "[[11 21 31] [42 52 62]]" -/
#guard_msgs in
#eval eval "(+ [1 2] [[10 20 30] [40 50 60]])"

/-- info: "[60 150]" -/
#guard_msgs in
#eval eval "(@sum _ ([3]) [[10 20 30] [40 50 60]])"

-- Explicit type application: instantiate the scalar identity at Int, then lift it.
/-- info: "[1 2]" -/
#guard_msgs in
#eval eval "(let ((t-fun (id (&a)) (fn ((x &a)) x))) ((t-app id Int) [1 2]))"

-- Explicit shape application: this identity consumes a whole [2] cell.
/-- info: "[1 2]" -/
#guard_msgs in
#eval eval "(let ((i-fun (id (@s)) (fn ((xs [Int @s])) xs))) ((i-app id [2]) [1 2]))"

-- Opening a shape box makes its shape available to instantiate sum.
-- This works in the interpreter; existential boxes remain unsupported by the backend.
/-- info: "3" -/
#guard_msgs in
#eval eval "(unbox (@s xs (box ([2]) [1 2] (Sigma (@s) (A Int @s)))) (@sum _ (@s) xs))"

/-! ## 5. Follow one program through the compiler

`identity` is polymorphic; `shift` is an ordinary named function capturing `offset`.
The result is [11 12]. Follow the same source through `compileWithASTs`, which runs
the real compilation sequence. Generating its IR needs no Futhark installation.
-/

def walkthrough : String := "
(entry (main)
  (let ((val offset 10)
        (t-fun (identity (&a)) (fn ((x &a)) x))
        (fun (shift (x Int)) (+ x offset)))
    (shift ((t-app identity Int) [1 2]))))"

private def parseWalkthrough : IO Remora.UncheckedProg := do
  let (_, p) ← require (Remora.Parser.parse "<walkthrough>" walkthrough)
  return p

/-- info: "[11 12]" -/
#guard_msgs in
#eval do
  let value ← require (← Remora.Pipeline.interpret "main" [] (← parseWalkthrough))
  return toString value

/-- Run the compiler and retain each actual stage plus the final SOACS text. -/
def compileWalkthrough : IO (List Remora.Pipeline.PassAST × String) := do
  require (← Remora.Pipeline.compileWithASTs (← parseWalkthrough))

/-- info: ["parse", "uniquify", "typecheck", "monomorphize", "lambdalift", "defunctionalize"] -/
#guard_msgs in
#eval do
  let (stages, _) ← compileWalkthrough
  return stages.map Remora.Pipeline.PassAST.passName

/-- Print one stage's JSON, or the final IR when called with "futhark". -/
def showStage (name : String) : IO Unit := do
  let (stages, ir) ← compileWalkthrough
  if name == "futhark" then
    IO.println ir
  else
    match stages.find? (fun stage ↦ stage.passName == name) with
    | some stage => IO.println stage.passJSON.pretty
    | none => throw (IO.userError s!"Unknown stage: {name}")

-- Uncomment one at a time; full ASTs are intentionally opt-in.
-- #eval showStage "typecheck"
-- #eval showStage "monomorphize"
-- #eval showStage "lambdalift"
-- #eval showStage "futhark"

-- Experiments: change offset to 20; change the input to a 2×3 matrix; change
-- shift's parameter to a vector cell and predict the frame and output shape.
-- The code map gives specific places to inspect before attempting a compiler change.

end Understanding
