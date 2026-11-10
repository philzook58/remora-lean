# Reading and changing the compiler

Start with [Understanding.lean](../Understanding.lean). It is an ordinary Lean file
with checked outputs, a shallow tensor model, one small lifting theorem, and calls
into the real implementation. Run it from the project root:

```sh
lake build
lake env lean Understanding.lean
```

Z3 must be available for the typechecker. Reading the file and emitting SOACS need
no Futhark installation. No new packages are needed. The tensor model describes
values and some application rules; it is not a verified semantics of the compiler.
Its `Int` arithmetic also differs from the implementation's machine integers.

## The central application rule

For each argument, split its shape into a leading **frame** and the **cell** shape
declared by the function parameter. The function array has a frame too. These frames
must agree by prefix: a shorter frame is extended along the additional trailing
dimensions. The longest agreeing frame is the **principal frame**. Apply the cell
functions there and append their result cell shape to that frame.

For a scalar function, input `[2,3]` has frame `[2,3]` and cell `[]`. For a function
consuming a length-three vector, that input has frame `[2]` and cell `[3]`.

In `(+ [1 2] [[10 20 30] [40 50 60]])`, addition consumes scalar cells. The frames
are `[2]` and `[2,3]`; the first vector extends across columns, giving
`[[11 21 31] [42 52 62]]`. A length-three vector has frame `[3]`, which would fail
prefix agreement here. For a Torch implementation, the length-two operand would
need a `[2,1]` shape to obtain this behavior.

## Read the implementation in this order

| Question | Code to read |
| --- | --- |
| What does a dimension or shape expression contain? | [ISpace.lean](../Remora/ISpace.lean): `Dim`, `Shape`, normalization |
| What syntax and types exist? | [Syntax.lean](../Remora/Syntax.lean): `AtomType`, `ArrayType`, `ExpBase`, AST aliases |
| What does source syntax become? | [Parser.lean](../Remora/Parser.lean): delimiter trees and lowering |
| How are scopes and shadowing represented? | [Uniquify.lean](../Remora/Uniquify.lean): source names to tagged `VName`s |
| Where are cells and frames inferred? | [TypeCheck.lean](../Remora/TypeCheck.lean): `checkExp'`, especially `.app` |
| How are shape relations decided? | [Prop.lean](../Remora/Prop.lean) and [Symbolic.lean](../Remora/Symbolic.lean) |
| How does a checked application execute? | [Interpreter.lean](../Remora/Interpreter.lean): `eval`'s `.app` branch, then `apply` |
| Which passes actually run, in which order? | [Pipeline.lean](../Remora/Pipeline.lean): `compile`, `compileWithASTs` |
| Where does lifting become loops/maps? | [Futhark.lean](../Remora/Futhark.lean): `compileExpM`, `applyName`, `mapNest` |

`UncheckedProg` uses source names and empty annotations. `UniqueProg` uses tagged
names but still has empty annotations. `Prog` uses tagged names and populated type
annotations. The compiler passes then reuse `Prog`; their output conditions are
conventions checked by subsequent code, rather than distinct Lean types for each IR.

## One program through the actual passes

This is `walkthrough` in the Lean file:

```scheme
(entry (main)
  (let ((val offset 10)
        (t-fun (identity (&a)) (fn ((x &a)) x))
        (fun (shift (x Int)) (+ x offset)))
    (shift ((t-app identity Int) [1 2]))))
```

It evaluates to `[11 12]`. Uncomment `#eval showStage "STAGE"` in the Lean file to
inspect any stage below. This uses the same sequence as whole-program compilation.

| Stage | What changes in this program | What later code relies on |
| --- | --- | --- |
| `parse` | Parentheses become applications, bindings, and lambdas; annotations are empty. | Source structure and positions are available. |
| `uniquify` | The two source parameters called `x` receive distinct tagged names. Uses of `offset` and `shift` resolve to their binders. | Substitution and free-variable analysis identify binders by tags. |
| `typecheck` | `identity` receives a universally quantified type. Scalar parameters and vector applications receive cell/frame information. The result has shape `[2]`. | Later passes can inspect types without inferring them again. |
| `monomorphize` | The type application of `identity` at `Int` becomes a concrete term lambda. The ordinary local `shift` definition stays shared. | This example has no remaining type abstractions/applications. General higher-order elimination is incomplete. |
| `lambdalift` | The local `shift` and instantiated identity become top-level `lifted` functions. `shift` gets an explicit scalar `offset` parameter; its closure reference supplies the captured value. | This example's local functions are named top-level code with explicit captures. |
| `defunctionalize` | No change here: the functions already return non-function values. | This pass currently eta-expands top-level partial applications; it is not a general closure-tag transformation. |
| `futhark` | The emitter generates two scalar functions and two `map` operations over the length-two input. The shift call supplies `10` as its capture argument. | The output must satisfy the selected external Futhark IR format. |

The corresponding code is [Monomorphize.lean](../Remora/Monomorphize.lean),
[LambdaLift.lean](../Remora/LambdaLift.lean), and
[Defunctionalize.lean](../Remora/Defunctionalize.lean). The emitted maps have not been
fused at this point; optimization happens in the external Futhark pipeline.

Use `showStage "futhark"` for the full emitted text. In simplified notation, its
structure is:

```text
lifted_shift(offset, x) = x + offset
lifted_identity(x) = x
main() = map (lifted_shift 10) (map lifted_identity [1, 2])
```

This is an explanation of the emitted IR, not another executable representation.
The `dev --lambdalift` convenience command runs that pass directly after checking;
for the complete compilation sequence, use `showStage` or `futhark --asts`.

## Small experiments before changing the machinery

1. Change `offset` to `20`. Predict the value, then inspect its explicit capture in
   the lifted AST and emitted IR.
2. Replace `[1 2]` with a `2×3` matrix. Predict the frame of both scalar functions
   and the result shape before running it.
3. Give `shift` a `[3]` input cell and make its body sum that cell before adding
   `offset`. A `2×3` input should now produce a length-two result.
4. Compare the type-polymorphic scalar identity with the shape-polymorphic identity
   in the Lean file. The first lifts over `[2]`; the second consumes `[2]` as a cell.

Try these in the notebook or your own scratch file. Then choose one small compiler
change whose expected AST and result you can describe in advance. The existing
[audit](../validation/compiler-audit.md) records coverage and the repeated
polymorphic-instantiation growth issue for when you want to return to pass work.
