# Port status

Upstream source: remora-lang/remora, commit cb245ea386f8ed8f5ac72957ed51990bebe84bc8.

## Direction for the next iteration

Keep the implementation small and readable. Upstream is the reference for language
behavior and compiler algorithms, but a literal port of its abstraction layers is
not a goal in itself. Prefer ordinary Lean definitions and explicit state over
compatibility shims or general frameworks.

Before replacing a simplified compiler pass, identify a concrete correctness,
coverage, or code-growth problem it must solve. Port the relevant upstream
algorithm when it addresses that problem; retain simplifications when their
behavior is adequate and easier to understand. Record substantive differences.

The compiler audit is recorded in [validation/compiler-audit.md](validation/compiler-audit.md).
It justifies preserving ordinary function definitions (implemented) and a focused
follow-up on sharing repeated polymorphic instances.

## Implemented

- Parameterized source/unique/checked syntax, source locations, JSON serialization.
- Complete surface grammar, including Unicode binder aliases, records, boxes,
  type/index applications, array-kind expansion, and `@` application/binding sugar.
- Uniquification, type checking, type/index substitution, alpha renaming, free variables.
- Interpreter with explicit lexical closures, partial intrinsic applications,
  lifting, records, boxes, and text/binary file operations.
- SMT-LIB/Z3 shape equality and prefix checks, retaining normalization fast paths.
- Specialization, lambda lifting, eta expansion, and a native SOACS emitter.
- CLI, REPL, import resolution, interpreter test directives, regex error matching,
  Futhark text/binary inputs, and C/CUDA subprocess adapters.

## Deliberate differences

Lean numeric operations and the Regex library's behavior are used directly; there
are no Haskell numeric/regex compatibility shims. `Int64` and `Float32` represent
language numbers; dimension expressions use Lean `Int`.

Effects are explicit IO. Function values use positive closure constructors rather
than recursively callback-valued constructors. Parsing uses Std Parsec to build
delimiter trees and then lowers those trees into the source AST.

The compiler transformation code is presently simpler than upstream: specialization
expands polymorphic definitions without upstream's memoized instance machinery,
while ordinary named functions stay shared. It runs before lambda lifting. Eta expansion
currently covers top-level function bindings. Further pass changes should follow
the audit findings. Modules that only wrap Haskell monad stacks have been
folded into their corresponding Lean subsystem.

## Outstanding work

- Add sharing for repeated polymorphic instances, with capture and fresh-name tests.
  Review higher-order handling and partial applications before adding more pass machinery.
- Finish backend record entry-point metadata and array-of-function cases.
- Resolve compiled corpus failures; interpreter tests passing does not certify the backend.
- Finish source-like AST pretty printing and the remaining mode-specific test
  exclusion switches. JSON stage output and program/expression AST dumps are implemented.
- Update Nix/container packaging and perform CUDA validation. GitHub CI builds Lean
  and runs the interpreter corpus.
- Verify the emitter against upstream's pinned Futhark revision
  16c6e619d1cc1a0fc2497a3da4573a89209ae0b9. Development validation so far uses the
  installed Futhark 0.25.33; its accepted IR is not a guarantee for every other version.

The exact upstream baseline and the port accept and reject the same files in the
106-file corpus: 85 valid lowerings and 21 rejections. See the audit for the
unsupported cases, negative fixtures, and important backend-version distinction.

## Module mapping

`Syntax`, `Parser`, `ISpace`, `Symbolic`, `VName`, `Imports`, `Serializer`,
`Substitute`, `Rename`, `Free`, `Prop`, `Uniquify`, `TypeCheck`, `Primitive`,
`Intrinsics`, `Interpreter`, `LambdaLift`, `Monomorphize`, `Defunctionalize`,
`Futhark`, and `Pipeline` map to corresponding files under `Remora/`.
Interpreter values/intrinsics/value parsing remain submodules. `Main.lean` owns
CLI dispatch and REPL; `Remora/CLI/Test.lean` owns test directives and execution.
`Remora/CLI/Futhark.lean` owns external backend compilation/execution.

The upstream ISC license is preserved in LICENSE. The SOACS syntax follows the
Futhark compiler, also distributed under the ISC license by DIKU, University of Copenhagen.

## Latest local validation

- `lake build` succeeds on Lean 4.35.0.
- `lake test`: 106 runs pass (99 upstream plus seven Lean-specific cases).
- C compilation and execution: 89 runs pass when excluding upstream `higher-order`
  and `existential` test tags and the Lean-native numeric behavior tests. Those
  exclusions are explicit; this is not a claim that the entire compiler corpus passes.
- AST JSON dumps and the basic C execution path have been exercised locally.

The named-function growth regression passes for top-level and local definitions;
polymorphic instance expansion remains an identified performance issue.
