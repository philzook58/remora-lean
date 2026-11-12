# Remora in Lean

A Lean 4 implementation of the Remora array language, based on
[remora-lang/remora](https://github.com/remora-lang/remora).
The port is in progress; the interpreter is covered by the upstream regression corpus,
while compiler coverage and close correspondence of compiler passes still need work.
See [PORTING.md](PORTING.md) for differences and outstanding work.

Start with [Understanding.lean](Understanding.lean) for an executable introduction
to tensors, lifting, quantifiers, and the compiler. The accompanying
[code map](docs/understanding.md) follows one program through the actual passes.
After building, run `lake env lean Understanding.lean` or open it in the Lean editor.

## AI Disclosure

This was heavily AI produced and heavily based on the human work in the remora implementation. I take no credit and no blame.

It is unlikely to be of interest to you and I'll probably abandon it.

I don't really intend to use Lean as a verification language, just as a Haskell stand-in I'm more comfortable with currently.

## Build and run

Install Elan, then run:

```sh
lake build
lake exe remora interpret -e '(+ 1 2)'
lake exe remora interpret -f tests/basic0.remora
lake exe remora repl
lake test
```

The toolchain is pinned to Lean 4.35.0. Lake fetches the pinned Lean Regex library and
its dependencies. Z3 is required for symbolic shape relations. The Lean implementation
does not invoke the original Haskell Remora implementation.

## Compiler

Install Futhark and a C compiler:

```sh
lake exe remora futhark -f tests/basic0.remora
lake exe remora futhark --run -f tests/basic0.remora
lake exe remora futhark --backend=c -f tests/basic0.remora
lake exe remora test --modes compile tests/basic0.remora
python3 scripts/check_backend.py
```

`futhark` emits SOACS IR. `--backend=cuda` selects Futhark's CUDA backend and requires
a CUDA toolchain; executing CUDA programs also requires a suitable GPU. Compiler
failures are reported as errors. Do not assume that every interpreter-supported
program can currently be compiled.

`lake test` runs the interpreter regression cases. Compiled regression tests are
available with `remora test --modes compile tests` and currently have failures.

Use `remora parse` for AST JSON and `remora dev` for checked AST JSON.
`dev --uniquify`, `--lambdalift`, `--monomorphize`, and `--defunctionalize` select stages.
`futhark --asts --asts-dir DIR` writes JSON snapshots of the actual compilation stages.
The `tests`, `examples`, `docs`, and `tutorial` directories retain the upstream
language examples and documentation; some upstream build/CLI descriptions in those
documents refer to the Haskell implementation.

For the currently supported C regression subset:

```sh
lake exe remora test --modes compile --exclude-tags higher-order,existential,native tests
```

The exclusions retain visibility of the upstream compiler limitations and the native
Lean numeric behavior checks. The full compiler corpus includes unsupported forms
and negative cases; see the compiler audit below for their classification.

The [compiler audit](validation/compiler-audit.md) compares the exact upstream
baseline, documents backend-version differences, and measures intermediate code
growth. Ordinary named functions now remain shared; repeated polymorphic instances
still need targeted sharing.
