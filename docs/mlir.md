# Experimental direct MLIR backend

Build normally, then emit standard MLIR:

```sh
lake build
lake exe remora mlir -e '(+ [1 2] [[10 20 30] [40 50 60]])'
lake exe remora mlir -f tests/basic0.remora
```

Emission needs no MLIR installation. It uses the existing typechecker (including
Z3), monomorphization, lambda lifting, and defunctionalization. It does not invoke
Futhark. Public entry symbols retain source names; internal functions use unique
tags. Scalars become `i64` or `f32`; non-scalar values become ranked tensors.

| Supported | Lowering |
| --- | --- |
| Static Int/Float arrays, including empty arrays | `arith.constant`, tensor construction/reshape |
| Computed one-axis frames, including frames of array cells | `tensor.insert` / `tensor.insert_slice` |
| Local values and known scalar functions, including captured scalar values | SSA values and shared `func.call` helpers |
| Integer `+`, `-`, `*`; Float `f.+`, `f.-`, `f.*` | `arith` operations |
| Scalar-function rank lifting and prefix extension | `linalg.generic` with prefix indexing maps |
| Integer sum, including empty and multidimensional cells | Initialized `linalg.generic` reduction |

The emitter is intentionally limited. It rejects Boolean/record values, residual
polymorphism, dynamic/existential shapes, runtime function arrays, general
vector-cell user functions, top-level value declarations, other intrinsics, and
multi-axis computed frame constructors. Nested ordinary frame construction works.
There is no `mlir --run`, C API, GPU pipeline, or optimization framework here.
The existing Futhark commands retain their behavior.

## Desired output for tensor kernels

The [handwritten target contracts](mlir-targets/README.md) demonstrate matmul and
padded, strided convolution with bias/activation. They verify and execute today,
but the Remora emitter does not generate them yet. They guide the next lowering
work and keep explicit primitives compatible with later pattern recognition.

## Validation

Normal CI checks emission, rejection, and interpreter fixture results:

```sh
python3 -m unittest discover -s scripts
```

CPU verification and execution are explicit and require a working MLIR 22 toolchain:

```sh
python3 scripts/test_mlir.py --native --mlir-bin-dir /path/to/llvm/build/bin
```

The harness requires sibling `mlir-opt` and `mlir-runner` binaries and
`../lib/libmlir_c_runner_utils.so`. Missing tools are an error. It prints the tool
version, verifies each module, generates a test-only wrapper checking every result
element, bufferizes, lowers to LLVM, and executes. Expected values also have
interpreter checks in normal tests. Integer overflow wraps at 64 bits; floating
fixtures use exactly representable values. No float compatibility shims are added.

Native tests currently cover 15 examples, including prefix extension in both
operand positions, captures, computed frames, scalar/row/whole-array sums, empty
outputs, empty reductions, and integer overflow, and a tensor-valued entry argument. The native harness targets Linux
shared libraries; portability and a general execution interface are future work.

## Design references

- [full-pipeline](https://github.com/remora-lang/full-pipeline/tree/23e6b75b85150d02aa40da0b10d3682670003325)
  routes SOACS through Futhark's GPU IR before MLIR. This backend bypasses those
  fusion/flattening transformations; equivalent performance is not claimed.
- [mlir-backend](https://github.com/remora-lang/mlir-backend/tree/2fe88f43d618db0a98cbb53e8003afa4b8d58774)
  supplies the reference for structured tensor/linalg lowering and CPU passes.
- [VeIR](https://github.com/opencompl/veir/tree/a37a0dc8c1d4c886c9cd39a5339ca1a3f70a28cf)
  offers Lean IR manipulation and rewriting. The prototype adds no dependency on
  it: emitting verified standard text currently needs only a small SSA builder.
  Revisit that choice when there is a concrete Lean-side rewriting requirement.

This is a compiler experiment, not a verified translation. The emitter reuses
checked Remora annotations; external MLIR verifies the generated target IR.
