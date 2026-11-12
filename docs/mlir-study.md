# Learning from Halide, IREE, and Mojo

Start here before reading the MLIR fixtures. The purpose is to choose a small
compiler boundary we understand. No Remora primitives are added by this study.

## 1. The computation is not its execution plan

Our running example is matrix multiplication, column bias, and ReLU:

```text
product[i,j] = sum over k of A[i,k] * B[k,j]
output[i,j]  = max(0, product[i,j] + bias[j])
```

For convolution, replace the matrix accesses by window accesses such as
`input[y+ky,x+kx,channel]` and reduce over kernel positions and channels.
The same distinction between output positions and summed positions applies.

These equations leave execution choices open:

- Compute and store every product before applying bias and ReLU?
- Finish a small output block, including its activation, before moving on?
- Reuse input blocks in cache, registers, or GPU shared memory?
- Assign output blocks to cores, threads, or specialized matrix instructions?

A **schedule** answers those questions. It can change without changing the
mathematical operation. Floating-point transformations also need a numerical
policy; mathematical equivalence does not imply bitwise equality.

## 2. What structured MLIR tells a compiler

A matrix product can be described without an operation named `matmul`:

```text
iteration positions: i, j, k
A reads:             (i,k)
B reads:             (k,j)
output updates:      (i,j)
i and j:             independent output positions
k:                   combine contributions by addition
```

In `linalg.generic`, these are indexing maps and iterator kinds. For example,
`(i,j,k) -> (i,k)` means "use the i and k coordinates to read A".
`parallel, parallel, reduction` distinguishes output axes from the summed axis.
The scalar region says to multiply two inputs and add to the accumulator.

This structure gives the optimizer information that an arbitrary sequence of
loads, stores, and jumps would obscure. Both the named and generic forms are
legitimate target representations. A source-level convolution primitive is not a
prerequisite for constructing the generic form.

Compare [named matmul](mlir-targets/matmul-bias-relu.mlir) and
[generic multiply/reduce](mlir-targets/matmul-generic.mlir). The scalar region of
the latter is the essential extra information to read; the bias/ReLU is unchanged.

## 3. Halide: separate the algorithm and the schedule

Halide lesson 8 studies a producer whose values feed a neighboring-window
consumer. Computing all producer values once uses an intermediate array;
inlining the producer avoids that array but repeats work. Computing portions
at a time offers intermediate choices. This is a concrete memory-versus-work
tradeoff, not just syntactic loop rearrangement.

Halide lets a programmer describe these choices separately from the algorithm.
Its autoschedulers can propose schedules using expected sizes and hardware
information. The MLIR Transform dialect offers an explicit scheduling language
for existing structured operations. These are related ideas; Halide's own
compiler is not an MLIR frontend in the workflow studied here.

The official MLIR [Halide convolution walkthrough][halide-mlir] is particularly
relevant: it expresses convolution using generic Linalg, then schedules it. It
also explains a difference between Halide's computation model and Linalg's
initially separate operations. Their schedules need not be identical scripts.

Our much smaller [row schedule](mlir-targets/matmul-row-schedule.mlir) does this:

```text
Before:                              After:
compute the full product             for each output row:
compute full bias/ReLU result           initialize a product row
                                       compute that product row
                                       apply bias/ReLU to that row
                                       put it in the output
```

It uses upstream tile-and-fuse transforms. This demonstrates separation of
algorithm and schedule; tile size one is deliberately educational, not an
optimized CPU recommendation. The runnable study saves the actual transformed
IR as `matmul-row-tiled.mlir` in its output directory and checks every element.

Why tiling can help: neighboring matrix outputs reuse entries of A and B.
Working on an output block makes that reuse explicit and can keep its inputs
near the arithmetic. Fusion can avoid writing and rereading a whole intermediate.
Whether those changes help depends on hardware, sizes, layouts, and overheads.

## 4. IREE: an existing compiler can choose the schedule

IREE accepts supported structured MLIR input and supplies CPU/GPU compilation
pipelines. It selects defaults for distribution, vectorization, and reduction
blocking, with tuning facilities for particular workloads.

We ran the same matrix/bias/ReLU computation in both representations:

| Experiment | Named operation | Generic multiply/reduce |
| --- | --- | --- |
| Small 2×3 by 3×2 numerical execution | Pass | Pass |
| 128×128 numerical execution | Pass | Pass |
| Configured dispatches, each containing the contraction and epilogue | 1 | 1 |
| Selected CPU pipeline | `DoubleTilingExpert` | `DoubleTilingExpert` |
| Classified root | `matmul` | `matmul_like` |

For 128×128, both received the same observed configuration:

```text
output distribution blocks:       [16,64]
common parallel vector tiles:     [8,16]
reduction vector tile:            [16]
```

These are logical, pipeline-specific configuration sizes, not a claim that the
CPU has an 8×16-wide SIMD instruction. They are observations on this CPU and
compiler version, not constants for our backend to emit. The configured generic
operation remained generic; classification did not require spelling it as
`linalg.matmul`.

This is concrete evidence that a source primitive is unnecessary for this case.
It does not show recognition of arbitrary Remora syntax, convolution recognition
from arbitrary loops, or identical final code/performance for the two forms.
Our task would be to translate Remora into the right structured computation.

We also compiled and numerically checked the unchanged padded-convolution target
through IREE. Its configured dump had two dispatches: a padding-related copy and
the convolution with epilogue. Thus a capable compiler did not remove every
intermediate automatically. The study script automates the matrix experiments;
the convolution result was a separate exploratory check.

Measured environment: local MLIR 22.0.0git (LLVM checkout `25ed9231159f`), IREE
compiler wheel 3.13.0 (reported revision `89ece87b89852100c0a4ac3d515e14842d090967`,
LLVM 24), runtime wheel 3.13.1. The two toolchains stay separate. This is
correctness and IR inspection, not a timing comparison or GPU benchmark.

## 5. Mojo and MAX: another allocation of responsibility

Mojo's compiler specializes parameterized programs and lowers its own MLIR
dialects toward LLVM. The public kernel examples and Modular's Blackwell matmul
worklog show substantial human-designed choices: tiles, layouts, shared memory,
asynchronous transfers, specialized matrix instructions, and pipelining.
Compile-time specialization makes those kernel designs reusable. Merely using
MLIR does not supply their designs automatically.

MAX's graph compiler is a separate layer that selects/composes kernels and
performs graph transformations. Its public architecture describes both structural
fusion and remaining named/builtin-specific paths. Do not confuse graph-level
fusion with optimization supplied by the Mojo language compiler alone.

For Remora, the useful lesson is to keep a well-defined boundary between a
computation and its implementation. We can learn from expert kernels without
writing our own equivalent of the entire Mojo/MAX stack. We have not established
Mojo as a supported drop-in consumer of the standard Linalg emitted here.
Halide and Mojo were studied from source/docs; neither was installed or executed.
The historical 2022 generative-compiler design document explicitly says it is not
a specification of the current Mojo system, so it is not used as one here.

## 6. What this suggests for Remora

The experiment proposed below is now implemented: ordinary Remora matmul and
YOLOv4 run through IREE. See the [backend guide](mlir.md) and
[full-network validation](../validation/mlir-yolo.md). The following records the
reasoning behind that direction.

| Responsibility | Proposed owner for the next experiment |
| --- | --- |
| Program meaning, types, and cell/frame agreement | Remora frontend |
| Explicit array accesses, independent axes, reduction domains and combination | Small Remora-to-structured-MLIR lowering |
| CPU scheduling and code generation | IREE experiment, alongside our simple upstream lowering |
| Target-specific expert kernels | A possible later implementation choice |

Do not add user-visible primitives yet. First try lowering an ordinary Remora dot
product to the generic contraction tested here. For example, preserve the meaning
of `reduce (+) (multiply xs ys)` as a structured reduction rather than expanding
it into unrelated opaque scalar calls. Then exercise a matrix product built from
those dots. The immediate missing pieces are array-cell handling, indexing, and
floating reductions, already identified in the backend review.

If the operation already has clear structure, pass it through. If that structure
is obscured, normalization or recognition can recover it before low-level
lowering. A designated primitive remains an optional escape hatch, to be justified
by a concrete failure or complexity problem. These are separate decisions from
choosing a named versus generic target operation.

## Reproduce the lesson

With the existing local LLVM/MLIR toolchain:

```sh
python3 scripts/study_mlir.py --mlir-bin-dir /path/to/llvm/build/bin
```

Optionally install IREE in a separate environment (not the Lean project):

```sh
python3 -m venv /tmp/remora-mlir-study
/tmp/remora-mlir-study/bin/python -m pip install iree-base-compiler==3.13.0 iree-base-runtime==3.13.1
python3 scripts/study_mlir.py --mlir-bin-dir /path/to/llvm/build/bin \
  --iree-python /tmp/remora-mlir-study/bin/python
```

Use `--output-dir DIR` to select the dump directory. Each IREE experiment uses a
fresh subdirectory. The report records configured IR and its location; LLVM IR,
assembly, and standalone dispatch artifacts are also available there. Tool
versions and numerical checks are printed. No large generated dumps are committed.

Suggested reading order: equations above → generic scalar region → row-scheduled
loop → IREE classification/configuration. The larger Mojo kernel examples can
wait until those pieces are comfortable.

## Sources to inspect

- [Halide lesson 8][halide8]: concrete intermediate-storage/recomputation tradeoffs.
- [Halide lesson 21][halide21]: autoscheduling inputs and generated schedules.
- [MLIR Transform tutorial, chapter 1][transform1]: tile/fuse a matrix/bias/ReLU pipeline.
- [MLIR's Halide convolution walkthrough][halide-mlir]: generic convolution and scheduling.
- [IREE developer overview][iree-overview] and [tuning][iree-tuning]: compiler boundaries and configurable choices.
- [Mojo compiler walkthrough][mojo-compiler]: specialization and compiler phases.
- [Modular matmul worklog, part 2][mojo-matmul]: explicitly designed GPU execution.
- [MAX architecture][max-architecture] and [transformation principles][max-patterns]: graph/kernel boundary and preserving structure.

[halide8]: https://github.com/halide/Halide/blob/main/tutorial/lesson_08_scheduling_2.cpp
[halide21]: https://github.com/halide/Halide/blob/main/tutorial/lesson_21_auto_scheduler_generate.cpp
[transform1]: https://mlir.llvm.org/docs/Tutorials/transform/Ch1/
[halide-mlir]: https://mlir.llvm.org/docs/Tutorials/transform/ChH/
[iree-overview]: https://iree.dev/developers/general/developer-overview/
[iree-tuning]: https://iree.dev/reference/tuning/
[mojo-compiler]: https://github.com/modular/modular/blob/main/Mojo/docs/compiler/MojoCompilerWalkthrough.md
[mojo-matmul]: https://github.com/modular/modular/blob/main/max/docs/design-docs/matmul-on-blackwell-part-2.md
[max-architecture]: https://github.com/modular/modular/blob/main/GraphCompiler/docs/ArchitectureSpecification.md
[max-patterns]: https://github.com/modular/modular/blob/main/GraphCompiler/docs/PatternTransformationPrinciples.md
