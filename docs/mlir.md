# Direct MLIR backend

The Lean compiler now lowers the ordinary Remora YOLOv4 demo to executable MLIR.
No convolution/matmul primitives were added to Remora. Futhark and VeIR are not
required. The existing parser, typechecker (including Z3), monomorphization,
lambda lifting, and defunctionalization feed a standard tensor/linalg emitter.

```sh
lake build
lake exe remora mlir -e '(+ [1 2] [[10 20 30] [40 50 60]])'
lake exe remora mlir -f examples/darknet/yolov4.remora > /tmp/yolov4.mlir
```

Emission needs no MLIR installation and adds no Lean dependencies. Public entry
symbols retain source names; helpers are private. Int, Float, and Bool become
i64, f32, and i1; arrays become ranked tensors. Record fields become parallel
primitive components, including function arguments/results.

## How ordinary Remora becomes useful tensor IR

The important step is preserving the distinction between a function's cell and
its lifted frame. A call of a vector-cell function across rows is lowered in that
outer frame, with invariant arguments broadcast into it. Internal operations
then see the whole tensor. We inline these lifted array-cell helpers during
lowering; ordinary calls still share `func.call` helpers.

For example, ordinary matmul contains nested applications of dot; dot itself
multiplies vectors and reduces them. These become parallel maps plus a structured
reduction. Broadcasts and transposes carry explicit indexing maps. IREE fuses the
maps, recognizes a generic contraction with `(m,k)`, `(k,n)`, `(m,n)` indexing,
and selects its tiled matrix-multiplication pipeline. There is no match on the
Remora names `dot`, `matmul`, or `conv`.

Convolution remains the source program's im2col plus matrix multiplication.
Padding, index arrays, gathers, reshapes, pooling, concatenation, upsampling,
batch normalization, activations, and detection decoding lower through the same
ordinary operations. This is a working initial representation; it does not yet
recognize convolution directly or prove that intermediates always disappear.

| Operations | Representation |
| --- | --- |
| Static literals/empty arrays, local/global values | Constants, tensors, SSA, shared accessors |
| Scalar arithmetic, comparisons, conversions, exp/log/sqrt/tanh/power | `arith`, `math`, scalar `linalg.generic` maps |
| Rank lifting and prefix extension | Principal-frame lowering, explicit broadcast maps |
| Transpose, reshape/flatten, append | Permutation maps, `tensor.reshape`, `tensor.concat` |
| Index/index2d/head/length, static iota, computed one-axis frames | Indexed linalg construction and extraction |
| Integer sum | Structured reduction over the cell axes |
| `reduce`/`reduce/zero` with builtin add/multiply/float min/max | Structured reduction over the leading cell axis |
| Pointwise eta expansions of those reducers | The same structured reduction, independent of function name |
| Other known folds/reducers | Ordered `scf.for` loops, including right-fold order |

Associative floating reductions may be reordered by the toolchain. We use native
f32 arithmetic and math operations without compatibility shims.

## Run YOLOv4

Use an isolated Python environment for the external compiler/runtime:

```sh
python3 -m venv /tmp/remora-iree
/tmp/remora-iree/bin/pip install iree-base-compiler==3.13.0 iree-base-runtime==3.13.1 numpy
export PATH=/tmp/remora-iree/bin:$PATH
```

The checked-in image and weights are Git LFS pointers. Supply their actual
contents; these are the upstream demo assets (about 4 MB and 258 MB):

```sh
curl -fL https://media.githubusercontent.com/media/remora-lang/remora/main/examples/darknet/input.bin -o /tmp/yolo-input.bin
curl -fL https://media.githubusercontent.com/media/remora-lang/remora/main/examples/darknet/yolov4.weights -o /tmp/yolov4.weights
/tmp/remora-iree/bin/python scripts/run_yolo_mlir.py \
  --image /tmp/yolo-input.bin --weights /tmp/yolov4.weights \
  --output-dir /tmp/remora-yolo
```

The runner emits MLIR, compiles for the host CPU, reads the exact exported
signature, and supplies the image followed by the 541 weight/BN/bias arrays in
Darknet storage order. It validates file sizes and rejects LFS pointers. Inputs
are raw RGB CHW float32; there is no resizing/preprocessing in this runner.

Artifacts are `yolov4.mlir`, `yolov4.vmfb`, `detections.npy`, compiler logs, and
`result.json`. The array contains 22,743 rows of 85 attributes, in the source's
anchor-major order. JSON uses objectness times class probability and classwise
NMS, with normalized center-x/center-y/width/height boxes and zero-based COCO
class IDs. `--module` can reuse a binary compiled for the exact same source.

IREE 3.13 needs `--iree-opt-const-expr-hoisting=false` for this program: its
constant-hoisting fixed-point pipeline otherwise exceeds its iteration limit.
The runner supplies that option; ordinary tensor optimization remains enabled.

The [validation record](../validation/mlir-yolo.md) describes the actual run and
independent OpenCV/Darknet comparison, including tool versions and asset hashes.

## Validation

```sh
lake test
python3 -m unittest discover -s scripts
python3 scripts/test_mlir.py --native --mlir-bin-dir /path/to/working/mlir/bin
/tmp/remora-iree/bin/python scripts/check_mlir_darknet.py
```

The native harness requires MLIR 22's `mlir-opt`, `mlir-runner`, and sibling
`../lib/libmlir_c_runner_utils.so`. It verifies and executes every result element,
including array cells, captured tensors, frame construction, records, tensor
arguments, left/right folds, empty reductions, comparisons, and integer overflow.
The explicit IREE check executes 12 neural-network examples against the
interpreter, plus parameterized matrix multiplication and non-symmetric,
multichannel convolution with padding, stride, and a remainder against NumPy.
The `conv_remainder.remora` standalone entry uses undefined weights; its operation
is exercised with real parameter values instead.

To compare the full output against an independent Darknet implementation:

```sh
/tmp/remora-iree/bin/pip install 'opencv-python-headless<5'
curl -fL https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg -o /tmp/yolov4.cfg
/tmp/remora-iree/bin/python scripts/check_yolo_opencv.py \
  --config /tmp/yolov4.cfg --image /tmp/yolo-input.bin \
  --weights /tmp/yolov4.weights --result /tmp/remora-yolo/detections.npy
```

This compares all attributes, not only selected detections. It reads OpenCV's raw
convolution heads and decodes them in NumPy because OpenCV's Region layer
thresholds class scores. OpenCV is only a validation dependency.

## Profiling and optimization priorities

The [initial warmed latency and layer/kernel breakdown](../validation/mlir-yolo-profile.md)
identified index arrays, broadcasts, and window gathers as major costs. The
[settings and representation experiments](../validation/mlir-options-fusion.md)
then tested existing IREE options and fixed one emitter issue: explicit index
operands now let standard Linalg fusion remove intermediates. The full network
improved from about 1.38 s to 1.12 s with default settings, or about 0.95 s with
the best tested opt-in data-tiling configuration. OpenCV remains faster overall.

For this YOLO experiment, add `--iree-flag=--iree-opt-level=O0` and
`--iree-flag=--iree-opt-data-tiling=true` to the runner. These are not global
defaults: some isolated workloads regress. Both the runner and
`scripts/profile_yolo_pieces.py` accept repeated compiler flags. The report
includes the numerical checks, counterexamples to blanket fusion, and remaining
pooling/window-representation work.

## Limits and further work

This is an experimental, unverified translation. External MLIR verifies target
IR, not correspondence with Remora semantics. Shapes must be static. Residual
polymorphism, dynamic/existential arrays, runtime function arrays, many unrelated
intrinsics, and multi-axis computed frame constructors remain unsupported.
Record operations are supported where components can be flattened; general
record reductions are not implemented. Runtime file IO and tracing are rejected.
There is no integrated `mlir --run`, general C/Python binding, or tested GPU
pipeline. Out-of-bounds gathers do not have a Remora runtime error wrapper.

The next optimization question is whether direct convolution recognition and
layout choices improve this representation enough to justify their complexity.
Start with the existing [scheduling study](mlir-study.md) and
[handwritten targets](mlir-targets/README.md). These remain useful contracts;
convolution recognition and optimal scheduling are separate from getting the
ordinary program to run correctly.
