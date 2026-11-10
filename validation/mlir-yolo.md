# Ordinary Remora → MLIR → YOLOv4

Validated 2026-11-10 on x86_64 Linux, AMD Ryzen 7 PRO 7840U. This is a local
correctness/integration run, not a controlled performance benchmark.

The compiler accepts the existing `examples/darknet/yolov4.remora` unchanged.
No source convolution/matmul primitives, Futhark executable, VeIR dependency,
network-specific compiler matches, or numeric compatibility shims were added.
The host supplies the real image and Darknet checkpoint as 542 tensor arguments.

## Full network

A fresh run of `scripts/run_yolo_mlir.py` emitted MLIR, compiled with IREE for
`llvm-cpu`/`host`, and executed the full 608×608 network:

| Measurement | Result |
| --- | --- |
| Emission plus compilation | 27.54 seconds |
| Single inference including argument transfer/result retrieval | 1.68 seconds |
| Result | 22,743 × 85 float32, all finite |
| Strong post-NMS detections | dog 0.9791; bicycle 0.9231; truck 0.9228 |

There was also a class-58 detection at 0.3253 with the runner's default 0.25
threshold. JSON reports class indices, confidences, and normalized xywh boxes.
Times depend on machine, thread configuration, and competing work; no comparative
speed claim follows from this run.

Independent OpenCV/Darknet execution used the identical raw image and weights.
`scripts/check_yolo_opencv.py` reads the three pre-YOLO convolution heads and
implements anchor/grid decoding in NumPy, preserving the Remora source's
anchor-major row order. This avoids OpenCV Region-layer confidence thresholding
and tests all 1,933,155 output attributes, including low-confidence rows.

| Difference from independent reference | Result |
| --- | ---: |
| Maximum absolute difference | 0.000203281641 |
| Mean absolute difference | 0.000000415380 |
| Elementwise check | `rtol=2e-3`, `atol=1e-4` passed |

These tolerances allow native f32 reductions/math and backend scheduling; there
is no claim of bitwise identity. An attempted Futhark reference was unavailable
because the existing Lean Futhark emitter rejects `index2d`; OpenCV provided an
independent full-model check instead. This does not affect the MLIR runtime path.

## Smaller checks

- `lake build` and `lake test`: 106 interpreter runs passed.
- Python unittest discovery: all checks passed, including existing compiler-growth checks.
- 37 emitted-MLIR native execution cases: passed with external verification and
  full-element result comparisons. These cover tensor cells/captures, records,
  prefix frames, booleans, transpose/reshape/index/append/iota, floating reduction,
  captured reduction operators, left/right fold order, and existing scalar/sum cases.
- `scripts/check_mlir_darknet.py`: all 12 runnable standalone neural examples
  agree with the interpreter. Parameterized matmul and multichannel,
  non-symmetric, padded/strided/remainder convolution agree with NumPy.
- Full 109-file fixture scan: 81 emitted modules verify; 28 reject unsupported
  source features. This is an emission/verification inventory, not 81 execution claims.
- Lean language-server diagnostics: no errors or warnings in `Remora/MLIR.lean`.

For a parameterized 2×3 by 3×4 ordinary Remora matrix product, IREE's configured
IR contains a `matmul_like_2x4x3_f32` dispatch with `DoubleTilingExpert`, and generic
indexing maps `(d0,d2)`, `(d2,d1)`, `(d0,d1)`. This verifies that the ordinary
multiply/reduce lowering exposes a contraction to IREE. It does not establish
optimal convolution layout or GPU performance.

## Versions and input identity

- Lean 4.35.0; local MLIR/LLVM 22.0.0git, revision `25ed9231159f`.
- `iree-base-compiler==3.13.0` (reported compiler revision
  `89ece87b89852100c0a4ac3d515e14842d090967`, LLVM 24).
- `iree-base-runtime==3.13.1`, `numpy==2.5.4`.
- Independent validation only: `opencv-python-headless==4.14.0.94`.
- IREE option `--iree-opt-const-expr-hoisting=false` avoids its constant-hoisting
  fixed-point iteration failure. The runner includes this flag.

SHA-256 identities:

| Input | SHA-256 |
| --- | --- |
| `examples/darknet/yolov4.remora` | `8c7e7fc60d2cb061f6870e8a433f07a96a58b25ed8bf536aa8929a77d67fed0b` |
| Raw demo image | `d40b65047677af79aed89ccfcf6ef0d6fa80cfc04649d46e671bb7078eee5100` |
| Darknet weights | `e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562` |
| Standard `yolov4.cfg` | `a6d0f8e5c62cc8378384f75a8159b95fa2964d4162e33351b00ac82e0fc46a34` |

The image and weights match their checked-in LFS pointers. Downloads and compiled
artifacts stayed in `/tmp`; they are not vendored into this repository.
Reproduction commands and known limits are in [the MLIR guide](../docs/mlir.md).
