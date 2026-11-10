# Existing MLIR optimizations versus the representation we emit

Follow-up to [the initial profile](mlir-yolo-profile.md), measured 2026-11-10 on
the same Ryzen CPU and isolated IREE 3.13 environment. Measurements and samples
are in [mlir-options-fusion.json](mlir-options-fusion.json). The parent compiler
commit is `a6a613a`; this change exposes index operands explicitly to Linalg.

## What the settings achieve

The isolated pieces were compiled with six configurations: explicit O0, O2, O3,
O2 plus data tiling, O2 plus aggressive reshape movement/multi-use fusion, and O2
with aggressive fusion disabled. All compiled and passed numerical comparisons.
The full old-emitter model was then tested at O0, O2, and O2 plus data tiling.

| Old emitter, warmed full model | Median |
| --- | ---: |
| Default/O0 | 1,442 ms |
| O2 | 1,564 ms |
| O2 plus data tiling | 1,302 ms |

This optimization level is IREE's global pipeline control; O0 does not mean that
LLVM optimization, vectorization, or normal IREE kernel scheduling is disabled.
The extra settings are not uniformly beneficial. Data tiling roughly halves the
representative matrix-only time, but regresses the 1×1 convolution path. Higher
levels help pooling modestly and barely affect the large stride-two convolution.

## A small reproducer identifies an emitter problem

The equivalent fixtures [index-capture.mlir](../docs/mlir-targets/index-capture.mlir)
and [index-input.mlir](../docs/mlir-targets/index-input.mlir) each implement an
identity gather through a generated index array. The difference is how that
index tensor reaches the consumer:

- Captured through `tensor.extract` inside a generic's scalar region: the
  upstream `linalg-fuse-elementwise-ops` pass retains the index array.
- Declared as a `linalg.generic` input with an indexing map: the same existing
  pass inlines the index calculation and removes the index array.

Both execute correctly. This is a representation issue with a demonstrable
existing optimizer solution, rather than a reason to write a new optimizer.
The ordinary Remora emitter now uses explicit input maps for rank-lifted `index`
and `index2d`. Coordinate pairs use two affine accesses to the same tensor, at
constant final-axis positions zero and one. Array-valued record fields retain
their trailing cell coordinates. Scalar-index cases keep their existing path.

This adds about 26 lines to the Lean backend and reuses standard Linalg fusion.
It introduces no new source primitives or Lean dependencies.

## Effect of that change

The following old/new comparisons used the same inputs, eight requested workers,
two warm-ups, and seven timed rounds, rotating variant order. Every result was
checked against the original emitter. Compilation and other test suites were
idle during timing. These isolated pieces exclude BN and activation.

| Piece | Old/default | New/default | New/O2 + tiling |
| --- | ---: | ---: | ---: |
| 3×3 stride-two conv, 32→64, 608→304 | 242 ms | 150 ms | 175 ms |
| 3×3 stride-one conv, 256→512, 38×38 | 39.7 ms | 14.6 ms | 11.6 ms |
| 1×1 conv, 128→64, 304×304 | 56.3 ms | 7.5 ms | 31.1 ms |
| 13×13 pool, 512 channels, 19×19 | 273 ms | 238 ms | 185 ms |
| Bare `512×2304` by `2304×1444` contraction | 15.8 ms | 15.2 ms | 8.6 ms |

A fresh full-network comparison then measured:

| Full network | Median |
| --- | ---: |
| Old emitter/default | 1,384 ms |
| Old emitter/O2 + tiling | 1,282 ms |
| New emitter/default | 1,123 ms |
| New emitter/O2 + tiling | 1,046 ms |
| New emitter/forced early fusion | 1,899 ms |

The emitter improvement is about 19% at default settings; combining it with
O2 plus tiling is about 24% faster than the old default in this run.
An unconditional early pipeline of `inline,canonicalize,cse,` then
`linalg-fuse-elementwise-ops,canonicalize,cse` regressed full-model performance.
The experiment does not isolate the precise cause of that regression. It is
excluded from the recommended configuration.

A final control separated data tiling from the O2 setting on the new emitter:

| New emitter, matched control | Median |
| --- | ---: |
| No tiling/O0 | 1,138 ms |
| Data tiling/O0 | **954 ms** |
| Data tiling/O2 | 1,041 ms |

Thus **O0 plus data tiling** was the best full-model configuration tested. The
1×1 piece still regresses with tiling at O0 (about 8 ms → 23 ms), so this remains
a workload-specific choice rather than a universal default. O2 helps pooling,
but its other effects outweigh that gain on this network. The code change plus
settings bring this demo from roughly 1.38 s to 0.95 s in these local experiments.

These are diagnostic laptop timings with unlocked frequency and unpinned threads;
small differences should not be treated as stable. The baseline varies between
runs. Use the matched old/new table for speedup estimates, rather than comparing
isolated numbers from different runs. The earlier warmed OpenCV reference was
about 465 ms, so matching its full-model performance remains unfinished work.

## The remaining pooling gap is also representational

[pool-window.mlir](../docs/mlir-targets/pool-window.mlir) expresses the same 13×13
max-pool directly as a padded input plus a structured affine window reduction.
IREE O2 executes this handwritten target in about **3–4 ms**, checked exactly
against NumPy. That is close to the earlier OpenCV result. This target is **not
yet generated automatically from ordinary Remora**; the actual helper still
requires 185–238 ms in these configurations.

The new full-model source dump shows gather/reduction fusion in the pooling
path, but a 62,473,216-element index tensor and repeated padded input still
remain. Therefore the next step is to preserve/recover affine window accesses
and logical broadcast views. Existing MLIR/IREE then has a useful representation
to tile and vectorize. Merely adding more optimization flags is insufficient.
No convolution recognition or Winograd implementation is added in this change.

## Reproduction and checks

The [MLIR guide](../docs/mlir.md) describes the environment and assets. The YOLO
runner and isolated-piece profiler now accept repeatable `--iree-flag` arguments:

```sh
/tmp/remora-iree/bin/python scripts/run_yolo_mlir.py \
  --image /tmp/yolo-input.bin --weights /tmp/yolov4.weights \
  --output-dir /tmp/yolo-tiling \
  --iree-flag=--iree-opt-level=O0 \
  --iree-flag=--iree-opt-data-tiling=true
PATH=/tmp/remora-iree/bin:$PATH /tmp/remora-iree/bin/python \
  scripts/profile_yolo_pieces.py --output-dir /tmp/pieces-tiling \
  --iree-flag=--iree-opt-level=O0 \
  --iree-flag=--iree-opt-data-tiling=true
/tmp/remora-iree/bin/python scripts/check_mlir_fusion.py \
  --tools /tmp/remora-iree/bin
```

Flags remain opt-in because of the per-workload regressions. The emitter change
applies by default. The fusion checker verifies and executes both index fixtures,
checks the existing pass's behavior, and numerically checks/times the structured
pooling target. External MLIR 22 verifies the fixture syntax as well.

Validation includes all 40 native MLIR execution cases, all 14 neural-network/kernel
checks, normal Python unittest discovery, Lean build and interpreter regressions,
and full real-weight YOLO comparisons. Both the default and opt-in tiling outputs
agree with independent OpenCV/Darknet within the existing tolerances (maximum
absolute difference about 0.000203, mean about 0.000000415).

The runner reports a cold first invocation with host arguments. Use
`scripts/profile_yolo_mlir.py` with its resulting `yolov4.vmfb` and `yolov4.mlir`
for warmed resident-input timings comparable to the tables above.
