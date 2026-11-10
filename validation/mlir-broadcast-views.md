# Broadcast views for ordinary indexed computations

Measured 2026-11-10 on the same Ryzen 7 PRO 7840U and IREE environment as the
[settings study](mlir-options-fusion.md). The parent emitter is `392589d`.
[Samples and correctness results](mlir-broadcast-views.json) accompany this report.

## What changed

A broadcast now records a source value and the axes it uses. For example, when
lifting a gather across kernel offsets, a logical read
`padded[c, ky, kx, y, x]` can read the stored `padded[c, y, x]` directly. The two
extra axes express reuse; they need not cause an array copy.

The emitter composes these projections with the existing Linalg input maps and
scalar tensor reads. A helper whose results are just projections of its arguments
can return those views through a call. Boundaries such as tensor returns,
nontrivial reshapes, and ordered loop state materialize an ordinary tensor when
required. Other operations keep using the original lowering.

This is about 30 net Lean lines after sharing the scalar-map builder. It adds
no new runtime tensor type, Remora primitive, user-function-name matching,
optimization pass, or Lean dependency. It doesn't try to recover every affine
access formula or turn arbitrary Remora reductions into named pooling operations.

## Measurements

Matched comparisons used eight requested workers, resident inputs, two warm-ups,
and seven measured rounds, alternating/rotating variant order. Compilation and
other tests were idle during timing. All outputs were checked against the parent
emitter. Frequencies and CPU placement were not pinned.

| Full YOLO network | Parent | With views |
| --- | ---: | ---: |
| Default IREE settings | 1,219 ms | 1,043 ms |
| O0 plus data tiling | 1,027 ms | **851 ms** |

With identical tiling settings the median fell by about 17%. The earlier
OpenCV reference remains about 465 ms; it was not retimed in this experiment.
Both new full-model outputs passed the independent OpenCV/Darknet comparison
(maximum absolute difference about 0.000203, mean about 0.000000415).

| Isolated ordinary Remora piece, default settings | Parent | With views |
| --- | ---: | ---: |
| 3×3 stride-two convolution, 32→64, 608→304 | 165 ms | 68 ms |
| 3×3 stride-one convolution, 256→512, 38×38 | 23.0 ms | 19.5 ms |
| 1×1 convolution, 128→64, 304×304 | 7.8 ms | 7.5 ms |
| 13×13 max-pool, 512 channels, 19×19 | 237 ms | 116 ms |
| Bare matrix product | 17.4 ms | 17.8 ms |

Small timing differences are within the normal variation of these runs. The
large stride and pooling gains correspond to actual changes in emitted kernels:

- The pooling gather reads the padded `512×31×31` tensor directly. Its old
  `512×13×13×31×31` repeated input (333 MB) is absent from the new dispatches.
- The stride-two convolution reads `32×610×610` directly; its old repeated input
  (429 MB) is absent as well.
- Pooling still builds 62,473,216 integer indices (500 MB). The handwritten
  [affine-window target](../docs/mlir-targets/pool-window.mlir) takes only a few
  milliseconds in the same toolchain. Reaching that target automatically still
  needs to expose window addresses directly and avoid those index tables.

The new [sliding-sum example](../docs/mlir-targets/sliding-sum.remora) exercises
ordinary indexing and float addition on `64×4096` input with a 13-element window.
It reads the original input directly even though the row is logically reused at
all 4,084 starts. IREE O2 executes the result in about 0.8 ms; all 261,376 values
agree with NumPy (maximum absolute difference below 0.000001). This is an
independent example of the same view lowering, rather than a pooling match.

## Would newer MLIR help?

The YOLO compiler is IREE 3.13.0 (`89ece87b…`), which already bundles LLVM/MLIR
**24.0.0git**. The separate `mlir-opt`/runner build is LLVM **22.0.0git** and is
used for native checks. Upgrading that standalone installation does not affect
the YOLO path. IREE pins its own LLVM revision, as described in
[its LLVM integration notes](https://github.com/iree-org/iree/blob/main/docs/website/docs/developers/integrating-llvm.md).

[Newer IREE builds exist](https://github.com/iree-org/iree/releases), including
3.14 release candidates. We did not benchmark those here. This experiment
kept the toolchain fixed, and the existing toolchain already runs the explicit
window target well. A toolchain comparison is separate from exposing the right
access representation; it should hold the emitted IR and options fixed.

## Reproduce

Use the environment, assets, and runner commands in [the MLIR guide](../docs/mlir.md).
To check the original fusion fixtures, handwritten pooling, and ordinary
sliding sum:

```sh
lake build
/tmp/remora-iree/bin/python scripts/check_mlir_fusion.py --tools /tmp/remora-iree/bin
```

`profile_yolo_pieces.py` reproduces the isolated kernels, and `profile_yolo_mlir.py`
measures warmed full-model runs using the VMFB emitted by the runner. Use the
same compiler options for parent/new comparisons. The current change passed
45 native MLIR cases, 14 neural-network/kernel checks, Python unittest discovery,
and all 106 interpreter runs. Both full models passed external MLIR verification
and real-weight inference. No upgrades or new dependencies were needed.
