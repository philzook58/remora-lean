# Where ordinary Remora YOLO spends its time

Measured 2026-11-10 on the same Ryzen 7 PRO 7840U and tool environment as the
[correctness run](mlir-yolo.md). Compiler revision: `f5e418b`. The measurements
and samples are saved in [mlir-yolo-profile.json](mlir-yolo-profile.json).
No compiler or Remora source changes were made for this experiment.

## Warm latency changes the comparison

Both engines requested eight worker threads. Each received two warm-up runs,
followed by seven measured runs, alternating engine order. Inputs and weights
were already loaded; IREE used resident device arrays. Both paths returned the
full decoded tensor and were checked for numerical agreement every iteration.
OpenCV ran the raw heads plus the existing NumPy decoder, so the comparison
includes decoding on both sides, without NMS. Compilation and model loading
were excluded and finished before timing began.

| Full network, median | Time |
| --- | ---: |
| IREE, resident arguments | 1,454 ms |
| OpenCV forward | 459 ms |
| OpenCV NumPy decoding | 6 ms |
| OpenCV forward + decoding | 465 ms |
| IREE, ordinary host NumPy arguments on each call | 1,566 ms |

The steady-state gap is about **3.1×**, rather than the earlier cold single-run
comparison of 1.68 s against 1.34 s. The IREE resident samples ranged from
1,400–1,563 ms; OpenCV total ranged from 429–489 ms. Keeping arguments resident
helps, but copying the weights is not the main explanation for the gap.

This is a local diagnostic benchmark. CPU frequency was not locked, threads
were not pinned, and the laptop's thermal state can change. Use the recorded
samples as approximate costs, not precise or universal speed ratios.

## What the IREE kernels are doing

A separate IREE HAL `host-execution` profiling run reports the execution duration
of each CPU worker tile. The following percentages sum those worker durations.
**They are not shares of inference wall time**: tiles run in parallel, dispatches
can overlap, and profiling itself has overhead. They locate work worth removing.
The categories come from kernel names and inspection of executable source dumps.

| Work | Share of measured worker-tile time |
| --- | ---: |
| Matrix multiplication, including fused epilogues | 41.4% |
| Gathers / materializing convolution and pooling windows | 25.6% |
| Integer index arrays | 14.9% |
| Other maps, broadcasts, and padding | 14.4% |
| Copies | 3.0% |
| Reductions and transposes | 0.7% |

About **55%** goes into gathers, index arrays, and other maps/broadcasts. The
most expensive individual kernel groups also include repeated contractions of
shapes `512×1444×2304` and `1024×361×4608`. Multiple source layers can share one
kernel, so kernel-group timing is not an individual source-layer timing.

The source dumps make the temporary-array cost concrete:

- `main_dispatch_29` writes a `32×9×372100` float array: **429 MB**, repeating the
  padded 32-channel image nine times for a 3×3 stride-two convolution.
- `main_dispatch_30` writes **53,231,616 i64 indices: 426 MB**. Its body computes
  row/column coordinates from loop indices and stores the resulting array.
- `main_dispatch_31` gathers the convolution windows into another **106 MB** array.
- The 13×13 pooling path writes **500 MB of indices** and **125 MB of gathered
  values**, besides repeating its padded input into a **333 MB** array.

These are sizes of individual materialized tensors, not a measurement of peak
memory usage. Much of this data can instead remain scalar address arithmetic
inside the consumer. IREE recognizes the final matrix product, but its current
pipeline does not eliminate all of this earlier array construction.

## Isolated pieces confirm the priorities

`scripts/profile_yolo_pieces.py` compiles the actual ordinary Remora helpers at
real network dimensions. OpenCV gets a minimal network and identical synthetic
input/weight arrays. Both engines use resident inputs, two warm-ups, and seven
trials. Every result is checked. These pieces exclude batch normalization and
activation; their timings cannot be added up to reconstruct the fused network.

| Piece | Remora/IREE | OpenCV |
| --- | ---: | ---: |
| 3×3 stride-two conv, 32→64 channels, 608→304 spatial | 258 ms | 16.7 ms |
| 3×3 stride-one conv, 256→512 channels, 38×38 | 41.6 ms | 5.2 ms |
| 1×1 conv, 128→64 channels, 304×304 | 51.5 ms | 27.4 ms |
| 13×13 max-pool, 512 channels, 19×19 | 277 ms | 3.5 ms |
| Matrix product `512×2304` by `2304×1444`, without spatial indexing | 16.8 ms | 15.8 ms |

The matrix-only OpenCV reference is a mathematically equivalent 1×1 convolution
with 2,304 input channels and a 38×38 spatial frame. It performs the same dense
contraction; it is not a measurement of OpenCV's 3×3 convolution algorithm.

To isolate that algorithm choice, a fresh OpenCV network disabled Winograd
**before its first inference and weight packing**. The stride-one 3×3 example
then took **15.1 ms**, versus **5.2 ms** normally. Both results pass the numerical
check. This is evidence of a separate convolution-algorithm advantage, in addition
to the array-construction overhead. Disabling the flag after initialization is
not a reliable control because packed kernels may already be cached.

## What to work on next

1. **Keep index/broadcast producers inside gathers.** Preserve logical repeated
   views and compute coordinates at each load, instead of emitting hundreds of
   megabytes of repeated pixels and indices. This is a general lowering/fusion
   problem; it benefits ordinary indexing, convolutions, and pooling alike.
2. **Fuse window gathers into reductions.** The 13×13 pooling case is an excellent
   small target: retain the reduction and indexed reads in one kernel, without
   materializing all windows. It offers a clear numerical and performance check.
3. **Evaluate convolution algorithms separately.** Even removing array plumbing
   does not imply matching OpenCV's Winograd performance. Benchmark handwritten
   structured convolution targets at these real shapes before implementing
   recognition or scheduling machinery. A named MLIR convolution operation alone
   does not guarantee a Winograd implementation.
4. **Tune matrix multiplication after those changes.** The representative bare
   contraction is already close to its OpenCV equivalent. It is a less compelling
   first target than the huge unnecessary intermediates.

These steps need not add source-language primitives. Start with one general
producer/consumer fusion case and keep it only if the existing correctness checks
and the isolated benchmark demonstrate a useful gain. No speedup beyond the
measured baseline is claimed yet.

## Reproduce

Use the environment and assets from [the MLIR guide](../docs/mlir.md). Set
`IREE=/tmp/remora-iree/bin` and `RUN=/tmp/remora-yolo` to your actual paths.
Run each thread count in a fresh process and keep other compilation/benchmarks
idle while measuring.

```sh
IREE=/tmp/remora-iree/bin
RUN=/tmp/remora-yolo
mkdir -p /tmp/yolo-dispatches
"$IREE/iree-compile" "$RUN/yolov4.mlir" \
  --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host \
  --iree-opt-const-expr-hoisting=false --compile-to=executable-sources \
  --iree-hal-dump-executable-sources-to=/tmp/yolo-dispatches \
  -o /tmp/yolo-dispatches.mlir
"$IREE/python" scripts/profile_yolo_mlir.py \
  --module "$RUN/yolov4.vmfb" --mlir "$RUN/yolov4.mlir" \
  --config /tmp/yolov4.cfg --image /tmp/yolo-input.bin \
  --weights /tmp/yolov4.weights --threads 8 \
  --output-dir /tmp/yolo-profile --profile-kernels \
  --tools "$IREE" --dispatch-sources /tmp/yolo-dispatches
PATH="$IREE:$PATH" "$IREE/python" scripts/profile_yolo_pieces.py --threads 8
```

The scripts save all samples and per-layer/kernel results. The optional kernel
capture also writes a HAL `.ireeprof` bundle and raw JSONL. Without executable
source dumps, the profiler leaves gather attribution in the broader map category.
Large dumps, input copies, and binaries stay in `/tmp`, outside the repository.
