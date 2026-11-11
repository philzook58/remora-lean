# PyTorch CPU comparison for YOLOv4

Measured November 10, 2026 on the Ryzen 7 PRO 7840U with PyTorch 2.15.0+cpu.
All paths use the same 608×608 image and Darknet weights. The medians include
decoding the full 22,743×85 result, with eight CPU threads, two warm-ups, and
seven measured runs. PyTorch was timed in a separate process from the alternating
OpenCV and IREE runs. Weight loading and model setup were outside the timers.

| CPU path | Median | Sample range |
| --- | ---: | ---: |
| PyTorch eager, channels-last | 418 ms | 405–429 ms |
| PyTorch eager, NCHW | 442 ms | 429–471 ms |
| OpenCV | 419 ms | 380–633 ms |
| Remora → IREE, resident arguments, data tiling | 812 ms | 767–873 ms |
| Remora → printed PyTorch, eager | 4,290 ms | 4,195–4,430 ms |
| Remora → printed PyTorch, TorchInductor | 850 ms | 817–1,073 ms |

The script builds a PyTorch FX graph from the Darknet configuration, folds
inference batch normalization into the convolution weights and biases during
setup, and calls PyTorch convolution, pooling, and other tensor operations.
It uses the same NumPy decoder as the OpenCV comparison. Every PyTorch trial
passed the existing full-output tolerance against the IREE reference.
Maximum absolute difference was 0.0000262 for NCHW and 0.0000248 for channels-last.

Run from the repository root with CPU PyTorch and NumPy installed. The
[MLIR guide](../docs/mlir.md) has the demo assets and IREE runner instructions.

```sh
python scripts/benchmark_yolo_torch.py \
  --config /tmp/yolov4.cfg --weights /tmp/yolov4.weights \
  --image /tmp/yolo-input.bin --reference /tmp/remora-yolo/detections.npy \
  --emit-forward /tmp/yolo-forward.py
```

The emitted forward method refers to the modules and weights built by the script.

## PyTorch printed from Remora

The first comparison below used the gather-and-contraction lowering.

The emitter consumes the same static tensor MLIR as IREE. It uses IREE's MLIR
parser and standard inlining and simplification passes at generation time. The
printed module calls PyTorch directly and runs with a PyTorch-only installation.

The emitter turns indexing maps into broadcasting, permutation, and advanced
indexing. A projected floating multiply consumed by a sum becomes an
`einsum`. It composes through broadcast views, avoiding the expanded product
tensor. YOLOv4's ordinary Remora convolutions still gather and
materialize the input patches, and its batch normalization, activations,
pooling, and decoder run as emitted tensor operations. The emitter supports
static maps, gathers, reshapes, concatenation, static slices, and sum/min/max
reductions. Ordered folds using `scf.for` fail at emission.

The full YOLOv4 module printed 110 calls to `torch.einsum`. Each of the nine
forward passes passed the 22,743×85 output comparison against IREE, with
maximum absolute error 0.0000206. In a separate CPU profile, advanced indexing
accounted for 61% of PyTorch operator self time and batched matrix multiply for
13%. The profile includes all indexing, including convolution patch gathers.

Use the environments and assets from the MLIR guide. Run the generator in the
environment with `iree-base-compiler` and the benchmark in the environment
with PyTorch and NumPy. The original MLIR provides the benchmark with the
signature it uses to unpack the weight file.

```sh
lake exe remora mlir --structured-conv -f examples/darknet/yolov4.remora > /tmp/yolov4.mlir
/tmp/remora-iree/bin/python scripts/emit_pytorch.py /tmp/yolov4.mlir > /tmp/yolov4_torch.py
python scripts/benchmark_yolo_torch.py \
  --printed /tmp/yolov4_torch.py --mlir /tmp/yolov4.mlir \
  --weights /tmp/yolov4.weights --image /tmp/yolo-input.bin \
  --reference /tmp/remora-yolo/detections.npy
```

Append `--compile` to use `torch.compile(fullgraph=True)` with the default
TorchInductor backend. The benchmark reports the first call separately. With
an empty Inductor cache and `TORCHINDUCTOR_COMPILE_THREADS=2`, the first call
took 69.5 s including compilation. All compiled runs passed the same
full-output tolerance, with maximum absolute error 0.0000200.

Inductor generated fused CPU kernels and retained all 110 batched matrix
multiplies. The generated wrapper still allocates patch tensors before those
calls. One 3×3 convolution uses a 32×3×3×304×304 float32 patch buffer, about
102 MiB. That intermediate identifies where recognizing sliding-window
contractions as native convolution could help.

## Structured convolution follow-up

The `--structured-conv` path recognizes all 110 convolutions in ordinary
YOLOv4. It reuses the pooling window analysis and verifies the row-major
flattening and multiply-and-sum axes. The printer calls `conv2d` directly,
passes symmetric zero padding as a convolution argument, and removes
concatenations with empty padding. The normal Remora entry still accepts
the image and all checkpoint tensors.

The stride-two 3×3 layer provided an initial target. On the same synthetic
image and weights, the gather-and-contraction IREE path took 63 ms, the
handwritten named Linalg convolution took 44 ms, and native PyTorch
`conv2d` took 18 ms in NCHW or 10 ms in channels-last. Full YOLOv4 behaved
differently in IREE. The direct NCHW op reached 4.55 s, and running
`iree-global-opt-convert-conv2d-to-img2col` before compilation reduced it
to 0.87 s versus 0.94 s for the pooling-only baseline in that run. The named
convolution remains opt-in.

A separate fixed-weight experiment moved weight-only calculations into setup.
In the same eager run, it measured 0.80 s with NCHW weights and 0.81 s with
channels-last weights, versus 0.82 s without preparation. The production
printer keeps the ordinary entry interface.

Final full-model CPU measurements used eight threads, two warm-ups, and seven
timed runs per path. Both paths passed the existing full-output tolerance.

| Printed PyTorch with structured convolution | Median | Range |
| --- | ---: | ---: |
| Eager | 736 ms | 689–785 ms |
| TorchInductor | 553 ms | 542–629 ms |

Maximum absolute error was 0.0000205 in eager mode and 0.0000260 with Inductor.
Inductor's first call took 17.2 s with a cache from an earlier convolution
experiment. A fresh cache was not timed for the final graph.

## Native PyTorch on ROCm

The AMD PyTorch 2.14.0 ROCm build recognized the Radeon 780M as `gfx1103` and
passed a small convolution check. During the full YOLOv4 benchmark on
November 10, the kernel logged repeated AMD GPU resets. The benchmark produced
no timing result. The cause of the later host reboot remains unverified.
