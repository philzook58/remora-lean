"""Compile ordinary yolov4.remora with IREE and run raw CHW float32 image data.

Run with a Python environment containing numpy and iree-runtime, with the matching
iree-compile on PATH. No Futhark, PyTorch, or compiler-side YOLO primitives.
"""
import argparse
import json
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'examples/darknet/yolov4.remora'


def command(args, log, timeout=300):
    with log.open('w') as errors:
        result = subprocess.run(list(map(str, args)), stdout=errors, stderr=errors,
                                cwd=ROOT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'Command failed; see {log}\n{log.read_text()[-4000:]}')


def entry_shapes(ir):
    # Read the static signature produced by our emitter, rather than maintaining
    # a second list of 542 parameters that could disagree with the compiler.
    header = next(line for line in ir.splitlines() if line.startswith('func.func @"main"('))
    params = header.split(') ->', 1)[0].split('(', 1)[1]
    types = re.findall(r': ([^,]+)', params)
    shapes = []
    for ty in types:
        match = re.fullmatch(r'tensor<((?:\d+x)+)f32>', ty.strip())
        if not match:
            raise ValueError(f'YOLO runner expects static Float arrays, got {ty}')
        shapes.append(tuple(map(int, match[1].rstrip('x').split('x'))))
    return shapes


def inputs(shapes, image_path, weights_path):
    import numpy as np
    for path in (image_path, weights_path):
        with path.open('rb') as f:
            if f.read(40).startswith(b'version https://git-lfs.github.com'):
                raise ValueError(f'{path} is a Git LFS pointer; fetch its actual data first')
    image = np.fromfile(image_path, dtype='<f4').reshape(shapes[0])
    with weights_path.open('rb') as f:
        major, minor, _ = struct.unpack('<iii', f.read(12))
    header = 20 if major*10+minor >= 2 and major < 1000 and minor < 1000 else 16
    weights = np.fromfile(weights_path, dtype='<f4', offset=header)
    result, offset = [image], 0
    for shape in shapes[1:]:
        count = int(np.prod(shape))
        result.append(weights[offset:offset+count].reshape(shape))
        offset += count
    if offset != weights.size:
        raise ValueError(f'Entry consumes {offset} weights, file has {weights.size}')
    return result


def detections(rows, threshold=0.25, iou_threshold=0.45):
    """Best class per box, confidence = objectness * class probability, class NMS."""
    import numpy as np
    classes = rows[:, 5:].argmax(axis=1)
    scores = rows[:, 4]*rows[np.arange(len(rows)), classes+5]
    order = np.flatnonzero(scores >= threshold)
    order = order[np.argsort(scores[order])[::-1]]
    lo, hi = rows[:, :2]-rows[:, 2:4]/2, rows[:, :2]+rows[:, 2:4]/2
    area = rows[:, 2]*rows[:, 3]
    kept = []
    while len(order):
        i, rest = int(order[0]), order[1:]
        kept.append({'class': int(classes[i]), 'confidence': float(scores[i]),
                     'box_xywh': rows[i, :4].tolist()})
        intersection = np.maximum(0, np.minimum(hi[i], hi[rest])-np.maximum(lo[i], lo[rest])).prod(axis=1)
        union = area[i]+area[rest]-intersection
        iou = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        order = rest[(classes[rest] != classes[i]) | (iou <= iou_threshold)]
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--image', type=Path, required=True, help='raw RGB CHW float32, 3x608x608')
    parser.add_argument('--weights', type=Path, required=True, help='Darknet yolov4.weights')
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/remora-yolo-mlir'))
    parser.add_argument('--compiler', default=shutil.which('iree-compile'))
    parser.add_argument('--rocm-target', help='Compile and run on HIP, e.g. gfx1103 for Radeon 780M')
    parser.add_argument('--module', type=Path, help='reuse an IREE executable for this exact source')
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--iree-flag', action='append', default=[],
                        help='extra compiler option; use --iree-flag=--option=value')
    args = parser.parse_args()
    if args.module and args.iree_flag:
        parser.error('--iree-flag applies when compiling; omit --module')
    import numpy as np
    import iree.runtime as rt
    out = args.output_dir.expanduser().absolute()
    out.mkdir(parents=True, exist_ok=True)
    ir, binary = out/'yolov4.mlir', args.module or out/'yolov4.vmfb'
    driver = 'hip' if args.rocm_target else 'local-task'
    remora_flags = ['--structured-conv'] if args.rocm_target else []
    start = time.monotonic()
    with ir.open('w') as stream, (out/'remora.log').open('w') as errors:
        subprocess.run([ROOT/'.lake/build/bin/remora', 'mlir', *remora_flags, '-f',
                        args.source.absolute()],
                       cwd=ROOT, stdout=stream, stderr=errors, check=True, timeout=120)
    if not args.module:
        if not args.compiler:
            parser.error('iree-compile is required on PATH or via --compiler')
        # IREE 3.13's constant-expression hoisting can fail its fixed-point
        # iteration limit on these index-array expressions. Disable that pass;
        # runtime constants and ordinary tensor optimization still work.
        if args.rocm_target:
            target_flags = ['--iree-hal-target-backends=rocm',
                            f'--iree-rocm-target={args.rocm_target}']
        else:
            target_flags = ['--iree-hal-target-backends=llvm-cpu', '--iree-llvmcpu-target-cpu=host']
        command([args.compiler, ir, *target_flags, '--iree-opt-const-expr-hoisting=false',
                 *args.iree_flag, '-o', binary], out/'iree.log')
    compile_seconds = time.monotonic()-start
    shapes = entry_shapes(ir.read_text())
    arguments = inputs(shapes, args.image, args.weights)
    module = rt.load_vm_flatbuffer(binary.read_bytes(), driver=driver)
    start = time.monotonic()
    result = module.main(*arguments).to_host()
    run_seconds = time.monotonic()-start
    if result.shape != (22743, 85) or not np.isfinite(result).all():
        raise RuntimeError(f'Invalid YOLO result: shape={result.shape}, finite={np.isfinite(result).all()}')
    if not result.any():
        raise RuntimeError('Invalid YOLO result: all output attributes are zero')
    np.save(out/'detections.npy', result)
    summary = {'driver': driver, 'rocm_target': args.rocm_target, 'iree_flags': args.iree_flag,
               'arguments': len(arguments), 'shape': list(result.shape),
               'compile_seconds': compile_seconds, 'run_seconds': run_seconds,
               'detections': detections(result, args.threshold)}
    (out/'result.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
