"""Execute ordinary Remora neural-network examples and parameterized kernels.

Run in a Python environment with numpy, iree-runtime and a matching iree-compile.
This is an explicit integration check, not part of ordinary unittest discovery.
"""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
from run_yolo_mlir import ROOT, SOURCE, command


def emit(source, target):
    with target.open('w') as output:
        subprocess.run([ROOT/'.lake/build/bin/remora', 'mlir', '-f', source], cwd=ROOT,
                       stdout=output, check=True, timeout=90)


def compile_load(compiler, ir):
    import iree.runtime as rt
    binary = ir.with_suffix('.vmfb')
    command([compiler, ir, '--iree-hal-target-backends=llvm-cpu',
             '--iree-llvmcpu-target-cpu=host', '--iree-opt-const-expr-hoisting=false',
             '-o', binary], ir.with_suffix('.log'))
    return rt.load_vm_flatbuffer(binary.read_bytes(), driver='local-task')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/remora-darknet-mlir'))
    parser.add_argument('--compiler', default=shutil.which('iree-compile'))
    args = parser.parse_args()
    if not args.compiler:
        parser.error('iree-compile must be on PATH or supplied via --compiler')
    import numpy as np
    out = args.output_dir.expanduser().absolute()
    out.mkdir(parents=True, exist_ok=True)
    # conv_remainder's standalone entry deliberately uses undefined weights.
    # Exercise its real stride/remainder path with random parameters below.
    examples = sorted((ROOT/'examples/darknet').glob('*.remora'))
    for source in examples:
        if source.stem in ('yolov4', 'conv_remainder'):
            continue
        ir = out/(source.stem+'.mlir')
        emit(source, ir)
        module = compile_load(args.compiler, ir)
        actual = module.main().to_host()
        text = subprocess.check_output([ROOT/'.lake/build/bin/remora', 'interpret', '-f', source],
                                       cwd=ROOT, text=True, timeout=120)
        expected = np.array([float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', text)])
        np.testing.assert_allclose(actual.flatten(), expected, rtol=1e-4, atol=2e-5)
        print(f'PASS {source.stem} {actual.shape}', flush=True)
    # Use the full demo's own polymorphic definitions, with small concrete
    # parameters. Arbitrary non-symmetric weights catch axis-order mistakes.
    definitions = SOURCE.read_text().split('(entry', 1)[0]
    rng = np.random.default_rng(0)
    for name, entry, shapes in [
        ('matmul', '(entry (main (a [Float 2 3]) (b [Float 3 4])) (@matmul _ (2 2 4) a b))', [(2,3),(3,4)]),
        ('conv_remainder', '(entry (main (im [Float 2 4 4]) (w [Float 3 2 3 3])) (@conv-2d/rem _ (1 3 2 3 1 2 1) im w))', [(2,4,4),(3,2,3,3)])]:
        source = out/(name+'-parameters.remora')
        source.write_text(definitions+entry+'\n')
        ir = source.with_suffix('.mlir')
        emit(source, ir)
        module = compile_load(args.compiler, ir)
        values = [rng.integers(-3, 4, shape).astype(np.float32) for shape in shapes]
        a, b = values
        if name == 'matmul':
            expected = a@b
        else:
            padded = np.pad(a, ((0,0),(1,1),(1,1)))
            expected = np.array([[[np.sum(padded[:, i*2:i*2+3, j*2:j*2+3]*f)
                                   for j in range(2)] for i in range(2)] for f in b]).reshape(3,4)
        actual = module.main(*values).to_host()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        print(f'PASS parameterized {name} {actual.shape}', flush=True)


if __name__ == '__main__':
    main()
