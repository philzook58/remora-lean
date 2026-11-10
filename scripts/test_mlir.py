"""Emission tests run in normal unittest discovery.

CPU tests: python3 scripts/test_mlir.py --native --mlir-bin-dir /path/to/llvm/bin
"""
import argparse
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / '.lake/build/bin/remora'
CASES = [
    ('empty', '(+ (array [0] Int) 1)', [0], [], 'i64'),
    ('empty_sum', '(@sum _ ([0]) (array [0] Int))', [], [0], 'i64'),
    ('empty_rows', '(@sum _ ([0]) (array [2 0] Int))', [2], [0,0], 'i64'),
    ('overflow', '(+ 9223372036854775807 1)', [], [-9223372036854775808], 'i64'),
    ('scalar', '(+ 1 2)', [], [3], 'i64'),
    ('map', '(+ [1 2] 10)', [2], [11, 12], 'i64'),
    ('matrix', '(+ [1 2] [[10 20 30] [40 50 60]])', [2, 3], [11,21,31,42,52,62], 'i64'),
    ('reverse', '(- [[10 20 30] [40 50 60]] [1 2])', [2, 3], [9,19,29,38,48,58], 'i64'),
    ('capture', '(let ((val n 10) (fun (f (x Int)) (+ n x))) (f [1 2]))', [2], [11,12], 'i64'),
    ('rowsum', '(@sum _ ([3]) [[10 20 30] [40 50 60]])', [2], [60,150], 'i64'),
    ('sum', '(@sum _ ([2 3]) [[10 20 30] [40 50 60]])', [], [210], 'i64'),
    ('float', '(f.* [1.5 2.0] 2.0)', [2], [3.0,4.0], 'f32'),
    ('computed', '[(+ 1 2) (* 2 3)]', [2], [3,6], 'i64'),
    ('computed_rows', '[(+ [1 2] 1) (* [3 4] 2)]', [2,2], [2,3,6,8], 'i64'),
]
PASSES = ('builtin.module(one-shot-bufferize{bufferize-function-boundaries '
          'function-boundary-type-conversion=identity-layout-map},buffer-deallocation-pipeline,'
          'convert-linalg-to-loops,convert-scf-to-cf,expand-strided-metadata,lower-affine,'
          'func.func(llvm-request-c-wrappers),finalize-memref-to-llvm,convert-func-to-llvm,'
          'convert-arith-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)')


def run(args, source=None):
    p = subprocess.run(list(map(str, args)), input=source, capture_output=True, text=True,
                       cwd=ROOT, timeout=60)
    if p.returncode:
        raise AssertionError(f'{args}\n{p.stderr}\n{p.stdout}')
    return p.stdout


def emit(source):
    return run([EXE, 'mlir', '-e', source])


def pretty(values, dims):
    if not dims:
        return format(values[0], ".6f") if isinstance(values[0],float) else str(values[0])
    size = 1
    for n in dims[1:]:
        size *= n
    return '[' + ' '.join(pretty(values[i*size:(i+1)*size], dims[1:])
                          for i in range(dims[0])) + ']'


def wrapper(dims, values, base):
    ty = 'tensor<' + ''.join(f'{n}x' for n in dims) + base + '>' if dims else base
    lines = ['func.func @__remora_test() -> i64 attributes {llvm.emit_c_interface} {',
             f'%result = func.call @"main"() : () -> {ty}',
             '%ok0 = arith.constant true']
    indices = list(itertools.product(*(range(n) for n in dims)))
    for k, (idx, value) in enumerate(zip(indices, values)):
        if dims:
            for axis, n in enumerate(idx):
                lines.append(f'%i{k}_{axis} = arith.constant {n} : index')
            loc = ', '.join(f'%i{k}_{axis}' for axis in range(len(dims)))
            lines.append(f'%actual{k} = tensor.extract %result[{loc}] : {ty}')
            actual = f'%actual{k}'
        else:
            actual = '%result'
        lines.append(f'%expected{k} = arith.constant {value} : {base}')
        op = 'cmpi eq' if base == 'i64' else 'cmpf oeq'
        lines.append(f'%cmp{k} = arith.{op}, {actual}, %expected{k} : {base}')
        lines.append(f'%ok{k+1} = arith.andi %ok{k}, %cmp{k} : i1')
    lines.extend([f'%answer = arith.extui %ok{len(values)} : i1 to i64',
                  'return %answer : i64', '}'])
    return '\n'.join(lines)


class EmissionTests(unittest.TestCase):
    def test_supported(self):
        for name, source, dims, values, base in CASES:
            with self.subTest(name=name):
                self.assertIn('func.func @"main"', emit(source))
                self.assertEqual(run([EXE, 'interpret', '-e', source]).strip(), pretty(values,dims))

    def test_program_and_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'input.remora'
            source.write_text('(entry (main (x [Int 2])) (+ x 1))')
            self.assertIn('tensor<2xi64>', run([EXE,'mlir','-f',source]))
        p = subprocess.run([EXE,'mlir','--run','-e','1'],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('emission only',p.stderr)

    def test_rejections(self):
        for source in ['(fn ((x Int)) x)', '(/ 4 2)', '(+ [1 2 3] [[1 2] [3 4]])',
                       '(box (2) [1 2] (Sigma ($n) [Int $n]))']:
            with self.subTest(source=source):
                p = subprocess.run([EXE,'mlir','-e',source],capture_output=True,text=True)
                self.assertNotEqual(p.returncode,0)
                self.assertTrue(p.stderr)


def native(directory):
    directory = Path(directory)
    opt, runner = directory/'mlir-opt', directory/'mlir-runner'
    lib = directory.parent/'lib/libmlir_c_runner_utils.so'
    for path in [opt,runner,lib]:
        if not path.exists():
            raise SystemExit(f'Required MLIR tool/runtime missing: {path}')
    print(run([opt,'--version']).strip())
    for name, source, dims, values, base in CASES:
        ir = emit(source)
        run([opt,'--verify-each'],ir)
        module = ir.rstrip().removesuffix('}') + wrapper(dims,values,base) + '\n}\n'
        lowered = run([opt,'--verify-each','--pass-pipeline='+PASSES],module)
        result = run([runner,'-e','__remora_test','-entry-point-result=i64',
                      '-shared-libs='+str(lib)],lowered).strip()
        if result != '1':
            raise AssertionError(f'{name}: element comparison failed: {result}')
        print(f'PASS {name}')
    # Exercise an exported entry with an actual tensor argument, not just constants.
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/'argument.remora'
        path.write_text('(entry (main (x [Int 2])) (+ x 1))')
        assert run([EXE,'interpret','-f',path,'[1 2]']).strip() == '[2 3]'
        ir = run([EXE,'mlir','-f',path])
        test = wrapper([2],[2,3],'i64').replace(
            '%result = func.call @"main"() : () -> tensor<2xi64>',
            '%input = arith.constant dense<[1,2]> : tensor<2xi64>\n'
            '%result = func.call @"main"(%input) : (tensor<2xi64>) -> tensor<2xi64>')
        module = ir.rstrip().removesuffix('}') + test + '\n}\n'
        lowered = run([opt,'--verify-each','--pass-pipeline='+PASSES],module)
        assert run([runner,'-e','__remora_test','-entry-point-result=i64',
                    '-shared-libs='+str(lib)],lowered).strip() == '1'
        print('PASS entry_argument')


if __name__ == '__main__':
    if '--native' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--native',action='store_true')
        parser.add_argument('--mlir-bin-dir',required=True)
        args = parser.parse_args()
        native(args.mlir_bin_dir)
    else:
        unittest.main()
