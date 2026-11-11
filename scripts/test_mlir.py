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
    ('empty_view_div', '(let ((fun (rep0 (x Int)) ((fn ((i Int)) x) (array [0] Int)))) (/ (rep0 1) (rep0 0)))', [0], [], 'i64'),
    ('sliding_sum', '(let ((val xs [1.0 2.0 3.0 4.0 5.0 6.0]) (fun (window (offset Int)) (@reduce/zero (Float) (3 []) f.+ 0.0 (@index (Float) (6) xs (+ offset (i-app iota/static [3])))))) (window (i-app iota/static [4])))', [4], [6.0,9.0,12.0,15.0], 'f32'),
    ('view_return', '((fn ((x [Int 2])) ((fn ((n Int)) x) (i-app iota/static [3]))) [4 5])', [3,2], [4,5,4,5,4,5], 'i64'),
    ('view_reshape', '(@reshape (Int) ([3 2] [6]) ((fn ((x [Int 2])) ((fn ((n Int)) x) (i-app iota/static [3]))) [4 5]))', [6], [4,5,4,5,4,5], 'i64'),
    ('view_transpose', '(@transpose2d (Int) (3 2) ((fn ((x [Int 2])) ((fn ((n Int)) x) (i-app iota/static [3]))) [4 5]))', [2,3], [4,4,4,5,5,5], 'i64'),
    ('record_index_cells', '(let ((type &row {(a [Int 2])})) (field (@index (&row) (2) {(a [2] [[1 2] [3 4]])} [1 0]) a))', [2,2], [3,4,1,2], 'i64'),
    ('framed_index2d', '(@index2d (Int) (2 2) [[[1 2] [3 4]] [[5 6] [7 8]]] [[[0 1] [1 0]] [[1 1] [0 0]]])', [2,2], [2,3,8,5], 'i64'),
    ('index_grid', '(@index (Int) (3) [10 20 30] [[0 2] [1 0]])', [2,2], [10,30,20,10], 'i64'),
    ('reducer_capture', '((fn ((n Int)) (@reduce/zero (Int) (2 []) (fn ((a Int) (b Int)) (+ a (+ b n))) 0 [1 2])) [10 20])', [2], [23,43], 'i64'),
    ('bool_literal', '(bool->i [#t #f])', [2], [1,0], 'i64'),
    ('framed_record', '(field ((fn ((x Int)) {(a [] x) (b [2] [x x])}) [1 2]) b)', [2,2], [1,1,2,2], 'i64'),
    ('cell_frame', '((fn ((x Int)) [x (+ x 1)]) [2 4])', [2,2], [2,3,4,5], 'i64'),
    ('cell_capture', '(let ((fun (f (x [Int 2])) ((fn ((n Int)) (+ x n)) [10 20 30]))) (f [[1 2] [3 4]]))', [2,3,2], [11,12,21,22,31,32,13,14,23,24,33,34], 'i64'),
    ('framed_transpose', '(@transpose2d (Int) (2 2) [[[1 2] [3 4]] [[5 6] [7 8]]])', [2,2,2], [1,3,2,4,5,7,6,8], 'i64'),
    ('framed_reduce', '(@reduce/zero (Float) (2 []) f.+ [10.0 20.0] [[1.0 2.0] [3.0 4.0]])', [2], [13.0,27.0], 'f32'),
    ('empty_float_reduce', '(@reduce/zero (Float) (0 []) f.+ 3.0 (array [0] Float))', [], [3.0], 'f32'),
    ('fold_left', '(@fold (Int Int) (2 [] []) - 10 [1 2 3])', [], [4], 'i64'),
    ('fold_right', '(@fold-right (Int Int) (2 [] []) - 10 [1 2 3])', [], [-8], 'i64'),
    ('bool_float', '(bool->f (f.== [1.0 2.0] 2.0))', [2], [0.0,1.0], 'f32'),
    ('array_cell', '((fn ((x [Int 2])) (+ x 1)) [[1 2] [3 4]])', [2,2], [2,3,4,5], 'i64'),
    ('float_reduce', '(@reduce/zero (Float) (3 []) f.+ 0.0 [1.0 2.0 3.0])', [], [6.0], 'f32'),
    ('vector_reduce', '(@reduce (Float) (1 [2]) (fn ((a [Float 2]) (b [Float 2])) (f.max a b)) [[1.0 4.0] [3.0 2.0]])', [2], [3.0,4.0], 'f32'),
    ('transpose', '(@transpose2d (Int) (2 3) [[1 2 3] [4 5 6]])', [3,2], [1,4,2,5,3,6], 'i64'),
    ('reshape', '(@reshape (Int) ([2 3] [3 2]) [[1 2 3] [4 5 6]])', [3,2], [1,2,3,4,5,6], 'i64'),
    ('index2d', '(@index2d (Int) (2 3) [[1 2 3] [4 5 6]] [[0 2] [1 0]])', [2], [3,4], 'i64'),
    ('append', '(@append (Int) (2 1 [2]) [[1 2] [3 4]] [[5 6]])', [3,2], [1,2,3,4,5,6], 'i64'),
    ('iota', '(i-app iota/static [2 3])', [2,3], [0,1,2,3,4,5], 'i64'),
    ('exp', '(f.exp [0.0 0.0])', [2], [1.0,1.0], 'f32'),
    ('sqrt', '(f.sqrt [1.0 4.0])', [2], [1.0,2.0], 'f32'),
    ('record', '(field {(x [2] [1 2]) (y [] 5)} x)', [2], [1,2], 'i64'),
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
RECTANGULAR_POOL = '''(entry (main (image [Float 1 5 7]))
  ((fn ((channel [Float 5 7]))
     (let ((val windows
              ((fn ((dy Int))
                 ((fn ((dx Int))
                    ((fn ((oy Int))
                       ((fn ((ox Int))
                          (@index2d (Float) (5 7) channel
                            [(+ (* oy 2) (* dy 2)) (+ (* ox 2) dx)]))
                        (i-app iota/static [3])))
                     (i-app iota/static [2])))
                  (i-app iota/static [3])))
               (i-app iota/static [2]))))
       (@reduce/zero (Float) (3 [2 3])
         (fn ((a [Float 2 3]) (b [Float 2 3])) (f.max a b))
         ((fn ((i Int)) -10.0) (i-app iota/static [2 3]))
         (@reduce/zero (Float) (2 [3 2 3])
           (fn ((a [Float 3 2 3]) (b [Float 3 2 3])) (f.max a b))
           ((fn ((i Int)) -20.0) (i-app iota/static [3 2 3]))
           windows))))
   image))
'''

PASSES = ('builtin.module(inline,canonicalize,cse,one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops '
          'function-boundary-type-conversion=identity-layout-map},buffer-deallocation-pipeline,convert-bufferization-to-memref,'
          'convert-linalg-to-loops,convert-scf-to-cf,expand-strided-metadata,lower-affine,'
          'convert-math-to-llvm,convert-math-to-libm,'
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

    def test_window_pooling(self):
        reversed_rows = RECTANGULAR_POOL.replace('(* dy 2)', '(- 2 (* dy 2))')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'pool.remora'
            source.write_text(RECTANGULAR_POOL)
            ir = run([EXE, 'mlir', '-f', source])
            spatial_map = '(d0,d1,d2 * 2 + d4 * 2,d3 * 2 + d5 * 1)'
            self.assertIn(spatial_map, ir)
            self.assertIn('"parallel", "reduction", "reduction"', ir)
            source.write_text(reversed_rows)
            ir = run([EXE, 'mlir', '-f', source])
            self.assertNotIn(spatial_map, ir)
            self.assertIn('arith.maximumf', ir)

    def test_window_contraction(self):
        definitions = (ROOT / 'examples/darknet/yolov4.remora').read_text().split('(entry', 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'conv.remora'
            source.write_text(definitions + """
(entry (main (x [Float 2 6 6]) (w [Float 3 2 3 3]))
  (@conv-2d/rem _ (2 3 2 3 1 2 1) x w))
""")
            ir = run([EXE, 'mlir', '--structured-conv', '-f', source])
            self.assertIn('linalg.conv_2d_nchw_fchw', ir)
            self.assertIn('strides = dense<[2, 2]>', ir)
            self.assertNotIn('linalg.conv_2d_nchw_fchw', run([EXE, 'mlir', '-f', source]))
            source.write_text(definitions + """
(entry (main (w [Float 3 18]) (x [Float 18 9]))
  (@matmul _ (3 17 9) w x))
""")
            ir = run([EXE, 'mlir', '--structured-conv', '-f', source])
            self.assertNotIn('linalg.conv_2d_nchw_fchw', ir)
            self.assertIn('arith.mulf', ir)

    def test_rejections(self):
        for source in ['(fn ((x Int)) x)', '(+ [1 2 3] [[1 2] [3 4]])',
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
