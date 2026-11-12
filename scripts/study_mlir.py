"""Small runnable lesson: explicit MLIR scheduling and optional IREE recognition.

All compiler dumps go to --output-dir. No Remora language changes or Mojo/Halide
installation are required. See docs/mlir-study.md for the guided explanation.
"""
import argparse
import json
import tempfile
from pathlib import Path
from test_mlir import ROOT, PASSES, run, wrapper

TARGETS = ROOT/'docs/mlir-targets'


def scheduled(opt, ir):
    # These are controlled lesson fixtures, not a general MLIR parser.
    schedule = (TARGETS/'matmul-row-schedule.mlir').read_text()
    body = schedule.split('\n',1)[1].rsplit('\n}',1)[0]
    combined = (ir.rstrip()[:-1] + body + '\n}').replace(
        'module {','module attributes {transform.with_named_sequence} {',1)
    return run([opt,'--transform-interpreter','--test-transform-dialect-erase-schedule',
                '--canonicalize','--cse'],combined)


def check_cpu(directory, ir):
    out_type = 'tensor<2x2xf32>'
    test = wrapper([2,2],[0.0,13.0,0.0,0.0],'f32').replace(
        f'%result = func.call @"main"() : () -> {out_type}',
        '''%a = arith.constant dense<[[1.0,-2.0,3.0],[-1.0,2.0,-3.0]]> : tensor<2x3xf32>
%b = arith.constant dense<[[1.0,2.0],[3.0,4.0],[5.0,6.0]]> : tensor<3x2xf32>
%bias = arith.constant dense<[-11.0,1.0]> : tensor<2xf32>
%result = func.call @"main"(%a,%b,%bias) : (tensor<2x3xf32>,tensor<3x2xf32>,tensor<2xf32>) -> tensor<2x2xf32>''')
    module = ir.rstrip()[:-1] + test + '\n}'
    llvm = run([directory/'mlir-opt','--verify-each','--pass-pipeline='+PASSES],module)
    result = run([directory/'mlir-runner','-e','__remora_test','-entry-point-result=i64',
                  '-shared-libs='+str(directory.parent/'lib/libmlir_c_runner_utils.so')],llvm)
    if result.strip() != '1':
        raise AssertionError('CPU element comparison failed: '+result)


def iree_check(python, source, output, large=False):
    compiler = python.parent/'iree-compile'
    dump = Path(tempfile.mkdtemp(
        prefix=source.stem + ('-128' if large else '') + '-iree-', dir=output))
    binary = dump/'module.vmfb'
    ir = source.read_text()
    if large:
        for old in ['2x3','3x2','2x2']:
            ir = ir.replace('tensor<'+old+'xf32>','tensor<128x128xf32>')
        ir = ir.replace('tensor<2xf32>','tensor<128xf32>')
    run([compiler,'-','--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',
         '--iree-hal-dump-executable-files-to='+str(dump),'-o',binary],ir)
    # Run inside the isolated environment containing IREE and NumPy.
    code = '''import sys, numpy as np, iree.runtime as rt
m=rt.load_vm_flatbuffer(open(sys.argv[1],'rb').read(),driver='local-task')
if sys.argv[2]=='large':
 rng=np.random.default_rng(0)
 a=rng.integers(-4,5,(128,128)).astype(np.float32)/4
 b=rng.integers(-4,5,(128,128)).astype(np.float32)/4
 bias=rng.integers(-4,5,128).astype(np.float32)/4
else:
 a=np.array([[1,-2,3],[-1,2,-3]],np.float32)
 b=np.array([[1,2],[3,4],[5,6]],np.float32)
 bias=np.array([-11,1],np.float32)
actual=m.main(a,b,bias).to_host()
np.testing.assert_allclose(actual,np.maximum(a@b+bias,0),rtol=1e-5,atol=1e-5)
print('PASS IREE numerical check',actual.shape)
'''
    print(run([python,'-c',code,binary,'large' if large else 'small']).strip())
    configs = sorted(dump.glob('configured*.mlir'))
    summaries = []
    for config in configs:
        lines = [line.strip() for line in config.read_text().splitlines()
                 if 'func.func' in line or 'lowering_config =' in line]
        summaries.extend(lines)
    print(f'{source.stem}: {len(configs)} configured dispatch(es)')
    for line in summaries:
        print(' ',line[:1000])
    return {'source':source.name,'large':large,'dispatches':len(configs),
            'configured_ir':summaries,'dump_directory':str(dump)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mlir-bin-dir',required=True,type=Path)
    parser.add_argument('--iree-python',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('/tmp/remora-mlir-study-output'))
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.mlir_bin_dir = args.mlir_bin_dir.expanduser().resolve()
    if args.iree_python:
        args.iree_python = args.iree_python.expanduser().absolute()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    opt = args.mlir_bin_dir/'mlir-opt'
    print(run([opt,'--version']).strip())
    named = TARGETS/'matmul-bias-relu.mlir'
    generic = TARGETS/'matmul-generic.mlir'
    after = scheduled(opt,named.read_text())
    (args.output_dir/'matmul-row-tiled.mlir').write_text(after)
    for label, ir in [('named',named.read_text()),('generic',generic.read_text()),('row-tiled',after)]:
        run([opt,'--verify-each'],ir)
        check_cpu(args.mlir_bin_dir,ir)
        print('PASS MLIR execution:',label)
    if args.iree_python:
        print(run([args.iree_python.parent/'iree-compile','--version']).strip())
        observations = [iree_check(args.iree_python,p,args.output_dir,large)
                        for large in (False,True) for p in (named,generic)]
        (args.output_dir/'iree-observations.json').write_text(json.dumps(observations,indent=2)+'\n')
    else:
        print('IREE experiment not requested; use --iree-python for that part.')


if __name__ == '__main__':
    main()
