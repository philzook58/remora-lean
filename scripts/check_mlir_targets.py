"""Execute handwritten target contracts; these do not measure emitter coverage."""
import argparse
import json
from pathlib import Path
from test_mlir import ROOT, PASSES, run, wrapper


def check(directory):
    opt, runner = directory/'mlir-opt', directory/'mlir-runner'
    lib = directory.parent/'lib/libmlir_c_runner_utils.so'
    for path in [opt,runner,lib]:
        if not path.exists():
            raise SystemExit(f'Required MLIR tool/runtime missing: {path}')
    print(run([opt,'--version']).strip())
    cases = [
        ('matmul-bias-relu', [2,2], [0.0,13.0,0.0,0.0],
         ['tensor<2x3xf32>','tensor<3x2xf32>','tensor<2xf32>'],
         [[[1.0,-2.0,3.0],[-1.0,2.0,-3.0]], [[1.0,2.0],[3.0,4.0],[5.0,6.0]], [-11.0,1.0]]),
        ('conv-bias-leaky', [1,2,2,2], [-0.8,-0.4,4.0,8.0,8.0,4.0,-0.4,-0.8],
         ['tensor<1x1x3x3xf32>','tensor<2x1x3x3xf32>','tensor<2xf32>'],
         [[[[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]]]],
          [[[[1.0]*3]*3],[[[-1.0]*3]*3]], [-16.0,16.0]])]
    for name, dims, expected, types, inputs in cases:
        ir = (ROOT/'docs/mlir-targets'/f'{name}.mlir').read_text()
        run([opt,'--verify-each'],ir)
        result_type = 'tensor<' + ''.join(f'{n}x' for n in dims) + 'f32>'
        constants = '\n'.join(f'%input{i} = arith.constant dense<{json.dumps(x)}> : {t}'
                              for i,(t,x) in enumerate(zip(types,inputs)))
        call = (f'%result = func.call @"main"(' + ', '.join(f'%input{i}' for i in range(len(types)))
                + f') : ({", ".join(types)}) -> {result_type}')
        test = wrapper(dims,expected,'f32').replace(
            f'%result = func.call @"main"() : () -> {result_type}', constants+'\n'+call)
        module = ir.rstrip().removesuffix('}') + test + '\n}'
        llvm = run([opt,'--verify-each','--pass-pipeline='+PASSES],module)
        actual = run([runner,'-e','__remora_test','-entry-point-result=i64',
                      '-shared-libs='+str(lib)],llvm).strip()
        if actual != '1':
            raise AssertionError(f'{name}: element comparison failed: {actual}')
        print(f'PASS handwritten target: {name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mlir-bin-dir',required=True,type=Path)
    check(parser.parse_args().mlir_bin_dir)
