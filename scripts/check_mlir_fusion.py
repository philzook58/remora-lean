"""Verify which indexing representation lets existing MLIR fusion remove arrays.

Also execute the equivalent structured pooling target against NumPy. Requires
IREE tools plus the isolated validation Python; this adds no Lean dependencies.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from run_yolo_mlir import ROOT, command
from profile_yolo_mlir import summary

TARGETS = ROOT/'docs/mlir-targets'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tools', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/remora-fusion-study'))
    args = parser.parse_args()
    import numpy as np
    import iree.runtime as rt
    rt.flags.parse_flags('--task_topology_group_count=8')
    out = args.output_dir.absolute()
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ['index-capture', 'index-input', 'pool-window']:
        source = TARGETS/(name+'.mlir')
        if name.startswith('index-'):
            text = subprocess.check_output([args.tools/'iree-opt',source,
                    '--linalg-fuse-elementwise-ops','--canonicalize','--cse'],text=True)
            (out/(name+'-fused.mlir')).write_text(text)
            remains = 'tensor<32xi64>' in text
            if remains != (name == 'index-capture'):
                raise AssertionError(f'{name}: unexpected index tensor after fusion')
            results[name] = {'materialized_index_tensor_after_fusion': remains}
        binary = out/(name+'.vmfb')
        command([args.tools/'iree-compile',source,'--iree-hal-target-backends=llvm-cpu',
                 '--iree-llvmcpu-target-cpu=host','--iree-opt-const-expr-hoisting=false',
                 '--iree-opt-level=O2','-o',binary],out/(name+'.log'))
        module = rt.load_vm_flatbuffer(binary.read_bytes(),driver='local-task')
        if name.startswith('index-'):
            a=np.arange(32,dtype=np.float32)-16
            np.testing.assert_array_equal(module.main(a).to_host(),a)
            print('PASS',name,results[name],flush=True)
        else:
            a=np.random.default_rng(0).uniform(-1,1,(512,19,19)).astype(np.float32)
            pad=np.pad(a,((0,0),(6,6),(6,6)),constant_values=-1e30)
            expected=np.lib.stride_tricks.sliding_window_view(pad,(13,13),axis=(1,2)).max(axis=(-1,-2))
            config=rt.Config('local-task')
            vm=rt.VmModule.copy_buffer(config.vm_instance,binary.read_bytes())
            module=rt.load_vm_module(vm,config)
            resident=rt.asdevicearray(config.device,a)
            for _ in range(2):module.main(resident).to_host()
            times=[]
            for _ in range(7):
                start=time.perf_counter();actual=module.main(resident).to_host()
                times.append(time.perf_counter()-start)
                np.testing.assert_array_equal(actual,expected)
            results[name]=summary(times)
            print('PASS',name,results[name],flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    main()
