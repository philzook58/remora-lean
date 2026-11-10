"""Isolate real-size convolution and pooling pieces from ordinary YOLO Remora.

Both engines receive identical synthetic data. OpenCV gets a minimal Darknet
network; IREE compiles the actual Remora helpers. This excludes BN/activation.
Run with the same isolated environment as profile_yolo_mlir.py.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time
from run_yolo_mlir import ROOT, SOURCE, command
from profile_yolo_mlir import summary

# name, input (C,H,W), output channels, kernel, stride, padding, Remora call
CASES = [
    ('conv_stride', (32,608,608), 64, 3, 2, 1, '@conv-2d/rem _ (303 3 32 64 1 2 1)'),
    ('conv_same', (256,38,38), 512, 3, 1, 1, '@conv-2d/rem _ (37 3 256 512 1 1 0)'),
    ('conv_1x1', (128,304,304), 64, 1, 1, 0, '@conv-2d/rem _ (303 1 128 64 0 1 0)'),
    ('pool13', (512,19,19), None, 13, 1, 6, '@max-pool-2d _ (6 13 512 6 1)'),
    ('matrix_only', (2304,38,38), 512, 1, 1, 0, '@matmul _ (512 2303 1444)'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/remora-yolo-pieces'))
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=7)
    parser.add_argument('--compiler', default=shutil.which('iree-compile'))
    args = parser.parse_args()
    if not args.compiler or args.threads < 1 or args.iterations < 1:
        parser.error('a compiler and positive thread/iteration counts are required')
    import cv2
    import numpy as np
    import iree.runtime as rt
    cv2.setNumThreads(args.threads)
    rt.flags.parse_flags('--task_topology_group_count='+str(args.threads))
    out = args.output_dir.absolute()
    out.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text().split('(entry', 1)[0]
    for name, (c,h,w), oc, k, _, _, call in CASES:
        if name == 'matrix_only':
            source += f'\n(entry ({name} (im [Float {c} {h*w}]) (w [Float {oc} {c}])) ({call} w im))\n'
        else:
            weights = f'(w [Float {oc} {c} {k} {k}])' if oc else ''
            source += f'\n(entry ({name} (im [Float {c} {h} {w}]) {weights}) ({call} im'+(' w' if oc else '')+'))\n'
    program, ir, binary = out/'pieces.remora', out/'pieces.mlir', out/'pieces.vmfb'
    program.write_text(source)
    with ir.open('w') as f:
        subprocess.run([ROOT/'.lake/build/bin/remora','mlir','-f',program], stdout=f, check=True, timeout=120)
    command([args.compiler,ir,'--iree-hal-target-backends=llvm-cpu',
             '--iree-llvmcpu-target-cpu=host','--iree-opt-const-expr-hoisting=false',
             '-o',binary],out/'iree.log')
    config = rt.Config('local-task')
    vm = rt.VmModule.copy_buffer(config.vm_instance,binary.read_bytes())
    module = rt.load_vm_module(vm,config)
    rng = np.random.default_rng(0)
    results = []
    for name, shape, oc, k, stride, pad, _ in CASES:
        c,h,w = shape
        image = rng.uniform(-0.5,0.5,shape).astype(np.float32)
        values = [image]
        cfg = f'[net]\nbatch=1\nwidth={w}\nheight={h}\nchannels={c}\n'
        if oc:
            weight = rng.uniform(-0.01,0.01,(oc,c,k,k)).astype(np.float32)
            values.append(weight)
            cfg += f'[convolutional]\nfilters={oc}\nsize={k}\nstride={stride}\npad={int(pad>0)}\nactivation=linear\n'
            checkpoint = struct.pack('<iiiQ',0,2,0,0)+np.zeros(oc,np.float32).tobytes()+weight.tobytes()
            net = cv2.dnn.readNetFromDarknet(np.frombuffer(cfg.encode(),np.uint8),np.frombuffer(checkpoint,np.uint8))
        else:
            cfg += f'[maxpool]\nsize={k}\nstride={stride}\npadding={2*pad}\n'
            net = cv2.dnn.readNetFromDarknet(np.frombuffer(cfg.encode(),np.uint8))
        net.setInput(image[None])
        if name == 'matrix_only':
            values = [image.reshape(c,h*w),weight.reshape(oc,c)]
        resident = [rt.asdevicearray(config.device,x) for x in values]
        function = module[name]
        for _ in range(2):
            function(*resident).to_host()
            net.forward()
        timings = {'iree':[], 'opencv':[]}
        for i in range(args.iterations):
            def a():
                start=time.perf_counter();result=function(*resident).to_host()
                timings['iree'].append(time.perf_counter()-start)
                return result
            def b():
                start=time.perf_counter();result=net.forward()
                timings['opencv'].append(time.perf_counter()-start)
                return result
            if i%2: actual=a();expected=b()
            else: expected=b();actual=a()
            np.testing.assert_allclose(actual.reshape(expected.shape),expected,rtol=2e-3,atol=1e-4)
        if oc and k == 3 and stride == 1:
            # Isolate algorithm choice from the im2col/gather overhead. OpenCV
            # exposes this switch; no assumption about its selected kernel.
            # A fresh network ensures the flag applies before weight packing.
            plain_net = cv2.dnn.readNetFromDarknet(np.frombuffer(cfg.encode(),np.uint8),
                                                  np.frombuffer(checkpoint,np.uint8))
            plain_net.enableWinograd(False)
            plain_net.setInput(image[None])
            for _ in range(2):
                plain_net.forward()
            timings['opencv_winograd_off'] = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                plain = plain_net.forward()
                timings['opencv_winograd_off'].append(time.perf_counter()-start)
            np.testing.assert_allclose(actual.reshape(plain.shape),plain,rtol=2e-3,atol=1e-4)
        row={'name':name,'input_shape':shape,'output_channels':oc,'kernel':k,
             'stride':stride,'timings':{k:summary(v) for k,v in timings.items()}}
        results.append(row)
        print(name,{k:round(summary(v)['median_ms'],3) for k,v in timings.items()},flush=True)
    (out/'pieces.json').write_text(json.dumps({'threads_per_engine':args.threads,
                       'iterations':args.iterations,'cases':results},indent=2)+'\n')


if __name__ == '__main__':
    main()
