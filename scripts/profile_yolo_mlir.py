"""Warmed YOLO latency and optional CPU kernel attribution; no compiler changes.

Use the isolated validation Python (IREE, NumPy, OpenCV 4). Run each thread count
in a fresh process. Kernel profiling runs separately from latency measurements.
"""
import argparse
import collections
import json
from pathlib import Path
import re
import statistics
import subprocess
import time
from run_yolo_mlir import command, entry_shapes, inputs
from check_yolo_opencv import decode, yolo_config


def summary(samples):
    return {'median_ms': statistics.median(samples)*1000,
            'min_ms': min(samples)*1000, 'max_ms': max(samples)*1000,
            'samples_ms': [x*1000 for x in samples]}


def kernel_profile(args, values):
    import numpy as np
    out = args.output_dir
    input_dir = out/'inputs'
    input_dir.mkdir(exist_ok=True)
    flags = []
    for i, value in enumerate(values):
        path = input_dir/f'{i}.npy'
        np.save(path, value)
        flags.append('--input=@'+str(path))
    bundle = out/'kernels.ireeprof'
    command([args.tools/'iree-benchmark-module', '--module='+str(args.module),
             '--device=local-task', '--task_topology_group_count='+str(args.threads),
             '--function=main', '--benchmark_min_time=3s',
             '--device_profiling_mode=host-execution,executable-metadata',
             '--device_profiling_output='+str(bundle)]+flags, out/'kernel-benchmark.log')
    raw = subprocess.check_output([args.tools/'iree-profile', 'dispatch',
                                   '--format=jsonl', bundle], text=True)
    (out/'kernels.jsonl').write_text(raw)
    rows = [json.loads(line) for line in raw.splitlines()]
    rows = [row for row in rows if row['type'] == 'host_dispatch_group']
    categories = collections.Counter()
    for row in rows:
        key = row['key']
        dispatch = re.search(r'dispatch_(\d+)_', key)[1]
        source = args.dispatch_sources/f'module_main_dispatch_{dispatch}.mlir' if args.dispatch_sources else None
        text = source.read_text() if source and source.exists() else ''
        if 'matmul' in key:
            category = 'matmul (including fused epilogues)'
        elif key.endswith('_i64'):
            category = 'integer index arrays'
        elif 'tensor.extract ' in text:
            category = 'gathers / window materialization'
        elif 'reduction' in key:
            category = 'reductions'
        elif 'memcpy' in key:
            category = 'copies'
        elif 'transpose' in key:
            category = 'transposes'
        else:
            category = 'other maps / broadcasts / padding'
        row['category'] = category
        categories[category] += row['total_tile_duration_sum_ns']
    total = sum(categories.values())
    return {'metric': 'sum of CPU worker-tile execution time; NOT wall-clock shares',
            'categories_percent': {k: v/total*100 for k, v in categories.most_common()},
            'dispatch_groups': sorted(rows, key=lambda r: r['total_tile_duration_sum_ns'], reverse=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', required=True, type=Path)
    parser.add_argument('--mlir', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=7)
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/remora-yolo-profile'))
    parser.add_argument('--profile-kernels', action='store_true')
    parser.add_argument('--tools', type=Path, help='matching IREE runtime tool directory')
    parser.add_argument('--dispatch-sources', type=Path, help='IREE executable source dumps, for finer attribution')
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1 or args.threads < 1:
        parser.error('iterations, warmup and threads must be positive')
    if args.profile_kernels and not args.tools:
        parser.error('--profile-kernels requires --tools')
    args.output_dir = args.output_dir.expanduser().absolute()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.module = args.module.absolute()
    import cv2
    import numpy as np
    import iree.runtime as rt
    cv2.setNumThreads(args.threads)
    rt.flags.parse_flags('--task_topology_group_count='+str(args.threads))
    values = inputs(entry_shapes(args.mlir.read_text()), args.image, args.weights)
    config = rt.Config('local-task')
    vm = rt.VmModule.copy_buffer(config.vm_instance, args.module.read_bytes())
    module = rt.load_vm_module(vm, config)
    start = time.perf_counter()
    resident = [rt.asdevicearray(config.device, x) for x in values]
    upload_ms = (time.perf_counter()-start)*1000
    net = cv2.dnn.readNetFromDarknet(str(args.config), str(args.weights))
    net.setInput(values[0][None])
    names = list(net.getLayerNames())
    heads = [next(n for n in reversed(names[:names.index(o)]) if n.startswith('conv_'))
             for o in net.getUnconnectedOutLayersNames()]
    configs = yolo_config(args.config)

    def iree(host_args=False):
        start = time.perf_counter()
        result = module.main(*(values if host_args else resident)).to_host()
        return result, time.perf_counter()-start

    def opencv():
        start = time.perf_counter()
        raw = net.forward(heads)
        mid = time.perf_counter()
        result = np.concatenate([decode(r, c, 608) for r, c in zip(raw, configs)])
        end = time.perf_counter()
        return result, mid-start, end-mid, end-start

    for _ in range(args.warmup):
        iree()
        opencv()
    samples = {k: [] for k in ('iree_resident', 'iree_host_args', 'opencv_forward', 'opencv_decode', 'opencv_total')}
    layer_times = []
    # Alternate order to reduce consistent thermal/cache ordering bias.
    for i in range(args.iterations):
        if i % 2:
            actual, elapsed = iree()
            expected, forward, decoding, total = opencv()
        else:
            expected, forward, decoding, total = opencv()
            actual, elapsed = iree()
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-4)
        for key, value in [('iree_resident', elapsed), ('opencv_forward', forward),
                           ('opencv_decode', decoding), ('opencv_total', total)]:
            samples[key].append(value)
        layer_times.append(net.getPerfProfile()[1]/cv2.getTickFrequency())
    for _ in range(args.iterations):
        samples['iree_host_args'].append(iree(True)[1])
    medians = np.median(np.array(layer_times), axis=0)
    ids, _, output_shapes = net.getLayersShapes([1,3,608,608])
    shapes = {int(i): list(map(int, s[0])) for i, s in zip(ids, output_shapes)}
    layers = []
    for i, (name, elapsed) in enumerate(zip(names, medians)):
        layer = net.getLayer(i+1)
        row = {'name': name, 'kind': layer.type, 'median_ms': float(elapsed*1000)}
        if layer.type == 'Convolution':
            weight_shape = list(layer.blobs[0].shape)
            output_shape = shapes[i+1]
            row['weight_shape'] = weight_shape
            row['output_shape'] = output_shape
            row['matmul_mnk'] = [weight_shape[0], output_shape[-2]*output_shape[-1],
                                 int(np.prod(weight_shape[1:]))]
        layers.append(row)
    result = {'threads_per_engine': args.threads, 'warmup': args.warmup,
              'iterations': args.iterations, 'initial_upload_ms': upload_ms,
              'latency': {k: summary(v) for k, v in samples.items()},
              'opencv_layers': sorted(layers, key=lambda r: r['median_ms'], reverse=True)}
    # Save usable baseline results even if optional profiling is unavailable.
    target = args.output_dir/'profile.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    if args.profile_kernels:
        result['iree_kernels'] = kernel_profile(args, values)
        target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('opencv_layers', 'iree_kernels')}, indent=2))
    print('Full layer/kernel results:', target)


if __name__ == '__main__':
    main()
