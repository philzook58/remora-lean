"""Benchmark the Darknet-config and Remora-generated YOLOv4 PyTorch paths."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import struct
import time

from check_yolo_opencv import config_sections, decode
from run_yolo_mlir import entry_shapes, inputs


def build(config, checkpoint):
    import numpy as np
    import torch
    from torch.nn import functional as F

    sections = config_sections(config)
    kind, net = sections[0]
    if kind != 'net' or net['width'] != net['height']:
        raise ValueError('Expected a square Darknet network')
    with checkpoint.open('rb') as stream:
        major, minor, _ = struct.unpack('<iii', stream.read(12))
    offset = 20 if major * 10 + minor >= 2 and major < 1000 and minor < 1000 else 16
    weights = np.fromfile(checkpoint, dtype='<f4', offset=offset)
    cursor = 0

    def take(shape):
        nonlocal cursor
        count = int(np.prod(shape))
        value = torch.from_numpy(weights[cursor:cursor + count].reshape(shape))
        cursor += count
        return value

    root = torch.nn.Module()
    graph = torch.fx.Graph()
    x = graph.placeholder('image')
    channels = int(net['channels'])
    outputs, heads, head_configs = [], [], []
    for i, (kind, opts) in enumerate(sections[1:]):
        if kind == 'convolutional':
            out = int(opts['filters'])
            kernel = int(opts['size'])
            stride = int(opts['stride'])
            padding = kernel // 2 if int(opts.get('pad', 0)) else int(opts.get('padding', 0))
            groups = int(opts.get('groups', 1))
            bias = take((out,))
            bn = int(opts.get('batch_normalize', 0))
            if bn:
                scale, mean, variance = (take((out,)) for _ in range(3))
            weight = take((out, channels // groups, kernel, kernel))
            if bn:
                # Fold inference BN using the same epsilon as the Remora demo.
                denom = (variance + 1e-5).sqrt()
                weight = weight * (scale / denom).reshape(-1, 1, 1, 1)
                bias = bias - scale * mean / denom
            conv = torch.nn.Conv2d(channels, out, kernel, stride, padding, groups=groups)
            conv.weight = torch.nn.Parameter(weight, requires_grad=False)
            conv.bias = torch.nn.Parameter(bias, requires_grad=False)
            name = f'conv_{i}'
            root.add_module(name, conv)
            x = graph.call_module(name, (x,))
            channels = out
        elif kind == 'route':
            refs = [int(ref) for ref in opts['layers'].split(',')]
            selected = [outputs[i + ref if ref < 0 else ref] for ref in refs]
            channels = sum(c for _, c in selected)
            x = selected[0][0] if len(selected) == 1 else graph.call_function(
                torch.cat, (tuple(value for value, _ in selected), 1))
            groups = int(opts.get('groups', 1))
            if groups != 1:
                channels //= groups
                x = graph.call_function(torch.narrow, (x, 1, int(opts['group_id']) * channels, channels))
        elif kind == 'shortcut':
            ref = int(opts['from'])
            other, other_channels = outputs[i + ref if ref < 0 else ref]
            if other_channels != channels:
                raise ValueError('Shortcut channel mismatch')
            x = graph.call_function(torch.add, (x, other))
        elif kind == 'maxpool':
            size = int(opts['size'])
            stride = int(opts['stride'])
            padding = int(opts.get('padding', size - 1))
            if padding % 2:
                raise ValueError('Expected symmetric max-pool padding')
            x = graph.call_function(F.max_pool2d, (x, size, stride, padding // 2))
        elif kind == 'upsample':
            x = graph.call_function(F.interpolate, (x,), {'scale_factor': int(opts['stride']), 'mode': 'nearest'})
        elif kind == 'yolo':
            heads.append(x)
            head_configs.append(opts)
        else:
            raise ValueError(f'Unsupported Darknet layer {kind}')

        activation = opts.get('activation', 'linear')
        if activation == 'mish':
            x = graph.call_function(F.mish, (x,))
        elif activation == 'leaky':
            x = graph.call_function(F.leaky_relu, (x, 0.1))
        elif activation != 'linear':
            raise ValueError(f'Unsupported activation {activation}')
        outputs.append((x, channels))

    if cursor != weights.size:
        raise ValueError(f'Consumed {cursor} weight values, expected {weights.size}')
    graph.output(tuple(heads))
    return torch.fx.GraphModule(root, graph).eval(), head_configs, int(net['width']), int(net['channels'])


def measure(run, reference, iterations):
    import numpy as np

    samples = []
    for i in range(iterations + 2):
        start = time.perf_counter()
        result = run()
        elapsed = (time.perf_counter() - start) * 1000
        np.testing.assert_allclose(result, reference, rtol=2e-3, atol=1e-4)
        if i >= 2:
            samples.append(elapsed)
    return {'median_ms': statistics.median(samples), 'samples_ms': samples,
            'max_absolute_error': float(np.max(np.abs(result - reference)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('weights', 'image', 'reference'):
        parser.add_argument('--' + name, required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--config', type=Path, help='Build the baseline from a Darknet configuration')
    mode.add_argument('--printed', type=Path, help='Run trusted source from emit_pytorch.py')
    parser.add_argument('--compile', action='store_true',
                        help='Compile the printed entry with TorchInductor')
    parser.add_argument('--mlir', type=Path, help='Original Remora MLIR for the printed signature')
    parser.add_argument('--device', default='cpu', help='PyTorch device, e.g. cuda:0 for ROCm')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=7)
    parser.add_argument('--emit-forward', type=Path, help='Write the generated Python forward method')
    args = parser.parse_args()
    if args.printed and (not args.mlir or args.emit_forward):
        parser.error('--printed requires --mlir and already supplies the Python source')
    if args.config and (args.mlir or args.compile):
        parser.error('--mlir and --compile are only used with --printed')
    if args.threads < 1 or args.iterations < 1:
        parser.error('threads and iterations must be positive')

    import numpy as np
    import torch

    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        parser.error('--device must be cpu or a PyTorch CUDA/ROCm device')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('PyTorch cannot access a CUDA/ROCm device')

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    reference = np.load(args.reference)
    results = {}
    with torch.inference_mode():
        if args.printed:
            spec = importlib.util.spec_from_file_location('printed_yolo', args.printed)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            shapes = entry_shapes(args.mlir.read_text())
            values = [torch.from_numpy(x).to(device) for x in inputs(shapes, args.image, args.weights)]
            entry = module.entries['main']
            if args.compile:
                entry = torch.compile(entry, fullgraph=True)
                start = time.perf_counter()
                result = entry(*values).cpu().numpy()
                first_call_ms = (time.perf_counter() - start) * 1000
                np.testing.assert_allclose(result, reference, rtol=2e-3, atol=1e-4)
                results['printed_inductor'] = {
                    'first_call_ms': first_call_ms,
                    **measure(lambda: entry(*values).cpu().numpy(), reference, args.iterations),
                }
            else:
                results['printed'] = measure(
                    lambda: entry(*values).cpu().numpy(), reference, args.iterations)
        else:
            model, heads, side, channels = build(args.config, args.weights)
            image = torch.from_numpy(
                np.fromfile(args.image, dtype='<f4').reshape(1, channels, side, side)).to(device)
            if args.emit_forward:
                args.emit_forward.write_text(model.code)
            for label, layout in [('nchw', torch.contiguous_format),
                                  ('channels_last', torch.channels_last)]:
                model.to(device=device, memory_format=layout)
                x = image.contiguous(memory_format=layout)

                def run():
                    return np.concatenate([decode(y.cpu().numpy(), config, side)
                                           for y, config in zip(model(x), heads, strict=True)])

                results[label] = measure(run, reference, args.iterations)
    name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    print(json.dumps({'torch': torch.__version__, 'hip': torch.version.hip,
                      'device': str(device), 'device_name': name, 'threads': args.threads,
                      'warmups': 2, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
