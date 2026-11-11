"""Compare all decoded YOLO rows to OpenCV's independent Darknet implementation.

Requires numpy and opencv-python-headless<5 in a separate validation environment.
Read raw pre-YOLO heads because OpenCV's Region layer thresholds class scores.
"""
import argparse
import json
from pathlib import Path
import time


def config_sections(path):
    sections = []
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('['):
            sections.append((line.strip('[]'), {}))
        elif '=' in line:
            key, value = line.split('=', 1)
            sections[-1][1][key.strip()] = value.strip()
    return sections


def yolo_config(path):
    return [fields for kind, fields in config_sections(path) if kind == 'yolo']


def decode(raw, config, side):
    import numpy as np
    s = raw.shape[-1]
    classes = int(config['classes'])
    masks = [int(x) for x in config['mask'].split(',')]
    anchors = np.array([float(x) for x in config['anchors'].split(',')], dtype=np.float32).reshape(-1, 2)[masks]
    blocks = raw.reshape(len(masks), classes+5, s, s).transpose(0, 2, 3, 1)
    result = 1/(1+np.exp(-blocks))
    scale = float(config.get('scale_x_y', '1'))
    result[..., 0] = (np.arange(s)[None, None, :]+result[..., 0]*scale-(scale-1)/2)/s
    result[..., 1] = (np.arange(s)[None, :, None]+result[..., 1]*scale-(scale-1)/2)/s
    result[..., 2:4] = np.exp(blocks[..., 2:4])*anchors[:, None, None, :]/side
    return result.reshape(-1, classes+5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--result', required=True, type=Path, help='runner detections.npy')
    args = parser.parse_args()
    import cv2
    import numpy as np
    net = cv2.dnn.readNetFromDarknet(str(args.config), str(args.weights))
    names = list(net.getLayerNames())
    heads = []
    for output in net.getUnconnectedOutLayersNames():
        # The standard Darknet importer inserts a permutation before each Region.
        preceding = names[:names.index(output)]
        heads.append(next(name for name in reversed(preceding) if name.startswith('conv_')))
    actual = np.load(args.result)
    side = 608
    net.setInput(np.fromfile(args.image, dtype='<f4').reshape(1, 3, side, side))
    start = time.monotonic()
    raw_heads = net.forward(heads)
    seconds = time.monotonic()-start
    configs = yolo_config(args.config)
    if len(raw_heads) != len(configs):
        raise ValueError('Expected one configuration per detection head')
    references = [decode(raw, config, side) for raw, config in zip(raw_heads, configs)]
    reference = np.concatenate(references)
    np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=1e-4)
    print(json.dumps({'opencv_version': cv2.__version__, 'reference_seconds': seconds,
                      'shape': list(actual.shape),
                      'max_absolute_error': float(np.max(np.abs(actual-reference))),
                      'mean_absolute_error': float(np.mean(np.abs(actual-reference)))}, indent=2))


if __name__ == '__main__':
    main()
