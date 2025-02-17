import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

sys.path.append(os.getcwd())

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='vits')
    parser.add_argument('--checkpoint-path', default='')
    parser.add_argument('--min-size', default=400, type=int)
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--height', default=400, type=int)
    parser.add_argument('--out-dir', default='output')
    parser.add_argument('--out-name', default='vits')
    args = parser.parse_args()

    # Model
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    max_depth = 250.0
    checkpoint_path = args.checkpoint_path

    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': max_depth})
    model.load_state_dict(
        torch.load(
            str(checkpoint_path),
            map_location='cpu',
        )
    )
    model = model.to(DEVICE).eval()

    # Input
    image_shape = (args.width, args.height)
    image = np.random.randint(0, 255, (image_shape[1], image_shape[0], 3), dtype=np.uint8)
    preprocessed_image, preprocessed_image_shape = model.image2tensor(image, input_size=args.min_size)

    # Convert
    onnx_program = torch.onnx.dynamo_export(model, preprocessed_image)
    onnx_program.optimize()

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir.joinpath(f'{args.out_name}_{preprocessed_image.shape[-1]}_{preprocessed_image.shape[-2]}.onnx')

    onnx_program.save(str(out_file))

    # Check
    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)
