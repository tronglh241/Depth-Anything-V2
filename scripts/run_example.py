import os
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

sys.path.append(os.getcwd())

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'  # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    out_dir = Path('output')
    out_dir.mkdir(parents=True, exist_ok=True)

    for file in tqdm(list(Path('assets/examples').glob('*.jpg'))):
        raw_img = cv2.imread(str(file))
        depth = model.infer_image(raw_img)  # HxW raw depth map in numpy

        cv2.imwrite(str(out_dir.joinpath(f'{file.stem}.png')), depth)
