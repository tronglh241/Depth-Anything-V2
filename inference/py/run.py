import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from liberyacs import CfgNode
from onnx_inference_engine import ONNXInferenceEngine

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-c', default='inference/configs/config.yml')
    args = parser.parse_args()

    config = CfgNode.load(args.config_file)

    # Load image using OpenCV
    image = cv2.imread(config.image_path)
    if image is None:
        raise ValueError(f'Image not found at {config.image_path}')

    engine = ONNXInferenceEngine(config.model_path, device=config.device)
    times = []

    for _ in range(20):
        # Initialize inference engine
        start = time.time()
        # Run inference
        output_image = engine.run(image)
        stop = time.time()
        times.append(stop - start)

    # Prepare output path
    output_dir = Path(config.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir.joinpath(f'{config.out_name}.png')

    # Save output image
    output_image = 255 - (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255
    output_image = output_image.astype(np.uint8)
    cv2.imwrite(str(output_path), output_image)
    print(f'Saved output image at {output_path}')

    print(sum(times[1:]) / len(times[1:]))
