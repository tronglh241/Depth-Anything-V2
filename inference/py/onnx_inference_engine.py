from typing import Any, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from inference_engine import InferenceEngine


class ONNXInferenceEngine(InferenceEngine):
    def __init__(self, model_path: str, size: Tuple[int, int] = (644, 406), device: str = 'cpu'):
        self.size = size
        self.device = device
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        super().__init__(model_path)

    def load_model(self, model_path: str):
        # Choose execution provider based on the device
        if self.device == 'cuda':
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Load ONNX model with selected providers
        self.session = ort.InferenceSession(model_path, providers=providers)
        print(providers, self.session.get_providers())

    def preprocess(self, input_data: npt.NDArray[Any]):
        image = cv2.resize(input_data, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        image = np.moveaxis(image, 2, 0)
        image = np.expand_dims(image, axis=0)
        return image

    def infer(self, preprocessed_data):
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: preprocessed_data})
        return outputs[0]

    def postprocess(self, raw_output):
        return raw_output[0]
