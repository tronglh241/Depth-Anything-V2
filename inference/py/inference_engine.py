from abc import ABC, abstractmethod


class InferenceEngine(ABC):
    def __init__(self, model_path: str):
        self.model = None
        self.model_path = model_path
        self.load_model(model_path)

    @abstractmethod
    def load_model(self, model_path: str):
        '''Load the model from the specified path'''
        pass

    @abstractmethod
    def preprocess(self, input_data):
        '''Preprocess input data before inference'''
        pass

    @abstractmethod
    def infer(self, preprocessed_data):
        '''Run inference on the preprocessed data'''
        pass

    @abstractmethod
    def postprocess(self, raw_output):
        '''Postprocess the raw model output'''
        pass

    def run(self, input_data):
        '''Run the full pipeline: preprocess, infer, postprocess'''
        preprocessed_data = self.preprocess(input_data)
        raw_output = self.infer(preprocessed_data)
        return self.postprocess(raw_output)
