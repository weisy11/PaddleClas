from abc import abstractmethod, ABC
from tests.infer_speed_test.infer_engines.paddle_inference_engine import PaddleInferEngine


def get_infer_engine(config_dict):
    engine_type = config_dict.get("engine_type")


class InferenceEngine(ABC):
    @abstractmethod
    def __init__(self, config_dict):
        raise NotImplemented

    @abstractmethod
    def preprocess(self, input_tensor):
        raise NotImplemented

    @abstractmethod
    def inference(self):
        raise NotImplemented

    @abstractmethod
    def postprocess(self):
        raise NotImplemented
