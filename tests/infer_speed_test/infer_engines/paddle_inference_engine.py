import os
import paddle.inference as paddle_infer

from tests.infer_speed_test.infer_engines import InferenceEngine


class PaddleInferEngine(InferenceEngine):
    def __init__(self, config_dict):
        model_dir = config_dict.get("model_dir", None)
        if not model_dir:
            raise IOError("Cannot found 'model_dir' in config file.")
        if os.path.exists(model_dir):
            model_file = os.path.join(model_dir, "Paddle", "inference.pdmodel")
            model_params = os.path.join(model_dir, "Paddle", "inference.pdiparams")
            paddle_config = paddle_infer.Config(model_file, model_params)
        else:
            raise FileNotFoundError(f"The model dir {model_dir} does not exist!")

        paddle_config.enable_memory_optim()
        paddle_config.switch_ir_optim(True)

        device = config_dict.get("device", "cpu")
        if device == "cpu":
            paddle_config.disable_gpu()
        elif device == "gpu":
            paddle_config.enable_use_gpu(256, 0)
        else:
            raise ValueError("Device {} not supported.".format(device))

        enable_mkldnn = config_dict.get("enable_mkldnn", False)
        if enable_mkldnn:
            paddle_config.enable_mkldnn()

        thread_num = config_dict.get("thread_num", None)
        if thread_num:
            paddle_config.set_cpu_math_library_num_threads(thread_num)

        enable_trt = config_dict.get("enable_trt", False)
        if enable_trt:
            paddle_config.enable_tensorrt_engine(
                precision_mode=paddle_infer.PrecisionType.Float32,
                max_batch_size=20,
                min_subgraph_size=3)

        self.predictor = paddle_infer.create_predictor(paddle_config)
        input_names = self.predictor.get_input_names()
        self.input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        self.output_handle_list = []
        for output_name in output_names:
            output_handle = self.predictor.get_output_handle(output_name)
            self.output_handle_list.append(output_handle)

    def preprocess(self, input_tensor):
        self.input_handle.copy_from_cpu(input_tensor)

    def inference(self):
        self.predictor.run()

    def postprocess(self):
        for output_handle in self.output_handle_list:
            output = output_handle.copy_to_cpu()
