import argparse
from deploy.utils.config import get_config
from tests.infer_speed_test.infer_engines import get_infer_engine
from tests.infer_speed_test.data_loader import get_dataloader


def infer_acc_test():
    pass


def normal_test(infer_engine, dataloader, test_iter_num, timer):
    infer_engine.mode_speed_test()
    for i in range(test_iter_num):
        input_data = dataloader.gen_data(i)
        infer_engine.preprocess(input_data)
        infer_engine.inference()
        infer_engine.postprocess()


def infer_speed_test(config_dict):
    warmup_iter_num = config_dict["Variables"]["warmup_iter_num"]
    test_iter_num = config_dict["Variables"]["test_iter_num"]
    timer = None
    infer_engine = get_infer_engine(config_dict)
    dataloader = get_dataloader(config_dict)
    infer_engine.mode_warm_up()
    for i in range(warmup_iter_num):
        input_data = dataloader.gen_data(i)
        infer_engine.preprocess(input_data)
        infer_engine.inference()
        infer_engine.postprocess()
    normal_test(infer_engine, dataloader, test_iter_num, timer)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    args = parser.parse_args()
    configs = get_config(args.config_path, args.override)
    return configs


if __name__ == '__main__':
    infer_speed_test(parse_args())
