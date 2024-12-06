import argparse
from run import *
from utils.utils import *
def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/RSAFormer/RSAFormer.yaml')
    return parser.parse_args()

if __name__ == '__main__':
    args = _args()
    opts = load_config(args.config)
    train(opts)
    test(opts)
    evaluate(opts)