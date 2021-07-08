from easydict import EasyDict as edict
import argparse
import numpy as np
import torch

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_conf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("-it", "--iterations", help = "training iterations", default=200000, type=int)
    parser.add_argument("-b", "--batch_size", help="batch_size", default = 32, type=int)
    parser.add_argument("-lp", "--load_path", help="path of model", default=None, type=str)
    parser.add_argument("-lr", "--lr", help="learning rate", default=1e-3, type=float)
    parser.add_argument("-m", "--mode", help="pose training or exp training", default="exp", type=str)
    parser.add_argument("-d", "--drop", help="prob of dropout", default=0.5, type=float)
    conf = parser.parse_args()
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.model_path = "./model/{}".format(conf.name)
    conf.log_path = "./log/{}".format(conf.name)
    conf.test_path = "./test/{}".format(conf.name)
    conf.fx = 50
    conf.fy = 25
    conf.xwin = 80
    conf.ywin = 32

    return conf

if __name__ == "__main__":
    conf = get_conf()
    print(conf)