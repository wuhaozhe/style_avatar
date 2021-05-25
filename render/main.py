import argparse
import numpy as np
import os
import random
import tensorboardX
import time
import torch
import config
from trainer import RenderTrainerAdv


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required = True)
parser.add_argument('--texturew', type=int, default=config.TEXTURE_W)
parser.add_argument('--textureh', type=int, default=config.TEXTURE_H)
parser.add_argument('--texture_dim', type=int, default=config.TEXTURE_DIM)
parser.add_argument('--it', type=int, default=config.ITERATIONS)
parser.add_argument('--batch', type=int, default=config.BATCH_SIZE)
parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
parser.add_argument('-tl', '--train_len', type=int, default=config.TRAIN_LEN)
args = parser.parse_args()

args.model_path = "./model/{}".format(args.name)
args.log_path = "./log/{}".format(args.name)
args.test_path = "./test/{}".format(args.name)
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    main_trainer = RenderTrainerAdv(args)