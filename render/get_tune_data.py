# 在这个文件下测试self reenactment的各个bench mark
# 生成音视频，测试音视频同步的benchmark
# 尝试更换纹理

import argparse
import numpy as np
import os
import random
import tensorboardX
import time
import torch
import torch.nn as nn
import torchvision
import config
import model
import pickle as pkl
import cv2
import lmdb

def get_data(lmdb_path, name, model_path):
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)
    align_db = env.open_db("align".encode())
    txn = env.begin(write = False)
    align_bin = txn.get(str(0).encode(), db = align_db)
    with open('../data/wav2lip_train/{}.mp4'.format(name), 'wb') as f:
        f.write(align_bin)
    

if __name__ == "__main__":

    # ted_lmdb_path = "../data/ted_hd/lmdb"
    # ted_env = lmdb.open(ted_lmdb_path, map_size=1099511627776, max_dbs = 64)
    # audio_db = ted_env.open_db("test_audio".encode())
    # ted_txn = ted_env.begin(write = False)

    # for i in range(35):
    #     test_audio_bin = ted_txn.get(str(i).encode(), db = audio_db)
    #     test_audio_path = "../data/wav2lip_train/{}.wav".format(i)
    #     with open(test_audio_path, 'wb') as f:
    #         f.write(test_audio_bin)

    # # cream dataset
    # folder_path = "/home/wuhz/dataset/CREMA-D"
    # id_list = ["1015", "1020", "1021", "1030", "1033", "1052", "1062", "1081", "1082", "1089"]
    # for idx, id_str in enumerate(id_list):
    #     lmdb_path = os.path.join(folder_path, id_str, "lmdb")
    #     name = "cream_{}".format(idx)
    #     model_path = "./tune_model/{}".format(name)
    #     get_data(lmdb_path, name, model_path)

    # # grid dataset
    # folder_path = "/home/wuhz/dataset/grid"
    # id_list = ["s11", "s13", "s15", "s18", "s19", "s2", "s25", "s31", "s33", "s4"]
    # for idx, id_str in enumerate(id_list):
    #     lmdb_path = os.path.join(folder_path, id_str, "lmdb")
    #     name = "grid_{}".format(idx)
    #     model_path = "./tune_model/{}".format(name)
    #     get_data(lmdb_path, name, model_path)

    # # tcd dataset
    # folder_path = "/home/wuhz/dataset/tcd-timit"
    # id_list = ["15", "18", "25", "28", "33", "41", "55", "56", "8", "9"]
    # for idx, id_str in enumerate(id_list):
    #     lmdb_path = os.path.join(folder_path, id_str, "lmdb")
    #     name = "tcd_{}".format(idx)
    #     model_path = "./tune_model/{}".format(name)
    #     get_data(lmdb_path, name, model_path)

    data_path = "../data/wav2lip_train"
    file_list = os.listdir(data_path)
    for file_name in file_list:
        if file_name.endswith('mp4'):
            file_path = os.path.join(data_path, file_name)
            os.system("ffmpeg -i {} -vframes 1 {}.jpg".format(file_path, file_path[:-4]))