import argparse
import numpy as np
import os
import random
import tensorboardX
import time
import torch
import config
import pickle as pkl
from trainer import TuneTrainer, NVPTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--texture_dim', type=int, default=config.TEXTURE_DIM)
parser.add_argument('--it', type=int, default=config.ITERATIONS)
parser.add_argument('--batch', type=int, default=config.BATCH_SIZE)
parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
parser.add_argument('-tl', '--train_len', type=int, default=config.TRAIN_LEN)
args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 一共30个identity，对于每一个identity，选取一个视频做训练集，再选一个视频做测试(check一下数据集，训练视频时长在3-5s左右)

def get_data(lmdb_path):
    import lmdb
    import cv2
    import deep_3drecon
    import pickle as pkl
    from io import BytesIO
    from utils import gen_randomwalk_list

    def getvideo(file_path):
        cap = cv2.VideoCapture(file_path)
        frame_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            frame_list.append(frame)
        return frame_list

    face_reconstructor = deep_3drecon.Reconstructor()
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)
    align_db = env.open_db("align".encode())
    uv_db = env.open_db("uv".encode())
    bg_db = env.open_db("bg".encode())
    texture_db = env.open_db("texture".encode())
    coeff_db = env.open_db("coeff".encode())
    mouth_db = env.open_db("mouth".encode())

    txn = env.begin(write = False)

    align_bin = txn.get(str(0).encode(), db = align_db)
    uv_bin = txn.get(str(0).encode(), db = uv_db)
    bg_bin = txn.get(str(0).encode(), db = bg_db)
    texture_bin = txn.get(str(0).encode(), db = texture_db)
    mouth_bin = txn.get(str(0).encode(), db = mouth_db)

    rand_idx = random.randint(1, txn.stat(db=align_db)['entries'] - 1)
    alignf_bin = txn.get(str(rand_idx).encode(), db = align_db)
    coeff_f_bin = txn.get(str(rand_idx).encode(), db = coeff_db)

    with open('test_align.mp4', 'wb') as f:
        f.write(align_bin)

    with open('test_align_f.mp4', 'wb') as f:
        f.write(alignf_bin)
    
    with open('test_uv.mp4', 'wb') as f:
        f.write(uv_bin)

    with open('test_bg.mp4', 'wb') as f:
        f.write(bg_bin)

    with open('test_texture.mp4', 'wb') as f:
        f.write(texture_bin)

    with open('test_mouth.mp4', 'wb') as f:
        f.write(mouth_bin)

    align_frames = np.array(getvideo('test_align.mp4'))
    alignf_frames = np.array(getvideo('test_align_f.mp4'))
    uv_frames = np.array(getvideo('test_uv.mp4'))
    bg_frames = np.array(getvideo('test_bg.mp4'))
    texture_frames = np.array(getvideo('test_texture.mp4'))
    mouth_frames = np.array(getvideo('test_mouth.mp4'))
    coeff_f = pkl.load(BytesIO(coeff_f_bin))

    tex = texture_frames[0]
    face_reconstructor.recon_uv_from_coeff(coeff_f, out_path = "test_uv2.mp4", bg_path = "test_bg2.mp4")
    test_uv = np.array(getvideo('test_uv2.mp4'))
    test_bg = np.array(getvideo('test_bg2.mp4'))

    os.system("rm test_align.mp4")
    os.system("rm test_align_f.mp4")
    os.system("rm test_uv.mp4")
    os.system("rm test_uv2.mp4")
    os.system("rm test_bg.mp4")
    os.system("rm test_bg2.mp4")
    os.system("rm test_texture.mp4")
    os.system("rm test_mouth.mp4")

    test_data = {
        'train_uv': uv_frames,
        'train_align': align_frames,
        'train_bg': bg_frames,
        'train_mouth': mouth_frames,
        'test_uv': test_uv,
        'test_bg': test_bg,
        'test_align': alignf_frames,
        'tex': tex
    }

    return test_data

def tune(lmdb_path, name, model_path, log_path):
    global args
    args.name = name
    args.model_path = model_path
    args.log_path = log_path
    test_data = get_data(lmdb_path)
    main_trainer = TuneTrainer(args, test_data)
    # main_trainer = NVPTrainer(args, test_data)
    main_trainer.train()
    del main_trainer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    if not os.path.exists("notune_model"):
        os.makedirs("notune_model")

    if not os.path.exists("notune_log"):
        os.makedirs("notune_log")

    # cream dataset
    # folder_path = "/home/wuhz/dataset/CREMA-D"
    # id_list = ["1015", "1020", "1021", "1030", "1033", "1052", "1062", "1081", "1082", "1089"]
    # for idx, id_str in enumerate(id_list):
    #     lmdb_path = os.path.join(folder_path, id_str, "lmdb")
    #     name = "cream_{}".format(idx)
    #     # model_path = "./notune_model/{}".format(name)
    #     # log_path = "./notune_log/{}".format(name)
    #     model_path = "./tune_model/{}".format(name)
    #     log_path = "./tune_log/{}".format(name)
    #     tune(lmdb_path, name, model_path, log_path)

    # # grid dataset
    # folder_path = "/home/wuhz/dataset/grid"
    # id_list = ["s11", "s13", "s15", "s18", "s19", "s2", "s25", "s31", "s33", "s4"]
    # for idx, id_str in enumerate(id_list):
    #     lmdb_path = os.path.join(folder_path, id_str, "lmdb")
    #     name = "grid_{}".format(idx)
    #     # model_path = "./notune_model/{}".format(name)
    #     # log_path = "./notune_log/{}".format(name)
    #     model_path = "./tune_model/{}".format(name)
    #     log_path = "./tune_log/{}".format(name)
    #     tune(lmdb_path, name, model_path, log_path)

    # tcd dataset
    folder_path = "/home/wuhz/dataset/tcd-timit"
    # id_list = ["15", "18", "25", "28", "33", "41", "55", "56", "8", "9"]
    id_list = ["9"]
    for idx, id_str in enumerate(id_list):
        lmdb_path = os.path.join(folder_path, id_str, "lmdb")
        name = "tcd_step_{}".format(idx)
        # model_path = "./notune_model/{}".format(name)
        # log_path = "./notune_log/{}".format(name)
        model_path = "./tune_model/{}".format(name)
        log_path = "./tune_log/{}".format(name)
        tune(lmdb_path, name, model_path, log_path)