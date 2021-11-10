import argparse
import numpy as np
import os
import random
import tensorboardX
import time
import torch
import config
import pickle as pkl
from trainer import RenderTrainerAdv, BlendTrainerAdv, TuneTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required = True)
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

def get_test_data():
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
    lmdb_path = "../data/ted_hd/lmdb"
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)
    align_data = env.open_db("test_align".encode())
    uv_data = env.open_db("test_uv".encode())
    bg_data = env.open_db("test_bg".encode())
    texture_data = env.open_db("test_texture".encode())
    mouth_data = env.open_db("test_mouth".encode())
    coeff_data = env.open_db("test_coeff".encode())
    txn = env.begin(write = False)

    align_bin = txn.get(str(0).encode(), db = align_data)
    uv_bin = txn.get(str(0).encode(), db = uv_data)
    bg_bin = txn.get(str(0).encode(), db = bg_data)
    texture_bin = txn.get(str(0).encode(), db = texture_data)
    mouth_bin = txn.get(str(0).encode(), db = mouth_data)
    coeff_bin = txn.get(str(0).encode(), db = coeff_data)

    with open('test_align.mp4', 'wb') as f:
        f.write(align_bin)

    with open('test_uv.mp4', 'wb') as f:
        f.write(uv_bin)

    with open('test_bg.mp4', 'wb') as f:
        f.write(bg_bin)

    with open('test_texture.mp4', 'wb') as f:
        f.write(texture_bin)

    with open('test_mouth.mp4', 'wb') as f:
        f.write(mouth_bin)

    align_frames = np.array(getvideo('test_align.mp4'))
    uv_frames = np.array(getvideo('test_uv.mp4'))
    bg_frames = np.array(getvideo('test_bg.mp4'))
    texture_frames = np.array(getvideo('test_texture.mp4'))
    mouth_frames = np.array(getvideo('test_mouth.mp4'))
    coeff = pkl.load(BytesIO(coeff_bin))

    tex = texture_frames[0]
    # 得先生成test时背景与前景blend的list(随机游走)，然后根据coeff然后才能确做光栅化
    train_len = 100
    train_align = align_frames[: train_len]
    train_uv = uv_frames[: train_len]
    train_bg = bg_frames[: train_len]
    train_mouth = mouth_frames[: train_len]
    train_coeff = coeff[: train_len]

    test_index = gen_randomwalk_list(list(range(train_len)), len(align_frames) - train_len)
    test_coeff = coeff[test_index]
    test_coeff[:, 80:144] = coeff[train_len:, 80:144]
    test_align = train_align[test_index]
    face_reconstructor.recon_uv_from_coeff(test_coeff, out_path = "test_uv2.mp4", bg_path = "test_bg2.mp4")
    test_uv = np.array(getvideo('test_uv2.mp4'))
    test_bg = np.array(getvideo('test_bg2.mp4'))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("./train_align.avi", fourcc, 25.0, (224, 224))
    for i in range(len(train_align)):
        out.write(train_align[i])
    out.release()

    os.system("rm test_align.mp4")
    os.system("rm test_uv.mp4")
    os.system("rm test_uv2.mp4")
    os.system("rm test_bg.mp4")
    os.system("rm test_bg2.mp4")
    os.system("rm test_texture.mp4")
    os.system("rm test_mouth.mp4")

    test_data = {
        'train_uv': train_uv,
        'train_align': train_align,
        'train_bg': train_bg,
        'train_mouth': train_mouth,
        'test_uv': test_uv,
        'test_bg': test_bg,
        'test_align': test_align,
        'tex': tex
    }

    pkl.dump(test_data, open('test.pkl', 'wb'))


if __name__ == "__main__":
    # 从ted test里面拿数据做测试
    # get_test_data()
    main_trainer = RenderTrainerAdv(args)
    # main_trainer = BlendTrainerAdv(args)
    # test_data = pkl.load(open('test.pkl', 'rb'))
    # main_trainer = TuneTrainer(args, test_data)
    # main_trainer.train()
    main_trainer.test()