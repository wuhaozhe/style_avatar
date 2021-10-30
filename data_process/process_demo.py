import sys
sys.path.append("..")
import lmdb
import os
import multiprocessing
import time
import imageio
import face_alignment
import pickle as pkl
import numpy as np
import cv2
import soundfile as sf
import deep_3drecon
import librosa
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from utils import lm68_2_lm5_batch, mean_eye_distance, read_video, write_video, filter_coeff, filter_lm5, lm68_mouth_contour
from io import BytesIO
from multiprocessing import Process
from align_img import align_lm68, align

if __name__ == "__main__":
    
    lmdb_path = "../data/demo_2"
    # file_list = ["./ted_0.mp4", "./ted_1.mp4"]
    file_list = ["./ted_2.mp4"]
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)
    wid = 0

    lm68_db = env.open_db("lm68".encode())
    lm5_db = env.open_db("lm5".encode())
    align_db = env.open_db("align".encode())
    uv_db = env.open_db("uv".encode())
    bg_db = env.open_db("bg".encode())
    texture_db = env.open_db("texture".encode())
    coeff_db = env.open_db("coeff".encode())
    mouth_db = env.open_db("mouth".encode())

    os.environ["CUDA_VISIBLE_DEVICES"] = str(wid)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, network_size=4, device='cuda')

    face_reconstructor = deep_3drecon.Reconstructor()
    tmp_dir = "../data/tmp/{}".format(wid)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    idx = 0
    for file_name in file_list:
        vid = imageio.get_reader(file_name,  'ffmpeg')
        lm_list = []
        img_list = []
        for image in vid.iter_data():
            img_list.append(image)
            preds = fa.get_landmarks(image)
            lm_list.append(preds)

        img_list = np.array(img_list)
        lm_list = np.array(lm_list)[:, 0]
        lm5_list = lm68_2_lm5_batch(lm_list)
        lm5_list = filter_lm5(lm5_list)

        img_list = np.array(img_list[:, :, :, ::-1])
        coeff, align_img_list = face_reconstructor.recon_coeff(img_list, lm5_list, return_image = True)
        coeff = filter_coeff(coeff)
        # 没有滤波的系数，需要重建uv, bg, texture，用于tune的训练
        face_reconstructor.recon_uv_from_coeff(coeff, "../data/tmp/{}_uv.mp4".format(wid), tmp_dir, "../data/tmp/{}_bg.mp4".format(wid))
        face_reconstructor.recon_texture_from_coeff(coeff, align_img_list, "../data/tmp/{}_texture.mp4".format(wid), tmp_dir)
        # 滤波的系数，只存coeff，align后的视频就可以了，后续根据音频再生成稳定的结果
        h, w, _ = img_list[0].shape
        lm3D = face_reconstructor.lm3D
        lm68_align = align_lm68(lm5_list, lm_list, lm3D, w, h)
        lm68_align = lm68_align.astype(np.int32)
        lm68_mouth_contour(lm68_align, "../data/tmp/{}_mouth.mp4".format(wid), tmp_dir)

        write_video(align_img_list, "../data/tmp/{}_align.mp4".format(wid), tmp_dir)

        with open("../data/tmp/{}_coeff.pkl".format(wid), 'wb') as f:
            pkl.dump(coeff, f)

        with open('../data/tmp/{}_lm68.pkl'.format(wid), 'rb') as f:
            lm_bin = f.read()
        with open('../data/tmp/{}_lm5.pkl'.format(wid), 'rb') as f:
            lm5_bin = f.read()
        with open("../data/tmp/{}_align.mp4".format(wid), 'rb') as f:
            align_bin = f.read()
        with open("../data/tmp/{}_uv.mp4".format(wid), 'rb') as f:
            uv_bin = f.read()
        with open("../data/tmp/{}_bg.mp4".format(wid), 'rb') as f:
            bg_bin = f.read()
        with open("../data/tmp/{}_texture.mp4".format(wid), 'rb') as f:
            texture_bin = f.read()
        with open("../data/tmp/{}_coeff.pkl".format(wid), 'rb') as f:
            coeff_bin = f.read()
        with open("../data/tmp/{}_mouth.mp4".format(wid), 'rb') as f:
            mouth_bin = f.read()

        txn = env.begin(write = True)
        txn.put(str(idx).encode(), lm5_bin, db = lm5_db)
        txn.put(str(idx).encode(), lm_bin, db = lm68_db)
        txn.put(str(idx).encode(), align_bin, db = align_db)
        txn.put(str(idx).encode(), uv_bin, db = uv_db)
        txn.put(str(idx).encode(), bg_bin, db = bg_db)
        txn.put(str(idx).encode(), texture_bin, db = texture_db)
        txn.put(str(idx).encode(), coeff_bin, db = coeff_db)
        txn.put(str(idx).encode(), mouth_bin, db = mouth_db)
        txn.commit()
        idx += 1