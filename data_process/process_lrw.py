
import sys
sys.path.append("..")
import os
import lmdb
import multiprocessing
import time
import numpy as np
import pickle as pkl
import deep_3drecon
from multiprocessing import Process
from io import BytesIO
from align_img import align_lm68, align
from utils import read_video, write_video, lm68_mouth_contour
from tqdm import tqdm

'''
    Data that will be stored in the preprocess:
    align_data: the aligned video data
    lm5_data: 5 facial landmarks
    lm68_data: 68 facial landmarks
    uv_data: the uv map
    bg_data: the background data
    texture_data: the unwrapped rgb texture (need to be re calculated)
    coeff_data: the reconstructed coeff
    mouth_data: the mouth region mask (need to be re calculated)
'''

def gather_worker(wid, data_list, src_path):
    lmdb_path = "../data/lrw/lmdb"
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
    
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)
    align_data = env.open_db("align".encode())
    lm5_data = env.open_db("lm5".encode())
    lm68_data = env.open_db("lm68".encode())
    uv_data = env.open_db("uv".encode())
    bg_data = env.open_db("bg".encode())
    texture_data = env.open_db("texture".encode())
    coeff_data = env.open_db("coeff".encode())
    mouth_data = env.open_db("mouth".encode())

    tmp_dir = "../data/tmp/{}".format(wid)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    os.environ["CUDA_VISIBLE_DEVICES"]=str(wid % 4)
    face_reconstructor = deep_3drecon.Reconstructor()

    if wid == 0:
        data_list = tqdm(data_list)

    for data_name in data_list:
        with open(os.path.join(src_path, str(data_name), "clip.mp4"), 'rb') as f:
            align_bin = f.read()
        with open(os.path.join(src_path, str(data_name), "lm_5.pkl"), 'rb') as f:
            lm5_bin = f.read()
        with open(os.path.join(src_path, str(data_name), "lm_68.pkl"), 'rb') as f:
            lm68_bin = f.read()
        with open(os.path.join(src_path, str(data_name), "uv.mp4"), 'rb') as f:
            uv_bin = f.read()
        with open(os.path.join(src_path, str(data_name), "bg.mp4"), 'rb') as f:
            bg_bin = f.read()
        with open(os.path.join(src_path, str(data_name), "recon.npy"), 'rb') as f:
            coeff_bin = f.read()

        lm5_list = pkl.load(BytesIO(lm5_bin))
        lm68_list = pkl.load(BytesIO(lm68_bin))
        lm3D = face_reconstructor.lm3D
        lm68_align = align_lm68(lm5_list, lm68_list, lm3D, 256, 256)
        lm68_align = lm68_align.astype(np.int32)
        lm68_mouth_contour(lm68_align, "../data/tmp/{}_mouth.mp4".format(wid), tmp_dir)
        with open("../data/tmp/{}_mouth.mp4".format(wid), 'rb') as f:
            mouth_bin = f.read()

        coeff = np.load(BytesIO((coeff_bin)))
        align_frames = read_video(os.path.join(src_path, str(data_name), "clip.mp4"))
        face_reconstructor.recon_texture_from_coeff(coeff, align_frames, "../data/tmp/{}_texture.mp4".format(wid), tmp_dir)
        with open("../data/tmp/{}_texture.mp4".format(wid), 'rb') as f:
            texture_bin = f.read()

        txn = env.begin(write = True)
        txn.put(str(data_name).encode(), align_bin, db = align_data)
        txn.put(str(data_name).encode(), lm5_bin, db = lm5_data)
        txn.put(str(data_name).encode(), lm68_bin, db = lm68_data)
        txn.put(str(data_name).encode(), uv_bin, db = uv_data)
        txn.put(str(data_name).encode(), bg_bin, db = bg_data)
        txn.put(str(data_name).encode(), texture_bin, db = texture_data)
        txn.put(str(data_name).encode(), coeff_bin, db = coeff_data)
        txn.put(str(data_name).encode(), mouth_bin, db = mouth_data)
        txn.commit()

def gather_data(num_worker = 1):
    src_path = "/home/wuhz/dataset/lrw/gather"
    folder_list = os.listdir(src_path)
    folder_num = len(folder_list)
    data_chunk = np.array_split(np.arange(folder_num), num_worker)

    w_list = []
    for wid in range(num_worker):
        w_list.append(Process(target = gather_worker, args = (wid, data_chunk[wid], src_path)))

    for wid in range(num_worker):
        w_list[wid].start()
        time.sleep(10)

    for wid in range(num_worker):
        w_list[wid].join()

def test():
    lmdb_path = "../data/lrw/lmdb"
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)

    # align_data = env.open_db("align".encode())
    align_data = env.open_db("pred_data".encode())
    lm5_data = env.open_db("lm5".encode())
    lm68_data = env.open_db("lm68".encode())
    uv_data = env.open_db("uv".encode())
    bg_data = env.open_db("bg".encode())
    texture_data = env.open_db("texture".encode())
    coeff_data = env.open_db("coeff".encode())
    mouth_data = env.open_db("mouth".encode())

    with env.begin(write = False) as txn:
        print(txn.stat(db=align_data))
        video1 = txn.get(str(5).encode(), db=align_data)
        video2 = txn.get(str(0).encode(), db=texture_data)
        # video_file1 = open("test.mp4", "wb")
        video_file1 = open("test.avi", "wb")
        video_file2 = open("test2.mp4", "wb")
        video_file1.write(video1)
        video_file2.write(video2)
        video_file1.close()
        video_file2.close()

if __name__ == "__main__":
    # gather_data(num_worker = 4)
    test()