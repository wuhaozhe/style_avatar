import sys
sys.path.append("..")
import torch
import numpy as np
import random
import time
import os
import cv2
import lmdb
random.seed(time.time())
from torch.utils.data import Dataset


class LRWTrainDataset(Dataset):
    def __init__(self, length = 8, require_appa = False):
        # the worker id and rand flag are used when read video from lmdb
        self.worker_id = None
        self.rand_flag = random.randint(0, 4096)

        self.length = length
        self.require_appa = require_appa
        lmdb_path = "../data/lrw/lmdb"
        self.env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)
        self.align_data = self.env.open_db("align".encode())
        self.uv_data = self.env.open_db("uv".encode())
        self.bg_data = self.env.open_db("bg".encode())
        self.texture_data = self.env.open_db("texture".encode())
        self.txn = self.env.begin(write = False)
        self.data_size = self.txn.stat(db=self.align_data)['entries']

    def __len__(self):
        return self.data_size

    def __getvideo__(self, file_path):
        cap = cv2.VideoCapture(file_path)
        frame_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            frame_list.append(frame)
        return frame_list

    def __get_data_from_db(self, idx):
        align_bin = self.txn.get(str(idx).encode(), db = self.align_data)
        uv_bin = self.txn.get(str(idx).encode(), db = self.uv_data)
        bg_bin = self.txn.get(str(idx).encode(), db = self.bg_data)
        texture_bin = self.txn.get(str(idx).encode(), db = self.texture_data)

        align_path = "../data/tmp/{}_{}_align.mp4".format(self.worker_id, self.rand_flag)
        uv_path = "../data/tmp/{}_{}_uv.mp4".format(self.worker_id, self.rand_flag)
        bg_path = "../data/tmp/{}_{}_bg.mp4".format(self.worker_id, self.rand_flag)
        texture_path = "../data/tmp/{}_{}_texture.mp4".format(self.worker_id, self.rand_flag)

        with open(align_path, 'wb') as f:
            f.write(align_bin)

        with open(uv_path, 'wb') as f:
            f.write(uv_bin)

        with open(bg_path, 'wb') as f:
            f.write(bg_bin)

        with open(texture_path, 'wb') as f:
            f.write(texture_bin)

        align_frames = np.array(self.__getvideo__(align_path))
        uv_frames = np.array(self.__getvideo__(uv_path))
        bg_frames = np.array(self.__getvideo__(bg_path))
        texture_frames = np.array(self.__getvideo__(texture_path))
        return align_frames, uv_frames, bg_frames, texture_frames

    # getitem中 tex frame与apparel frames是前length个，uv, vid与bg是后面随机
    def __getitem__(self, idx):
        align_frames, uv_frames, bg_frames, texture_frames = self.__get_data_from_db(idx)

        texture_frames = texture_frames[0: self.length]
        appa_frames = align_frames[0]

        rand_start = random.randint(self.length, len(align_frames) - self.length)
        appa_bg = bg_frames[0]
        bg_frames = bg_frames[rand_start: rand_start + self.length]
        uv_frames = uv_frames[rand_start: rand_start + self.length]
        align_frames = align_frames[rand_start: rand_start + self.length]
        
        align_frames = torch.from_numpy(align_frames).float().permute(0, 3, 1, 2) / 128 - 1
        uv_frames = torch.from_numpy(uv_frames).float().permute(0, 3, 1, 2) / 255
        uv_frames = uv_frames[:, :2]
        bg_frames = torch.from_numpy(bg_frames > 127).float()[:, :, :, 0].unsqueeze(1)
        texture_frames = torch.from_numpy(texture_frames).float().permute(0, 3, 1, 2) / 128 - 1
        if self.require_appa:
            appa_bg = torch.from_numpy(appa_bg < 127).float()[:, :, 0].unsqueeze(0)
            appa_frames = torch.from_numpy(appa_frames).float().permute(2, 0, 1) / 128 - 1
            appa_frames = appa_frames * appa_bg
            return bg_frames, uv_frames, align_frames, texture_frames, appa_frames
        else:
            return bg_frames, uv_frames, align_frames, texture_frames


class TedTestDataset(Dataset):
    def __init__(self, length = 8):
        self.worker_id = None
        self.rand_flag = random.randint(0, 4096)
        self.length = length

        lmdb_path = "../data/ted_hd/lmdb"
        self.env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)
        self.align_data = self.env.open_db("test_align".encode())
        self.uv_data = self.env.open_db("test_uv".encode())
        self.bg_data = self.env.open_db("test_bg".encode())
        self.texture_data = self.env.open_db("test_texture".encode())
        self.txn = self.env.begin(write = False)
        self.data_size = self.txn.stat(db=self.align_data)['entries']
        
    def __len__(self):
        return self.data_size

    def __getvideo__(self, file_path):
        cap = cv2.VideoCapture(file_path)
        frame_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            frame_list.append(frame)
        return frame_list

    def __get_data_from_db(self, idx):
        align_bin = self.txn.get(str(idx).encode(), db = self.align_data)
        uv_bin = self.txn.get(str(idx).encode(), db = self.uv_data)
        bg_bin = self.txn.get(str(idx).encode(), db = self.bg_data)
        texture_bin = self.txn.get(str(idx).encode(), db = self.texture_data)

        align_path = "../data/tmp/{}_{}_align.mp4".format(self.worker_id, self.rand_flag)
        uv_path = "../data/tmp/{}_{}_uv.mp4".format(self.worker_id, self.rand_flag)
        bg_path = "../data/tmp/{}_{}_bg.mp4".format(self.worker_id, self.rand_flag)
        texture_path = "../data/tmp/{}_{}_texture.mp4".format(self.worker_id, self.rand_flag)

        with open(align_path, 'wb') as f:
            f.write(align_bin)

        with open(uv_path, 'wb') as f:
            f.write(uv_bin)

        with open(bg_path, 'wb') as f:
            f.write(bg_bin)

        with open(texture_path, 'wb') as f:
            f.write(texture_bin)

        align_frames = np.array(self.__getvideo__(align_path))
        uv_frames = np.array(self.__getvideo__(uv_path))
        bg_frames = np.array(self.__getvideo__(bg_path))
        texture_frames = np.array(self.__getvideo__(texture_path))
        return align_frames, uv_frames, bg_frames, texture_frames

    def __getitem__(self, idx):
        align_frames, uv_frames, bg_frames, texture_frames = self.__get_data_from_db(idx)

        uv_frames = torch.from_numpy(uv_frames).float().permute(0, 3, 1, 2) / 255
        uv_frames = uv_frames[:, :2]
        bg_frames = torch.from_numpy(bg_frames > 127).float()[:, :, :, 0].unsqueeze(1)
        texture_frames = texture_frames[0: self.length]
        texture_frames = torch.from_numpy(texture_frames).float().permute(0, 3, 1, 2) / 128 - 1

        return bg_frames, uv_frames, texture_frames
    

if __name__ == "__main__":
    train_data = LRWTrainDataset(length = 3, require_appa = True)
    bg_frames, uv_frames, vid_frames, tex_frames, appa_frames = train_data.__getitem__(0)
    print(bg_frames.shape, uv_frames.shape, vid_frames.shape, tex_frames.shape, appa_frames.shape)

    test_data = TedTestDataset(length = 3)
    bg_frames, uv_frames, tex_frames = test_data.__getitem__(0)
    print(bg_frames.shape, uv_frames.shape, tex_frames.shape)