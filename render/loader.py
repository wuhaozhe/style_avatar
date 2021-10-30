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

class LRWDataset(Dataset):
    def __init__(self, require_appa = False):
        # the worker id and rand flag are used when read video from lmdb
        self.worker_id = None
        self.rand_flag = random.randint(0, 4096)

        self.require_appa = require_appa
        lmdb_path = "../data/lrw/lmdb"
        self.env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)
        self.align_data = self.env.open_db("align".encode())
        self.uv_data = self.env.open_db("uv".encode())
        self.bg_data = self.env.open_db("bg".encode())
        self.texture_data = self.env.open_db("texture".encode())
        self.mouth_data = self.env.open_db("mouth".encode())
        self.pred_data = self.env.open_db("pred_data".encode())
        self.txn = self.env.begin(write = False)
        self.data_size = self.txn.stat(db=self.pred_data)['entries']

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

    def get_data_from_db(self, idx):
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
        raise Exception("method should be implemented in sub classes")

class LRWTrainDataset(LRWDataset):
    def __init__(self, length = 8, require_appa = False):
        # the worker id and rand flag are used when read video from lmdb
        LRWDataset.__init__(self, require_appa)
        self.length = length

    def __len__(self):
        return self.data_size

    # getitem中 tex frame与apparel frames是前length个，uv, vid与bg是后面随机
    def __getitem__(self, idx):
        align_frames, uv_frames, bg_frames, texture_frames = self.get_data_from_db(idx)

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
        texture_frames = torch.flip(texture_frames, dims = [2])
        if self.require_appa:
            appa_bg = torch.from_numpy(appa_bg < 127).float()[:, :, 0].unsqueeze(0)
            appa_frames = torch.from_numpy(appa_frames).float().permute(2, 0, 1) / 128 - 1
            appa_frames = appa_frames * appa_bg
            return bg_frames, uv_frames, align_frames, texture_frames, appa_frames
        else:
            return bg_frames, uv_frames, align_frames, texture_frames

class LRWTestDataset(LRWDataset):
    def __init__(self, length = 8, require_appa = False):
        LRWDataset.__init__(self, require_appa)
        self.length = length

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        align_frames, uv_frames, bg_frames, texture_frames = self.get_data_from_db(idx)

        texture_frames = texture_frames[0: self.length]
        appa_frames = align_frames[0]
        appa_bg = bg_frames[0]

        align_frames = torch.from_numpy(align_frames).float().permute(0, 3, 1, 2) / 128 - 1
        uv_frames = torch.from_numpy(uv_frames).float().permute(0, 3, 1, 2) / 255
        uv_frames = uv_frames[:, :2]
        bg_frames = torch.from_numpy(bg_frames > 127).float()[:, :, :, 0].unsqueeze(1)
        texture_frames = torch.from_numpy(texture_frames).float().permute(0, 3, 1, 2) / 128 - 1
        texture_frames = torch.flip(texture_frames, dims = [2])
        if self.require_appa:
            appa_bg = torch.from_numpy(appa_bg < 127).float()[:, :, 0].unsqueeze(0)
            appa_frames = torch.from_numpy(appa_frames).float().permute(2, 0, 1) / 128 - 1
            appa_frames = appa_frames * appa_bg
            return bg_frames, uv_frames, align_frames, texture_frames, appa_frames
        else:
            return bg_frames, uv_frames, align_frames, texture_frames


class LRWBlendDataset(LRWDataset):
    def __init__(self):
        LRWDataset.__init__(self, False)

    def __len__(self):
        return self.data_size

    def get_data_from_db(self, idx):
        align_bin = self.txn.get(str(idx).encode(), db = self.align_data)
        bg_bin = self.txn.get(str(idx).encode(), db = self.bg_data)
        mouth_bin = self.txn.get(str(idx).encode(), db = self.mouth_data)
        pred_bin = self.txn.get(str(idx).encode(), db = self.pred_data)

        align_path = "../data/tmp/{}_{}_align.mp4".format(self.worker_id, self.rand_flag)
        bg_path = "../data/tmp/{}_{}_bg.mp4".format(self.worker_id, self.rand_flag)
        mouth_path = "../data/tmp/{}_{}_mouth.mp4".format(self.worker_id, self.rand_flag)
        pred_path = "../data/tmp/{}_{}_pred.avi".format(self.worker_id, self.rand_flag)

        with open(align_path, 'wb') as f:
            f.write(align_bin)

        with open(bg_path, 'wb') as f:
            f.write(bg_bin)

        with open(mouth_path, 'wb') as f:
            f.write(mouth_bin)

        with open(pred_path, 'wb') as f:
            f.write(pred_bin)

        align_frames = np.array(self.__getvideo__(align_path))
        bg_frames = np.array(self.__getvideo__(bg_path))
        mouth_frames = np.array(self.__getvideo__(mouth_path))
        pred_frames = np.array(self.__getvideo__(pred_path))
        return align_frames, bg_frames, mouth_frames, pred_frames

    # 随机选取两帧
    def __getitem__(self, idx):
        align_frames, bg_frames, mouth_frames, pred_frames = self.get_data_from_db(idx)
        rand_idx1 = random.randint(0, len(align_frames) - 2)
        rand_idx2 = random.randint(rand_idx1 + 1, len(align_frames) - 1)
        im1, bg1, mouth1, pred1 = align_frames[rand_idx1], bg_frames[rand_idx1], mouth_frames[rand_idx1], pred_frames[rand_idx1]
        im2, bg2, mouth2, pred2 = align_frames[rand_idx2], bg_frames[rand_idx2], mouth_frames[rand_idx2], pred_frames[rand_idx2]

        im1 = torch.from_numpy(im1).float().permute(2, 0, 1) / 128 - 1
        im2 = torch.from_numpy(im2).float().permute(2, 0, 1) / 128 - 1
        bg1 = torch.from_numpy(bg1 > 127).float()[:, :, 0].unsqueeze(0)
        bg2 = torch.from_numpy(bg2 > 127).float()[:, :, 0].unsqueeze(0)
        mouth1 = torch.from_numpy(mouth1 > 127).float()[:, :, 0].unsqueeze(0)
        mouth2 = torch.from_numpy(mouth2 > 127).float()[:, :, 0].unsqueeze(0)
        pred1 = torch.from_numpy(pred1).float().permute(2, 0, 1) / 128 - 1
        pred2 = torch.from_numpy(pred2).float().permute(2, 0, 1) / 128 - 1
        return im1, im2, bg1, bg2, mouth1, mouth2, pred1, pred2


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
        texture_frames = torch.flip(texture_frames, dims = [2])

        return bg_frames, uv_frames, texture_frames
    
class TedDemoDataset(Dataset):
    def __init__(self):
        self.worker_id = None
        self.rand_flag = random.randint(0, 4096)

        lmdb_path = "../data/ted_hd/lmdb"
        self.env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)
        self.align_data = self.env.open_db("test_align".encode())
        self.uv_data = self.env.open_db("test_uv".encode())
        self.bg_data = self.env.open_db("test_bg".encode())
        self.texture_data = self.env.open_db("test_texture".encode())
        self.txn = self.env.begin(write = False)

    def __len__(self):
        return 1

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
        align_frames, _, bg_frames, _ = self.__get_data_from_db(idx)
        pred_frames = np.array(self.__getvideo__("/home/wuhz/mnt/avatar/style_avatar/render/test_render_lrw_lr_2e-4_tl_1.mp4"))

        bg_frames = torch.from_numpy(bg_frames > 127).float()[:, :, :, 0].unsqueeze(1)
        align_frames = torch.from_numpy(align_frames).float().permute(0, 3, 1, 2) / 128 - 1
        pred_frames = torch.from_numpy(pred_frames).float().permute(0, 3, 1, 2) / 128 - 1

        return bg_frames, align_frames, pred_frames

# 这一段的输入有uv, bg, gt, mouth (没有tex)
# 直接传入frames
class TuneTrainset(Dataset):
    def __init__(self, uv_frames, bg_frames, align_frames, mouth_frames):
        self.uv_frames = uv_frames
        self.bg_frames = bg_frames
        self.align_frames = align_frames
        self.mouth_frames = mouth_frames
        self.bg_frames = torch.from_numpy(self.bg_frames > 127).float()[:, :, :, 0].unsqueeze(1)
        self.uv_frames = torch.from_numpy(self.uv_frames).float().permute(0, 3, 1, 2) / 255
        self.uv_frames = self.uv_frames[:, :2]
        self.align_frames = torch.from_numpy(self.align_frames).float().permute(0, 3, 1, 2) / 128 - 1
        self.mouth_frames = torch.from_numpy(self.mouth_frames > 127).float()[:, :, :, 0].unsqueeze(1)

    def __len__(self):
        return len(self.align_frames)

    def __getitem__(self, idx):
        return self.uv_frames[idx], self.bg_frames[idx], self.align_frames[idx], self.mouth_frames[idx]


# 这一段的输入只有gt(用来生成背景)，bg，uv
class TuneTestset(Dataset):
    def __init__(self, uv_frames, bg_frames, align_frames):
        self.uv_frames = uv_frames
        self.bg_frames = bg_frames
        self.align_frames = align_frames
        self.bg_frames = torch.from_numpy(self.bg_frames > 127).float()[:, :, :, 0].unsqueeze(1)
        self.uv_frames = torch.from_numpy(self.uv_frames).float().permute(0, 3, 1, 2) / 255
        self.uv_frames = self.uv_frames[:, :2]
        self.align_frames = torch.from_numpy(self.align_frames).float().permute(0, 3, 1, 2) / 128 - 1

    def __len__(self):
        return len(self.align_frames)

    def __getitem__(self, idx):
        return self.uv_frames[idx], self.bg_frames[idx], self.align_frames[idx]

if __name__ == "__main__":
    # train_data = LRWTrainDataset(length = 3, require_appa = True)
    # bg_frames, uv_frames, vid_frames, tex_frames, appa_frames = train_data.__getitem__(0)
    # print(bg_frames.shape, uv_frames.shape, vid_frames.shape, tex_frames.shape, appa_frames.shape)

    # test_data = LRWTestDataset(require_appa = True)
    # print(len(test_data))
    # bg_frames, uv_frames, vid_frames, tex_frames, appa_frames = test_data.__getitem__(0)
    # print(bg_frames.shape, uv_frames.shape, vid_frames.shape, tex_frames.shape, appa_frames.shape)

    test_data = TedTestDataset(length = 3)
    print(len(test_data))
    bg_frames, uv_frames, tex_frames = test_data.__getitem__(0)
    print(bg_frames.shape, uv_frames.shape, tex_frames.shape)

    # lrw_blend_data = LRWBlendDataset()
    # im1, im2, bg1, bg2, mouth1, mouth2, pred1, pred2 = lrw_blend_data.__getitem__(10)
    # print(im1.shape, im2.shape, bg1.shape, bg2.shape, mouth1.shape, mouth2.shape, pred1.shape, pred2.shape)