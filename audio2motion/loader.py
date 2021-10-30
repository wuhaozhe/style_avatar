import sys
sys.path.append("..")
import torch
import numpy as np
import random
import time
import os
random.seed(time.time())
import pickle as pkl
import math
import lmdb
import quaternion
from torch.utils.data import Dataset
from io import BytesIO

class TedDataset(Dataset):
    def __init__(self, fx, fy, x_win, y_win, train = True, step = 8):
        self.fx = fx
        self.fy = fy
        self.y_win = y_win
        self.x_win = x_win
        self.train = train
        self.step = step    # the test step
        if train:
            self.db_key = "train"
        else:
            self.db_key = "test"

        exp_mean_std = pkl.load(open("../data/ted_hd/exp_mean_std.pkl", 'rb'))
        self.exp_std_mean = exp_mean_std['s_m']
        self.exp_std_std = exp_mean_std['s_s']
        self.exp_diff_std_mean = exp_mean_std['d_s_m']
        self.exp_diff_std_std = exp_mean_std['d_s_s']

        pose_mean_std = pkl.load(open("../data/ted_hd/pose_mean_std.pkl", 'rb'))
        self.pose_diff_std_mean = pose_mean_std['d_s_m']
        self.pose_diff_std_std = pose_mean_std['d_s_s']

        lmdb_path = "../data/ted_hd/lmdb"
        self.env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)
        self.coeff_norm_data = self.env.open_db("{}_coeff_norm".format(self.db_key).encode())
        self.deepspeech_data = self.env.open_db("{}_deepspeech".format(self.db_key).encode())
        self.energy_data = self.env.open_db("{}_energy".format(self.db_key).encode())
        self.txn = self.env.begin(write = False)
        self.length = self.txn.stat(db=self.coeff_norm_data)['entries']

    def __len__(self):
        return self.length

    def __split_coeff(self, coeff):
        exp = coeff[:,80:144]
        translation = coeff[:,254:257]
        angles = quaternion.euler_to_quaternion(coeff[:, 224:227], 'xyz')
        return exp, translation, angles

    def __get_feat(self, exp, pose):
        diff_exp = exp[:-1, :] - exp[1:, :]
        exp_std = (np.std(exp, axis = 0) - self.exp_std_mean) / self.exp_std_std
        diff_exp_std = (np.std(diff_exp, axis = 0) - self.exp_diff_std_mean) / self.exp_diff_std_std

        diff_pose = pose[:-1, :] - pose[1:, :]
        diff_pose_std = (np.std(diff_pose, axis = 0) - self.pose_diff_std_mean) / self.pose_diff_std_std

        return np.concatenate((exp_std, diff_exp_std, diff_pose_std))

    def __get_data_from_db(self, idx):
        audio_bin = self.txn.get(str(idx).encode(), db = self.deepspeech_data)
        energy_bin = self.txn.get(str(idx).encode(), db = self.energy_data)
        coeff_bin = self.txn.get(str(idx).encode(), db = self.coeff_norm_data)
        audio = pkl.load(BytesIO(audio_bin))
        energy = pkl.load(BytesIO(energy_bin))
        coeff = pkl.load(BytesIO(coeff_bin))
        return audio, energy, coeff

    def __get_train_data(self, idx):
        audio, energy, coeff = self.__get_data_from_db(idx)
        exp, translation, angles = self.__split_coeff(coeff)
        pose = np.concatenate((angles, translation), axis = 1)
        
        x_sty = torch.from_numpy(self.__get_feat(exp, pose)).float()

        y_len = len(pose)
        y_left = random.randint(0, y_len - self.y_win)
        y_right = y_left + self.y_win

        pose_clip = pose[y_left: y_right]
        exp_clip = exp[y_left: y_right]

        audio_clip = self.__get_sync_data(audio, 29, y_left, y_right)   # 29 is the deepspeech dim
        
        energy = np.transpose(energy)
        energy_clip = self.__get_sync_data(energy, 1, y_left, y_right)  # 1 is the energy dim

        pose_clip = torch.from_numpy(pose_clip).transpose(0, 1).float()
        exp_clip = torch.from_numpy(exp_clip).transpose(0, 1).float()
        energy_clip = torch.from_numpy(energy_clip).transpose(0, 1).float()
        audio_clip = torch.from_numpy(audio_clip).transpose(0, 1).float()

        return exp_clip, pose_clip, audio_clip, energy_clip, x_sty

    def __get_test_data(self, idx):
        audio, energy, coeff = self.__get_data_from_db(idx)
        exp, translation, angles = self.__split_coeff(coeff)
        pose = np.concatenate((angles, translation), axis = 1)
        energy = np.transpose(energy)

        y_len = len(pose)
        audio_clip_list = []
        energy_clip_list = []
        pose_clip_list = []
        exp_clip_list = []

        for i in range(0, y_len, self.step):
            if i > y_len - self.y_win:
                y_left = y_len - self.y_win
                y_right = y_len
            else:
                y_left = i
                y_right = i + self.y_win

            audio_clip = self.__get_sync_data(audio, 29, y_left, y_right)   # 29 is the deepspeech dim
            energy_clip = self.__get_sync_data(energy, 1, y_left, y_right)  # 1 is the energy dim
            pose_clip = pose[y_left: y_right]
            exp_clip = exp[y_left: y_right]

            audio_clip_list.append(audio_clip)
            energy_clip_list.append(energy_clip)
            pose_clip_list.append(pose_clip)
            exp_clip_list.append(exp_clip)

        audio_clip_list = np.array(audio_clip_list)
        energy_clip_list = np.array(energy_clip_list)
        pose_clip_list = np.array(pose_clip_list)
        exp_clip_list = np.array(exp_clip_list)

        audio_clip_list = torch.from_numpy(audio_clip_list).transpose(1, 2).float()
        energy_clip_list = torch.from_numpy(energy_clip_list).transpose(1, 2).float()
        pose_clip_list = torch.from_numpy(pose_clip_list).transpose(1, 2).float()
        exp_clip_list = torch.from_numpy(exp_clip_list).transpose(1, 2).float()

        return exp_clip_list, pose_clip_list, audio_clip_list, energy_clip_list, y_len

    def __get_sync_data(self, x, x_dim, y_left, y_right):
        x_len = len(x)
        x_left = math.floor(y_left * self.fx / self.fy)
        x_right = math.floor(y_right * self.fx / self.fy)
        pad_len = self.x_win - x_right + x_left

        if pad_len % 2 == 0:
            pad_left = pad_len // 2
            pad_right = pad_len // 2
        else:
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left

        x_left = x_left - pad_left
        x_right = x_right + pad_right
        if x_left < 0:
            if x_right > x_len:
                x_data = np.concatenate((np.tile(x[0], -1 * x_left).reshape(-1, x_dim), x, np.tile(x[-1], x_right - x_len).reshape(-1, x_dim)), axis = 0)
            else:
                x_data = x[0: x_right]
                x_data = np.concatenate((np.tile(x[0], -1 * x_left).reshape(-1, x_dim), x_data), axis = 0)
        elif x_right > x_len:
            x_data = x[x_left: x_len]
            x_data = np.concatenate((x_data, np.tile(x[-1], x_right - x_len).reshape(-1, x_dim)), axis = 0)
        else:
            x_data = x[x_left: x_right]
        
        return x_data

    def __getitem__(self, idx):
        if self.train:
            return self.__get_train_data(idx)
        else:
            return self.__get_test_data(idx)

if __name__ == "__main__":
    train_data = TedDataset(50, 25, 80, 32, True)
    test_data = TedDataset(50, 25, 80, 32, False)
    print(len(train_data), len(test_data))
    from tqdm import tqdm
    for i in tqdm(range(100)):
        exp_clip, pose_clip, audio_clip, energy_clip, x_sty = train_data.__getitem__(i)
        # print(exp_clip.shape, pose_clip.shape, audio_clip.shape, energy_clip.shape, x_sty.shape)

    exp_clip_list, pose_clip_list, audio_clip_list, energy_clip_list, y_len = test_data.__getitem__(0)
    print(exp_clip_list.shape, pose_clip_list.shape, audio_clip_list.shape, energy_clip_list.shape, y_len)
