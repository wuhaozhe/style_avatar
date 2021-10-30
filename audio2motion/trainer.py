import sys
sys.path.append("..")
# from .loader import TedDataset
from loader import TedDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch import optim
# from .model import StyleFusionModel
# from .utils import mkdirs
from model import StyleFusionModel
from utils import mkdirs
import json
import numpy as np
import torch
import torch.nn as nn
import pickle as pkl
import os
import math
import quaternion

class Trainer(object):
    def __init__(self, conf):
        self.mode = conf.mode
        if self.mode != "exp" and self.mode != 'pose':
            raise Exception("Unknown trainer mode")

        self.fx = conf.fx
        self.fy = conf.fy
        self.xwin = conf.xwin
        self.ywin = conf.ywin

        # config train test param
        self.milestones = set([25000])
        self.board_info_every = 100
        self.test_every = 100
        self.test_step = 8
        self.iterations = conf.iterations

        self.trainset = TedDataset(self.fx, self.fy, self.xwin, self.ywin, train = True)
        self.testset = TedDataset(self.fx, self.fy, self.xwin, self.ywin, train = False, step = self.test_step)

        self.batch_size = conf.batch_size
        self.trainloader = DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True, pin_memory = True, drop_last = True)

        self.device = conf.device

        # config saver
        self.model_path = conf.model_path
        self.log_path = conf.log_path
        mkdirs(self.model_path)
        mkdirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)

        # config model
        if self.mode == "exp":
            self.out_dim = 64
        elif self.mode == "pose":
            self.out_dim = 7

        self.mseloss = nn.MSELoss().to(self.device)
        self.l1loss = nn.L1Loss().to(self.device)
        self.drop = conf.drop

        self.motion_mean = torch.Tensor([-9.9823493e-01, -1.0308119e-05,  5.8148009e-05, -5.5828014e-06, -1.1329491e-06,  2.2419929e-06, -2.5054207e-05]).float().to(self.device)
        self.motion_std = torch.Tensor([0.006208, 0.03814816, 0.05244023, 0.04612421, 0.00423496, 0.00747705, 0.03932659]).float().to(self.device)
        self.motion_mean = self.motion_mean.unsqueeze(0).unsqueeze(2)
        self.motion_std = self.motion_std.unsqueeze(0).unsqueeze(2)

        self.backbone = StyleFusionModel(self.drop, self.out_dim, self.ywin).to(self.device)

        self.optimizer = optim.Adam(
            [
                {'params': self.backbone.parameters()}
            ],
            lr = conf.lr,
            weight_decay = 1e-4
        )
        self.load_state(self.backbone, "/home/wuhz/mnt/avatar/fouriermap/model/arbsty_drop_0.5_exp/backbone.pkl")

    def save_state(self, model, save_path):
        torch.save(model.state_dict(), save_path)

    def load_state(self, model, model_path, strict = True):
        model.load_state_dict(torch.load(model_path), strict = strict)

    def test_all(self):
        trainloader_iter = iter(self.trainloader)
        _, _, _, _, sty = next(trainloader_iter)
        sty = sty.to(self.device)
        sty = sty[0:10]

        # self.backbone.eval()
        self.backbone.train()
        coeff = np.load("../data/ted_hd/test_coeff.npy")
        for idx in range(len(self.testset)):
            exp_batch, pose_batch, audio_batch, ene_batch, y_len = self.testset.__getitem__(idx)
            exp_batch, pose_batch, audio_batch, ene_batch = exp_batch.to(self.device), pose_batch.to(self.device), audio_batch.to(self.device), ene_batch.to(self.device)
            coeff = np.repeat(np.expand_dims(coeff[0], axis = 0), y_len, axis = 0)

            for j in range(10):
                sty_tmp = sty[j].unsqueeze(0).repeat(len(exp_batch), 1)
                pred_exp_batch = self.backbone(audio_batch, ene_batch, sty_tmp)

                y_repeat = torch.zeros(y_len).int().to(self.device)
                predict_exp_cat = torch.zeros((y_len, self.out_dim)).float().to(self.device)
                for counter, i in enumerate(range(0, y_len, 8)):
                    if i > y_len - self.ywin:
                        y_left = y_len - self.ywin
                        y_right = y_len
                    else:
                        y_left = i
                        y_right = i + self.ywin
                    y_repeat[y_left: y_right] += 1
                    predict_exp_cat[y_left: y_right] += pred_exp_batch[counter].transpose(0, 1)
                y_repeat = y_repeat.float()
                predict_exp_cat = predict_exp_cat / y_repeat.unsqueeze(1)
                coeff[:, 80:144] = predict_exp_cat.detach().cpu().numpy()
                np.save("../data/ted_hd/test/{}_{}.npy".format(idx, j), coeff)

        self.backbone.train()

    def test(self, sty = None):
        # Given light, shape of test coeff, synthesize video
        if sty is None:
            trainloader_iter = iter(self.trainloader)
            _, _, _, _, sty = next(trainloader_iter)
            sty = sty.to(self.device)
            sty = sty[0:10]

        self.backbone.eval()
        coeff = np.load("../data/ted_hd/test_coeff.npy")
        exp_batch, pose_batch, audio_batch, ene_batch, y_len = self.testset.__getitem__(0)
        exp_batch, pose_batch, audio_batch, ene_batch = exp_batch.to(self.device), pose_batch.to(self.device), audio_batch.to(self.device), ene_batch.to(self.device)
        coeff = np.repeat(np.expand_dims(coeff[0], axis = 0), y_len, axis = 0)

        if self.mode == 'exp':
            for idx in range(10):
                sty_tmp = sty[idx].unsqueeze(0).repeat(len(exp_batch), 1)
                pred_exp_batch = self.backbone(audio_batch, ene_batch, sty_tmp)

                y_repeat = torch.zeros(y_len).int().to(self.device)
                predict_exp_cat = torch.zeros((y_len, self.out_dim)).float().to(self.device)
                for counter, i in enumerate(range(0, y_len, 8)):
                    if i > y_len - self.ywin:
                        y_left = y_len - self.ywin
                        y_right = y_len
                    else:
                        y_left = i
                        y_right = i + self.ywin
                    y_repeat[y_left: y_right] += 1
                    predict_exp_cat[y_left: y_right] += pred_exp_batch[counter].transpose(0, 1)
                y_repeat = y_repeat.float()
                predict_exp_cat = predict_exp_cat / y_repeat.unsqueeze(1)
                coeff[:, 80:144] = predict_exp_cat.detach().cpu().numpy()
                np.save("../data/ted_hd/test_{}_{}.npy".format(self.drop, idx), coeff)
        else:
            for idx in range(10):
                sty_tmp = sty[idx].unsqueeze(0).repeat(len(exp_batch), 1)
                pred_pose_batch = self.backbone(audio_batch, ene_batch, sty_tmp)

                y_repeat = torch.zeros(y_len).int().to(self.device)
                predict_pose_cat = torch.zeros((y_len, self.out_dim)).float().to(self.device)
                for counter, i in enumerate(range(0, y_len, 8)):
                    if i > y_len - self.ywin:
                        y_left = y_len - self.ywin
                        y_right = y_len
                    else:
                        y_left = i
                        y_right = i + self.ywin
                    y_repeat[y_left: y_right] += 1
                    predict_pose_cat[y_left: y_right] += pred_pose_batch[counter].transpose(0, 1)
                y_repeat = y_repeat.float()
                predict_pose_cat = predict_pose_cat / y_repeat.unsqueeze(1)
                predict_pose_cat = (predict_pose_cat * self.motion_std[:, :, 0]) + self.motion_mean[:, :, 0]
                predict_pose_cat = predict_pose_cat.detach().cpu().numpy()
                coeff[:, 224:227] = quaternion.qeuler_np(predict_pose_cat[:, 0:4], order = 'xyz', epsilon = 1e-5)
                coeff[:, 254:257] = predict_pose_cat[:, 4:]
                np.save("../data/ted_hd/test_pose_{}_{}.npy".format(self.drop, idx), coeff)

        self.backbone.train()

    def train(self):
        self.backbone.train()

        iter_idx = 0
        p_bar = tqdm(total = self.iterations)
        trainloader_iter = iter(self.trainloader)
        running_loss = 0.

        while iter_idx < self.iterations:
            try:
                exp_batch, pose_batch, audio_batch, ene_batch, sty = next(trainloader_iter)
            except Exception:
                trainloader_iter = iter(self.trainloader)
                exp_batch, pose_batch, audio_batch, ene_batch, sty = next(trainloader_iter)

            exp_batch = exp_batch.to(self.device)
            pose_batch = pose_batch.to(self.device)
            pose_batch = (pose_batch - self.motion_mean) / self.motion_std
            audio_batch = audio_batch.to(self.device)
            ene_batch = ene_batch.to(self.device)
            sty = sty.to(self.device)

            pred_batch = self.backbone(audio_batch, ene_batch, sty)

            if self.mode == 'exp':
                loss = self.mseloss(pred_batch, exp_batch)
            else:
                loss = self.mseloss(pred_batch, pose_batch)

            if iter_idx >= 3000:
                loss = 50 * loss

            self.optimizer.zero_grad()
            loss.backward()
            running_loss += loss.item()
            self.optimizer.step()

            if iter_idx % self.board_info_every == 0 and iter_idx != 0:
                mse_board = running_loss / self.board_info_every
                self.writer.add_scalar('train_loss', mse_board, iter_idx)
                tqdm.write('[{}] loss: {:.5f}'.format(iter_idx, mse_board))
                running_loss = 0.

            if iter_idx % self.test_every == 0 and iter_idx != 0:
                self.test(sty[0:10])
                
                self.save_state(self.backbone, os.path.join(self.model_path, 'backbone.pkl'))

            iter_idx += 1
            p_bar.update(1)

        p_bar.close()