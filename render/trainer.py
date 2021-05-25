import os
import torch.nn as nn
import torch
import torchvision
import numpy as np
from loader import LRWTrainDataset, TedTestDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import model
from torch import optim
from utils import AverageMeter, worker_init_fn


class RenderTrainerAdv(object):
    def __init__(self, args):
        self.length = args.train_len
        self.trainset = LRWTrainDataset(self.length, require_appa = True)
        self.testset = TedTestDataset(self.length)

        self.batch_size = args.batch
        self.trainloader = DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True,
                                     pin_memory = True, drop_last = True, num_workers = 16, worker_init_fn=worker_init_fn)

        self.device = args.device
        self.name = args.name

        self.model_path = args.model_path
        self.log_path = args.log_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.writer = SummaryWriter(self.log_path)

        self.board_info_every = 50
        self.test_every = 5000
        self.milestones = set([600000, 640000, 680000, 720000, 760000, 800000, 840000, 880000, 920000, 960000])
        self.iterations = args.it

        if torch.cuda.device_count() > 1:
            self.multi_gpu = True
        else:
            self.multi_gpu = False

        self.tex_encoder = model.UNet(self.length * 3, args.texture_dim).to(self.device)
        self.tex_sampler = model.TexSampler().to(self.device)
        self.face_unet = model.define_G(args.texture_dim, 3, 64, 'local').to(self.device)
        self.discriminator = model.define_D(input_nc = 3 + 3 * self.length, ndf = 64, n_layers_D = 3, num_D = 2, getIntermFeat = True).to(self.device)

        self.criterion_l1 = nn.L1Loss().to(self.device)
        self.criterion_gan = model.GANLoss(use_lsgan = True, tensor = torch.cuda.FloatTensor).to(self.device)
        self.criterion_vgg = model.VGGLoss().to(self.device)

        if self.multi_gpu:
            self.tex_encoder = nn.DataParallel(self.tex_encoder)
            self.tex_sampler = nn.DataParallel(self.tex_sampler)
            self.face_unet = nn.DataParallel(self.face_unet)
            self.discriminator = nn.DataParallel(self.discriminator)
            self.criterion_vgg = nn.DataParallel(self.criterion_vgg)

        model.init_weights(self.tex_encoder)
        model.init_weights(self.tex_sampler)
        model.init_weights(self.face_unet)
        model.init_weights(self.discriminator)


        self.optimizer_G = optim.Adam([
            {'params': self.tex_encoder.parameters()},
            {'params': self.tex_sampler.parameters()},
            {'params': self.face_unet.parameters()}
        ], lr = args.lr, betas = (0.5, 0.999))

        self.optimizer_D = optim.Adam([
            {'params': self.discriminator.parameters()}
        ], lr = args.lr, betas = (0.5, 0.999))

        print(self.optimizer_G)
        print(self.optimizer_D)

    def save_state(self, model, save_path):
        if self.multi_gpu:
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)

    def load_state(self, model, model_path, strict = True):
        if self.multi_gpu:
            model.module.load_state_dict(torch.load(model_path), strict = strict)
        else:
            model.load_state_dict(torch.load(model_path), strict = strict)

    def schedule_lr(self):
        for params in self.optimizer_D.param_groups:
            params['lr'] *= 0.7

        for params in self.optimizer_G.param_groups:
            params['lr'] *= 0.7
        
        print(self.optimizer_D)
        print(self.optimizer_G)

    def test(self):
        self.tex_encoder.eval()
        self.tex_sampler.eval()
        self.face_unet.eval()
        
        batch_size = 32
        
        with torch.no_grad():
            for idx in range(1):
                bg_img_batch, uv_img_batch, tex_img_batch = self.testset.__getitem__(idx)
                bg_img_batch, uv_img_batch, tex_img_batch = bg_img_batch.unsqueeze(0), uv_img_batch.unsqueeze(0), tex_img_batch.unsqueeze(0)
                tex_img_batch = tex_img_batch.reshape(tex_img_batch.shape[0], -1, tex_img_batch.shape[3], tex_img_batch.shape[4]).to(self.device)
                tex = self.tex_encoder(tex_img_batch)
                # tex = tex_img_batch[:, 0]
                # torchvision.utils.save_image(tex[0], "test.png", normalize = True, range = (-1, 1))
                pred_img_batch = torch.zeros((uv_img_batch.shape[0], uv_img_batch.shape[1], 3, uv_img_batch.shape[3], uv_img_batch.shape[4])).float()
                start_idx = 0
                while start_idx < pred_img_batch.shape[1]:
                    if start_idx + batch_size > pred_img_batch.shape[1]:
                        end_idx = pred_img_batch.shape[1]
                    else:
                        end_idx = start_idx + batch_size
                    bg_tmp_batch = bg_img_batch[:, start_idx: end_idx].to(self.device)
                    uv_tmp_batch = uv_img_batch[:, start_idx: end_idx].to(self.device)
                    bg_tmp_batch = bg_tmp_batch.reshape(-1, bg_tmp_batch.shape[2], bg_tmp_batch.shape[3], bg_tmp_batch.shape[4])
                    uv_tmp_batch = uv_tmp_batch.reshape(-1, uv_tmp_batch.shape[2], uv_tmp_batch.shape[3], uv_tmp_batch.shape[4])
                    tex_tmp = tex.unsqueeze(1).repeat(1, uv_tmp_batch.shape[0], 1, 1, 1)
                    tex_tmp = tex_tmp.reshape(-1, tex_tmp.shape[2], tex_tmp.shape[3], tex_tmp.shape[4])
                    sample_image = self.tex_sampler(uv_tmp_batch, tex_tmp)
                    pred_image = self.face_unet(sample_image) * (1 - bg_tmp_batch)
                    # pred_image = sample_image * (1 - bg_tmp_batch)
                    pred_img_batch[:, start_idx:end_idx] = pred_image.cpu()
                    start_idx += batch_size
                pred_img_batch = pred_img_batch[0].cpu().detach()
                pred_img_batch = torch.flip(pred_img_batch, dims = [1])

                os.system("rm ../data/tmp/test/*")
                for i in range(len(pred_img_batch)):
                    torchvision.utils.save_image(pred_img_batch[i], "../data/tmp/test/{}.png".format(i), normalize = True, range = (-1, 1))
                os.system("ffmpeg -loglevel warning -framerate 25 -start_number 0 -i ../data/tmp/test/%d.png -c:v libx264 -pix_fmt yuv420p -b:v 2000k ./test.mp4")

        self.tex_encoder.train()
        self.tex_sampler.train()
        self.face_unet.train()

    def train(self):
        self.tex_encoder.train()
        self.tex_sampler.train()
        self.face_unet.train()
        self.discriminator.train()

        iter_idx = 0
        p_bar = tqdm(total = self.iterations)
        trainloader_iter = iter(self.trainloader)
        loss_d_meter = AverageMeter()
        loss_d_real_meter = AverageMeter()
        loss_d_fake_meter = AverageMeter()
        loss_g_meter = AverageMeter()
        loss_gan_meter = AverageMeter()
        loss_match_meter = AverageMeter()
        loss_rgb_meter = AverageMeter()
        loss_pred_meter = AverageMeter()

        while iter_idx < self.iterations:
            try:
                bg_img_batch, uv_img_batch, img_batch, tex_img_batch, appa_img_batch = next(trainloader_iter)
            except Exception:
                trainloader_iter = iter(self.trainloader)
                bg_img_batch, uv_img_batch, img_batch, tex_img_batch, appa_img_batch = next(trainloader_iter)

            bg_img_batch, uv_img_batch, img_batch, tex_img_batch, appa_img_batch = bg_img_batch.to(self.device), uv_img_batch.to(self.device), \
                                                        img_batch.to(self.device), tex_img_batch.to(self.device), appa_img_batch.to(self.device)
            img_batch = img_batch * (1 - bg_img_batch)
            img_batch_con = img_batch.reshape(-1, img_batch.shape[1] * img_batch.shape[2], img_batch.shape[3], img_batch.shape[4])  # con means continue
            img_batch = img_batch.reshape(-1, img_batch.shape[2], img_batch.shape[3], img_batch.shape[4])

            uv_img_batch = uv_img_batch.reshape(-1, uv_img_batch.shape[2], uv_img_batch.shape[3], uv_img_batch.shape[4])
            bg_img_batch = bg_img_batch.reshape(-1, bg_img_batch.shape[2], bg_img_batch.shape[3], bg_img_batch.shape[4])

            # img for generate texture
            tex_img_batch = tex_img_batch.reshape(tex_img_batch.shape[0], -1, tex_img_batch.shape[3], tex_img_batch.shape[4])
            tex = self.tex_encoder(tex_img_batch)
            tex = tex.unsqueeze(1).repeat(1, self.length, 1, 1, 1)
            tex = tex.reshape(-1, tex.shape[2], tex.shape[3], tex.shape[4])

            
            sample_image = self.tex_sampler(uv_img_batch, tex)
            pred_image = self.face_unet(sample_image) * (1 - bg_img_batch)
            pred_image_con = pred_image.reshape(-1, img_batch_con.shape[1], img_batch_con.shape[2], img_batch_con.shape[3])
            rgb_img = sample_image[:, 0:3, :, :] * (1 - bg_img_batch)

            # backward D
            # self.set_requires_grad(self.discriminator, True)
            self.optimizer_D.zero_grad()
            fake_AB = torch.cat((appa_img_batch, pred_image_con), 1)
            pred_fake = self.discriminator(fake_AB.detach())
            loss_d_fake = self.criterion_gan(pred_fake, False)

            real_AB = torch.cat((appa_img_batch, img_batch_con), 1)
            pred_real = self.discriminator(real_AB.detach())
            loss_d_real = self.criterion_gan(pred_real, True)

            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            self.optimizer_D.step()

            # backward G
            # self.set_requires_grad(self.discriminator, False)
            self.optimizer_G.zero_grad()
            fake_AB = torch.cat((appa_img_batch, pred_image_con), 1)
            pred_fake = self.discriminator(fake_AB)
            loss_gan = self.criterion_gan(pred_fake, True)

            # rgb loss
            loss_rgb = self.criterion_l1(rgb_img, img_batch) * 10

            # feat matching loss
            feat_weights = 1
            D_weights = 0.5
            loss_G_GAN_Feat = 0
            for i in range(2):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterion_l1(pred_fake[i][j], pred_real[i][j].detach()) * 10

            # vgg loss
            loss_pred = torch.mean(self.criterion_vgg(pred_image, img_batch) * 10) + self.criterion_l1(pred_image, img_batch) * 10

            loss_g = loss_rgb + loss_pred + loss_G_GAN_Feat + loss_gan
            loss_g.backward()
            self.optimizer_G.step()

            loss_d_meter.update(loss_d.item())
            loss_d_real_meter.update(loss_d_real.item())
            loss_d_fake_meter.update(loss_d_fake.item())
            loss_g_meter.update(loss_g.item())
            loss_gan_meter.update(loss_gan.item())
            loss_match_meter.update(loss_G_GAN_Feat.item())
            loss_rgb_meter.update(loss_rgb.item())
            loss_pred_meter.update(loss_pred.item())

            if iter_idx % self.board_info_every == 0 and iter_idx != 0:
                self.writer.add_scalar('loss_d', loss_d_meter(), iter_idx)
                self.writer.add_scalar('loss_d_real', loss_d_real_meter(), iter_idx)
                self.writer.add_scalar('loss_d_fake', loss_d_fake_meter(), iter_idx)
                self.writer.add_scalar('loss_g', loss_g_meter(), iter_idx)
                self.writer.add_scalar('loss_gan', loss_gan_meter(), iter_idx)
                self.writer.add_scalar('loss_match', loss_match_meter(), iter_idx)
                self.writer.add_scalar('loss_rgb', loss_rgb_meter(), iter_idx)
                self.writer.add_scalar('loss_pred', loss_pred_meter(), iter_idx)
                tqdm.write('[{}] loss_d: {:.5f} loss_d_real: {:.5f} loss_d_fake: {:.5f} loss_g: {:.5f} loss_gan: {:.5f} loss_match: {:.5f} loss_rgb: {:.5f} loss_pred: {:.5f}'.format(
                    iter_idx, loss_d_meter(), loss_d_real_meter(), loss_d_fake_meter(), loss_g_meter(), loss_gan_meter(), loss_match_meter(), loss_rgb_meter(), loss_pred_meter()))

            if iter_idx % self.test_every == 0 and iter_idx != 0:
                self.test()
                self.save_state(self.tex_encoder, os.path.join(self.model_path, 'tex_encoder.pkl'))
                self.save_state(self.tex_sampler, os.path.join(self.model_path, 'tex_sampler.pkl'))
                self.save_state(self.face_unet, os.path.join(self.model_path, 'face_unet.pkl'))
                self.save_state(self.discriminator, os.path.join(self.model_path, 'discriminator.pkl'))

            if iter_idx in self.milestones:
                self.schedule_lr()

            iter_idx += 1
            p_bar.update(1)

        p_bar.close()