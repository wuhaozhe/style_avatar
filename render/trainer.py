import os
import torch.nn as nn
import torch
import torchvision
import numpy as np
import cv2
from loader import LRWTrainDataset, LRWTestDataset, LRWBlendDataset, TedTestDataset, TedDemoDataset, TuneTrainset, TuneTestset
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

        # self.load_state(self.face_unet, "/home/wuhz/mnt/avatar/NeuralRender/model/render_lrw_lr_2e-4_vggsingle/face_unet.pkl")
        # self.load_state(self.tex_sampler, "/home/wuhz/mnt/avatar/NeuralRender/model/render_lrw_lr_2e-4_vggsingle/tex_sampler.pkl")
        # self.load_state(self.tex_encoder, "/home/wuhz/mnt/avatar/NeuralRender/model/render_lrw_lr_2e-4_vggsingle/tex_encoder.pkl")

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

    def infer_trainset(self):
        self.load_state(self.face_unet, "/home/wuhz/mnt/avatar/style_avatar/render/model/render_lrw_lr_2e-4_tl_1/face_unet.pkl")
        self.load_state(self.tex_sampler, "/home/wuhz/mnt/avatar/style_avatar/render/model/render_lrw_lr_2e-4_tl_1/tex_sampler.pkl")
        self.load_state(self.tex_encoder, "/home/wuhz/mnt/avatar/style_avatar/render/model/render_lrw_lr_2e-4_tl_1/tex_encoder.pkl")

        self.tex_encoder.eval()
        self.tex_sampler.eval()
        self.face_unet.eval()

        import lmdb
        lmdb_path = "../data/lrw/lmdb"
        env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)
        pred_data = env.open_db("pred_data".encode())

        batch_size = 32
        lrw_test_set = LRWTestDataset(self.length, require_appa = False)

        with torch.no_grad():
            for idx in tqdm(range(len(lrw_test_set))):
                bg_frames, uv_frames, vid_frames, tex_frames = lrw_test_set.__getitem__(idx)
                bg_frames, uv_frames, vid_frames, tex_frames = bg_frames.cuda(), uv_frames.cuda(), vid_frames.cuda(), tex_frames.cuda()
                bg_img_batch, uv_img_batch, tex_img_batch = bg_frames.unsqueeze(0), uv_frames.unsqueeze(0), tex_frames.unsqueeze(0)
                tex_img_batch = tex_img_batch.reshape(tex_img_batch.shape[0], -1, tex_img_batch.shape[3], tex_img_batch.shape[4])
                tex = self.tex_encoder(tex_img_batch)
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
                    pred_img_batch[:, start_idx:end_idx] = pred_image.cpu()
                    start_idx += batch_size

                pred_img_batch = pred_img_batch[0].cpu().detach()
                pred_img_batch = torch.flip(pred_img_batch, dims = [1])

                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter("./test_{}.avi".format(self.name), fourcc, 25.0, (224, 224))
                for i in range(len(pred_img_batch)):
                    pred_img = 255 * (pred_img_batch[i].detach().data.numpy() * 0.5 + 0.5)
                    pred_img = np.transpose(pred_img, (1, 2, 0))[:, :, ::-1].astype(np.uint8)
                    out.write(pred_img)
                out.release()

                with open("./test_{}.avi".format(self.name), 'rb') as f:
                    pred_bin = f.read()

                txn = env.begin(write = True)
                txn.put(str(idx).encode(), pred_bin, db = pred_data)
                txn.commit()


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
                # tex = tex_img_batch
                # torchvision.utils.save_image(torch.flip(tex[0], dims = [0]), "test_tex.png", normalize = True, range = (-1, 1))
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
                    # torchvision.utils.save_image(torch.flip(uv_tmp_batch[0, 0], dims = [0]), "test_uv.png", normalize = True, range = (-1, 1))
                    bg_tmp_batch = bg_tmp_batch.reshape(-1, bg_tmp_batch.shape[2], bg_tmp_batch.shape[3], bg_tmp_batch.shape[4])
                    uv_tmp_batch = uv_tmp_batch.reshape(-1, uv_tmp_batch.shape[2], uv_tmp_batch.shape[3], uv_tmp_batch.shape[4])
                    tex_tmp = tex.unsqueeze(1).repeat(1, uv_tmp_batch.shape[0], 1, 1, 1)
                    tex_tmp = tex_tmp.reshape(-1, tex_tmp.shape[2], tex_tmp.shape[3], tex_tmp.shape[4])
                    sample_image = self.tex_sampler(uv_tmp_batch, tex_tmp)
                    # pred_image = sample_image
                    pred_image = self.face_unet(sample_image) * (1 - bg_tmp_batch)
                    pred_image = sample_image * (1 - bg_tmp_batch)
                    pred_img_batch[:, start_idx:end_idx] = pred_image.cpu()
                    start_idx += batch_size
                pred_img_batch = pred_img_batch[0].cpu().detach()
                pred_img_batch = torch.flip(pred_img_batch, dims = [1])

                os.system("rm ../data/tmp/test/*")
                for i in range(len(pred_img_batch)):
                    torchvision.utils.save_image(pred_img_batch[i], "../data/tmp/test/{}_{}.png".format(i, self.name), normalize = True, range = (-1, 1))
                os.system("ffmpeg -y -loglevel warning -framerate 25 -start_number 0 -i ../data/tmp/test/%d_{}.png -c:v libx264 -pix_fmt yuv420p -b:v 2000k ./test_{}.mp4".format(self.name, self.name))

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


class BlendTrainerAdv(object):
    def __init__(self, args):
        # 暂时testset，给一个video path直接生成demo
        self.trainset = LRWBlendDataset()
        self.testset = TedDemoDataset()

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

        self.generator = model.define_G(3, 3, 64, 'local').to(self.device)
        # self.generator = model.UNetnop(3, 3).to(self.device)
        self.discriminator = model.define_D(input_nc = 6, ndf = 64, n_layers_D = 3, num_D = 2, getIntermFeat = True).to(self.device)

        self.criterion_l1 = nn.L1Loss()
        self.criterion_gan = model.GANLoss(use_lsgan = True, tensor = torch.cuda.FloatTensor).to(self.device)
        self.criterion_vgg = model.VGGLoss().to(self.device)
        self.criterion_lap = model.LapLoss(device = self.device)
        self.bgerode = model.BgErode().to(self.device)

        if self.multi_gpu:
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)
            self.criterion_vgg = nn.DataParallel(self.criterion_vgg)

        model.init_weights(self.generator)
        model.init_weights(self.discriminator)

        self.optimizer_G = optim.Adam([
            {'params': self.generator.parameters()}
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
        self.generator.eval()
        
        batch_size = 32
        
        with torch.no_grad():

            bg_frames, align_frames, pred_frames = self.testset.__getitem__(0)
            # bg_im = bg_frames * align_frames

            pred_img_batch = torch.zeros((align_frames.shape[0], align_frames.shape[1], align_frames.shape[2], align_frames.shape[3])).float()
            start_idx = 0
            while start_idx < pred_img_batch.shape[0]:
                if start_idx + batch_size > pred_img_batch.shape[0]:
                    end_idx = pred_img_batch.shape[0]
                else:
                    end_idx = start_idx + batch_size
                
                align_batch = align_frames[start_idx: end_idx].cuda()
                bg_batch = bg_frames[start_idx: end_idx].cuda()
                bg_erode = self.bgerode(bg_batch).detach()
                bg_im = bg_erode * align_batch
                # in_batch = torch.cat((bg_im[start_idx: end_idx], pred_frames[start_idx: end_idx]), dim = 1).cuda()
                # in_batch = bg_im[start_idx: end_idx] + pred_frames[start_idx: end_idx].cuda()
                in_batch = bg_im + pred_frames[start_idx: end_idx].cuda()
                pred_im = self.generator(in_batch) * (1 - bg_erode) + bg_im
                pred_img_batch[start_idx:end_idx] = pred_im.cpu().detach()
                start_idx += batch_size

            pred_img_batch = torch.flip(pred_img_batch, dims = [1])

            os.system("rm ../data/tmp/test/*")
            for i in range(len(pred_img_batch)):
                torchvision.utils.save_image(pred_img_batch[i], "../data/tmp/test/{}_{}.png".format(i, self.name), normalize = True, range = (-1, 1))
            os.system("ffmpeg -y -loglevel warning -framerate 25 -start_number 0 -i ../data/tmp/test/%d_{}.png -c:v libx264 -pix_fmt yuv420p -b:v 2000k ./test_{}.mp4".format(self.name, self.name))

        self.generator.train()

    def train(self):
        self.generator.train()
        self.discriminator.train()

        iter_idx = 0
        p_bar = tqdm(total = self.iterations)
        trainloader_iter = iter(self.trainloader)
        loss_g_meter = AverageMeter()
        loss_pred_meter = AverageMeter()
        loss_pred_mouth_meter = AverageMeter()
        loss_lap_meter = AverageMeter()

        while iter_idx < self.iterations:
            try:
                im1_batch, im2_batch, bg1_batch, bg2_batch, mouth1_batch, mouth2_batch, pred1_batch, pred2_batch = next(trainloader_iter)
            except Exception:
                trainloader_iter = iter(self.trainloader)
                im1_batch, im2_batch, bg1_batch, bg2_batch, mouth1_batch, mouth2_batch, pred1_batch, pred2_batch = next(trainloader_iter)

            im1_batch = im1_batch.to(self.device)
            im2_batch = im2_batch.to(self.device)
            bg1_batch = bg1_batch.to(self.device)
            bg2_batch = bg2_batch.to(self.device)
            mouth1_batch = mouth1_batch.to(self.device)
            mouth2_batch = mouth2_batch.to(self.device)
            pred1_batch = pred1_batch.to(self.device)
            pred2_batch = pred2_batch.to(self.device)

            bg1_erode = self.bgerode(bg1_batch).detach()
            bg2_erode = self.bgerode(bg2_batch).detach()
            bg_erode = torch.cat((bg1_erode, bg2_erode), dim = 0)

            im1_bg = im1_batch * bg1_erode
            im2_bg = im2_batch * bg2_erode
            im_bg = torch.cat((im1_bg, im2_bg), dim = 0)
            
            in_im1_pred1 = im1_bg + pred1_batch
            in_im2_pred2 = im2_bg + pred2_batch

            in_im_pred = torch.cat((in_im1_pred1, in_im2_pred2), dim = 0)

            real_im = torch.cat((im1_batch, im2_batch), dim = 0)
            # if iter_idx == 0:
            #     torchvision.utils.save_image(real_im[0].cpu().data, "test.png", normalize = True, range = (-1, 1))
            fake_im = self.generator(in_im_pred) * (1 - bg_erode) + im_bg
            mouth_mask = torch.cat((mouth1_batch, mouth2_batch), dim = 0)
            real_im_mouth = real_im * mouth_mask
            fake_im_mouth = fake_im * mouth_mask

            self.optimizer_G.zero_grad()

            # vgg loss
            loss_pred = torch.mean(self.criterion_vgg(fake_im, real_im) * 10) + self.criterion_l1(fake_im, real_im) * 10
            loss_pred_mouth = self.criterion_l1(fake_im_mouth, real_im_mouth) * 10
            loss_lap = self.criterion_lap(fake_im, real_im) * 50

            loss_g = loss_pred + loss_pred_mouth + loss_lap
            loss_g.backward()
            self.optimizer_G.step()

            loss_g_meter.update(loss_g.item())
            loss_pred_meter.update(loss_pred.item())
            loss_pred_mouth_meter.update(loss_pred_mouth.item())
            loss_lap_meter.update(loss_lap.item())

            if iter_idx % self.board_info_every == 0 and iter_idx != 0:
                self.writer.add_scalar('loss_g', loss_g_meter(), iter_idx)
                self.writer.add_scalar('loss_pred', loss_pred_meter(), iter_idx)
                self.writer.add_scalar('loss_pred_mouth', loss_pred_mouth_meter(), iter_idx)
                self.writer.add_scalar('loss_lap', loss_lap_meter(), iter_idx)
                tqdm.write('[{}] loss_g: {:.5f} loss_mouth: {:.5f} loss_pred: {:.5f} loss_lap: {:.5f}'.format(
                    iter_idx, loss_g_meter(), loss_pred_mouth_meter(), loss_pred_meter(), loss_lap_meter()))

            if iter_idx % self.test_every == 0 and iter_idx != 0:
                self.test()
                self.save_state(self.generator, os.path.join(self.model_path, 'generator.pkl'))
                self.save_state(self.discriminator, os.path.join(self.model_path, 'discriminator.pkl'))

            if iter_idx in self.milestones:
                self.schedule_lr()

            iter_idx += 1
            p_bar.update(1)

        p_bar.close()

class TuneTrainer(object):
    def __init__(self, args, in_data):
        # the in data is a dict
        self.length = args.train_len
        train_uv, train_align, train_bg, train_mouth = in_data['train_uv'], in_data['train_align'], in_data['train_bg'], in_data['train_mouth']
        test_uv, test_align, test_bg = in_data['test_uv'], in_data['test_align'], in_data['test_bg']
        rgb_texture = in_data['tex']
        self.rgb_texture = torch.from_numpy(rgb_texture).float().permute(2, 0, 1) / 128 - 1
        self.rgb_texture = torch.flip(self.rgb_texture.unsqueeze(0), dims = [2]).cuda()
        self.trainset = TuneTrainset(train_uv, train_bg, train_align, train_mouth)
        self.testset = TuneTestset(test_uv, test_bg, test_align)

        self.batch_size = args.batch
        self.trainloader = DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True,
                                     pin_memory = True, drop_last = True, num_workers = 16, worker_init_fn=worker_init_fn)
        self.testloader = DataLoader(self.testset, batch_size = 16, shuffle = False,
                                     pin_memory = True, drop_last = False, num_workers = 16, worker_init_fn=worker_init_fn)

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
        # self.test_every = 500
        self.test_every = 100

        self.milestones = set([2000, 4000])
        # self.milestones = set([6000])
        self.iterations = args.it

        if torch.cuda.device_count() > 1:
            self.multi_gpu = True
        else:
            self.multi_gpu = False

        self.tex_encoder = model.UNet(self.length * 3, args.texture_dim).to(self.device)
        self.tex_sampler = model.TexSampler().to(self.device)
        self.face_unet = model.define_G(args.texture_dim, 3, 64, 'local').to(self.device)
        self.blender = model.define_G(3, 3, 64, 'local').to(self.device)
        self.bgerode = model.BgErode().to(self.device)

        self.criterion_l1 = nn.L1Loss().to(self.device)
        self.criterion_vgg = model.VGGLoss().to(self.device)
        self.criterion_lap = model.LapLoss(device = self.device)

        if self.multi_gpu:
            self.tex_sampler = nn.DataParallel(self.tex_sampler)
            self.tex_encoder = nn.DataParallel(self.tex_encoder)
            self.face_unet = nn.DataParallel(self.face_unet)
            self.blender = nn.DataParallel(self.blender)
            self.criterion_vgg = nn.DataParallel(self.criterion_vgg)

        self.load_state(self.face_unet, "/home/wuhz/mnt/avatar/style_avatar/render/model/render_lrw_lr_2e-4_tl_1/face_unet.pkl")
        self.load_state(self.tex_sampler, "/home/wuhz/mnt/avatar/style_avatar/render/model/render_lrw_lr_2e-4_tl_1/tex_sampler.pkl")
        self.load_state(self.tex_encoder, "/home/wuhz/mnt/avatar/style_avatar/render/model/render_lrw_lr_2e-4_tl_1/tex_encoder.pkl")
        # self.load_state(self.blender, "/home/wuhz/mnt/avatar/style_avatar/render/model/render_lrw_lr_2e-4_blend_lap/generator.pkl")
        self.load_state(self.blender, "/home/wuhz/mnt/avatar/style_avatar/render/model/render_lrw_lr_2e-4_blend_4/generator.pkl")

        self.neural_texture = self.tex_encoder(self.rgb_texture).detach().requires_grad_(True)
        torch.save(self.neural_texture, os.path.join(self.model_path, 'neural_texture.pkl'))

        self.optimizer_G = optim.Adam([
            {'params': self.tex_sampler.parameters()},
            {'params': self.face_unet.parameters()},
            {'params': self.blender.parameters()},
            {'params': self.neural_texture}
        ], lr = args.lr, betas = (0.5, 0.999))

        print(self.optimizer_G)

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

        for params in self.optimizer_G.param_groups:
            params['lr'] *= 0.5
        
        print(self.optimizer_G)

    def test(self):
        self.tex_sampler.eval()
        self.face_unet.eval()
        self.blender.eval()

        with torch.no_grad():
            testloader_iter = iter(self.testloader)
            pred_img_batch = torch.zeros((len(self.testset), 3, 224, 224)).float()
            idx = 0
            while True:
                try:
                    uv_batch, bg_batch, align_batch = next(testloader_iter)
                except Exception:
                    break
                uv_batch = uv_batch.to(self.device)
                bg_batch = bg_batch.to(self.device)
                align_batch = align_batch.to(self.device)
                tex = self.neural_texture.repeat(len(align_batch), 1, 1, 1)
                bg_erode = self.bgerode(bg_batch).detach()
                
                sample_image = self.tex_sampler(uv_batch, tex)
                pred_image_mask = self.face_unet(sample_image) * (1 - bg_batch)
                in_im_pred = pred_image_mask + bg_erode * align_batch
                # pred_image = self.blender(in_im_pred)
                pred_image = self.blender(in_im_pred) * (1 - bg_erode) + bg_erode * align_batch

                pred_img_batch[idx:idx + len(pred_image)] = pred_image.cpu()
                idx += len(pred_image)

            pred_img_batch = pred_img_batch.detach()
            pred_img_batch = torch.flip(pred_img_batch, dims = [1])

            os.system("rm ../data/tmp/test/*")
            for i in range(len(pred_img_batch)):
                torchvision.utils.save_image(pred_img_batch[i], os.path.join(self.log_path, "{}_{}.png".format(i, self.name)), normalize = True, range = (-1, 1))
            os.system("ffmpeg -y -loglevel warning -framerate 25 -start_number 0 -i ../data/tmp/test/%d_{}.png -c:v libx264 -pix_fmt yuv420p -b:v 2000k ./test_{}.mp4".format(self.name, self.name))

        self.tex_sampler.train()
        self.face_unet.train()
        self.blender.train()

    def test_step(self, iter_idx):
        self.tex_sampler.eval()
        self.face_unet.eval()
        self.blender.eval()
        with torch.no_grad():
            testloader_iter = iter(self.testloader)
            pred_img_batch = torch.zeros((16, 3, 224, 224)).float()
            idx = 0
            while True:
                try:
                    uv_batch, bg_batch, align_batch = next(testloader_iter)
                except Exception:
                    break
                uv_batch = uv_batch.to(self.device)
                bg_batch = bg_batch.to(self.device)
                align_batch = align_batch.to(self.device)
                tex = self.neural_texture.repeat(len(align_batch), 1, 1, 1)
                bg_erode = self.bgerode(bg_batch).detach()
                
                sample_image = self.tex_sampler(uv_batch, tex)
                pred_image_mask = self.face_unet(sample_image) * (1 - bg_batch)
                in_im_pred = pred_image_mask + bg_erode * align_batch
                # pred_image = self.blender(in_im_pred)
                pred_image = self.blender(in_im_pred) * (1 - bg_erode) + bg_erode * align_batch

                pred_img_batch[idx:idx + len(pred_image)] = pred_image.cpu()
                idx += len(pred_image)
                break

            pred_img_batch = pred_img_batch.detach()
            pred_img_batch = torch.flip(pred_img_batch, dims = [1])

            for i in range(len(pred_img_batch)):
                torchvision.utils.save_image(pred_img_batch[i], os.path.join(self.log_path, "{}_{}.png".format(iter_idx, i)), normalize = True, range = (-1, 1))

        self.tex_sampler.train()
        self.face_unet.train()
        self.blender.train()

    def train(self):
        self.tex_sampler.train()
        self.face_unet.train()
        self.blender.train()

        iter_idx = 0
        p_bar = tqdm(total = self.iterations)
        trainloader_iter = iter(self.trainloader)
        loss_g_meter = AverageMeter()
        loss_pred_mask_meter = AverageMeter()
        loss_pred_mouth_meter = AverageMeter()
        loss_pred_meter = AverageMeter()
        loss_lap_meter = AverageMeter()

        while iter_idx < self.iterations:
            try:
                uv_batch, bg_batch, align_batch, mouth_batch = next(trainloader_iter)
            except Exception:
                trainloader_iter = iter(self.trainloader)
                uv_batch, bg_batch, align_batch, mouth_batch = next(trainloader_iter)


            uv_batch = uv_batch.to(self.device)
            bg_batch = bg_batch.to(self.device)
            align_batch = align_batch.to(self.device)
            mouth_batch = mouth_batch.to(self.device)
            bg_erode = self.bgerode(bg_batch).detach()

            tex = self.neural_texture.repeat(len(align_batch), 1, 1, 1)
            sample_image = self.tex_sampler(uv_batch, tex)
            pred_image_mask = self.face_unet(sample_image) * (1 - bg_batch)
            mask_image = align_batch * (1 - bg_batch)
            in_im_pred = pred_image_mask + bg_erode * align_batch
            # pred_image = self.blender(in_im_pred)
            pred_image = self.blender(in_im_pred) * (1 - bg_erode) + bg_erode * align_batch
            pred_mouth = pred_image * mouth_batch
            real_mouth = align_batch * mouth_batch

            self.optimizer_G.zero_grad()
            loss_pred_mask = torch.mean(self.criterion_vgg(pred_image_mask, mask_image) * 10) + self.criterion_l1(pred_image_mask, mask_image) * 10
            loss_pred_mouth = self.criterion_l1(pred_mouth, real_mouth) * 10
            loss_pred = torch.mean(self.criterion_vgg(pred_image, align_batch) * 10) + self.criterion_l1(pred_image, align_batch) * 10
            loss_pred_lap = self.criterion_lap(pred_image, align_batch) * 50

            loss_g = loss_pred + loss_pred_mouth + loss_pred_mask + loss_pred_lap

            loss_g.backward()
            self.optimizer_G.step()

            loss_g_meter.update(loss_g.item())
            loss_pred_meter.update(loss_pred.item())
            loss_pred_mouth_meter.update(loss_pred_mouth.item())
            loss_pred_mask_meter.update(loss_pred_mask.item())
            loss_lap_meter.update(loss_pred_lap.item())

            if iter_idx % self.board_info_every == 0 and iter_idx != 0:
                self.writer.add_scalar('loss_g', loss_g_meter(), iter_idx)
                self.writer.add_scalar('loss_pred', loss_pred_meter(), iter_idx)
                self.writer.add_scalar('loss_pred_mask', loss_pred_mask_meter(), iter_idx)
                self.writer.add_scalar('loss_pred_mouth', loss_pred_mouth_meter(), iter_idx)
                self.writer.add_scalar('loss_lap', loss_lap_meter(), iter_idx)
                # tqdm.write('[{}] loss_g: {:.5f} loss_mouth: {:.5f} loss_pred: {:.5f} loss_mask: {:.5f} loss_lap: {:.5f}'.format(
                #     iter_idx, loss_g_meter(), loss_pred_mouth_meter(), loss_pred_meter(), loss_pred_mask_meter() ,loss_lap_meter()))

            # if iter_idx % self.test_every == 0 and iter_idx != 0:
            if iter_idx % self.test_every == 0:
                # self.test()
                # self.save_state(self.blender, os.path.join(self.model_path, 'blender.pkl'))
                # self.save_state(self.face_unet, os.path.join(self.model_path, 'face_unet.pkl'))
                # torch.save(self.neural_texture, os.path.join(self.model_path, 'neural_texture.pkl'))
                self.test_step(iter_idx)

            if iter_idx in self.milestones:
                self.schedule_lr()

            iter_idx += 1
            p_bar.update(1)

        p_bar.close()

class NVPTrainer(object):
    def __init__(self, args, in_data):
        self.length = args.train_len
        train_uv, train_align, train_bg, train_mouth = in_data['train_uv'], in_data['train_align'], in_data['train_bg'], in_data['train_mouth']
        test_uv, test_align, test_bg = in_data['test_uv'], in_data['test_align'], in_data['test_bg']

        self.trainset = TuneTrainset(train_uv, train_bg, train_align, train_mouth)
        self.testset = TuneTestset(test_uv, test_bg, test_align)

        self.batch_size = args.batch
        self.trainloader = DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True,
                                     pin_memory = True, drop_last = True, num_workers = 16, worker_init_fn=worker_init_fn)
        self.testloader = DataLoader(self.testset, batch_size = 16, shuffle = False,
                                     pin_memory = True, drop_last = False, num_workers = 16, worker_init_fn=worker_init_fn)

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
        self.test_every = 500

        self.milestones = set([5000])
        self.iterations = args.it

        if torch.cuda.device_count() > 1:
            self.multi_gpu = True
        else:
            self.multi_gpu = False

        # 参考nvp的实现
        self.tex_sampler = model.TexSampler().to(self.device)
        self.face_unet = model.UNetnop(args.texture_dim, 3).to(self.device)
        self.blender = model.UNetnop(3, 3).to(self.device)

        self.criterion_l1 = nn.L1Loss().to(self.device)
        self.criterion_vgg = model.VGGLoss().to(self.device)

        if self.multi_gpu:
            self.tex_sampler = nn.DataParallel(self.tex_sampler)
            self.face_unet = nn.DataParallel(self.face_unet)
            self.blender = nn.DataParallel(self.blender)
            self.criterion_vgg = nn.DataParallel(self.criterion_vgg)


        self.neural_texture = torch.randn((16, 256, 256), dtype = torch.float).cuda().requires_grad_(True)

        self.optimizer_G = optim.Adam([
            {'params': self.tex_sampler.parameters()},
            {'params': self.face_unet.parameters()},
            {'params': self.blender.parameters()},
            {'params': self.neural_texture}
        ], lr = args.lr, betas = (0.5, 0.999))

        print(self.optimizer_G)

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

        for params in self.optimizer_G.param_groups:
            params['lr'] *= 0.5
        
        print(self.optimizer_G)

    def test(self):
        self.tex_sampler.eval()
        self.face_unet.eval()
        self.blender.eval()

        with torch.no_grad():
            testloader_iter = iter(self.testloader)
            pred_img_batch = torch.zeros((len(self.testset), 3, 224, 224)).float()
            idx = 0
            while True:
                try:
                    uv_batch, bg_batch, align_batch = next(testloader_iter)
                except Exception:
                    break
                uv_batch = uv_batch.to(self.device)
                bg_batch = bg_batch.to(self.device)
                align_batch = align_batch.to(self.device)
                tex = self.neural_texture.repeat(len(align_batch), 1, 1, 1)
                
                sample_image = self.tex_sampler(uv_batch, tex)
                pred_image_mask = self.face_unet(sample_image) * (1 - bg_batch)
                mask_image = align_batch * (1 - bg_batch)
                in_im_pred = pred_image_mask + bg_batch * align_batch
                pred_image = self.blender(in_im_pred)

                pred_img_batch[idx:idx + len(pred_image)] = pred_image.cpu()
                idx += len(pred_image)

            pred_img_batch = pred_img_batch.detach()
            pred_img_batch = torch.flip(pred_img_batch, dims = [1])

            os.system("rm ../data/tmp/test/*")
            for i in range(len(pred_img_batch)):
                torchvision.utils.save_image(pred_img_batch[i], "../data/tmp/test/{}_{}.png".format(i, self.name), normalize = True, range = (-1, 1))
            os.system("ffmpeg -y -loglevel warning -framerate 25 -start_number 0 -i ../data/tmp/test/%d_{}.png -c:v libx264 -pix_fmt yuv420p -b:v 2000k ./test_{}.mp4".format(self.name, self.name))

        self.tex_sampler.train()
        self.face_unet.train()
        self.blender.train()

    def train(self):
        self.tex_sampler.train()
        self.face_unet.train()
        self.blender.train()

        iter_idx = 0
        p_bar = tqdm(total = self.iterations)
        trainloader_iter = iter(self.trainloader)
        loss_g_meter = AverageMeter()
        loss_pred_mask_meter = AverageMeter()
        loss_pred_mouth_meter = AverageMeter()
        loss_pred_meter = AverageMeter()

        while iter_idx < self.iterations:
            try:
                uv_batch, bg_batch, align_batch, mouth_batch = next(trainloader_iter)
            except Exception:
                trainloader_iter = iter(self.trainloader)
                uv_batch, bg_batch, align_batch, mouth_batch = next(trainloader_iter)


            uv_batch = uv_batch.to(self.device)
            bg_batch = bg_batch.to(self.device)
            align_batch = align_batch.to(self.device)
            mouth_batch = mouth_batch.to(self.device)

            tex = self.neural_texture.repeat(len(align_batch), 1, 1, 1)
            sample_image = self.tex_sampler(uv_batch, tex)
            pred_image_mask = self.face_unet(sample_image) * (1 - bg_batch)
            mask_image = align_batch * (1 - bg_batch)
            in_im_pred = pred_image_mask + bg_batch * align_batch
            pred_image = self.blender(in_im_pred)
            pred_mouth = pred_image * mouth_batch
            real_mouth = align_batch * mouth_batch

            self.optimizer_G.zero_grad()
            loss_pred_mask = torch.mean(self.criterion_vgg(pred_image_mask, mask_image) * 10) + self.criterion_l1(pred_image_mask, mask_image) * 10
            loss_pred_mouth = self.criterion_l1(pred_mouth, real_mouth) * 10
            loss_pred = torch.mean(self.criterion_vgg(pred_image, align_batch) * 10) + self.criterion_l1(pred_image, align_batch) * 10

            loss_g = loss_pred + loss_pred_mouth + loss_pred_mask

            loss_g.backward()
            self.optimizer_G.step()

            loss_g_meter.update(loss_g.item())
            loss_pred_meter.update(loss_pred.item())
            loss_pred_mouth_meter.update(loss_pred_mouth.item())
            loss_pred_mask_meter.update(loss_pred_mask.item())

            if iter_idx % self.board_info_every == 0 and iter_idx != 0:
                self.writer.add_scalar('loss_g', loss_g_meter(), iter_idx)
                self.writer.add_scalar('loss_pred', loss_pred_meter(), iter_idx)
                self.writer.add_scalar('loss_pred_mask', loss_pred_mask_meter(), iter_idx)
                self.writer.add_scalar('loss_pred_mouth', loss_pred_mouth_meter(), iter_idx)
                tqdm.write('[{}] loss_g: {:.5f} loss_mouth: {:.5f} loss_pred: {:.5f} loss_mask: {:.5f}'.format(
                    iter_idx, loss_g_meter(), loss_pred_mouth_meter(), loss_pred_meter(), loss_pred_mask_meter()))

            if iter_idx % self.test_every == 0 and iter_idx != 0:
                self.test()
                self.save_state(self.blender, os.path.join(self.model_path, 'blender.pkl'))
                self.save_state(self.face_unet, os.path.join(self.model_path, 'face_unet.pkl'))
                torch.save(self.neural_texture, os.path.join(self.model_path, 'neural_texture.pkl'))

            if iter_idx in self.milestones:
                self.schedule_lr()

            iter_idx += 1
            p_bar.update(1)

        p_bar.close()