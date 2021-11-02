import deep_3drecon
import face_alignment
import cv2
import os
import numpy as np
import librosa
import torch
import math
import audio2motion
import render
import torch.nn as nn
import torchvision
import argparse
from align_img import align_lm68, align

parser = argparse.ArgumentParser()
parser.add_argument("--in_img", help = "input portrait", default='example/example.png', type=str)
parser.add_argument("--in_audio", help="input audio", default='example/example.wav', type=str)
parser.add_argument("--output_path", help="output path of videos", default='output', type=str)
conf = parser.parse_args()

def lm68_2_lm5(in_lm):
    lm_idx = np.array([31,37,40,43,46,49,55]) - 1
    lm = np.stack([in_lm[lm_idx[0],:],np.mean(in_lm[lm_idx[[1,2]],:],0),np.mean(in_lm[lm_idx[[3,4]],:],0),in_lm[lm_idx[5],:],in_lm[lm_idx[6],:]], axis = 0)
    lm = lm[[1,2,0,3,4],:2]
    return lm

def recon_texture(uv_file, img_file, out_path):
    uv_img = cv2.imread(uv_file).astype(np.int32)
    img = cv2.imread(img_file).astype(np.int32)
    
    x = uv_img[:, :, 0].reshape(-1)
    y = uv_img[:, :, 1].reshape(-1)
    index = y * 256 + x
    img = img.reshape(-1, 3)
    texture = np.zeros((256 * 256, 3), dtype = np.int32)
    texture_count = np.zeros((256 * 256), dtype = np.int32)
    
    np.add.at(texture_count, index, 1)
    np.add.at(texture, index, img)
    texture_count[texture_count == 0] = 1
    texture = texture / np.expand_dims(texture_count, 1)

    texture = texture.reshape(256, 256, 3)
    cv2.imwrite(out_path, texture)


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        frame_list.append(frame)
    return np.array(frame_list)

os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
img_path = conf.in_img
audio_path = conf.in_audio
tmp_path = conf.output_path

if len(os.listdir(tmp_path)) != 0:
    raise Exception("Output path must be empty")

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, network_size=4, device='cuda')
face_reconstructor = deep_3drecon.Reconstructor()
im_bgr = cv2.imread(img_path)
im_rgb = im_bgr[:, :, ::-1]
lm68 = fa.get_landmarks(im_rgb)[0]
lm5 = lm68_2_lm5(lm68)
coeff, align_img = face_reconstructor.recon_coeff(np.array([im_bgr]), np.array([lm5]), return_image = True)
# Due to one bug, we use tex_noise.png in our implementation, currently the tex_no_noise.mp4 is not used, which will be fixed further.
face_reconstructor.recon_texture_from_coeff(coeff, align_img, os.path.join(tmp_path, "tex_no_noise.mp4"), tmp_dir = tmp_path)
face_reconstructor.recon_uv_from_coeff(coeff, os.path.join(tmp_path, "uv_src.mp4"), tmp_dir = tmp_path)

align_img = align_img[0]
cv2.imwrite(os.path.join(tmp_path, "align.png"), align_img)
uv_img = read_video(os.path.join(tmp_path, "uv_src.mp4"))[0]
recon_texture(os.path.join(tmp_path, "0.png"), os.path.join(tmp_path, "align.png"), os.path.join(tmp_path, "tex_noise.png"))
tex_img = cv2.imread(os.path.join(tmp_path, "tex_noise.png")).astype(np.int32)

del fa
torch.cuda.empty_cache()

import deepspeech
deepspeech_prob = deepspeech.get_prob(audio_path)
del deepspeech
audio, sr = librosa.load(audio_path, sr = 16000)
audio_energy = librosa.feature.rms(y = audio, frame_length = 512, hop_length = 320, center = False)
audio_energy = np.transpose(audio_energy)
coeff_len = len(deepspeech_prob) // 2
coeff_array = np.tile(coeff, (coeff_len, 1))

y_len = len(coeff_array)
audio_clip_list = []
energy_clip_list = []

def get_sync_data(x, x_dim, y_left, y_right):
    x_len = len(x)
    x_left = math.floor(y_left * 50 / 25)
    x_right = math.floor(y_right * 50 / 25)
    pad_len = 80 - x_right + x_left

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

for i in range(0, y_len, 8):
    if i > y_len - 32:
        y_left = y_len - 32
        y_right = y_len
    else:
        y_left = i
        y_right = i + 32

    audio_clip = get_sync_data(deepspeech_prob, 29, y_left, y_right)
    energy_clip = get_sync_data(audio_energy, 1, y_left, y_right)

    audio_clip_list.append(audio_clip)
    energy_clip_list.append(energy_clip)

audio_clip_list = np.array(audio_clip_list)
energy_clip_list = np.array(energy_clip_list)

audio_batch = torch.from_numpy(audio_clip_list).transpose(1, 2).float().cuda()
energy_batch = torch.from_numpy(energy_clip_list).transpose(1, 2).float().cuda()
sty_batch = torch.from_numpy(np.load("./example/sty.npy")).float().cuda()

torch.cuda.empty_cache()

# os.environ["CUDA_VISIBLE_DEVICES"]=str(1)
model = audio2motion.StyleFusionModel().cuda()
model.load_state_dict(torch.load("./audio2motion/model/backbone.pkl"), strict = True)
model.eval()

with torch.no_grad():
    for idx in range(10):
        sty_tmp = sty_batch[idx].unsqueeze(0).repeat(len(audio_batch), 1)
        pred_exp_batch = model(audio_batch, energy_batch, sty_tmp)

        y_len = len(coeff_array)
        y_repeat = torch.zeros(y_len).int().cuda()
        predict_exp_cat = torch.zeros((y_len, 64)).float().cuda()
        for counter, i in enumerate(range(0, y_len, 8)):
            if i > y_len - 32:
                y_left = y_len - 32
                y_right = y_len
            else:
                y_left = i
                y_right = i + 32
            y_repeat[y_left: y_right] += 1
            predict_exp_cat[y_left: y_right] += pred_exp_batch[counter].transpose(0, 1)
        y_repeat = y_repeat.float()
        predict_exp_cat = predict_exp_cat / y_repeat.unsqueeze(1)
        coeff_array[:, 80:144] = predict_exp_cat.detach().cpu().numpy()
        np.save(os.path.join(tmp_path, "test_{}.npy".format(idx)), coeff_array)
        face_reconstructor.recon_uv_from_coeff(coeff_array, 
            out_path = os.path.join(tmp_path, "test_uv_{}.mp4".format(idx)), 
            bg_path = os.path.join(tmp_path, "test_bg_{}.mp4".format(idx)), 
            tmp_dir = tmp_path
        )

del model
del face_reconstructor
del audio_batch
del energy_batch
del sty_batch
del coeff_array
del predict_exp_cat
torch.cuda.empty_cache()

tex_encoder = render.UNet(9, 16).cuda()
tex_sampler = render.TexSampler().cuda()
face_unet = render.define_G(16, 3, 64, 'local').cuda()
tex_encoder.load_state_dict(torch.load("./render/model/tex_encoder.pkl"), strict = True)
face_unet.load_state_dict(torch.load("./render/model/face_unet.pkl"), strict = True)
tex_encoder.eval()
tex_sampler.eval()
face_unet.eval()
tex_img = torch.from_numpy(tex_img).float().permute(2, 0, 1) / 128 - 1
tex_img_batch = tex_img.unsqueeze(0).repeat(3, 1, 1, 1).cuda()

with torch.no_grad():
    batch_size = 8
    for idx in range(10):
        uv_path = os.path.join(tmp_path, "test_uv_{}.mp4".format(idx))
        bg_path = os.path.join(tmp_path, "test_bg_{}.mp4".format(idx))
        bg_frames = np.array(read_video(bg_path))
        uv_frames = np.array(read_video(uv_path))
        uv_frames = torch.from_numpy(uv_frames).float().permute(0, 3, 1, 2) / 255
        uv_img_batch = uv_frames[:, :2]
        bg_img_batch = torch.from_numpy(bg_frames > 127).float()[:, :, :, 0].unsqueeze(1)
        bg_img_batch, uv_img_batch, tex_img_batch = bg_img_batch.unsqueeze(0), uv_img_batch.unsqueeze(0), tex_img_batch.unsqueeze(0)
        tex_img_batch = tex_img_batch.reshape(tex_img_batch.shape[0], -1, tex_img_batch.shape[3], tex_img_batch.shape[4]).cuda()
        tex = tex_encoder(tex_img_batch)
        pred_img_batch = torch.zeros((uv_img_batch.shape[0], uv_img_batch.shape[1], 3, uv_img_batch.shape[3], uv_img_batch.shape[4])).float()
        start_idx = 0
        while start_idx < pred_img_batch.shape[1]:
            if start_idx + batch_size > pred_img_batch.shape[1]:
                end_idx = pred_img_batch.shape[1]
            else:
                end_idx = start_idx + batch_size
            bg_tmp_batch = bg_img_batch[:, start_idx: end_idx].cuda()
            uv_tmp_batch = uv_img_batch[:, start_idx: end_idx].cuda()
            bg_tmp_batch = bg_tmp_batch.reshape(-1, bg_tmp_batch.shape[2], bg_tmp_batch.shape[3], bg_tmp_batch.shape[4])
            uv_tmp_batch = uv_tmp_batch.reshape(-1, uv_tmp_batch.shape[2], uv_tmp_batch.shape[3], uv_tmp_batch.shape[4])
            tex_tmp = tex.unsqueeze(1).repeat(1, uv_tmp_batch.shape[0], 1, 1, 1)
            tex_tmp = tex_tmp.reshape(-1, tex_tmp.shape[2], tex_tmp.shape[3], tex_tmp.shape[4])
            sample_image = tex_sampler(uv_tmp_batch, tex_tmp)
            pred_image = face_unet(sample_image) * (1 - bg_tmp_batch)
            pred_img_batch[:, start_idx:end_idx] = pred_image.cpu()
            start_idx += batch_size
        pred_img_batch = pred_img_batch[0].cpu().detach()
        pred_img_batch = torch.flip(pred_img_batch, dims = [1])
        os.system("rm {}".format(os.path.join(tmp_path, "*.png")))
        for i in range(len(pred_img_batch)):
            torchvision.utils.save_image(pred_img_batch[i], "./{}/{}.png".format(tmp_path, i), normalize = True, range = (-1, 1))
        os.system("ffmpeg -y -loglevel warning -framerate 25 -start_number 0 -i {}/%d.png -c:v libx264 -pix_fmt yuv420p -b:v 2000k {}/render_{}.mp4".format(tmp_path, tmp_path, idx))
        os.system("ffmpeg -y -loglevel warning -i {} -i {} -map 0:v -map 1:a -c:v copy -shortest {}".format(
            "{}/render_{}.mp4".format(tmp_path, idx),
            audio_path,
            "{}/result_{}.mp4".format(tmp_path, idx),
        ))

os.system("rm {}".format(os.path.join(tmp_path, "*.png")))
os.system("rm {}".format(os.path.join(tmp_path, "test_*")))
os.system("rm {}".format(os.path.join(tmp_path, "render_*")))
os.system("rm {}".format(os.path.join(tmp_path, "tex_no_noise.mp4")))
os.system("rm {}".format(os.path.join(tmp_path, "uv_src.mp4")))