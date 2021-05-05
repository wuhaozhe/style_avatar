import tensorflow as tf 
import numpy as np
import cv2
import os
import time
from scipy.io import loadmat,savemat
from tqdm import tqdm
from .preprocess_img import Preprocess_cv2
from .load_data import *
from .face_decoder import Face3D

def load_graph(graph_filename):
	with tf.gfile.GFile(graph_filename,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def

if not os.path.isfile('./BFM/BFM_model_front.mat'):
	transferBFM09()

lm3D = load_lm3d()
batchsize = 128

g = tf.Graph()

with g.as_default() as graph,tf.device('/gpu:0'):
    FaceReconstructor = Face3D()
    images = tf.placeholder(name = 'input_imgs', shape = [batchsize,224,224,3], dtype = tf.float32)
    graph_def = load_graph('network/FaceReconModel.pb')
    tf.import_graph_def(graph_def,name='resnet',input_map={'input_imgs:0': images})

    # output coefficients of R-Net (dim = 257) 
    coeff = graph.get_tensor_by_name('resnet/coeff:0')

    # reconstructing faces
    id_coeff,ex_coeff,tex_coeff,angles,translation,gamma = FaceReconstructor.Split_coeff(coeff)
    FaceReconstructor.Reconstruction_Block(coeff,batchsize)
    face_shape = FaceReconstructor.face_shape_t
    face_texture = FaceReconstructor.face_texture
    face_color = FaceReconstructor.face_color
    landmarks_2d = FaceReconstructor.landmark_p
    recon_img = FaceReconstructor.render_imgs
    tri = FaceReconstructor.facemodel.face_buf
    uv_img = FaceReconstructor.render_uvs

def pad_batch(frame_batch, lm_batch):
    frame_zeros = np.ones((batchsize - len(frame_batch), frame_batch.shape[1], frame_batch.shape[2], frame_batch.shape[3]), dtype = np.uint8)
    lm_zeros = np.tile(np.expand_dims(lm_batch[0], 0), (batchsize - len(lm_batch), 1, 1))
    frame_batch = np.concatenate((frame_batch, frame_zeros), axis=0)
    lm_batch = np.concatenate((lm_batch, lm_zeros), axis=0)

    return frame_batch, lm_batch

def pad_coeff(coeff_batch):
    coeff_zeros = np.tile(np.expand_dims(coeff_batch[0], 0), (batchsize - len(coeff_batch), 1))
    coeff_batch = np.concatenate((coeff_batch, coeff_zeros), axis=0)
    return coeff_batch

def recon_video(frame_array, lm_array):
    with g.as_default() as graph, tf.device('/gpu:0'):
        with tf.Session() as sess:
            idx = 0
            while idx < len(frame_array):
                if idx + batchsize <= len(frame_array):
                    end_idx = batchsize
                    frame_batch = frame_array[idx: idx + batchsize]
                    lm_batch = lm_array[idx: idx + batchsize]
                else:
                    # pad
                    end_idx = len(frame_array) - idx
                    frame_batch, lm_batch = pad_batch(frame_array[idx: ], lm_array[idx: ])

                input_img_list, lm_new_list, transform_param_list = [], [], []
                for i in range(batchsize):
                    input_img,lm_new,transform_params = Preprocess_cv2(frame_batch[i],lm_batch[i],lm3D)
                    input_img_list.append(input_img)
                    lm_new_list.append(lm_new)
                    transform_param_list.append(transform_params)
                
                input_img_array = np.concatenate(input_img_list, axis = 0)
                lm_new_array = np.array(lm_new_list)
                transform_param_array = np.array(transform_param_list)
                coeff_, id_coeff_, ex_coeff_, tex_coeff_, angles_, translation_, gamma_ = sess.run([coeff,\
					id_coeff,ex_coeff,tex_coeff,angles,translation,gamma],feed_dict = {images: input_img_array})
                
                if idx == 0:
                    coeff_array = coeff_[:end_idx]
                else:
                    coeff_array = np.concatenate((coeff_array, coeff_[:end_idx]), axis = 0)

                # recon_img_ = sess.run([recon_img], feed_dict = {images: input_img_array})
                # recon_img_ = recon_img_[0]
                # if idx == 0:
                #     recon_img_list = recon_img_[:end_idx, :, :]
                # else:
                #     recon_img_list = np.concatenate((recon_img_list, recon_img_[:end_idx, :, :]), axis = 0)
                idx += batchsize
                

            # os.system("rm ./test/*.jpg")
            # for i in range(len(recon_img_list)):
            #     cv2.imwrite("./test/{}.jpg".format(i), recon_img_list[i][:, :, :3])

            # os.system("ffmpeg -loglevel warning -framerate 25 -start_number 0 -i ./test/%d.jpg -c:v libx264 -b:v 2000k test.mp4")

            return coeff_array

def recon_uv(frame_array, lm_array, out_folder_path):
    
    with g.as_default() as graph, tf.device('/gpu:0'):
        with tf.Session() as sess:
            idx = 0
            while idx < len(frame_array):
                if idx + batchsize <= len(frame_array):
                    end_idx = batchsize
                    frame_batch = frame_array[idx: idx + batchsize]
                    lm_batch = lm_array[idx: idx + batchsize]
                else:
                    # pad
                    end_idx = len(frame_array) - idx
                    frame_batch, lm_batch = pad_batch(frame_array[idx: ], lm_array[idx: ])

                input_img_list, lm_new_list, transform_param_list = [], [], []
                for i in range(batchsize):
                    input_img,lm_new,transform_params = Preprocess_cv2(frame_batch[i],lm_batch[i],lm3D)
                    input_img_list.append(input_img)
                    lm_new_list.append(lm_new)
                    transform_param_list.append(transform_params)
                
                input_img_array = np.concatenate(input_img_list, axis = 0)
                lm_new_array = np.array(lm_new_list)
                transform_param_array = np.array(transform_param_list)
                coeff_, id_coeff_, ex_coeff_, tex_coeff_, angles_, translation_, gamma_, uv_img_ = sess.run([coeff,\
					id_coeff,ex_coeff,tex_coeff,angles,translation,gamma,uv_img],feed_dict = {images: input_img_array})
                
                if idx == 0:
                    coeff_array = coeff_[:end_idx]
                    input_array = input_img_array[:end_idx]
                    output_array = uv_img_[:end_idx]
                else:
                    coeff_array = np.concatenate((coeff_array, coeff_[:end_idx]), axis = 0)
                    input_array = np.concatenate((input_array, input_img_array[:end_idx]), axis = 0)
                    output_array = np.concatenate((output_array, uv_img_[:end_idx]), axis = 0)

                # recon_img_ = sess.run([recon_img], feed_dict = {images: input_img_array})
                # recon_img_ = recon_img_[0]
                # if idx == 0:
                #     recon_img_list = recon_img_[:end_idx, :, :]
                # else:
                #     recon_img_list = np.concatenate((recon_img_list, recon_img_[:end_idx, :, :]), axis = 0)
                idx += batchsize
                

            # os.system("rm ./test/*.jpg")
            # for i in range(len(recon_img_list)):
            #     cv2.imwrite("./test/{}.jpg".format(i), recon_img_list[i][:, :, :3])

            # os.system("ffmpeg -loglevel warning -framerate 25 -start_number 0 -i ./test/%d.jpg -c:v libx264 -b:v 2000k test.mp4")

            for i in range(len(input_array)):
                cv2.imwrite(os.path.join(out_folder_path, "{}.png".format(i)), input_array[i])
                tmp = output_array[i][::-1, :, :]
                cv2.imwrite(os.path.join(out_folder_path, "{}_uv.png".format(i)), tmp)
                num_labels, labels = cv2.connectedComponents((tmp[:, :, 0] == 0).astype(np.uint8))
                cv2.imwrite(os.path.join(out_folder_path, "{}_bg.png".format(i)), (labels == 1).astype(np.uint8) * 255)
                recon_texture(os.path.join(out_folder_path, "{}_uv.png".format(i)), os.path.join(out_folder_path, "{}.png".format(i)))
                # print(num_labels, np.sum(labels == 0), np.sum(labels == 1), np.sum(labels == 2))
            
            return coeff_array

def recon_coeff_uv(coeff_array, out_path, out_name):
    with g.as_default() as graph, tf.device('/gpu:0'):
        with tf.Session() as sess:
            idx = 0
            while idx < len(coeff_array):
                if idx + batchsize <= len(coeff_array):
                    end_idx = batchsize
                    coeff_batch = coeff_array[idx: idx + batchsize]
                else:
                    # pad
                    end_idx = len(coeff_array) - idx
                    coeff_batch = pad_coeff(coeff_array[idx: ])

                uv_img_ = sess.run([uv_img],feed_dict = {coeff: coeff_batch})
                uv_img_ = uv_img_[0]
                
                if idx == 0:
                    uv_array = uv_img_[:end_idx]
                else:
                    uv_array = np.concatenate((uv_array, uv_img_[:end_idx]), axis = 0)

                idx += batchsize
                
            os.system("rm ./test/*.png")

            for i in range(len(uv_array)):
                tmp = uv_array[i][::-1, :, :]
                cv2.imwrite(os.path.join("./test", "{}_uv.png".format(i)), tmp)
                num_labels, labels = cv2.connectedComponents((tmp[:, :, 0] == 0).astype(np.uint8))
                cv2.imwrite(os.path.join("./test", "{}_bg.png".format(i)), (labels == 1).astype(np.uint8) * 255)
                # print(num_labels, np.sum(labels == 0), np.sum(labels == 1), np.sum(labels == 2))
            
            os.system("ffmpeg -loglevel warning -y -framerate 25 -start_number 0 -i ./test/%d_uv.png -c:v libx264 -pix_fmt yuv420p -b:v 1000k {}_uv.mp4".format(os.path.join(out_path, out_name)))
            os.system("ffmpeg -loglevel warning -y -framerate 25 -start_number 0 -i ./test/%d_bg.png -c:v libx264 -pix_fmt yuv420p -b:v 1000k {}_bg.mp4".format(os.path.join(out_path, out_name)))

# given uv img array and img array, generate texture image
def recon_texture(uv_file, img_file):
    # print(uv_file, img_file)
    uv_img = cv2.imread(uv_file).astype(np.int32)
    img = cv2.imread(img_file).astype(np.int32)
    
    x = uv_img[:, :, 0].reshape(-1)
    y = uv_img[:, :, 1].reshape(-1)
    index = y * 256 + x
    img = img.reshape(-1, 3)
    texture = np.zeros((256 * 256, 3), dtype = np.int32)
    texture_count = np.zeros((256 * 256), dtype = np.int32)
    
    # for i in range(len(index)):
    #     texture_count[index[i]] += 1
    np.add.at(texture_count, index, 1)
    np.add.at(texture, index, img)
    texture_count[texture_count == 0] = 1
    texture = texture / np.expand_dims(texture_count, 1)

    texture = texture.reshape(256, 256, 3)
    cv2.imwrite(uv_file[:-6] + 'tex.png', texture)
    
    # print((uv_img / 128 - 1)[112])

    # import torch
    # import torch.nn.functional as F
    # from torchvision import transforms
    # totensor = transforms.ToTensor()
    # uv_img = totensor(Image.open(uv_file))[1:] * 2 - 1
    # texture_img = totensor(Image.open("/home/wuhz/mnt/avatar/dataset/ted-2020/clip_video_sep3/0/0/1_tex.png"))
    # uv_img = uv_img.permute(1,2,0).unsqueeze(0)
    # print(uv_img[0, 112, :])
    # texture_img = texture_img.unsqueeze(0)
    # uv_img = torch.from_numpy(uv_img[:, :, :2]).unsqueeze(0).float()
    # uv_img = uv_img / 128 - 1
    # print(uv_img)
    # texture = torch.from_numpy(texture).permute(2, 0, 1).unsqueeze(0).float()
    # out_img = F.grid_sample(texture_img, uv_img)[0].permute(1,2,0).detach().numpy() * 255
    # cv2.imwrite("test2.png", out_img)

    # cv2.imwrite("test.png", texture)

    # xandy, count = np.unique(index, return_counts = True)
    # for i in range(8):
    #     print(np.sum(texture_count==i))

    # for i in range(8):
    #     print(len(xandy[count==i]))

def recon_coeff(coeff_array, out_path = "test.mp4"):

    with tf.Graph().as_default() as graph, tf.device('/gpu:0'):
        FaceReconstructor = Face3D()
        batchsize = 1
        coeff = tf.placeholder(name = 'input_coeff', shape = [batchsize, 257], dtype = tf.float32)

        # reconstructing faces
        FaceReconstructor.Reconstruction_Block(coeff, batchsize)
        recon_img = FaceReconstructor.render_imgs

        with tf.Session() as sess:
            image_array = []
            os.system("rm ./test/*.jpg")
            for idx, coeff_ in tqdm(enumerate(coeff_array)):
                recon_img_ = sess.run([recon_img], feed_dict = {coeff: np.expand_dims(coeff_, 0)})
                recon_img_ = recon_img_[0][0]
                image_array.append(recon_img_)
                cv2.imwrite("./test/{}.jpg".format(idx), recon_img_[:, :, :3])

            os.system("ffmpeg -loglevel warning -framerate 25 -start_number 0 -i ./test/%d.jpg -c:v libx264 -b:v 2000k {}".format(out_path))

def recon_coeff_notex(coeff_array, out_path = "test.mp4"):

    with tf.Graph().as_default() as graph, tf.device('/gpu:0'):
        FaceReconstructor = Face3D()
        batchsize = 1
        coeff = tf.placeholder(name = 'input_coeff', shape = [batchsize, 257], dtype = tf.float32)
        texture = tf.placeholder(name = 'input_texture', shape = [batchsize, 35709, 3], dtype = tf.float32)

        # reconstructing faces
        FaceReconstructor.Reconstruction_Block(coeff, batchsize, texture)
        recon_img = FaceReconstructor.render_imgs
        with tf.Session() as sess:
            image_array = []
            os.system("rm ./test/*.jpg")
            norm_texture = np.ones((batchsize, 35709, 3), dtype = np.float32) * 128
            for idx, coeff_ in tqdm(enumerate(coeff_array)):
                recon_img_ = sess.run([recon_img], feed_dict = {coeff: np.expand_dims(coeff_, 0), texture: norm_texture})
                recon_img_ = recon_img_[0][0]
                image_array.append(recon_img_)
                cv2.imwrite("./test/{}.jpg".format(idx), recon_img_[:, :, :3][:, :, ::-1])

            os.system("ffmpeg -y -loglevel warning -framerate 25 -start_number 0 -i ./test/%d.jpg -c:v libx264 -b:v 2000k {}".format(out_path))

def recon_lm(coeff_array):

    with tf.Graph().as_default() as graph, tf.device('/gpu:0'):
        FaceReconstructor = Face3D()
        batchsize = 1
        coeff = tf.placeholder(name = 'input_coeff', shape = [batchsize, 257], dtype = tf.float32)

        # reconstructing faces
        FaceReconstructor.Reconstruction_Block(coeff, batchsize)
        landmark_p = FaceReconstructor.landmark_p

        landmark_array = []

        with tf.Session() as sess:
            for idx, coeff_ in tqdm(enumerate(coeff_array)):
                landmark_p_ = sess.run([landmark_p], feed_dict = {coeff: np.expand_dims(coeff_, 0)})
                landmark_p_ = landmark_p_[0][0]
                landmark_array.append(landmark_p_)
        
        landmark_array = np.array(landmark_array)
        
        return landmark_array


def recon_coeff_single(coeff_array, max_len):

    with tf.Graph().as_default() as graph, tf.device('/gpu:0'):
        FaceReconstructor = Face3D()
        batchsize = 1
        coeff = tf.placeholder(name = 'input_coeff', shape = [batchsize, 257], dtype = tf.float32)

        # reconstructing faces
        FaceReconstructor.Reconstruction_Block(coeff, batchsize)
        recon_img = FaceReconstructor.render_imgs

        with tf.Session() as sess:
            image_array = []
            # os.system("rm ./test/*.jpg")
            for idx, coeff_ in tqdm(enumerate(coeff_array)):
                recon_img_ = sess.run([recon_img], feed_dict = {coeff: np.expand_dims(coeff_, 0)})
                recon_img_ = recon_img_[0][0]
                image_array.append(recon_img_)
        return image_array

model = loadmat('BFM/BFM_model_front.mat')
ex_base = np.transpose(model['exBase'].reshape(-1, 3, 64), (0, 2, 1))
lm_68 = model['keypoints'][0].astype(int)
upper_lip_top = ex_base[lm_68[51]]
upper_lip_bottom = ex_base[lm_68[62]]
lower_lip_top = ex_base[lm_68[66]]
lower_lip_bottom = ex_base[lm_68[57]]
left_lip = ex_base[lm_68[48]]
right_lip = ex_base[lm_68[54]]

def get_feat(coeff_array):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    exp_coeff = coeff_array[:, 80:144]
    upper_lip_top_diff = exp_coeff @ upper_lip_top
    print(upper_lip_top_diff[0:100, :])
    upper_lip_bottom_diff = exp_coeff @ upper_lip_bottom
    lower_lip_top_diff = exp_coeff @ lower_lip_top
    lower_lip_bottom_diff = exp_coeff @ lower_lip_bottom
    left_lip_diff = exp_coeff @ left_lip
    right_lip_diff = exp_coeff @ right_lip
    x = np.arange(len(upper_lip_top_diff)) / 25
    plt.plot(x, upper_lip_top_diff[:, 0])
    plt.plot(x, upper_lip_bottom_diff[:, 0])
    plt.savefig('test.jpg')
    # 0是横向轴 (left lip diff与right lip diff分别的均值与方差)
    # 1是上下轴 (left lip diff与right lip diff分别的均值与方差，upper bottom 与 lower top的均值与方差)
    # 2是外突轴 (left lip diff与right lip diff分别均值与方差，upper top 与 lower bottom的均值与方差)

recon_texture("/home/wuhz/mnt/avatar/dataset/ted-2020/clip_video_sep3/0/0/1_uv.png", "/home/wuhz/mnt/avatar/dataset/ted-2020/clip_video_sep3/0/0/1.png")