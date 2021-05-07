import tensorflow as tf 
import numpy as np
import cv2
import os
import time
from scipy.io import loadmat,savemat
from tqdm import tqdm
from .align_img import align
from .face_decoder import Face3D

def load_graph(graph_filename):
	with tf.gfile.GFile(graph_filename,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def

# load landmarks for standard face, which is used for image preprocessing
def load_lm3d():

	Lm3D = loadmat(os.path.join(bfm_dir, 'similarity_Lm3D_all.mat'))
	Lm3D = Lm3D['lm']

	# calculate 5 facial landmarks using 68 landmarks
	lm_idx = np.array([31,37,40,43,46,49,55]) - 1
	Lm3D = np.stack([Lm3D[lm_idx[0],:],np.mean(Lm3D[lm_idx[[1,2]],:],0),np.mean(Lm3D[lm_idx[[3,4]],:],0),Lm3D[lm_idx[5],:],Lm3D[lm_idx[6],:]], axis = 0)
	Lm3D = Lm3D[[1,2,0,3,4],:]

	return Lm3D


bfm_dir = os.path.join(os.path.dirname(__file__), 'BFM')
network_dir = os.path.join(os.path.dirname(__file__), 'network')
lm3D = load_lm3d()
batchsize = 128
g = tf.Graph()

with g.as_default() as graph,tf.device('/gpu:0'):
    FaceReconstructor = Face3D(bfm_dir)
    images = tf.placeholder(name = 'input_imgs', shape = [batchsize,224,224,3], dtype = tf.float32)
    graph_def = load_graph(os.path.join(network_dir, 'FaceReconModel.pb'))
    tf.import_graph_def(graph_def,name='resnet',input_map={'input_imgs:0': images})

    # output coefficients of R-Net (dim = 257) 
    coeff = graph.get_tensor_by_name('resnet/coeff:0')
    # reconstructing faces
    id_coeff,ex_coeff,tex_coeff,angles,translation,gamma = FaceReconstructor.Split_coeff(coeff)
    FaceReconstructor.Reconstruction_Block(coeff,batchsize, None, images)
    face_shape = FaceReconstructor.face_shape_t
    face_texture = FaceReconstructor.face_texture
    face_color = FaceReconstructor.face_color
    landmarks_2d = FaceReconstructor.landmark_p
    recon_img = FaceReconstructor.render_imgs
    tri = FaceReconstructor.facemodel.face_buf
    uv_img = FaceReconstructor.render_uvs
    recon_textures = FaceReconstructor.recon_textures


def pad_frame(frame_batch):
    '''
        pad frame to batchsize
    '''
    frame_zeros = np.ones((batchsize - len(frame_batch), frame_batch.shape[1], frame_batch.shape[2], frame_batch.shape[3]), dtype = np.uint8)
    frame_batch = np.concatenate((frame_batch, frame_zeros), axis=0)
    return frame_batch

def pad_lm(lm_batch):
    '''
        pad landmark to batchsize
    '''
    lm_zeros = np.tile(np.expand_dims(lm_batch[0], 0), (batchsize - len(lm_batch), 1, 1))
    lm_batch = np.concatenate((lm_batch, lm_zeros), axis=0)
    return lm_batch

def pad_coeff(coeff_batch):
    '''
        pad coeff to batchsize
    '''
    coeff_zeros = np.tile(np.expand_dims(coeff_batch[0], 0), (batchsize - len(coeff_batch), 1))
    coeff_batch = np.concatenate((coeff_batch, coeff_zeros), axis=0)
    return coeff_batch

def recon_coeff(frame_array, lm_array, return_image = False):
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
                    frame_batch = pad_frame(frame_array[idx: ])
                    lm_batch = pad_lm(lm_array[idx: ])

                input_img_list = []
                for i in range(batchsize):
                    input_img,lm_new,transform_params = align(frame_batch[i],lm_batch[i],lm3D)
                    input_img_list.append(input_img)
                
                input_img_array = np.concatenate(input_img_list, axis = 0)
                coeff_, id_coeff_, ex_coeff_, tex_coeff_, angles_, translation_, gamma_ = sess.run([coeff,\
					id_coeff,ex_coeff,tex_coeff,angles,translation,gamma],feed_dict = {images: input_img_array})
                
                if idx == 0:
                    input_array = input_img_array[:end_idx]
                    coeff_array = coeff_[:end_idx]
                else:
                    input_array = np.concatenate((input_array, input_img_array[:end_idx]), axis = 0)
                    coeff_array = np.concatenate((coeff_array, coeff_[:end_idx]), axis = 0)

                idx += batchsize

        if return_image:
            return coeff_array, input_array
        else:
            return coeff_array

def recon_video_from_coeff(coeff_array, out_path = "test.mp4", tmp_dir = "./test"):

    with g.as_default() as graph, tf.device('/gpu:0'):
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        FaceReconstructor = Face3D(bfm_dir)
        batchsize = 1
        coeff = tf.placeholder(name = 'input_coeff', shape = [batchsize, 257], dtype = tf.float32)

        FaceReconstructor.Reconstruction_Block(coeff, batchsize)
        recon_img = FaceReconstructor.render_imgs
        with tf.Session() as sess:
            image_array = []
            os.system("rm {}".format(os.path.join(tmp_dir, "*.jpg")))
            for idx, coeff_ in tqdm(enumerate(coeff_array)):
                recon_img_ = sess.run([recon_img], feed_dict = {coeff: np.expand_dims(coeff_, 0)})
                recon_img_ = recon_img_[0][0]
                image_array.append(recon_img_)
                cv2.imwrite("{}/{}.jpg".format(tmp_dir, idx), recon_img_[:, :, :3][:, :, ::-1])

            os.system("ffmpeg -loglevel warning -framerate 25 -start_number 0 -i {}/%d.jpg -c:v libx264 -b:v 2000k {}".format(tmp_dir, out_path))

def recon_video_from_coeff_notex(coeff_array, out_path = "test.mp4", tmp_dir = "./test"):
    with tf.Graph().as_default() as graph, tf.device('/gpu:0'):
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        FaceReconstructor = Face3D(bfm_dir)
        batchsize = 1
        coeff = tf.placeholder(name = 'input_coeff', shape = [batchsize, 257], dtype = tf.float32)
        texture = tf.placeholder(name = 'input_texture', shape = [batchsize, 35709, 3], dtype = tf.float32)

        # reconstructing faces
        FaceReconstructor.Reconstruction_Block(coeff, batchsize, texture)
        recon_img = FaceReconstructor.render_imgs
        with tf.Session() as sess:
            image_array = []
            os.system("rm {}".format(os.path.join(tmp_dir, "*.jpg")))
            norm_texture = np.ones((batchsize, 35709, 3), dtype = np.float32) * 128
            for idx, coeff_ in tqdm(enumerate(coeff_array)):
                recon_img_ = sess.run([recon_img], feed_dict = {coeff: np.expand_dims(coeff_, 0), texture: norm_texture})
                recon_img_ = recon_img_[0][0]
                image_array.append(recon_img_)
                cv2.imwrite("{}/{}.jpg".format(tmp_dir, idx), recon_img_[:, :, :3][:, :, ::-1])

            os.system("ffmpeg -y -loglevel warning -framerate 25 -start_number 0 -i {}/%d.jpg -c:v libx264 -b:v 2000k {}".format(tmp_dir, out_path))

def recon_video(frame_array, lm_array, out_path = "test.mp4"):
    coeff_array = recon_coeff(frame_array, lm_array)
    recon_video_from_coeff(coeff_array, out_path)

def recon_uv(frame_array, lm_array, out_path = "test.mp4", tmp_dir = "./test"):
    coeff_array = recon_coeff(frame_array, lm_array)
    recon_uv_from_coeff(coeff_array, out_path, tmp_dir)

def recon_uv_from_coeff(coeff_array, out_path = "test.mp4", tmp_dir = "./test"):
    with g.as_default() as graph, tf.device('/gpu:0'):
        with tf.Session() as sess:
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
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
                
            os.system("rm {}".format(os.path.join(tmp_dir, "*.jpg")))

            for i in range(len(uv_array)):
                tmp_uv_img = uv_array[i][::-1, :, :]
                cv2.imwrite("{}/{}.png".format(tmp_dir, i), tmp_uv_img)

            os.system("ffmpeg -loglevel warning -y -framerate 25 -start_number 0 -i {}/%d.png -c:v libx264 -pix_fmt yuv420p -b:v 1000k {}".format(tmp_dir, out_path))

# given uv img array and img array, generate texture image
def recon_texture_from_coeff(coeff_array, img_array, out_path = "test.mp4", tmp_dir = "./test"):
    '''
        reconstruct texture from reconstructed coeff and aligned image
    '''
    with g.as_default() as graph, tf.device('/gpu:0'):
        with tf.Session() as sess:
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            idx = 0
            while idx < len(coeff_array):
                if idx + batchsize <= len(coeff_array):
                    end_idx = batchsize
                    coeff_batch = coeff_array[idx: idx + batchsize]
                    img_batch = img_array[idx: idx + batchsize]
                else:
                    # pad
                    end_idx = len(coeff_array) - idx
                    coeff_batch = pad_coeff(coeff_array[idx: ])
                    img_batch = pad_frame(img_array[idx: ])

                texture = sess.run([recon_textures], feed_dict = {coeff: coeff_batch, images: img_batch})
                texture = texture[0].astype(np.uint8)

                if idx == 0:
                    texture_array = texture[:end_idx]
                else:
                    texture_array = np.concatenate((texture_array, texture[:end_idx]), axis = 0)

                idx += batchsize

            os.system("rm {}".format(os.path.join(tmp_dir, "*.jpg")))

            for i in range(len(texture_array)):
                tmp_texture_img = texture_array[i][::-1, :, :]
                cv2.imwrite("{}/{}.png".format(tmp_dir, i), tmp_texture_img)

            os.system("ffmpeg -loglevel warning -y -framerate 25 -start_number 0 -i {}/%d.png -c:v libx264 -pix_fmt yuv420p -b:v 1000k {}".format(tmp_dir, out_path))


def recon_texture(frame_array, lm_array, out_path = "test.mp4", tmp_dir = "./test"):
    coeff, align_img = recon_coeff(frame_array, lm_array, return_image = True)
    recon_texture_from_coeff(coeff, align_img)