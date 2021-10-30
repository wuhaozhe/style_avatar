import sys
sys.path.append("..")
import tensorflow as tf 
import numpy as np
import cv2
import os
import time
from scipy.io import loadmat,savemat
from tqdm import tqdm
from align_img import align, align_video
from .face_decoder import Face3D

def load_graph(graph_filename):
	with tf.gfile.GFile(graph_filename,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def

# load landmarks for standard face, which is used for image preprocessing
def load_lm3d(bfm_dir):

	Lm3D = loadmat(os.path.join(bfm_dir, 'similarity_Lm3D_all.mat'))
	Lm3D = Lm3D['lm']

	# calculate 5 facial landmarks using 68 landmarks
	lm_idx = np.array([31,37,40,43,46,49,55]) - 1
	Lm3D = np.stack([Lm3D[lm_idx[0],:],np.mean(Lm3D[lm_idx[[1,2]],:],0),np.mean(Lm3D[lm_idx[[3,4]],:],0),Lm3D[lm_idx[5],:],Lm3D[lm_idx[6],:]], axis = 0)
	Lm3D = Lm3D[[1,2,0,3,4],:]

	return Lm3D

def pad_frame(frame_batch, batchsize):
    '''
        pad frame to batchsize
    '''
    frame_zeros = np.ones((batchsize - len(frame_batch), frame_batch.shape[1], frame_batch.shape[2], frame_batch.shape[3]), dtype = np.uint8)
    frame_batch = np.concatenate((frame_batch, frame_zeros), axis=0)
    return frame_batch

def pad_lm(lm_batch, batchsize):
    '''
        pad landmark to batchsize
    '''
    lm_zeros = np.tile(np.expand_dims(lm_batch[0], 0), (batchsize - len(lm_batch), 1, 1))
    lm_batch = np.concatenate((lm_batch, lm_zeros), axis=0)
    return lm_batch

def pad_coeff(coeff_batch, batchsize):
    '''
        pad coeff to batchsize
    '''
    coeff_zeros = np.tile(np.expand_dims(coeff_batch[0], 0), (batchsize - len(coeff_batch), 1))
    coeff_batch = np.concatenate((coeff_batch, coeff_zeros), axis=0)
    return coeff_batch

class Reconstructor():
    def __init__(self):

        self.bfm_dir = os.path.join(os.path.dirname(__file__), 'BFM')
        self.network_dir = os.path.join(os.path.dirname(__file__), 'network')
        self.lm3D = load_lm3d(self.bfm_dir)
        self.batchsize = 32
        self.g = tf.Graph()

        with self.g.as_default() as graph,tf.device('/gpu:0'):
            FaceReconstructor = Face3D(self.bfm_dir)
            self.images = tf.placeholder(name = 'input_imgs', shape = [self.batchsize,224,224,3], dtype = tf.float32)
            self.graph_def = load_graph(os.path.join(self.network_dir, 'FaceReconModel.pb'))
            tf.import_graph_def(self.graph_def,name='resnet',input_map={'input_imgs:0': self.images})

            # output coefficients of R-Net (dim = 257) 
            self.coeff = graph.get_tensor_by_name('resnet/coeff:0')
            # reconstructing faces
            FaceReconstructor.Reconstruction_Block(self.coeff,self.batchsize, None, self.images)
            self.face_shape = FaceReconstructor.face_shape_t
            self.face_texture = FaceReconstructor.face_texture
            self.face_color = FaceReconstructor.face_color
            self.landmarks_2d = FaceReconstructor.landmark_p
            self.recon_img = FaceReconstructor.render_imgs
            self.tri = FaceReconstructor.facemodel.face_buf
            self.uv_img = FaceReconstructor.render_uvs
            self.recon_textures = FaceReconstructor.recon_textures

        gpu_options = tf.GPUOptions(allow_growth = True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph = graph)


    def recon_coeff(self, frame_array, lm_array, return_image = False):
        with self.g.as_default() as graph, tf.device('/gpu:0'):
            with tf.Session() as sess:
                input_img_list, _, _ = align_video(frame_array, lm_array, self.lm3D)
                idx = 0
                while idx < len(frame_array):
                    if idx + self.batchsize <= len(frame_array):
                        end_idx = self.batchsize
                        input_img_array = input_img_list[idx: idx + self.batchsize]
                        # lm_batch = lm_array[idx: idx + self.batchsize]
                    else:
                        # pad
                        end_idx = len(frame_array) - idx
                        input_img_array = pad_frame(input_img_list[idx: ], self.batchsize)
                        # lm_batch = pad_lm(lm_array[idx: ], self.batchsize)

                    # input_img_list = []
                    # for i in range(self.batchsize):
                    #     input_img,lm_new,transform_params = align(frame_batch[i],lm_batch[i],self.lm3D)
                    #     input_img_list.append(input_img)
                    
                    # input_img_array = np.concatenate(input_img_list, axis = 0)
                    coeff_ = sess.run([self.coeff],feed_dict = {self.images: input_img_array})
                    coeff_ = coeff_[0]
                    
                    if idx == 0:
                        input_array = input_img_array[:end_idx]
                        coeff_array = coeff_[:end_idx]
                    else:
                        input_array = np.concatenate((input_array, input_img_array[:end_idx]), axis = 0)
                        coeff_array = np.concatenate((coeff_array, coeff_[:end_idx]), axis = 0)

                    idx += self.batchsize

            if return_image:
                return coeff_array, input_array
            else:
                return coeff_array

    def recon_video_from_coeff(self, coeff_array, out_path = "test.mp4", tmp_dir = "./test"):
        # use local face reconstructor
        with tf.Graph().as_default() as graph, tf.device('/gpu:0'):
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)

            FaceReconstructor = Face3D(self.bfm_dir)
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

                os.system("ffmpeg -y -loglevel warning -framerate 25 -start_number 0 -i {}/%d.jpg -c:v libx264 -b:v 2000k {}".format(tmp_dir, out_path))

    def recon_video(self, frame_array, lm_array, out_path = "test.mp4"):
        coeff_array = self.recon_coeff(frame_array, lm_array)
        self.recon_video_from_coeff(coeff_array, out_path)

    def recon_video_from_coeff_notex(self, coeff_array, out_path = "test.mp4", tmp_dir = "./test"):
        # use local face reconstructor
        with tf.Graph().as_default() as graph, tf.device('/gpu:0'):
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)

            FaceReconstructor = Face3D(self.bfm_dir)
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


    def recon_uv(self, frame_array, lm_array, out_path = "test.mp4", tmp_dir = "./test"):
        coeff_array = self.recon_coeff(frame_array, lm_array)
        self.recon_uv_from_coeff(coeff_array, out_path, tmp_dir)

    def recon_uv_from_coeff(self, coeff_array, out_path = "test.mp4", tmp_dir = "./test", bg_path = None):
        '''
            if bg path is not none, also write background video
        '''
        with self.g.as_default() as graph, tf.device('/gpu:0'):
            with tf.Session() as sess:
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                idx = 0
                while idx < len(coeff_array):
                    if idx + self.batchsize <= len(coeff_array):
                        end_idx = self.batchsize
                        coeff_batch = coeff_array[idx: idx + self.batchsize]
                    else:
                        # pad
                        end_idx = len(coeff_array) - idx
                        coeff_batch = pad_coeff(coeff_array[idx: ], self.batchsize)

                    uv_img_ = sess.run([self.uv_img],feed_dict = {self.coeff: coeff_batch})
                    uv_img_ = uv_img_[0]
                    
                    if idx == 0:
                        uv_array = uv_img_[:end_idx]
                    else:
                        uv_array = np.concatenate((uv_array, uv_img_[:end_idx]), axis = 0)

                    idx += self.batchsize
                    
                os.system("rm {}".format(os.path.join(tmp_dir, "*.png")))

                for i in range(len(uv_array)):
                    tmp_uv_img = uv_array[i][::-1, :, :]
                    cv2.imwrite("{}/{}.png".format(tmp_dir, i), tmp_uv_img)
                    if not (bg_path is None):
                        num_labels, labels = cv2.connectedComponents((tmp_uv_img[:, :, 0] == 0).astype(np.uint8))
                        cv2.imwrite(os.path.join(tmp_dir, "{}_bg.png".format(i)), (labels == 1).astype(np.uint8) * 255)

                os.system("ffmpeg -loglevel warning -y -framerate 25 -start_number 0 -i {}/%d.png -c:v libx264 -pix_fmt yuv420p -b:v 1000k {}".format(tmp_dir, out_path))
                if not (bg_path is None):
                    os.system("ffmpeg -loglevel warning -y -framerate 25 -start_number 0 -i {}/%d_bg.png -c:v libx264 -pix_fmt yuv420p -b:v 1000k {}".format(tmp_dir, bg_path))


    # given uv img array and img array, generate texture image
    def recon_texture_from_coeff(self, coeff_array, img_array, out_path = "test.mp4", tmp_dir = "./test"):
        '''
            reconstruct texture from reconstructed coeff and aligned image
        '''
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        idx = 0
        while idx < len(coeff_array):
            if idx + self.batchsize <= len(coeff_array):
                end_idx = self.batchsize
                coeff_batch = coeff_array[idx: idx + self.batchsize]
                img_batch = img_array[idx: idx + self.batchsize]
            else:
                # pad
                end_idx = len(coeff_array) - idx
                coeff_batch = pad_coeff(coeff_array[idx: ], self.batchsize)
                img_batch = pad_frame(img_array[idx: ], self.batchsize)

            texture = self.sess.run([self.recon_textures], feed_dict = {self.coeff: coeff_batch, self.images: img_batch})
            texture = texture[0].astype(np.uint8)

            if idx == 0:
                texture_array = texture[:end_idx]
            else:
                texture_array = np.concatenate((texture_array, texture[:end_idx]), axis = 0)

            idx += self.batchsize

        os.system("rm {}".format(os.path.join(tmp_dir, "*.png")))

        for i in range(len(texture_array)):
            tmp_texture_img = texture_array[i][::-1, :, :]
            cv2.imwrite("{}/{}.png".format(tmp_dir, i), tmp_texture_img)

        os.system("ffmpeg -loglevel warning -y -framerate 25 -start_number 0 -i {}/%d.png -c:v libx264 -pix_fmt yuv420p -b:v 1000k {}".format(tmp_dir, out_path))


    def recon_texture(self, frame_array, lm_array, out_path = "test.mp4", tmp_dir = "./test"):
        coeff, align_img = self.recon_coeff(frame_array, lm_array, return_image = True)
        self.recon_texture_from_coeff(coeff, align_img, out_path, tmp_dir)