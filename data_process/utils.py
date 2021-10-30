import sys
sys.path.append("..")
# import face_alignment
# import librosa
# from deepspeech import get_prob
import numpy as np
import cv2
import os
import align_img
import quaternion
from scipy import signal

def lm68_2_lm5(in_lm):
    lm_idx = np.array([31,37,40,43,46,49,55]) - 1
    lm = np.stack([in_lm[lm_idx[0],:],np.mean(in_lm[lm_idx[[1,2]],:],0),np.mean(in_lm[lm_idx[[3,4]],:],0),in_lm[lm_idx[5],:],in_lm[lm_idx[6],:]], axis = 0)
    lm = lm[[1,2,0,3,4],:2]
    return lm

def lm68_2_lm5_batch(in_lm_batch):
    lm5_list = []
    for i in range(len(in_lm_batch)):
        lm = in_lm_batch[i]
        lm5 = lm68_2_lm5(lm)
        lm5_list.append(lm5)

    return np.array(lm5_list)

def mean_eye_distance(lm_list):
    lm_list = np.array(lm_list)
    left_eye = lm_list[:, 0, :].reshape(-1, 2)
    right_eye = lm_list[:, 1, :].reshape(-1, 2)
    eye_dis = np.linalg.norm(left_eye - right_eye, ord=2, axis = 1)
    return np.mean(eye_dis)

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        frame_list.append(frame)
    return np.array(frame_list)

def write_video(frame_list, video_path, tmp_dir):
    os.system("rm {}".format(os.path.join(tmp_dir, "*.png")))
    for i in range(len(frame_list)):
        tmp_frame_img = frame_list[i]
        cv2.imwrite("{}/{}.png".format(tmp_dir, i), tmp_frame_img)
    os.system("ffmpeg -y -loglevel warning -framerate 25 -start_number 0 -i {}/%d.png -c:v libx264 -pix_fmt yuv420p -b:v 2000k {}".format(tmp_dir, video_path))

def filter_norm_coeff(coeff_array):
    '''
        filter the pose, and normalize the pose of the coeff
    '''
    coeff_array_copy = np.copy(coeff_array)
    angles = coeff_array[:, 224:227] # euler angles for pose, order(x y z)
    translation = coeff_array[:, 254:257] # translation
    angle_normed = normalize_euler(angles)
    translation_normed = translation - np.mean(translation, axis = 0, keepdims=True)
    angle_normed = filter_batch(angle_normed, 0.7)
    translation_normed = filter_batch(translation_normed, 0.7)

    coeff_array_copy[:, 224:227] = angle_normed
    coeff_array_copy[:, 254:257] = translation_normed
    return coeff_array_copy

def filter_coeff(coeff_array):
    coeff_array_copy = np.copy(coeff_array)
    angles = coeff_array[:, 224:227] # euler angles for pose, order(x y z)
    translation = coeff_array[:, 254:257] # translation
    angle_filter = filter_batch(angles, 0.7)
    translation_filter = filter_batch(translation, 0.7)

    coeff_array_copy[:, 224:227] = angle_filter
    coeff_array_copy[:, 254:257] = translation_filter
    return coeff_array_copy

def filter_lm5(lm_array):
    lm_flat = lm_array.reshape(lm_array.shape[0], -1)
    lm_filtered = filter_batch(lm_flat, 0.5)
    lm_filtered = lm_filtered.reshape(lm_array.shape[0], -1, 2)
    return lm_filtered

# Take euler array as input, output normalized quaternion
# the average of quaternion is normalized to I
def normalize_euler(euler_array):
    qua = quaternion.euler_to_quaternion(euler_array, 'xyz')
    qua_avg = quaternion.average_quaternion_np(qua)
    qua_avg_inv = quaternion.qinv_np(qua_avg)
    qua_avg_inv = np.tile(qua_avg_inv, (len(qua), 1))
    qua_normed = quaternion.qmul_np(qua, qua_avg_inv)
    euler_normed = quaternion.qeuler_np(qua_normed, order = 'xyz', epsilon = 1e-5)
    return euler_normed

def filter_array(arr, bw):
    b, a = signal.butter(8, bw, 'lowpass', fs = 5)
    arr_low = signal.filtfilt(b, a, arr)
    return arr_low

def filter_batch(batch_array, bw):
    video_roi_arr_f = np.zeros_like(batch_array)
    for i in range(batch_array.shape[1]):
        video_roi_arr_f[:, i] = filter_array(batch_array[:, i], bw)
    return video_roi_arr_f

def lm68_eye_contour(lm68_list, video_path, tmp_dir):
    os.system("rm {}".format(os.path.join(tmp_dir, "*.png")))
    for i in range(len(lm68_list)):
        lm68 = lm68_list[i]
        left_eye = lm68[36:42]
        right_eye = lm68[42:48]
        image = np.zeros((224, 224, 3), dtype = np.uint8)
        cv2.fillPoly(image, pts = [left_eye, right_eye], color=(255,255,255))
        cv2.imwrite("{}/{}.png".format(tmp_dir, i), image)
    
    os.system("ffmpeg -y -loglevel warning -framerate 25 -start_number 0 -i {}/%d.png -c:v libx264 -pix_fmt yuv420p -b:v 2000k {}".format(tmp_dir, video_path))

def lm68_mouth_contour(lm68_list, video_path, tmp_dir):
    os.system("rm {}".format(os.path.join(tmp_dir, "*.png")))
    for i in range(len(lm68_list)):
        lm68 = lm68_list[i]
        mouth = lm68[48:60]
        image = np.zeros((224, 224, 3), dtype = np.uint8)
        cv2.fillPoly(image, pts = [mouth], color=(255,255,255))
        cv2.imwrite("{}/{}.png".format(tmp_dir, i), image)
    
    os.system("ffmpeg -y -loglevel warning -framerate 25 -start_number 0 -i {}/%d.png -c:v libx264 -pix_fmt yuv420p -b:v 2000k {}".format(tmp_dir, video_path))