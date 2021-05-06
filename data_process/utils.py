# import sys
# sys.path.append("..")
# import face_alignment
# import librosa
# from deepspeech import get_prob
import numpy as np
import cv2

def lm68_2_lm5(in_lm):
    lm_idx = np.array([31,37,40,43,46,49,55]) - 1
    lm = np.stack([in_lm[lm_idx[0],:],np.mean(in_lm[lm_idx[[1,2]],:],0),np.mean(in_lm[lm_idx[[3,4]],:],0),in_lm[lm_idx[5],:],in_lm[lm_idx[6],:]], axis = 0)
    lm = lm[[1,2,0,3,4],:2]
    return lm

def lm68_2_lm5_batch(in_lm_batch):
    lm5_list = []
    for i in range(len(in_lm_batch)):
        lm = in_lm_batch[i][0]
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