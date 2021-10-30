import os
import cv2
import cpbd
import numpy as np
import imutils
import dlib
from skimage.metrics import structural_similarity
from skimage import data
from tqdm import tqdm
from imutils import face_utils


# 评测主要是评估tune之前，tune之后的LMD,CPBD,SSIM

data_path = "./data/notune_re"
# data_path = "./data/ours_re"

file_list = os.listdir(data_path)
file_list.sort()

def getvideo(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        frame_list.append(frame)
    return frame_list

def calculate_ssim(pred_list, gt_list):
    ssim_array = []
    for idx in tqdm(range(len(pred_list))):
        pred_file_path = os.path.join(data_path, pred_list[idx])
        gt_file_path = os.path.join(data_path, gt_list[idx])
        pred_frames = getvideo(pred_file_path)
        gt_frames = getvideo(gt_file_path)
        for j in range(len(pred_frames)):
            pred_image = cv2.cvtColor(pred_frames[j], cv2.COLOR_BGR2GRAY)
            gt_image = cv2.cvtColor(gt_frames[j], cv2.COLOR_BGR2GRAY)
            ssim = structural_similarity(gt_image, pred_image, data_range = pred_image.max() - pred_image.min())
            ssim_array.append(ssim)
    ssim_array = np.array(ssim_array)
    print(np.mean(ssim_array))

def calculate_cpbd(pred_list):
    cpbd_array = []
    for idx in tqdm(range(len(pred_list))):
        pred_file_path = os.path.join(data_path, pred_list[idx])
        pred_frames = getvideo(pred_file_path)
        for j in range(len(pred_frames)):
            pred_image = cv2.cvtColor(pred_frames[j], cv2.COLOR_BGR2GRAY)
            cpbd_array.append(cpbd.compute(pred_image))
    cpbd_array = np.array(cpbd_array)
    print(np.mean(cpbd_array))

def calculate_lmd(pred_list, gt_list):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    dis_array = []
    for idx in tqdm(range(len(pred_list))):
        pred_file_path = os.path.join(data_path, pred_list[idx])
        gt_file_path = os.path.join(data_path, gt_list[idx])
        pred_frames = getvideo(pred_file_path)
        gt_frames = getvideo(gt_file_path)
        for j in range(len(pred_frames)):
            pred_image = pred_frames[j]
            gt_image = gt_frames[j]
            pred_gray = cv2.cvtColor(pred_image, cv2.COLOR_BGR2GRAY)
            gt_gray = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
            pred_rects = detector(pred_gray, 1)
            gt_rects = detector(gt_gray, 1)
            if len(gt_rects) != len(pred_rects) or len(gt_rects) == 0:
                continue
            pred_shape = predictor(pred_gray, pred_rects[len(pred_rects) - 1])
            gt_shape = predictor(gt_gray, gt_rects[len(pred_rects) - 1])
            pred_shape = face_utils.shape_to_np(pred_shape)
            gt_shape = face_utils.shape_to_np(gt_shape)
            for (name, (start, end)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                if name != 'mouth':
                    continue
                pred_mouth_land = pred_shape[start:end].copy()
                gt_mouth_land = gt_shape[start:end].copy()
                pred_original = np.sum(pred_mouth_land, axis = 0) / 20.0
                gt_original = np.sum(gt_mouth_land, axis = 0) / 20.0
                pred_mouth_land = pred_mouth_land - pred_original
                gt_mouth_land = gt_mouth_land - gt_original
                dis = (pred_mouth_land-gt_mouth_land)**2
                dis = np.sum(dis,axis=1)
                dis = np.sqrt(dis)

                dis = np.mean(dis,axis=0)
                dis_array.append(dis)
    dis_array = np.array(dis_array)
    print(np.mean(dis_array))


# cream
pred_list = []
gt_list = []
for file_name in file_list:
    if file_name.startswith("cream") and file_name.endswith("gt.mp4"):
        gt_list.append(file_name)
        pred_list.append(file_name[:-7] + ".mp4")

# calculate_ssim(pred_list, gt_list)
# calculate_cpbd(pred_list)
# calculate_lmd(pred_list, gt_list)

# tcd
pred_list = []
gt_list = []
for file_name in file_list:
    if file_name.startswith("tcd") and file_name.endswith("gt.mp4"):
        gt_list.append(file_name)
        pred_list.append(file_name[:-7] + ".mp4")

calculate_ssim(pred_list, gt_list)
calculate_cpbd(pred_list)
# calculate_lmd(pred_list, gt_list)

# grid
pred_list = []
gt_list = []
for file_name in file_list:
    if file_name.startswith("grid") and file_name.endswith("gt.mp4"):
        gt_list.append(file_name)
        pred_list.append(file_name[:-7] + ".mp4")

calculate_ssim(pred_list, gt_list)
calculate_cpbd(pred_list)
# calculate_lmd(pred_list, gt_list)