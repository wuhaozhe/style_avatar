
import os
import lmdb
import multiprocessing
import time
import imageio
import face_alignment
import pickle as pkl
import numpy as np
import cv2
import soundfile as sf
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from utils import lm68_2_lm5_batch, mean_eye_distance

'''
    the process of ted hd dataset contains the following stages:
    (1) clip video to slices according to manually labeled time stamps
    (2) cut slices to more fine-grained piece according to scene change and face size
    (3) extract audio and texture features, store in lmdb dataset
'''

def coarse_slice():
    '''
        clip video to slices according to manually labeled time stamps
    '''

    src_video_path = "../data/ted_hd/clip_video"
    file_list = os.listdir(src_video_path)
    file_list.sort()
    clip_video_path = "../data/ted_hd/slice_video_coarse"
    head_tail_crop = 0.2

    if not os.path.exists(clip_video_path):
        os.makedirs(clip_video_path)

    for file_name in file_list:
        if file_name.endswith("_clip.mp4"):
            time_stamps = os.path.join(src_video_path, file_name[:-8] + "time.txt")
            time_file = open(time_stamps, 'r')
            video = VideoFileClip(os.path.join(src_video_path, file_name[:-9] + ".mp4"))

            directory = os.path.join(clip_video_path, file_name[:-9])
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            for idx, line in enumerate(time_file):
                line = line.strip().split(',')
                start, end = float(line[0]) + head_tail_crop, float(line[1]) - head_tail_crop

                clip = video.subclip(start, end)
                clip.write_videofile(os.path.join(directory, "{}.mp4".format(idx)),
                            codec='libx264',
                            audio_codec='aac',
                            temp_audiofile='temp-audio.m4a',
                            remove_temp=True, 
                            bitrate = "1500k", 
                            fps = 25)

def detect_worker(wid, folder_list, src_video_path):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(wid % 4)
    src_video_path = "../data/ted_hd/slice_video_coarse"
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, network_size=4, device='cuda')

    if wid == 0:
        folder_list = tqdm(folder_list)

    for folder_name in folder_list:
        folder_path = os.path.join(src_video_path, folder_name)
        file_list = os.listdir(folder_path)
        for file_name in file_list:
            if file_name.endswith(".mp4"):
                file_path = os.path.join(folder_path, file_name)
                vid = imageio.get_reader(file_path,  'ffmpeg')
                lm_list = []
                for image in vid.iter_data():
                    preds = fa.get_landmarks(image)
                    lm_list.append(preds)
                pkl.dump(lm_list, open(file_path[:-4] + "_lm.pkl", 'wb'))

    
def detect(wnum = 1):
    '''
        detect face landmarks
        param:
        wnum: the number of threads for processing
    '''
    src_video_path = "../data/ted_hd/slice_video_coarse"
    folder_list = os.listdir(src_video_path)
    folder_list.sort()

    p_array = []
    for wid in range(wnum):
        start_idx = (len(folder_list) // wnum) * wid
        if wid == wnum - 1:
            end_idx = len(folder_list)
        else:
            end_idx = (len(folder_list) // wnum) * (wid + 1)
        sub_folder_list = folder_list[start_idx: end_idx]

        p = multiprocessing.Process(target = detect_worker, args = (wid, sub_folder_list, src_video_path))
        p_array.append(p)
        p.start()

    for i in range(wnum):
        p_array[i].join()


def split_lm(lm_list):
    '''
        Scenario split:
        split landmark5 according to the landmark distance, 
        if distance is too large, conduct split
    '''
    lm_list = np.array(lm_list).reshape(-1, 10)
    lm_diff = lm_list[1:] - lm_list[:-1]
    diff_norm = np.linalg.norm(lm_diff, ord = 2, axis = 1)
    diff_mask = diff_norm > 100
    split_index = np.nonzero(diff_mask)[0]
    return split_index


def save_fine_clip(src_video_path, dst_video_path, folder, file_name, lm_list, sep_list, file_cnt):
    dst_path = os.path.join(dst_video_path, folder)
    src_path = "{}/{}/{}.mp4".format(src_video_path, folder, file_name)
    os.system("rm ../data/tmp/test.wav")
    os.system("ffmpeg -loglevel warning -i {} -ar 16000 ../data/tmp/test.wav".format(src_path))
    cap = cv2.VideoCapture(src_path)
    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        frame_list.append(frame)
    cap.release()

    audio, sr = sf.read("../data/tmp/test.wav")
    
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for sep in sep_list:
        frame_sep = frame_list[sep[0]: sep[1]]
        lm_sep = lm_list[sep[0]: sep[1]]
        audio_sep = audio[int(sep[0] * sr / 25): min(int(sep[1] * sr / 25), audio.shape[0])]

        # save
        sf.write(os.path.join(dst_path, "{}.wav".format(file_cnt[0])), audio_sep, sr)
        pkl.dump(lm_sep, open(os.path.join(dst_path, "{}.pkl".format(file_cnt[0])), 'wb'))
        
        for idx, frame in enumerate(frame_sep):
            cv2.imwrite("../data/tmp/{}.jpg".format(idx), frame)

        os.system("ffmpeg -loglevel warning -framerate 25 -start_number 0 -i ../data/tmp/%d.jpg -c:v libx264 -b:v 2000k {}".format(os.path.join(dst_path, "{}.mp4".format(file_cnt[0]))))
        os.system("rm ../data/tmp/*.jpg")
        file_cnt[0] += 1

def fine_slice():
    '''
        cut slices to more fine-grained piece according to scene change and face size
        save results to folders
    '''
    src_video_path = "../data/ted_hd/slice_video_coarse"
    dst_video_path = "../data/ted_hd/slice_video_fine"
    if not os.path.exists(dst_video_path):
        os.makedirs(dst_video_path)

    src_folder_list = os.listdir(src_video_path)
    src_folder_list.sort()

    for folder in tqdm(src_folder_list):
        folder_path = os.path.join(src_video_path, folder)
        file_list = os.listdir(folder_path)

        file_num = 0
        for file_name in file_list:
            if file_name.endswith("mp4"):
                file_num += 1
        
        file_cnt = [0]
        for i in range(file_num):
            lm_list = pkl.load(open(os.path.join(folder_path, "{}_lm.pkl".format(i)), 'rb'))
            lm5_list = lm68_2_lm5_batch(lm_list)
            split_index = split_lm(lm5_list)

            start_idx = 1
            if len(split_index) == 0:
                end_idx = len(lm5_list) - 1
            else:
                end_idx = split_index[0]

            sep_list = []
            for j in range(len(split_index) + 1):
                if end_idx - start_idx < 70:
                    continue
                else:
                    # drop if face is too small
                    eye_distance = mean_eye_distance(lm5_list[start_idx: end_idx])
                    if eye_distance < 35:
                        continue
                    else:
                        sep_list.append([start_idx, end_idx])
                    # save mp3, mp4 and landmark
                
                # update index
                if j < len(split_index) - 1:
                    start_idx = split_index[j] + 2
                    end_idx = split_index[j + 1]
                elif j == len(split_index) - 1:
                    start_idx = split_index[j] + 2
                    end_idx = len(lm5_list) - 1
                else:
                    break

            save_fine_clip(src_video_path, dst_video_path, folder, i, lm5_list, sep_list, file_cnt)


def split_train_test():
    '''
        gather dataset and split to training and testing
        no overlap on identity between training and testing set
    '''
    


def get_features_worker(wid):
    lmdb_path = "../data/ted_hd/lmdb"
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)

    env = lmdb.open(lmdb_path, map_size=1099511627776)

def get_features(wnum = 8):
    '''
        For video:
            reconstruct 3d param
            normalize pose
            filter pose
            mean and std for calculating style code
            rgb texture
            uv map
        For audio:
            deepspeech feature
            energy feature
    '''
    pass

def main():
    # coarse_slice()
    # detect(4)
    # fine_slice()
    split_train_test()

if __name__ == "__main__":
    main()