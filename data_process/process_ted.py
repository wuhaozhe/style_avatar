
import sys
sys.path.append("..")
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
import deep_3drecon
import librosa
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from utils import lm68_2_lm5_batch, mean_eye_distance, read_video, write_video, filter_norm_coeff, lm68_mouth_contour
from io import BytesIO
from multiprocessing import Process
from align_img import align_lm68, align

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
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, network_size=4, device='cuda')

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
                lm_list = np.array(lm_list)[:, 0]
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


def save_fine_clip(src_video_path, dst_video_path, folder, file_name, lm5_list, lm68_list, sep_list, file_cnt):
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
        lm5_sep = lm5_list[sep[0]: sep[1]]
        lm68_sep = lm68_list[sep[0]: sep[1]]
        audio_sep = audio[int(sep[0] * sr / 25): min(int(sep[1] * sr / 25), audio.shape[0])]

        # save
        sf.write(os.path.join(dst_path, "{}.wav".format(file_cnt[0])), audio_sep, sr)
        pkl.dump(lm5_sep, open(os.path.join(dst_path, "{}_5.pkl".format(file_cnt[0])), 'wb'))
        pkl.dump(lm68_sep, open(os.path.join(dst_path, "{}_68.pkl".format(file_cnt[0])), 'wb'))
        
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

            save_fine_clip(src_video_path, dst_video_path, folder, i, lm5_list, lm_list, sep_list, file_cnt)


def split_train_test():
    '''
        gather dataset and split to training and testing
        no overlap on identity between training and testing set
        build lmdb
    '''
    src_path = "../data/ted_hd/slice_video_fine"
    folder_list = os.listdir(src_path)
    folder_list.sort()
    split_folder = 68

    # if folder number is smaller than split folder, group to train set
    # if folder number is no smaller than split folder, group to test set
    lmdb_path = "../data/ted_hd/lmdb"
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
    
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)

    train_video = env.open_db("train_video".encode())
    train_audio = env.open_db("train_audio".encode())
    train_lm5 = env.open_db("train_lm5".encode())
    train_lm68 = env.open_db("train_lm68".encode())
    test_video = env.open_db("test_video".encode())
    test_audio = env.open_db("test_audio".encode())
    test_lm5 = env.open_db("test_lm5".encode())
    test_lm68 = env.open_db("test_lm68".encode())

    with env.begin(write = True) as txn:
        train_cnt, test_cnt = 0, 0
        for folder_name in tqdm(folder_list):
            folder_path = os.path.join(src_path, folder_name)
            file_list = os.listdir(folder_path)
            file_list.sort()
            file_num = 0
            for file_name in file_list:
                if file_name.endswith("mp4"):
                    file_num += 1

            for idx in range(file_num):
                video_path = os.path.join(folder_path, "{}.mp4".format(idx))
                lm5_path = os.path.join(folder_path, "{}_5.pkl".format(idx))
                lm68_path = os.path.join(folder_path, "{}_68.pkl".format(idx))
                audio_path = os.path.join(folder_path, "{}.wav".format(idx))
                with open(video_path, 'rb') as f:
                    video_bin = f.read()
                with open(lm5_path, 'rb') as f:
                    lm5_bin = f.read()
                with open(lm68_path, 'rb') as f:
                    lm68_bin = f.read()
                with open(audio_path, 'rb') as f:
                    audio_bin = f.read()

                if int(folder_name) < split_folder:
                    txn.put(str(train_cnt).encode(), video_bin, db = train_video)
                    txn.put(str(train_cnt).encode(), lm5_bin, db = train_lm5)
                    txn.put(str(train_cnt).encode(), lm68_bin, db = train_lm68)
                    txn.put(str(train_cnt).encode(), audio_bin, db = train_audio)
                    train_cnt += 1
                else:
                    txn.put(str(test_cnt).encode(), video_bin, db = test_video)
                    txn.put(str(test_cnt).encode(), lm5_bin, db = test_lm5)
                    txn.put(str(test_cnt).encode(), lm68_bin, db = test_lm68)
                    txn.put(str(test_cnt).encode(), audio_bin, db = test_audio)
                    test_cnt += 1


def recon3d_worker(wid, data_list, train):
    if wid == 0:
        data_list = tqdm(data_list)

    lmdb_path = "../data/ted_hd/lmdb"
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)
    if train:
        video_data = env.open_db("train_video".encode())
        lm5_data = env.open_db("train_lm5".encode())
        lm68_data = env.open_db("train_lm68".encode())
        align_data = env.open_db("train_align".encode())
        uv_data = env.open_db("train_uv".encode())
        bg_data = env.open_db("train_bg".encode())
        texture_data = env.open_db("train_texture".encode())
        coeff_data = env.open_db("train_coeff".encode())
        coeff_norm_data = env.open_db("train_coeff_norm".encode())
        mouth_data = env.open_db("train_mouth".encode())
    else:
        video_data = env.open_db("test_video".encode())
        lm5_data = env.open_db("test_lm5".encode())
        lm68_data = env.open_db("test_lm68".encode())
        align_data = env.open_db("test_align".encode())
        uv_data = env.open_db("test_uv".encode())
        bg_data = env.open_db("test_bg".encode())
        texture_data = env.open_db("test_texture".encode())
        coeff_data = env.open_db("test_coeff".encode())
        coeff_norm_data = env.open_db("test_coeff_norm".encode())
        mouth_data = env.open_db("test_mouth".encode())

    tmp_dir = "../data/tmp/{}".format(wid)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(wid % 4)
    face_reconstructor = deep_3drecon.Reconstructor()

    for data_name in data_list:
        txn = env.begin(write = False)
        video_bin = txn.get(str(data_name).encode(), db=video_data)
        lm5_bin = txn.get(str(data_name).encode(), db=lm5_data)
        lm68_bin = txn.get(str(data_name).encode(), db=lm68_data)
        txn.abort()
        video_file = open("../data/tmp/{}_src.mp4".format(wid), "wb")
        video_file.write(video_bin)
        video_file.close()
        frames = read_video("../data/tmp/{}_src.mp4".format(wid))
        h, w, _ = frames[0].shape
        lm5 = pkl.load(BytesIO(lm5_bin))
        lm68_list = pkl.load(BytesIO(lm68_bin))

        lm3D = face_reconstructor.lm3D
        lm68_align = align_lm68(lm5, lm68_list, lm3D, w, h)
        lm68_align = lm68_align.astype(np.int32)
        lm68_mouth_contour(lm68_align, "../data/tmp/{}_mouth.mp4".format(wid), tmp_dir)

        coeff, align_img = face_reconstructor.recon_coeff(frames, lm5, return_image = True)

        with open("../data/tmp/{}_coeff.pkl".format(wid), 'wb') as f:
            pkl.dump(coeff, f)
        with open("../data/tmp/{}_coeff_norm.pkl".format(wid), 'wb') as f:
            coeff_norm = filter_norm_coeff(coeff)
            pkl.dump(coeff_norm, f)
        write_video(align_img, "../data/tmp/{}_align.mp4".format(wid), tmp_dir)
        face_reconstructor.recon_uv_from_coeff(coeff, "../data/tmp/{}_uv.mp4".format(wid), tmp_dir, "../data/tmp/{}_bg.mp4".format(wid))
        face_reconstructor.recon_texture_from_coeff(coeff, align_img, "../data/tmp/{}_texture.mp4".format(wid), tmp_dir)

        with open("../data/tmp/{}_align.mp4".format(wid), 'rb') as f:
            align_bin = f.read()
        with open("../data/tmp/{}_uv.mp4".format(wid), 'rb') as f:
            uv_bin = f.read()
        with open("../data/tmp/{}_bg.mp4".format(wid), 'rb') as f:
            bg_bin = f.read()
        with open("../data/tmp/{}_texture.mp4".format(wid), 'rb') as f:
            texture_bin = f.read()
        with open("../data/tmp/{}_coeff.pkl".format(wid), 'rb') as f:
            coeff_bin = f.read()
        with open("../data/tmp/{}_coeff_norm.pkl".format(wid), 'rb') as f:
            coeff_norm_bin = f.read()
        with open("../data/tmp/{}_mouth.mp4".format(wid), 'rb') as f:
            mouth_bin = f.read()
            
        txn = env.begin(write = True)
        txn.put(str(data_name).encode(), align_bin, db = align_data)
        txn.put(str(data_name).encode(), uv_bin, db = uv_data)
        txn.put(str(data_name).encode(), bg_bin, db = bg_data)
        txn.put(str(data_name).encode(), texture_bin, db = texture_data)
        txn.put(str(data_name).encode(), coeff_bin, db = coeff_data)
        txn.put(str(data_name).encode(), coeff_norm_bin, db = coeff_norm_data)
        txn.put(str(data_name).encode(), mouth_bin, db = mouth_data)
        txn.commit()

def audio_worker(wid, data_list, train):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(wid % 4)
    import deepspeech
    if wid == 0:
        data_list = tqdm(data_list)

    lmdb_path = "../data/ted_hd/lmdb"
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)
    if train:
        audio_data = env.open_db("train_audio".encode())
        deepspeech_data = env.open_db("train_deepspeech".encode())
        energy_data = env.open_db("train_energy".encode())
    else:
        audio_data = env.open_db("test_audio".encode())
        deepspeech_data = env.open_db("test_deepspeech".encode())
        energy_data = env.open_db("test_energy".encode())

    tmp_dir = "../data/tmp/{}".format(wid)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    

    for data_name in data_list:
        txn = env.begin(write = False)
        audio_bin = txn.get(str(data_name).encode(), db=audio_data)
        txn.abort()

        with open("../data/tmp/{}.wav".format(wid), 'wb') as f:
            f.write(audio_bin)

        deepspeech_prob = deepspeech.get_prob("../data/tmp/{}.wav".format(wid))
        audio, sr = librosa.load("../data/tmp/{}.wav".format(wid), sr = 16000)
        audio_energy = librosa.feature.rms(y = audio, frame_length = 512, hop_length = 320, center = False)

        with open("../data/tmp/{}_deepspeech.pkl".format(wid), 'wb') as f:
            pkl.dump(deepspeech_prob, f)
        with open("../data/tmp/{}_energy.pkl".format(wid), 'wb') as f:
            pkl.dump(audio_energy, f)

        with open("../data/tmp/{}_deepspeech.pkl".format(wid), 'rb') as f:
            deepspeech_bin = f.read()
        with open("../data/tmp/{}_energy.pkl".format(wid), 'rb') as f:
            energy_bin = f.read()

        txn = env.begin(write = True)
        txn.put(str(data_name).encode(), deepspeech_bin, db = deepspeech_data)
        txn.put(str(data_name).encode(), energy_bin, db = energy_data)
        txn.commit()

def get_features_worker(wid, data_list, train):
    audio_worker(wid, data_list, train)
    recon3d_worker(wid, data_list, train)

def get_features(train = True, num_worker = 8):
    '''
        For video:
            reconstruct 3d param
            normalize pose
            filter pose
            mean and std for calculating style code
            rgb texture
            uv map
            mouth region mask
            background mask
        For audio:
            deepspeech feature
            energy feature
    '''

    def assign_job(data_size, num_worker):
        data_list = range(data_size)
        data_chunk = np.array_split(np.array(data_list), num_worker)
        return data_chunk

    lmdb_path = "../data/ted_hd/lmdb"
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)
    if train:
        video = env.open_db("train_video".encode())
    else:
        video = env.open_db("test_video".encode())
    
    with env.begin(write = False) as txn:
        data_size = txn.stat(db=video)['entries']
        data_chunk = assign_job(data_size, num_worker)

    w_list = []
    for wid in range(num_worker):
        w_list.append(Process(target = get_features_worker, args = (wid, data_chunk[wid], train)))

    for wid in range(num_worker):
        w_list[wid].start()
        time.sleep(10)

    for wid in range(num_worker):
        w_list[wid].join()


def main():
    # coarse_slice()
    # detect(4)
    # fine_slice()
    split_train_test()
    get_features(train = True, num_worker = 2)
    get_features(train = False, num_worker = 2)

def test():
    lmdb_path = "../data/ted_hd/lmdb"
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)

    train_video = env.open_db("train_video".encode())
    train_audio = env.open_db("train_audio".encode())
    train_lm5 = env.open_db("train_lm5".encode())
    test_video = env.open_db("test_video".encode())
    test_audio = env.open_db("test_audio".encode())
    test_lm5 = env.open_db("test_lm5".encode())
    train_align_video = env.open_db("test_align".encode())

    with env.begin(write = False) as txn:
        video = txn.get(str(0).encode(), db=train_align_video)
        lm = txn.get(str(0).encode(), db=test_lm5)
        audio = txn.get(str(0).encode(), db=test_audio)
        video_file = open("test.mp4", "wb")
        audio_file = open("test.wav", "wb")
        video_file.write(video)
        audio_file.write(audio)
        video_file.close()
        audio_file.close()
        # print(txn.stat(db=test_video))
    
    # lm = pkl.load(BytesIO(lm))
    # video_file = open("test.mp4", "wb")
    # video_file.write(video)
    # video_file.close()
    # frames = read_video("test.mp4")
    # reconstructor = deep_3drecon.Reconstructor()
    # coeff, align_img = reconstructor.recon_coeff(frames, lm, return_image = True)
    # print(coeff.shape)
    # reconstructor.recon_uv_from_coeff(coeff, "test2.mp4")
    # reconstructor.recon_video_from_coeff_notex(coeff, "test2.mp4")
    # reconstructor.recon_texture(frames, lm, out_path = "test2.mp4")
    # print(coeff.shape, align_img.shape)
    # reconstructor.recon_video(frames, lm, out_path = "test2.mp4")

    
    # deep_3drecon.recon_uv_from_coeff(coeff, "test3.mp4")
    # frames = frames[0:128]
    # lm = lm[0:128]
    # deep_3drecon.recon_texture(frames, lm)
    

if __name__ == "__main__":
    # main()
    test()