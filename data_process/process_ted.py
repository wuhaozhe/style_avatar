
import os
import lmdb
import multiprocessing
import time
from moviepy.editor import VideoFileClip

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

def fine_slice_worker(wid, folder, src_video_path):

    print(wid, folder)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(wid % 4)
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # lmdb_path = "../data/ted_hd/lmdb"
    # if not os.path.exists(lmdb_path):
    #     os.makedirs(lmdb_path)

    # env = lmdb.open(lmdb_path, map_size=1099511627776)

    
def fine_slice(wnum = 8):
    '''
        detect face landmarks
        cut slices to more fine-grained piece according to scene change and face size
        save results to folders
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
        sub_folder = folder_list[start_idx: end_idx]

        p = multiprocessing.Process(target = fine_slice_worker, args = (wid, sub_folder, src_video_path))
        p_array.append(p)
        p.start()
        # force sleep 3 seconds of eahc thread to prevent conflicts when face_alignment model occupies GPU
        time.sleep(3)

    for i in range(wnum):
        p_array[i].join()


def main():
    # coarse_slice()
    fine_slice()

if __name__ == "__main__":
    main()