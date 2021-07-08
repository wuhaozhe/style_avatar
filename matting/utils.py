import cv2
import torch
import os
import torch
import torch.nn as nn
import numpy as np
from models.modnet import MODNet

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        frame_list.append(frame)
    return np.array(frame_list)

class VideoMatting():
    def __init__(self, device, batchsize = 16):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        self.modnet = MODNet(backbone_pretrained=False)
        self.modnet = nn.DataParallel(self.modnet).cuda()
        self.modnet.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'modnet_webcam_portrait_matting.ckpt')))
        self.modnet.eval()
        self.batchsize = batchsize

    def mat_video(self, video_path, out_path):
        '''
            Given input video path
            output video matting results
        '''
        video_frames = read_video(video_path)
        video_frames = torch.from_numpy(video_frames).float().permute(0, 3, 1, 2) / 128 - 1
        c, h, w = video_frames.shape[1:]

        for i in range(0, len(video_frames), self.batchsize):
            if i + self.batchsize > len(video_frames):
                pad_frame = torch.zeros((self.batchsize + i - len(video_frames), c, h, w), dtype = torch.float).cuda()
                video_batch = torch.cat((video_frames[i:].cuda(), pad_frame), dim = 0)
            else:
                video_batch = video_frames[i: i + self.batchsize].cuda()

            with torch.no_grad():
                _, _, pred_batch = self.modnet(video_batch, True)

            matte_np = pred_batch.detach().data.cpu().numpy()
            
            if i == 0:
                matte_array = matte_np
            else:
                matte_array = np.concatenate((matte_array, matte_np), axis = 0)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_path, fourcc, 25.0, (h, w), 0)

        matte_array = matte_array[:len(video_frames)] < 0.5
        for i in range(len(matte_array)):
            num_labels, labels = cv2.connectedComponents(matte_array[i][0].astype(np.uint8))
            left_up_corner_label = labels[0, 0]
            right_up_corner_label = labels[0, -1]
            mask = (np.logical_and(labels != left_up_corner_label, labels != right_up_corner_label) * 255).astype(np.uint8)
            out.write(mask)

        out.release()

if __name__ == "__main__":
    video_path = "/home/wuhz/mnt/avatar/style_avatar/data_process/test.mp4"
    video_matting = VideoMatting(2)
    video_matting.mat_video(video_path, 'test.avi')