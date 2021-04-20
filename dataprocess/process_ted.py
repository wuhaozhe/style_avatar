import sys
sys.path.append("..")

import lmdb
import librosa
from deepspeech import get_prob

'''
    the process of ted hd dataset contains the following stages:
    (1) clip video to slices according to manually labeled videos
    (2) cut slices to more fine-grained piece according to scene change and face size
    (3) extract audio and texture features
'''

