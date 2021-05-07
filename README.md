# style_avatar
A repository for generating stylized talking 3D and 3D face

------

### Overview

Our project organizes the files as follows:

```
├── README.md
├── data
├── data_process
├── deepspeech

The deepspeech folder provides api for extracting deepspeech probabilities.

The data_process folder provides methods for preprocessing video and audio files.

The data folder contains processed data.

```

- `Python 3.6`
- Install necessary packages through `pip install -r requirements.txt`

------
### DeepSpeech

Please download the pretrained deepspeech model from the [Link](https://github.com/mozilla/DeepSpeech/releases/download/v0.9.2/deepspeech-0.9.2-checkpoint.tar.gz) to the deepspeech folder and unzip the file.

------
### Dataprocess

#### Ted-HD data
We leverage `lmdb` to store the fragmented data. 
You can obtain the train/test video with the code bellow

```python
import lmdb

def test():
    lmdb_path = "../data/ted_hd/lmdb"
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs = 64)

    train_video = env.open_db("train_video".encode())
    train_audio = env.open_db("train_audio".encode())
    train_lm5 = env.open_db("train_lm5".encode())
    test_video = env.open_db("test_video".encode())
    test_audio = env.open_db("test_audio".encode())
    test_lm5 = env.open_db("test_lm5".encode())

    with env.begin(write = False) as txn:
        video = txn.get(str(0).encode(), db=test_video)
        audio = txn.get(str(0).encode(), db=test_audio)
        video_file = open("test.mp4", "wb")
        audio_file = open("test.wav", "wb")
        video_file.write(video)
        audio_file.write(audio)
        video_file.close()
        audio_file.close()
        print(txn.stat(db=train_video))
        print(txn.stat(db=test_video)) # we can obtain the database size here  
```

------
### Deep3DReconstruction
We refer to the [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction) to conduct the face reconstruction. 
We implement several batch-wise api in `deep_3drecon/utils.py`, including UV image generation and RGB texture generation. 


Notice that the function `recon_texture` generates texture through aligning input image and 3D model, rather than generating texture from 3DMM space.