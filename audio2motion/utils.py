import os

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)