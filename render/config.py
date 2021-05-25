# =============== Basic Configurations ===========
TEXTURE_DIM = 16


# =============== Train Configurations ===========
DATA_DIR = 'data'
CHECKPOINT_DIR = ''
LOG_DIR = ''
TRAIN_SET = ['{:04d}'.format(i) for i in range(899)]
ITERATIONS = 50000
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
TRAIN_LEN = 3