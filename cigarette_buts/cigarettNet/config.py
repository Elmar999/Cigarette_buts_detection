import os
import getpass
user = getpass.getuser()


nb_image = 3
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_MAX_VAL = 255


OUT_PATH = '/home/'+f'{user}'+'/Cigarette_buts_detection/cigarette_buts/out'
DB_PATH = '/home/'+f'{user}'+'/Cigarette_buts_detection/cigarette_buts/cigarettNet/buts.db'
IMAGE_DIRECTORY = os.path.join(OUT_PATH, 'image/buts')
MASK_DIRECTORY = os.path.join(OUT_PATH, 'mask')
BACKGROUND_DIRECTORY = os.path.join(OUT_PATH, 'image/bg')
JSON_DIRECTORY = os.path.join(OUT_PATH, 'image/js')
TEST_DIRECTORY = os.path.join(OUT_PATH, 'image/test')


UNET_CONFIG = {
    'START_FILTER': 16,
    'MIDDLE_FILTERS': 256,
    'DROP_RATE': 0.3,
    'EPOCHS': 30,
    'BATCH_SIZE': 32,
    'NUM_THRESHOLDS': 50
}
