import getpass
user = getpass.getuser()


IMG_WIDTH = 256
IMG_HEIGHT = 256
nb_image = 3
IMG_MAX_VAL = 255




table_name = 'paths'
OUT_PATH = '/home/'+f'{user}'+'/Cigarette_buts_detection/cigarette_buts/cigarettNet/out
DB_PATH = os.path.join(OUT_PATH, 'buts.db')
IMAGE_DIRECTORY = os.path.join(OUT_PATH, 'buts')
MASK_DIRECTORY = os.path.join(OUT_PATH, 'mask')
BACKGROUND_DIRECTORY = os.path.join(OUT_PATH, 'bg')
JSON_DIRECTORY = os.path.join(OUT_PATH, 'js')


UNET_CONFIG = {
    'START_FILTER': 16,
    'MIDDLE_FILTERS': 128,
    'DROP_RATE': 0.3,
    'EPOCHS': 15,
    'BATCH_SIZE': 32,
    'NUM_THRESHOLDS': 50
}

