import getpass
user = getpass.getuser()


IMG_WIDTH = 256
IMG_HEIGHT = 256
nb_image = 3




table_name = 'paths'
DB_PATH = '/home/elmar/Cigarette_buts_detection/cigarette_buts/cigarettNet/buts.db'
IMAGE_DIRECTORY = '/home/'+f'{user}'+'/Cigarette_buts_detection/cigarette_buts/out/image/buts'
MASK_DIRECTORY = '/home/'+f'{user}'+'/Cigarette_buts_detection/cigarette_buts/out/mask'
BACKGROUND_DIRECTORY = '/home/'+f'{user}'+'/Cigarette_buts_detection/cigarette_buts/out/image/bg'
JSON_DIRECTORY = '/home/'+f'{user}'+'/Cigarette_buts_detection/cigarette_buts/out/image/js'


UNET_CONFIG = {
    'START_FILTER': 16,
    'MIDDLE_FILTERS': 128,
    'DROP_RATE': 0.3,
    'EPOCHS': 15,
    'BATCH_SIZE': 32,
    'NUM_THRESHOLDS': 50
}

