import getpass
user = getpass.getuser()


IMG_WIDTH = 256
IMG_HEIGHT = 256
db_name = '/home/elmar/Cigarette_buts_detection/cigarette_buts/cigarettNet/buts.db'
table_name = 'paths'

nb_image = 3

IMAGE_DIRECTORY = '/home/'+f'{user}'+'/Cigarette_buts_detection/cigarette_buts/out/image/buts'
MASK_DIRECTORY = '/home/'+f'{user}'+'/Cigarette_buts_detection/cigarette_buts/out/mask'
BACKGROUND_DIRECTORY = '/home/'+f'{user}'+'/Cigarette_buts_detection/cigarette_buts/out/image/bg'
JSON_DIRECTORY = '/home/'+f'{user}'+'/Cigarette_buts_detection/cigarette_buts/out/image/js'
