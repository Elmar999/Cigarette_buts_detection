import getpass
user = getpass.getuser()



IMG_WIDTH = 256
IMG_HEIGHT = 256



IMAGE_DIRECTORY = '/home/'+f'{user}'+'/Cigarette_buts_detection/cigarette_buts/out/image/buts'
MASK_DIRECTORY = '/home/'+f'{user}'+'/Cigarette_buts_detection/cigarette_buts/out/mask'
BACKGROUND_DIRECTORY = '/home/'+f'{user}'+'/Cigarette_buts_detection/cigarette_buts/out/image/bg'


