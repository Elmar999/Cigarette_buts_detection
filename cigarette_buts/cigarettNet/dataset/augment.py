import os
import random
import config
import numpy as np
import matplotlib.pyplot as plt
from albumentations.augmentations import transforms


def augment(tr_ds, init_step, nb_augment, augment_method):
    for count in range(nb_augment): 
        tr_ds.append(augment_method(tr_ds[init_step+count][0], tr_ds[init_step+count][1]))    
    return tr_ds

def generate_img_over_bg(image , mask , bg_image):
    '''
    project a segmented objects over background image.
    Args: 
        image(nd.array)
        mask(nd.array)
        bg_image(nd.array)

    Returns: 
        (merged, mask) (tuple) : tuple of generated image and mask.
    '''

    positions = np.where(mask > 0)
    merged = bg_image.copy()
    for pos_x , pos_y in zip(positions[0] , positions[1]):
        merged[pos_x , pos_y] = image[pos_x , pos_y]

    return (merged, mask)

def over_bg_augment(tr_ds, init_step, nb_augment):
    bg_images = os.listdir(config.BACKGROUND_DIRECTORY)
    nb_images = len(os.listdir(config.BACKGROUND_DIRECTORY))

    for count in range(nb_augment):
        rnd = random.randint(0, nb_images - 1)
        bg_image = plt.imread(os.path.join(config.BACKGROUND_DIRECTORY, bg_images[rnd]))
        tr_ds.append(generate_img_over_bg(tr_ds[init_step+count][0], tr_ds[init_step+count][1], bg_image))
    return tr_ds


def flip_augment(image, mask):
    '''
    flip an image vertically or horizontally.
    Args: 
        image(nd.array)
        mask(nd.array)
    Returns:
        (flip_image, flip_mask) (tuple) : tuple of flipped image and mask.
    '''

    if(random.randint(0,1) == 0):       
        flip_img = transforms.HorizontalFlip().apply(image)
        flip_mask = transforms.HorizontalFlip().apply_to_mask(mask)
    else:
        flip_img = transforms.VerticalFlip().apply(image)
        flip_mask = transforms.VerticalFlip().apply_to_mask(mask)

    return (flip_img, flip_mask)


def rotate_augment(image, mask):
    '''
    takes an image and rotates it by multiple of 90 degree.
    Args:
        image(nd.array)
        mask(nd.array)
    Returns:
        (rot_img, rot_mask) (tuple): tuple of rotated image and mask
    '''

    factor = random.randint(1,3)
    rot_img = transforms.RandomRotate90().apply(image, factor)
    rot_mask = transforms.RandomRotate90().apply(mask, factor)

    return (rot_img, rot_mask)

