import matplotlib.pyplot as plt
import os
from albumentations.augmentations import transforms
import random
import numpy as np



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

