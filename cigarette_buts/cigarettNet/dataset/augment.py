import matplotlib.pyplot as plt
import os

class Augment():
    def __init__(self):
        pass


    def generate_img_over_bg(image_path , mask_path , bg_image_path):
        '''
        project a segmented objects over background image
        Args: 
            image_path(str)
            mask_path(str)
            bg_image_path(str)
        Returns: 
            (merged, mask) (tuple) : tuple of generated image and mask.
        '''
        img = plt.imread(image_path)
        mask = np.load(mask_path)
        bg_img = plt.imread(bg_image_path)

        positions = np.where(mask > 0)
        merged = bg_img.copy()
        for pos_x , pos_y in zip(positions[0] , positions[1]):
            merged[pos_x , pos_y] = img[pos_x , pos_y]

        return (merged, mask)

    def flip_augment(image_path, mask_path):
        pass


    def rotate_augment(image_path, mask_path):
        pass

