import matplotlib.pyplot as plt
import numpy as np
import json
import config
from matplotlib.path import Path
import os
import cv2

class Utils:
    def __init__(self, paths = None):
        '''
        initialize paths that we loaded from database.
        '''
        self.paths = paths


    
    def load_images(self):
        '''
        load images from paths that contains image and mask path, to the list of tuples.
        '''
        
        tr_ds = []
        for path in self.paths:
            img_pixel, mask_pixel = plt.imread(path[0]), np.load(path[1])
            tr_ds.append((img_pixel, mask_pixel))
        return tr_ds


def resize_image(directory):
    '''
    resizes the images in directory according to dimensions in the config file.
    Args: directory(str)
    
    '''
    for filename in os.listdir(directory):
        if filename.split('.')[1] == 'jpg':
            filename = os.path.join(directory , filename)
            img = cv2.imread(filename)
            img = cv2.resize(img , (config.IMG_HEIGHT , config.IMG_WIDTH))
            cv2.imwrite(filename , img)

    
def rename_files(file_directory):
    ''' 
    renames the file names in image directory in this format -> cgrt_0.jpg , cgrt_1.jpg .... 
    Args: file_directory 

    '''
    i = 1
    for filename in os.listdir(file_directory):
        if filename.split('.')[1] == 'jpg':
            filename = os.path.join(file_directory , filename)
            renamed_file = os.path.join(file_directory , f'cgrt_{i}.jpg')
            print(filename , "   " ,  renamed_file)
            os.rename(filename , renamed_file)
            i+=1

def generate_mask(filename , json_file):
    '''
    generates a binary mask as ground truth for each image.
    Args: json_file
    '''

    with open(json_file, "r") as read_file:
        data = json.load(read_file)
    shapes = data['shapes']

    polygons = dict()
    for polygon_index in range(len(shapes)):
        polygons[polygon_index] = shapes[polygon_index]['points']
    
    for points in polygons:
        for index in range(len(polygons[points])):
            polygons[points][index][0] = round(polygons[points][index][0])
            polygons[points][index][1] = round(polygons[points][index][1]) 
            polygons[points][index] = tuple(polygons[points][index])

    x, y = np.meshgrid(np.arange(config.IMG_WIDTH), np.arange(config.IMG_HEIGHT))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
    grid = np.zeros(config.IMG_HEIGHT*config.IMG_WIDTH)
    for polygon in polygons:
        path = Path(polygons[polygon])
        grid = grid.astype(float)
        grid += path.contains_points(points).astype(float)
        
    grid = grid.reshape((config.IMG_HEIGHT, config.IMG_WIDTH))
    grid = grid.astype(bool).astype(float)

    filename = filename.split('/')[-1]
    mask_file_name = os.path.join(config.MASK_DIRECTORY, filename.split('.')[0]+'.npy')
    # print(mask_file_name)
    np.save(mask_file_name , grid)


def crop_images(directory_path):
    def _crop_images(image_path):
        img = plt.imread(image_path)
        crop_img = img[:config.IMG_HEIGHT - 30, :] 
        plt.imsave(image_path, crop_img)
        
    '''
    for a given directory crop images from bottom to get rid of noise that we had because of camera.
    Args: directory_path(str)
    '''
    for image_path in os.listdir(directory_path):
        image_path = os.path.join(directory_path , image_path)
        _crop_images(image_path)


def split_train_val(data):
    '''
    Splitting the data into train and validation set for u-net segmentation model.
    Args: 
        data
    Returns:
        (tr_ds, mask_ds), (val_ds, mask_val_ds)
    '''
    def _split(data, init_step, len_new):
        x_ds, y_ds = list(), list()
        for i in range(len_new):
            x_ds.append(data[i + init_step][0])
            y_ds.append(data[i + init_step][1])
        y_ds = np.array(y_ds).reshape(len_new, 128, 128, 1)

        return x_ds, y_ds
        

    data_len = len(data)
    tr_len = int(data_len * 0.75)
    val_len = data_len - tr_len

    tr_ds, mask_ds = _split(data, 0, tr_len)
    val_ds, val_mask_ds = _split(data, tr_len, val_len)
    
    return (tr_ds, mask_ds), (val_ds, val_mask_ds)



