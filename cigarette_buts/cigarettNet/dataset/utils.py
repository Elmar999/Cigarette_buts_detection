import matplotlib.pyplot as plt
import numpy as np

class Utils:
    def __init__(self, paths):
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
