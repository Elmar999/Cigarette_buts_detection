import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt




def view_img_mask(img, mask, thres_val, model=False):
    if model:
        pred = model.predict(np.expand_dims(img, axis=0)).reshape((1, img.shape[0], img.shape[1], 1))
        pred = np.squeeze(pred)
        fig, ax = plt.subplots(1 , 3, figsize=(18, 8))
        ax[0].imshow(img)
        ax[1].imshow(mask)
        ax[2].imshow(pred > thres_val)
    else:
        fig, ax = plt.subplots(1 , 2, figsize=(12, 8))
        ax[0].imshow(img)
        ax[1].imshow(mask)
        plt.show()


def predict_img(img_path, thres_val, base_model=False):
    img = plt.imread(img_path)
    img = cv2.resize(img, (128, 128))
    pred = base_model.predict(np.expand_dims(img, axis=0)).reshape((1, 128, 128, 1))
    pred = np.squeeze(pred)
    fig, ax = plt.subplots(1 , 2, figsize=(18, 8))
    ax[0].imshow(img)
    ax[1].imshow(pred > thres_val)
    plt.show()


def video_predict(file_path, base_model, thresh_val):
    '''
    Read video and predict on each frame.
    Args:
        filename(str)
        model(h5 file)
        thresh_val(int): threshold value
    '''
    def getFrame(sec , img_mask_ds, video_path):

        WIDTH, HEIGHT = 128, 128
        vidcap = cv2.VideoCapture(file_path)
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            image_real = image
            image = cv2.resize(image, (WIDTH, HEIGHT))
            pred_img = base_model.predict(np.expand_dims(image, axis=0)).reshape((1, WIDTH, HEIGHT, 1))
            pred_img = np.squeeze(pred_img)
            img_mask_ds.append((image, (pred_img > thresh_val).astype(float)*255))

        return hasFrames, img_mask_ds

    img_mask_ds = list()
    sec = 0
    frameRate = 0.5 #//it will capture image in each 0.5 second
    count = 1
    success = getFrame(sec, img_mask_ds)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success, img_mask_ds = getFrame(sec, img_mask_ds)

    return img_mask_ds