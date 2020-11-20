import cv2
import config
import argparse
from model.unet import *
from dataset.utils import *
from dataset.augment import *
from dataset.dataset import Dataset
from model.predict import predict_img
from tensorflow.keras.models import load_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--train', action='store_true', dest='flag_train')
    parser.add_argument('--predict', action='store_true', dest='flag_predict')
    parser.add_argument('--model_path', help='give .h5 model file path', type=str, default=None)
    parser.add_argument('--img_path', help='image path', type=str, default=None)
    
    args = parser.parse_args()

    if args.flag_train is True:
        db = Dataset(config.DB_PATH)
        util = Utils(db.load_paths(db.con, 600)) 
        train_ds = util.load_images()
        random.shuffle(train_ds)

        train_ds = augment(train_ds, 0, 200, flip_augment)
        train_ds = augment(train_ds, 609, 1000, rotate_augment)
        train_ds = over_bg_augment(train_ds, 0, 1000)

        resized_ds = list()

        for i in range(len(train_ds)):
            img = cv2.resize(train_ds[i][0], dsize=(128, 128))
            mask = cv2.resize(train_ds[i][1], dsize=(128, 128))
            resized_ds.append((img, mask.astype(bool).astype(float)))

        base_model = get_model(resized_ds, **config.UNET_CONFIG)
        base_model[1].save('model.h5')

       

    if args.flag_predict is True:
        assert args.model_path is not None
        assert args.img_path is not None
        #predict the image
        thresh_val = 0.34
        base_model = load_model(args.model_path)
        predict_img(img_path, 0.34, base_model)


