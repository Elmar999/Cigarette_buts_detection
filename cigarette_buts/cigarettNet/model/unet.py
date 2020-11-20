from .unet import *


import os
import sys
import time
import random
import pprint
import config
from dataset.utils import *
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K



class CustomMetrics(Callback):
    def __init__(self, model, X_val, y_val, num_thresholds):
        super(CustomMetrics, self).__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.num_thresholds = num_thresholds

    def on_epoch_end(self, epoch, logs={}):
        num_thresholds = self.num_thresholds
        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
        custom_val_metrics = list()

        y_pred = tf.reshape(self.model.predict(self.X_val), [-1])
        y_true = tf.reshape(self.y_val, [-1])

        for thresh in thresholds:
            pred_classes = tf.cast(tf.math.greater(y_pred, tf.constant([thresh])), dtype=tf.float32)
            cm = tf.math.confusion_matrix(y_true, pred_classes, num_classes=2)

            tn = cm[0][0].numpy()
            fp = cm[0][1].numpy()
            fn = cm[1][0].numpy()
            tp = cm[1][1].numpy()

            precision = tp / (tp + fp + K.epsilon())
            recall = tp / (tp + fn + K.epsilon())
            f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

            payload = \
                OrderedDict([('val_auc', logs.get('val_auc')),
                             ('val_f1', f1),
                             ('precision', precision),
                             ('recall', recall),
                             ('val_tp', tp),
                             ('val_tn', tn),
                             ('val_fp', fp),
                             ('val_fn', fn),
                             ('thresh', thresh)])

            custom_val_metrics.append(payload)
        best_val_f1 = sorted(custom_val_metrics, key=lambda x: x['val_f1'], reverse=True)[0]
        pprint.pprint(best_val_f1)

        
def downsample(x, dconvs, depth,  kernel_size = [(7, 7) , (5, 5) , (5, 5) , (3, 3)]):
    start_filter = 16
    drop_rate = 0.3
    k_init = tf.initializers.glorot_normal(seed=313)
    b_init = tf.initializers.glorot_normal(seed=313)
    i = 0
    for i in range(depth):
        x = Conv2D(filters=start_filter * np.power(2, i), kernel_size= kernel_size[i], strides=1, padding='same',
                            activation='relu', kernel_initializer=k_init, bias_initializer=b_init,
                            name=f'dconv_1_level_{i}')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=start_filter * np.power(2, i) , kernel_size=kernel_size[i], strides=1, padding='same',
                            activation='relu', kernel_initializer=k_init, bias_initializer=b_init,
                            name=f'dconv_2_level_{i}')(x)
        x = BatchNormalization()(x)
        dconvs.append(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(drop_rate, seed=313)(x)
    return x, dconvs


def bottleneck(x , depth):
    drop_rate = 0.3
    k_init = tf.initializers.glorot_normal(seed=313)
    b_init = tf.initializers.glorot_normal(seed=313)
    start_filter = 16
    convm_1 = Conv2D(filters=start_filter * np.power(2, depth), kernel_size=(3, 3), strides=1, padding='same',
                     activation='relu', kernel_initializer=k_init, bias_initializer=b_init, name='convm_1')(x)
    convm_2 = Conv2D(filters=start_filter * np.power(2, depth), kernel_size=(3, 3), strides=1, padding='same',
                     activation='relu', kernel_initializer=k_init, bias_initializer=b_init, name='convm_2')(convm_1)
   
    return convm_2

def Upsample(x, dconvs, depth, kernel_size = [(3, 3), (5, 5), (5, 5), (7, 7)]):
    start_filter = 16
    drop_rate = 0.3
    k_init = tf.initializers.glorot_normal(seed=313)
    b_init = tf.initializers.glorot_normal(seed=313)
    for i in range(depth - 1, -1, -1):
        deconv = Conv2DTranspose(start_filter * np.power(2, i), kernel_size=kernel_size[i], strides=(2, 2), padding="same",
                                 kernel_initializer=k_init, bias_initializer=b_init, name=f'deconv_level_{i}')(x)
        
        # deconv = UpSampling2D((2,2) , interpolation='bilinear')(x)
        x = concatenate([deconv, dconvs.pop()], name=f'concat_level_{i}')
        x = Dropout(drop_rate, seed=313)(x)
        x = Conv2D(start_filter * np.power(2, i),kernel_size=kernel_size[i], activation="relu", padding="same",
                kernel_initializer=k_init, bias_initializer=b_init, name=f'uconv_1_level_{i}')(x)
        x = BatchNormalization()(x)
        x = Conv2D(start_filter * np.power(2, i) ,kernel_size=kernel_size[i], activation="relu", padding="same",
                kernel_initializer=k_init, bias_initializer=b_init, name=f'uconv_2_level_{i}')(x)
        x = BatchNormalization()(x)
       
    return x



def get_model(data, name="UNET-BASE", **params):
    def _get_log_weight_dirs(name, dt):
        _dt = dt.strftime("%Y_%m_%d_T%H_%M_%S")
        relative_dt_dir = os.path.join(name, _dt)
        dt_dir = os.path.join(config.MODEL_OUT_DIR, relative_dt_dir)
        return os.path.join(dt_dir, 'logs'), os.path.join(dt_dir, 'weights')


    start_filter = params.get('START_FILTER')
    middle_filter = params.get('MIDDLE_FILTERS')
    drop_rate = params.get('DROP_RATE')
    epochs = params.get('EPOCHS')
    batch_size = params.get('BATCH_SIZE')
    num_thresholds = params.get('NUM_THRESHOLDS')

    MODEL_OUT = '/home/elmar/Cigarette_buts_detection/cigarette_buts/cigarettNet/model/out'
    assert np.mod(np.log2(start_filter), 1) == 0.0
    assert np.mod(np.log2(middle_filter), 1) == 0.0
    assert middle_filter >= start_filter

    depth = np.int(np.log2(middle_filter) - np.log2(start_filter))
    
    (tr_ds, mask_ds),(val_ds, val_mask_ds) = split_train_val(data)
    
    
    X_train = tf.cast(np.array(tr_ds), tf.float32)
    y_train = tf.cast(mask_ds, tf.float32)
    X_val = tf.cast(np.array(val_ds), tf.float32)
    y_val = tf.cast(val_mask_ds, tf.float32)


    input_channels = X_train.shape[-1]
    output_channels = y_train.shape[-1]

    weight_dir = MODEL_OUT
    log_dir = MODEL_OUT
    weight_path = str(os.path.join(weight_dir, 'weight.h5'))

    inputs = Input(shape=(128, 128, 3), name="inputs")
    temp = Lambda(lambda x: x / config.IMG_MAX_VAL, name="normalize")(inputs)

    k_init = tf.initializers.glorot_normal(seed=313)
    b_init = tf.initializers.glorot_normal(seed=313)
    dconvs = list()

    x, dconvs = downsample(temp, dconvs, depth)
    x = bottleneck(x, depth)
    x = Upsample(x, dconvs, depth)


    outputs = Conv2D(output_channels,(1, 1), padding="same", activation="sigmoid", kernel_initializer=k_init,
                     bias_initializer=b_init, name="outputs")(x)


    model = Model(inputs=inputs, outputs=outputs)


    auc = AUC(num_thresholds=num_thresholds, curve='PR', name='auc')
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[auc])

    tensorboard_callback = TensorBoard(log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
    model_checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1,
                                       save_weights_only=True)
    custom_metrics = CustomMetrics(model, X_val, y_val, num_thresholds)

    history = model.fit(X_train, y_train,
                        validation_data=[X_val, y_val],
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[model_checkpoint, reduce_lr, custom_metrics, tensorboard_callback],
                        shuffle=False,
                        verbose=1)

    model.load_weights(weight_path)


    return history, model
