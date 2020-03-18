from .unet import *


import os
import sys
import time
import random
import pprint
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate
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

    assert np.mod(np.log2(start_filter), 1) == 0.0
    assert np.mod(np.log2(middle_filter), 1) == 0.0
    assert middle_filter >= start_filter

    depth = np.int(np.log2(middle_filter) - np.log2(start_filter))

    # X_train = data[0][0]
    # y_train = data[0][1]
    # X_val = data[1][0]
    # y_val = data[1][1]

    input_channels = X_train.shape[-1]
    output_channels = y_train.shape[-1]

    log_dir, weight_dir = _get_log_weight_dirs(name, dt)
    utils.instantiate_store(weight_dir)
    weight_path = str(os.path.join(weight_dir, 'weight.h5'))

    inputs = Input(shape=(config.IMG_WIDTH, config.IMG_HEIGHT, input_channels), name="inputs")
    temp = Lambda(lambda x: x / config.IMG_MAX_VAL, name="normalize")(inputs)

    k_init = tf.initializers.glorot_normal(seed=config.TERRANET_SEED)
    b_init = tf.initializers.glorot_normal(seed=config.TERRANET_SEED)
    dconvs = list()

    i = 0
    for i in range(depth):
        dconv_1 = Conv2D(filters=start_filter * np.power(2, i), kernel_size=(3, 3), strides=1, padding='same',
                         activation='relu', kernel_initializer=k_init, bias_initializer=b_init,
                         name=f'dconv_1_level_{i}')(temp)
        dconv_2 = Conv2D(filters=start_filter * np.power(2, i), kernel_size=(3, 3), strides=1, padding='same',
                         activation='relu', kernel_initializer=k_init, bias_initializer=b_init,
                         name=f'dconv_2_level_{i}')(dconv_1)
        dconvs.append(dconv_2)
        pool = MaxPooling2D((2, 2))(dconv_2)
        drop = Dropout(drop_rate, seed=config.TERRANET_SEED)(pool)
        temp = drop

    convm_1 = Conv2D(filters=start_filter * np.power(2, i + 1), kernel_size=(3, 3), strides=1, padding='same',
                     activation='relu', kernel_initializer=k_init, bias_initializer=b_init, name='convm_1')(temp)
    convm_2 = Conv2D(filters=start_filter * np.power(2, i + 1), kernel_size=(3, 3), strides=1, padding='same',
                     activation='relu', kernel_initializer=k_init, bias_initializer=b_init, name='convm_2')(convm_1)

    temp = convm_2
    for i in range(depth - 1, -1, -1):
        deconv = Conv2DTranspose(start_filter * np.power(2, i), (4, 4), strides=(2, 2), padding="same",
                                 kernel_initializer=k_init, bias_initializer=b_init, name=f'deconv_level_{i}')(temp)
        concat = concatenate([deconv, dconvs.pop()], name=f'concat_level_{i}')
        drop = Dropout(drop_rate, seed=config.TERRANET_SEED)(concat)
        uconv_1 = Conv2D(start_filter * np.power(2, i), (3, 3), activation="relu", padding="same",
                         kernel_initializer=k_init, bias_initializer=b_init, name=f'uconv_1_level_{i}')(drop)
        uconv_2 = Conv2D(start_filter * np.power(2, i), (3, 3), activation="relu", padding="same",
                         kernel_initializer=k_init, bias_initializer=b_init, name=f'uconv_2_level_{i}')(uconv_1)
        temp = uconv_2

    outputs = Conv2D(output_channels, (1, 1), padding="same", activation="sigmoid", kernel_initializer=k_init,
                     bias_initializer=b_init, name="outputs")(temp)

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
                        callbacks=[tensorboard_callback, early_stopping, reduce_lr, model_checkpoint, custom_metrics],
                        shuffle=False,
                        verbose=1)

    model.load_weights(weight_path)


    return history, model
