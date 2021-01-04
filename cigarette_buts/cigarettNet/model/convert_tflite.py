import argparse
import sys
sys.path.append("..")
import config
import tensorflow as tf



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--model_path', help='give .h5 model file path', type=str, default=None)
    parser.add_argument('--new_tflite_path', help='give a new path to store the tflite', type=str, default=None)   
    args = parser.parse_args()


    assert args.model_path is not None
    assert args.new_tflite_path is not None

    model = tf.keras.models.load_model(args.model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(args.new_tflite_path, "wb").write(tflite_model)

