# TerraNet
***Detecting Objects In High-Resolution Multi-Spectral Satellite Imagery***

# Table of contents
* [Introduction](https://github.com/robmarkcole/satellite-image-deep-learning#datasets)
* [Interesting deep learning projects](https://github.com/robmarkcole/satellite-image-deep-learning#interesting-deep-learning-projects)
* [Techniques](https://github.com/robmarkcole/satellite-image-deep-learning#techniques)

## Introduction
U-net semantic segmentation(end-to-end fully convolutional network) is implemented with *Keras* functional API, which makes it extremely easy to experiment with different interesting architectures. Output from the network is a 64*64 which represents tile masks that should be learned. Sigmoid activation function makes sure that mask pixels are in [0, 1] range.


## Installation

First of all, after connecting to server activate the virtual environment and check whether you have required packages or not in order to run the app. If not then install requirements.
```
source ~/venvs/deep_env_0/bin/activate
cd /home/rdc/terranet
pip3 install -r requirements.txt
```
Then `cd` into terranet directory and run `app.py -h` to see the usage and options.
```
cd /home/rdc/terranet/terranet
python3 app.py -h
```



## Usage
```
usage: app [-h] [--metadata] [--enhance] [--tile] [--no-augment] [--no-mixup]
           [--clean_run] [--dilated] [--model_name MODEL_NAME]
           [--model_path MODEL_PATH] [--scaler_path SCALER_PATH] [--train]
           [--f1] [--hem] [--loss LOSS] [--ks KS] [--predict] [--no-rotate]
           [--burn] [--image_path IMAGE_PATH] [--shapefile SHAPEFILE]
           [--burn_dir BURN_DIR] [--name_tag NAME_TAG] [--evaluate]
```
##### Options

* `--help` - shows the help message, usage and options.
* `--metadata` - loading metadata to generate tiles and masks later on.
* `--tile` - generates tiles and masks from TIF files.
* `--enhance` - enhancing the images with ***Clahe Histogram Equalization*** method.
* `--no-augment` - not any ***Flip, Rotate, Gamma, Brightness*** augmentation will be used during training phase.
* `--no-mixup` - augmentation by projecting a shapefile over background image will not be used during training phase.
* `--clean_run` - clean previous models data.
* `--dilated` - using ***dilated convolution neural network*** during training phase
* `--model_name MODEL_NAME` - specify the model name that can be used for later purposes.
* `--train` - train a model with ***U-net semantic segmentation***.
*  `--f1` - custom callback will be used during training phase.
*  `--hem` - Hard Example Mining can be used while training.
*  `--loss LOSS_NAME` - specify loss function.
*  `--ks KS_SIZE` - specify maximum kernel size.
*  `--predict` - predict given TIF file with specified *model name*.
*  `--no-rotate` - not any rotation will be used while predicting.
*  `--burn` - burn mask onto JPG file after prediction.
*  `--evaluate` - evaluate the model.

#### Usage example:


```
python /home/rdc/terranet/terranet/app.py --metadata --enhance --tile --train --f1 --model_name
BASE_AZE_NO_HEM_BLKFE --loss balanced_base_loss --ks 7 --clean_run --predict --evaluate --burn 
```


