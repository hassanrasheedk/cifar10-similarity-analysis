import random
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D
from keras.layers import MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Activation, Concatenate
from keras import optimizers
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model, model_to_dot
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from IPython.display import SVG
from matplotlib import pyplot as plt
import math
from PIL import Image
import imageio
import os
import sys
import argparse
import numpy as np

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1-y_true) * margin_square)


parser = argparse.ArgumentParser(description='''Give path to an RGB image file to find similar images in the CIFAR-10 dataset''')
required_named = parser.add_argument_group('required named arguments')
required_named.add_argument(
    '-i', '--image_path', help='Path to image', required=True)
required_named.add_argument(
    '-d', '--data_path', help='Path to the CIFAR-10 data', required=True)
args = parser.parse_args()

# Take input files

image_path = args.image_path
data_path = args.data_path

if os.path.exists(data_path):
    print('\nVerified, images in path:' + str(len([i for i in os.listdir(data_path) if '.png' in i])))
else:
    print("Error! Images path does not exist.")
    sys.exit()

if not os.path.exists(image_path):
    print("Error reading image at the specified path")
    sys.exit()

 
test_image = imageio.imread(image_path)
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
test_image = test_image.reshape(1, 32, 32, 3)
model = load_model('model.hfs5', custom_objects={'contrastive_loss': contrastive_loss})

similar_images = []

for image_name in os.listdir(data_path):
    ref_image_path = os.path.join(data_path, image_name)
    ref_image = imageio.imread(ref_image_path)
    ref_image = np.array(ref_image)
    ref_image = ref_image.astype('float32')
    ref_image /= 255
    ref_image = ref_image.reshape(1, 32, 32, 3)
    prediction = model.predict([test_image, ref_image])
    pred_tuple = (ref_image_path, prediction[0][0])
    print(pred_tuple)
    similar_images.append(pred_tuple)

similar_images_sorted = sorted(similar_images, key=lambda x: x[1])

print([i[0] for i in similar_images_sorted[:5]])
