import os
import json
import datetime as dt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from tensorflow.keras.backend import set_session
set_session(session)

from tensorflow import keras

import pandas as pd
import seaborn as sns

# from keras.applications.xception import Xception
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19
# from keras.applications.resnet50 import ResNet50
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.mobilenet import MobileNet
# from keras.applications.densenet import DenseNet121
# from keras.applications.densenet import DenseNet169
# from keras.applications.densenet import DenseNet201
# from keras.applications.nasnet import NASNetLarge
# from keras.applications.nasnet import NASNetMobile
# from keras.applications.mobilenet_v2 import MobileNetV2

from keras.applications.densenet import DenseNet121, preprocess_input
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import (categorical_accuracy, categorical_crossentropy,
                           top_k_categorical_accuracy)
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
# from tensorflow.keras.applications.nasnet import preprocess_input
start = dt.datetime.now()

BASE_SIZE = 256
NCSVS = 100
NCATS = 340
np.random.seed(seed=1987)
tf.set_random_seed(seed=1987)

def f2cat(filename: str) -> str:
    return filename.split('.')[0]

def list_all_categories():
    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)

print(list_all_categories())

STEPS = 1000
EPOCHS = 16
size = 64
batchsize = 256

base_model = DenseNet121(include_top=False, weights=None,
                         input_shape=(size, size, 3), classes=NCATS)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(NCATS, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=[x, predictions])
model.load_weights("/Users/justryit/Desktop/dense121_back.h5")

# model.compile(optimizer=Adam(lr=1e-4, decay=1e-9), loss='categorical_crossentropy', metrics=[
#               categorical_crossentropy, categorical_accuracy, top_3_accuracy])

print(model.summary())

def readimage(img_path, size=64):
    img = cv2.imread(img_path)
    x = np.zeros((1, size, size, 3))
    x[0, :, :, :] = cv2.resize(img, (size, size))
    x = preprocess_input(x).astype(np.float32)
    return x

IMAGE_FILE = '3.png'

# test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
# test.head()
# x_test = df_to_image_array_xd(test, size)
# print(test.shape, x_test.shape)
# print('Test array memory {:.2f} GB'.format(x_test.nbytes / 1024.**3 ))

x_test = readimage(IMAGE_FILE)
hidden, test_predictions = model.predict(x_test, batch_size=1, verbose=1)


incat = 'book'

cats = list_all_categories()
id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
cat2id = {cat: k for k, cat in enumerate(cats)}
print(cat2id[incat])

print(test_predictions[0])
print(hidden[0])