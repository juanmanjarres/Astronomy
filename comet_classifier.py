import keras.activations
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from astropy.io import fits
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate

from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tf_fits.image import image_decode_fits

import random
import glob

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

def CNN():
    input_list = []
    # Creates the CNN for 5 images
    for i in range(5):
        input_list.append(Input(shape=(1024, )))
    merged = Concatenate()(input_list)
    dense1 = Dense(5, input_dim=5, activation='sigmoid', use_bias=True)(merged)
    output = Dense(1, activation='relu')(dense1)
    model = Model(inputs=input_list, outputs=output)
    print("Model Summary:")
    print("==============")
    print(model.summary())

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


def train_CNN(images):
    batch_size = 30
    epochs = 100

    img_data_npy = []
    expected_res = []
    # If train_CNN is called, then we know the number of images is 5
    for i in range(5):
        #data = images[i][1].data  # This is a numpy.ndarray
        #print("The data on image: " + str(data))
        #img_data_npy.append(data)
        img_data_npy.append(images[i])
        expected_res.append(tf.Tensor(True, dtype=bool))
    model = CNN()

    #print("The image data:" + str(img_data_npy))
    model.fit(img_data_npy, expected_res,  batch_size=batch_size, epochs=epochs)




fits_image = fits.open("NASA_data/train/cmt0038/22257681.fts")
image_data = fits.getdata("NASA_data/train/cmt0038/22257681.fts", ext=0)

filepath = ""



for comet_number in range(38, 2000):
    if 1000 > comet_number >= 100:
        filepath = "NASA_data/train/cmt0" + str(comet_number) + "/*.fts"
    elif comet_number < 100:
        filepath = "NASA_data/train/cmt00" + str(comet_number) + "/*.fts"
    else:
        filepath = "NASA_data/train/cmt" + str(comet_number) + "/*.fts"

    number_of_images = 0
    images = []
    for filename in glob.glob(filepath):
        with fits.open(filename) as opened_img:
            number_of_images += 1

            # Normalizes imgs to the exposure time
            #TODO figure out how to use normalized imgs with the tf-fits
            exposure_time = opened_img[0].header['EXPTIME']
            normalized_img = opened_img[0].data / exposure_time
            #images.append(normalized_img)

            img = tf.io.read_file(filename)
            img = image_decode_fits(img, 0)  # 0 for the header

            images.append(img)

            #print(opened_img.info())

    print("Number of images for comet " + str(comet_number) + " :" + str(number_of_images))
    if number_of_images == 5:
        # Create the neural network with 5 inputs
        train_CNN(images)



    



plt.figure()
plt.imshow(image_data, cmap='gray')
plt.colorbar()
plt.show()
