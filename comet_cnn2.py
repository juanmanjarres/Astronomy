import os.path

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

import visualkeras

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


def train_CNN(image_set, truth, filenames):
    batch_size = 30
    epochs = 70

    print("Filenames:")
    print(filenames)
    print("#### Image_set shape:" + str(image_set.shape))
    # print(image_set)
    img_data_npy = []
    expected_res = []

    comet_count = 0
    for comet_num in range(len(image_set)):
        if (comet_num + 38) in truth.keys():
            comet_res = []
            img_data_per_comet = []
            for img_num in range(len(image_set[0])):
                # data = images[i][1].data  # This is a numpy.ndarray
                # print("The data on image: " + str(data))
                # img_data_npy.append(data)
                # TODO fix this, appending one per image instead of one per comet
                img_data_per_comet.append(image_set[comet_num][img_num])
                print("Truth values: ")
                print(truth)
                image_filename = filenames[comet_count][img_num]
                print("Index: " + str(comet_num + 38))
                print("Image filename:")
                print(image_filename)
                comet_res.append(truth[comet_num + 38][image_filename])
            img_data_npy.append(np.asarray(img_data_per_comet))
            expected_res.append(np.asarray(comet_res))
            comet_count += 1

    train_dataset = tf.data.Dataset.from_tensor_slices((img_data_npy, expected_res))

    # input_list = []
    # for i in range(5):
    #     input_list.append(Input(shape=(1024, 1024, )))
    input = Input(shape=(1024, 1024, ))
    #merged = Concatenate()(input_list)
    dense1 = Dense(7)(input)
    dense2 = Dense(7)(dense1)
    flatten = Flatten()(dense2)

    output = Dense(2)(flatten)

    model = Model(inputs=input, outputs=output)

    # model = tf.keras.Sequential([
    #     [Input(shape=(1024, 1024)),
    #      Input(shape=(1024, 1024)),
    #      Input(shape=(1024, 1024)),
    #      Input(shape=(1024, 1024)),
    #      Input(shape=(1024, 1024))],
    #     Flatten(),
    #     Dense(128),
    #     Dense(2)
    # ])

    model.compile(optimizer='adam',
                  loss='mean_absolute_error',
                  metrics=['accuracy'])

    #visualkeras.layered_view(model).show()
    history = model.fit(train_dataset, batch_size=batch_size, epochs=epochs)

    plt.plot(history.history['accuracy'])
    plt.title('Mean Squared Error accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(history.history['loss'])
    plt.title("Mean Squared Error loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


filepath = ""

truth_file = open("NASA_data/train-gt.txt", 'r')
truth_data = {}
for line in truth_file:
    info = line.split(',')
    # each line contains {cometid} [{image_name}, {x_coord}, {y_coord}] ... {confidence}
    comet_id = info[0]
    truth_data[comet_id] = {}

    # this accounts for the initial cometid and confidence at the end
    # and because there is 3 objects per image
    img_amount = (len(info) - 2) // 3
    for i in range(0, img_amount):
        image_name = info[(3 * i) + 1]
        x_coord = info[(3 * i) + 2]
        y_coord = info[(3 * i) + 3]
        truth_data[comet_id][image_name] = [float(x_coord), float(y_coord)]

# print(truth_data)

set_5_image = []
set_5_image_truth = {}
set_5_image_filenames = []

# TODO change this to 38, 2000
for comet_number in range(38, 139):
    comet_truth = {}

    if 1000 > comet_number >= 100:
        filepath = "NASA_data/train/cmt0" + str(comet_number) + "/*.fts"
        comet_truth = truth_data["cmt0" + str(comet_number)]
    elif 10 <= comet_number < 100:
        filepath = "NASA_data/train/cmt00" + str(comet_number) + "/*.fts"
        comet_truth = truth_data["cmt00" + str(comet_number)]
    elif comet_number < 10:
        filepath = "NASA_data/train/cmt000" + str(comet_number) + "/*.fts"
        comet_truth = truth_data["cmt000" + str(comet_number)]
    else:
        filepath = "NASA_data/train/cmt" + str(comet_number) + "/*.fts"
        comet_truth = truth_data["cmt" + str(comet_number)]

    number_of_images = 0
    images = []
    image_names = []
    for filename in glob.glob(filepath):
        with fits.open(filename) as opened_img:
            number_of_images += 1

            # Normalizes imgs to the exposure time
            # TODO figure out how to use normalized imgs with the tf-fits
            exposure_time = opened_img[0].header['EXPTIME']
            normalized_img = opened_img[0].data / exposure_time
            # images.append(normalized_img)

            img = tf.io.read_file(filename)
            img = image_decode_fits(img, 0)  # 0 for the header

            images.append(img)
            image_names.append(os.path.basename(filename))

            # print(opened_img.info())

    print("Number of images for comet " + str(comet_number) + " :" + str(number_of_images))
    if number_of_images >= 5:
        # If the number of images is equal or more than 5, add the comet to the list for training
        set_5_image.append(images[0:5])
        set_5_image_truth[comet_number] = comet_truth
        set_5_image_filenames.append(image_names[0:5])

set_5_image = np.asarray(set_5_image)

train_CNN(set_5_image, set_5_image_truth, set_5_image_filenames)


images = ["NASA_data/train/cmt0039/22291228.fts",
          "NASA_data/train/cmt0039/22291229.fts",
          "NASA_data/train/cmt0039/22291230.fts",
          "NASA_data/train/cmt0039/22291231.fts",
          "NASA_data/train/cmt0039/22291232.fts"]

fig = plt.figure(figsize=(10, 7))

rows = 5
columns = 1
counter = 1
for img_filename in images:
    img_data = fits.getdata(img_filename, ext=0)
    fig.add_subplot(rows, columns, counter)
    plt.imshow(img_data, cmap='gray')
    plt.axis('off')
    counter += 1

#plt.show()
#image_data1 = fits.getdata("NASA_data/train/cmt0038/22257681.fts", ext=0)

#
# plt.figure()
# plt.imshow(image_data, cmap='gray')
# #plt.colorbar()
# plt.show()
