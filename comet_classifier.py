import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from astropy.io import fits
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Convolution2D, MaxPooling2D

import random
import glob

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


def read_savefile(filename):
    '''Read npy save file containing images or labels of galaxies'''
    return np.load(filename)


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

    for filename in glob.glob(filepath):
        with fits.open(filename) as opened_img:
            print(opened_img.info())

    



plt.figure()
plt.imshow(image_data, cmap='gray')
plt.colorbar()
plt.show()
