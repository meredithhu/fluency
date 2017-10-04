#!/usr/bin/env python
print "system argument 1: number of training epochs"
print "system argument 2: which model"

import os
from PIL import Image
import numpy as np
import sys
import codecs
import pandas as pd
import pickle

#from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
np.random.seed(seed)
os.chdir("/mnt/saswork/sh2264/vision/code")

out = np.load('X_train.npy')
index_array = np.load('X_train_index.npy')
ca = np.load("category_crosswalk.npy")
ca = pd.DataFrame({'name': ca[:,1], 'category': ca[:,0], 'index': [int(x) for x in ca[:,2]]})
co = np.load("country_crosswalk.npy")
co = pd.DataFrame({'name': co[:,1], 'country': co[:,0], 'index': [int(x) for x in co[:,2]]})
#X_train_name = pd.DataFrame({'name': index_array})

def mergeLeftInOrder(x, y, on=None):
	x = x.copy()
	x["Order"] = np.arange(len(x))
	z = x.merge(y, how='left', on=on).sort_values(by="Order")
	return z.drop("Order", 1)

# 276108 logos down to 251793?
X_train_category = mergeLeftInOrder(X_train_name, ca, on="name")
#np.save('X_train_category.npy', np.array( [ [x] for x in X_train_category['index'] ] ) )
X_train_country = mergeLeftInOrder(X_train_name, co, on="name")
#np.save('X_train_country.npy', np.array( [ [x] for x in X_train_country['index'] ] ) )


# remove null records
X_train_class_label = X_train_category[X_train_category.category.notnull()]
# 251793 logos down to 148220

y_train_category_label = X_train_class_label['index'] # category index, not instance index

X_train_class_label = np.array(X_train_class_label)

# take a 5% subsample for now 'cuz memory
y_train_sample = y_train_category_label#[::20]
# down to 7411 logos

X_train = np.array([ out[index] for index in y_train_sample.index ])
X_train_name = np.array([ index_array[index] for index in y_train_sample.index])

# slice into equal bins, let's say, 10
#num_bins = 10
#for i in xrange(num_bins):
#	np.save('data_batch_%d.bin' % i, np.array_split(X_train, num_bins)[i])

# normalization
X_train = X_train.astype('float32')
# one hot encode outputs
#y_train = np_utils.to_categorical(np.array(y_train))
y_train = np_utils.to_categorical(np.array(y_train_sample))

print "data augmentation..."
datagen = ImageDataGenerator(
	featurewise_center=True, 
	featurewise_std_normalization=True,
	zca_whitening = True,
	rotation_range = 90,
	width_shift_range=0,
	height_shift_range=0,
	horizontal_flip = True,
	vertical_flip=True
	)


num_classes = y_train.shape[1]
# 39

X_train = X_train/255.0

train_x = X_train[:118576] # 80%
train_y = y_train[:118576] # 80%
train_x_name = X_train_name[:118576]
test_x = X_train[118576:] # 20%
test_y = y_train[118576:] # 20%
# 7411*0.8=5929
#train_x = X_train[:5929] # 80%
#train_y = y_train[:5929] # 80%
#test_x = X_train[5929:] # 20%
#test_y = y_train[5929:] # 20%