#!/usr/bin/env python
import os
from PIL import Image
import numpy as np
import sys
import codecs
import pandas as pd
import pickle
import cPickle
import statsmodels.api as sm
import scipy.stats
from scipy.stats import entropy
from statsmodels.iolib.summary2 import summary_col
from itertools import izip
from keras.utils import np_utils

os.chdir("/mnt/saswork/sh2264/vision/code")
# combine X_train_resnet_folder.npy's and X_train_resnet_index_folder.npy's in subfolders 
# into X_train_resnet.npy and X_train_index_resnet.npy under /code folder
# and pre-processs both save as:
# ("X_train_resnet_processed.npy")
# ("X_train_name_resnet_processed.npy")
# ("y_train_resnet_processed.npy")


#folder = sys.argv[1]
#img_color = Image.open(image_file)
#img_grey = img_color.convert('L')
#img_color = np.array(img_color)
#img_grey = np.array(img_grey)

input_dir = "/mnt/saswork/sh2264/vision/data/"
#input_dir = "/Users/sheng/image"

directories = os.listdir(input_dir)

index = 0
#index2 = 0

for folder in directories:
	if folder == '.DS_Store':
		pass

	else:
		print "folder "+folder
		#images = os.listdir(input_dir + '/' + folder)
		os.chdir(input_dir + '/' + folder)
		index += 1

		try:
			X_train_sub = np.load('X_train_resnet_'+folder+'.npy')
			X_train_index_sub = np.load('X_train_resnet_index_'+folder+'.npy')
		except IOError:
			continue
		
		if index == 1:
			X_train = X_train_sub
			X_train_index = X_train_index_sub
		else:
			X_train = np.concatenate((X_train, X_train_sub), axis = 0)
			X_train_index = np.concatenate((X_train_index, X_train_index_sub), axis = 0)


os.chdir("/mnt/saswork/sh2264/vision/code")
np.save("X_train_resnet.npy", X_train)
np.save("X_train_index_resnet.npy", X_train_index)

# and pre-processs both save as:
# ("X_train_resnet_processed.npy")
# ("X_train_name_resnet_processed.npy")
# ("y_train_resnet_processed.npy")

out = X_train
index_array = X_train_index
ca = np.load("category_crosswalk.npy")
ca = pd.DataFrame({'name': ca[:,1], 'category': ca[:,0], 'index': [int(x) for x in ca[:,2]]})
co = np.load("country_crosswalk.npy")
co = pd.DataFrame({'name': co[:,1], 'country': co[:,0], 'index': [int(x) for x in co[:,2]]})
X_train_name = pd.DataFrame({'name': [index.tolist()[0] for index in index_array]})

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

y_train_category_label = X_train_class_label['index']

X_train_class_label = np.array(X_train_class_label)

# take a 5% subsample for now 'cuz memory
y_train_sample = y_train_category_label#[::20]
# down to 7411 logos

X_train = np.array([ out[index] for index in y_train_sample.index ])
X_train_name = np.array([ index_array[index] for index in y_train_sample.index ])


# slice into equal bins, let's say, 10
#num_bins = 10
#for i in xrange(num_bins):
#	np.save('data_batch_%d.bin' % i, np.array_split(X_train, num_bins)[i])

# normalization
X_train = X_train.view('uint8') #rather than:X_train.astype('uint8'), which is more memory efficient and faster: https://stackoverflow.com/questions/1888870/numpy-how-to-convert-an-array-type-quickly
# one hot encode outputs
#y_train = np_utils.to_categorical(np.array(y_train))
y_train = np_utils.to_categorical(np.array(y_train_sample))

print "data augmentation..."
# datagen = ImageDataGenerator(
# 	featurewise_center=True, 
# 	featurewise_std_normalization=True,
# 	zca_whitening = True,
# 	rotation_range = 90,
# 	width_shift_range=0,
# 	height_shift_range=0,
# 	horizontal_flip = True,
# 	vertical_flip=True
# 	)


num_classes = y_train.shape[1]
# 39

X_train = X_train/255.0
### saved to code folder!!! ###
np.save("X_train_resnet_processed.npy",X_train)
np.save("X_train_name_resnet_processed.npy",X_train_name)
np.save("y_train_resnet_processed.npy",y_train) # this is the one-hot vector