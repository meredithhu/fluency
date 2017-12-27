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

		memscore = []
		memname = []

		try:
			ms = codecs.open(input_dir+folder+"/memscore.txt",encoding = "utf-8")
			for line in ms:
				x = line.split("\t")
				try:
					memscore.append(float(x[1].split("\n")[0]))
					memname.append(x[0])
				except ValueError:
					pass

			mem = pd.DataFrame({'name':memname, 'mscore': memscore})
		except IOError:
			pass

		popscore = []
		popname = []

		try:
			pop = codecs.open(input_dir+folder+"/popscore.txt",encoding = "utf-8")
			for line in pop:
				x = line.split("\t")
				try:
					popscore.append(float(x[1].split("\n")[0]))
					popname.append(x[0])
				except ValueError:
					pass

			#adjusted_popscore = [x if x>0 else 0.016587944219523178 for x in popscore]
			pop = pd.DataFrame({'name':popname, 'pscore':popscore}) #'pscore':adjusted_popscore})
		except IOError:
			pass
		#entropy = pd.DataFrame({'name': entropies_name, 'mean':entropy_mean, 'std':entropy_std})

		if index == 1:
			merged = pd.merge(pop, mem, how="inner",on="name")#, merged_imagenet, how = "inner", on = "name")
		else:
			merged = merged.append(pd.merge(pop, mem, how="inner",on="name"))
		# merged.keys(): name, mscore, pscore, mean, std

		# 	X_train_sub = np.load('X_train_resnet_'+folder+'.npy')
		# 	X_train_index_sub = np.load('X_train_resnet_index_'+folder+'.npy')
		# except IOError:
		# 	continue
		
		# if index == 1:
		# 	X_train = X_train_sub
		# 	X_train_index = X_train_index_sub
		# else:
		# 	X_train = np.concatenate((X_train, X_train_sub), axis = 0)
		# 	X_train_index = np.concatenate((X_train_index, X_train_index_sub), axis = 0)


os.chdir("/mnt/saswork/sh2264/vision/code")
# np.save("X_train_resnet.npy", X_train)
# np.save("X_train_index_resnet.npy", X_train_index)

X_train = np.load("X_train_resnet.npy")
X_train_name = np.load("X_train_index_resnet.npy")
# and pre-processs both save as:
# ("X_train_resnet_processed.npy")
# ("X_train_name_resnet_processed.npy")
# ("y_train_resnet_processed.npy")
X_train_index = pd.DataFrame({'index': [x for x in range(len(X_train_name))], 'name': [x[0][:-5] for x in X_train_name]})

# out = X_train
# index_array = X_train_index
# ca = np.load("category_crosswalk.npy")
# ca = pd.DataFrame({'name': ca[:,1], 'category': ca[:,0], 'index': [int(x) for x in ca[:,2]]})
# co = np.load("country_crosswalk.npy")
# co = pd.DataFrame({'name': co[:,1], 'country': co[:,0], 'index': [int(x) for x in co[:,2]]})
# X_train_name = pd.DataFrame({'name': [index.tolist()[0] for index in index_array]})

def mergeLeftInOrder(x, y, on=None):
	x = x.copy()
	x["Order"] = np.arange(len(x))
	z = x.merge(y, how='left', on=on).sort_values(by="Order")
	return z.drop("Order", 1)


X_train_label = pd.merge(X_train_index, merged, how = "inner", on = "name")
# # 276108 logos down to 251793?
# X_train_category = mergeLeftInOrder(X_train_name, ca, on="name")
# #np.save('X_train_category.npy', np.array( [ [x] for x in X_train_category['index'] ] ) )
# X_train_country = mergeLeftInOrder(X_train_name, co, on="name")
# #np.save('X_train_country.npy', np.array( [ [x] for x in X_train_country['index'] ] ) )

X_train_matched = np.array([X_train[index] for index in X_train_label.index])


# # remove null records
# X_train_class_label = X_train_category[X_train_category.category.notnull()]
# # 251793 logos down to 148220
# y_train_category_label = X_train_class_label['index']
# X_train_class_label = np.array(X_train_class_label)

# # take a 5% subsample for now 'cuz memory
# y_train_sample = y_train_category_label#[::20]
# # down to 7411 logos

# X_train = np.array([ out[index] for index in y_train_sample.index ])
# X_train_name = np.array([ index_array[index] for index in y_train_sample.index ])


# slice into equal bins, let's say, 10
#num_bins = 10
#for i in xrange(num_bins):
#	np.save('data_batch_%d.bin' % i, np.array_split(X_train, num_bins)[i])

# normalization
X_train_matched = X_train_matched.view('uint8') #rather than:X_train.astype('uint8'), which is more memory efficient and faster: https://stackoverflow.com/questions/1888870/numpy-how-to-convert-an-array-type-quickly
X_train_matched = X_train_matched/255.0
# save them up!
os.chdir("/mnt/saswork/sh2264/vision/code")
np.save("X_train_scores.npy", X_train_matched)
np.save("X_train_scores_label.npy", X_train_label)
# one hot encode outputs
#y_train = np_utils.to_categorical(np.array(y_train))
##y_train = np_utils.to_categorical(np.array(y_train_sample))
y_train_mem = X_train_label.mscore
y_train_pop = X_train_label.pscore