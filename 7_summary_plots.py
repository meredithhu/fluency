#! usr/bin env python
# -*- coding: utf-8 -*-


#!/usr/sheng/env python
print "system argument 1: which feature file to use, no need to include .npy"
print "system argument 2: take log of memscore or not: 0, don't; 1, do."
print "system argument 3: take log of merged['mean'] or not: 0, don't; 1, do"
print "system argument 4: take log of popscore or not: 0, don't; 1, do."
print "system argument 0: depend variable being memorability score - 1, or popularity score - 2"

import os
from PIL import Image
import numpy as np
import sys
import pandas as pd
import codecs
import statsmodels.api as sm
import scipy.stats
from scipy.stats import entropy
from statsmodels.iolib.summary2 import summary_col
from itertools import izip
import pickle
import ggplot
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
			entropies_name = [x[:-5] for x in np.load("entropy_name.npy")] # remove .jpeg
		except IOError:
			continue
		
		entropies = np.load("entropy.npy")
		entropy_mean = [x.mean() for x in entropies]
		entropy_std = [x.std() for x in entropies]

		memscore = []
		memname = []

		ms = codecs.open(input_dir+folder+"/memscore.txt",encoding = "utf-8")
		for line in ms:
			x = line.split("\t")
			try:
				memscore.append(float(x[1].split("\n")[0]))
				memname.append(x[0])
			except ValueError:
				pass


		mem = pd.DataFrame({'name':memname, 'mscore':memscore})

		popscore = []
		popname = []
		pop = codecs.open(input_dir+folder+"/popscore.txt",encoding = "utf-8")
		for line in pop:
			x = line.split("\t")
			try:
				popscore.append(float(x[1].split("\n")[0]))
				popname.append(x[0])
			except ValueError:
				pass

		pop = pd.DataFrame({'name':popname, 'pscore':popscore})


		entropy = pd.DataFrame({'name': entropies_name, 'mean':entropy_mean, 'std':entropy_std})

		if index == 1:
			if :
			merged = pd.merge(pd.merge(pop, entropy, how="inner", on="name"),mem, how="inner",on="name")
		else:
			merged = merged.append(pd.merge(pd.merge(pop, entropy, how="inner", on="name"),mem, how="inner",on="name"))

		# merged.keys(): name, mscore, pscore, mean, std

os.chdir("/mnt/saswork/sh2264/vision/code/")
ca = np.load("category_crosswalk.npy")
co = np.load("country_crosswalk.npy")
category = pd.DataFrame({'name':[x[1][:-5] for x in ca],'category':[x[2] for x in ca]})
country = pd.DataFrame({'name':[x[1][:-5] for x in co],'country':[x[2] for x in co]})

co_ca = pd.merge(category,country,how="inner",on="name")
merged = pd.merge(merged, co_ca, how="inner", on="name")
merged0 = merged
features = np.load(sys.argv[1]+".npy")
print features.shape
y_train = np.load("y_train_processed.npy")
print y_train.shape
features_name = np.load("X_train_name_processed.npy")
print features_name.shape
print "the above three should have the same first dimension..."

## temp when results for only 118576 instances are available:
#features0 = features
#features = features[:118576]
#y_train0 = y_train
#y_train = y_train[:118576]
#features_name0=features_name
#features_name = features_name[:118576]
## end temp

features_entropy = [scipy.stats.entropy(x) for x in features]
# kullback leibler divergence of x,y --- scipy.stats.entropy(x,y): x is truth, y is candidate
features_kl = [scipy.stats.entropy(x,y) for x,y in izip(y_train,features)]

merged_features = pd.DataFrame({'entropy':features_entropy,'kl':features_kl,'name':[x[:-5] for x in features_name]})
merged = pd.merge(merged0, merged_features, how="inner", on="name")

memoscore = merged['mscore']
memoscore_ln = np.log(memoscore)
memoname = merged['name']
popscore = merged['pscore']
# normalize popscore onto [0,1]
popscore_max = popscore.max()
popscore_min = popscore.min()
popscore = [float(x - popscore_min)/(popscore_max-popscore_min) for x in popscore]
popscore_ln = np.log(popscore)
popname = merged['name']
del merged['pscore']
del merged['name']
del merged['mscore']



#d as some pandas dataframe

k = [2,3,4,5]

for i in k:
    p = ggplot(d, aes(x='x', y='y', color='cluster'+str(i))) + geom_point(size=75) + ggtitle("Cluster Result: "+str(i))
    file_name = "Clusters_"+str(i)+'.png'  
    #this is not saving to any directory  
	ggsave(plot = p, filename = file_name, path = "C:\Documents\Graphs")