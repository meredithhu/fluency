#!/usr/sheng/env python
print "system argument 1: which feature file to use, no need to include .npy"
print "system argument 2: take log of memscore or not: 0, don't; 1, do."
print "system argument 3: take log of merged['mean'] or not: 0, don't; 1, do"
print "system argument 4: take log of popscore or not: 0, don't; 1, do."
print "system argument 5: take log of entropy/kl or not: 0, don't; 1, do."

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

		imagenet = np.load("imagenet_"+folder+".npy")
		imagenet_name = np.load("imagenet_index_"+folder+".npy")
		merged_imagenet = pd.DataFrame({'imagenet':[scipy.stats.entropy(x) for x in imagenet],'name':[x[0][:-5] for x in imagenet_name]})


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

		adjusted_popscore = [x if x>0 else 0.016587944219523178 for x in popscore]
		pop = pd.DataFrame({'name':popname, 'pscore':adjusted_popscore})

		entropy = pd.DataFrame({'name': entropies_name, 'mean':entropy_mean, 'std':entropy_std})

		if index == 1:
			merged = pd.merge(pd.merge(pd.merge(pop, entropy, how="inner", on="name"),mem, how="inner",on="name"), merged_imagenet, how = "inner", on = "name")
		else:
			merged = merged.append(pd.merge(pd.merge(pd.merge(pop, entropy, how="inner", on="name"),mem, how="inner",on="name"), merged_imagenet, how = "inner", on = "name"))
		# merged.keys(): name, mscore, pscore, mean, std



os.chdir("/mnt/saswork/sh2264/vision/code/")
ca = np.load("category_crosswalk.npy")
co = np.load("country_crosswalk.npy")
category = pd.DataFrame({'name':[x[1][:-5] for x in ca],'category':[x[0] for x in ca]})
country = pd.DataFrame({'name':[x[1][:-5] for x in co],'country':[x[0] for x in co]})

co_ca = pd.merge(category,country,how="inner", on="name")
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

# take the log:
if sys.argv[5] == "1":
	features_entropy = [np.log(x) for x in features_entropy]
	features_kl = [np.log(x) for x in features_kl]

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

if sys.argv[3] == "1":
	logmean = np.log(merged['mean'])
	del merged['mean']
	merged['mean'] = logmean
elif sys.argv[3] == "0":
	pass


# dummy variables for category and country

dummies = pd.concat([pd.get_dummies(merged['category']).reset_index(drop=True), pd.get_dummies(merged['country'])], axis=1)
category_index = merged['category']
del merged['category']
country_index = merged['country']
del merged['country']
merged = pd.concat([merged.reset_index(drop=True), dummies], axis = 1)



print "checkpoint 1"
# squared terms
### model0
print "model0: no squared, no interaction"
if sys.argv[2] == "1":
	#modelm = sm.OLS(np.array(memoscore_ln), np.array(sm.add_constant(merged)))
	modelm0 = sm.OLS(np.array(memoscore_ln), np.array(sm.add_constant(merged)), missing = "drop")
elif sys.argv[2] == "0":
	#modelm = sm.OLS(np.array(memoscore), np.array(sm.add_constant(merged)))
	modelm0 = sm.OLS(np.array(memoscore), np.array(sm.add_constant(merged)), missing = "drop")

print "checkpoint 3"
if sys.argv[4] == "1":
	modelp0 = sm.OLS(np.array(popscore_ln), np.array(sm.add_constant(merged)), missing = "drop")
	#modelp = sm.OLS(np.array(memoscore_ln), np.array(sm.add_constant(merged)))
elif sys.argv[4] == "0":
	modelp0 = sm.OLS(np.array(popscore), np.array(sm.add_constant(merged)), missing = "drop")
	#modelp = sm.OLS(np.array(memoscore), np.array(sm.add_constant(merged)))


# model1: squared
merged['meansq'] = pd.DataFrame({'meansq':[x**2 for x in merged['mean']]})
merged['stdsq'] = pd.DataFrame({'stdsq':[x**2 for x in merged['std']]})
merged['entropysq'] = pd.DataFrame({'entropysq':[x**2 for x in merged['entropy']]})
merged['imagenetsq'] = pd.DataFrame({'imagenetsq':[x**2 for x in merged['imagenet']]})
merged['klsq'] = pd.DataFrame({'klsq':[x**2 for x in merged['kl']]})

print "model1: squared"
if sys.argv[2] == "1":
	#modelm = sm.OLS(np.array(memoscore_ln), np.array(sm.add_constant(merged)))
	modelm1 = sm.OLS(np.array(memoscore_ln), np.array(sm.add_constant(merged)), missing = "drop")
elif sys.argv[2] == "0":
	#modelm = sm.OLS(np.array(memoscore), np.array(sm.add_constant(merged)))
	modelm1 = sm.OLS(np.array(memoscore), np.array(sm.add_constant(merged)), missing = "drop")

print "checkpoint 3"
if sys.argv[4] == "1":
	modelp1 = sm.OLS(np.array(popscore_ln), np.array(sm.add_constant(merged)), missing = "drop")
	#modelp = sm.OLS(np.array(memoscore_ln), np.array(sm.add_constant(merged)))
elif sys.argv[4] == "0":
	modelp1 = sm.OLS(np.array(popscore), np.array(sm.add_constant(merged)), missing = "drop")
	#modelp = sm.OLS(np.array(memoscore), np.array(sm.add_constant(merged)))


# interaction terms!

# complexity * content ambiguity
merged['mean-entropy'] = pd.DataFrame({'mean-entropy':[x*y for x,y in izip(merged['mean'],merged['entropy'])]})
# complexity * content ambiguity (relative)
merged['mean-kl'] = pd.DataFrame({'mean-kl':[x*y for x,y in izip(merged['mean'],merged['kl'])]})
# imagenet entropy * content ambiguity
merged['imagenet-entropy'] = pd.DataFrame({'imagenet-entropy':[x*y for x,y in izip(merged['imagenet'],merged['entropy'])]})
# imagenet entropy * content ambiguity (relative)
merged['imagenet-kl'] = pd.DataFrame({'imagenet-kl':[x*y for x,y in izip(merged['imagenet'],merged['kl'])]})
# imagenet entropy * complexity * content ambiguity
merged['imagenet-mean-entropy'] = pd.DataFrame({'imagenet-mean-entropy':[x*y*z for x,y,z in izip(merged['imagenet'], merged['mean'], merged['entropy'])]})
# imagenet entropy * complexity * content ambiguity (relative)
merged['imagenet-mean-kl'] = pd.DataFrame({'imagenet-mean-kl':[x*y*z for x,y,z in izip(merged['imagenet'], merged['mean'], merged['kl'])]})

merged['std-entropy'] = pd.DataFrame({'std-entropy':[x*y for x,y in izip(merged['std'],merged['entropy'])]})
merged['std-kl'] = pd.DataFrame({'std-kl':[x*y for x,y in izip(merged['std'],merged['kl'])]})


print "model2: squared, interactions"
if sys.argv[2] == "1":
	#modelm = sm.OLS(np.array(memoscore_ln), np.array(sm.add_constant(merged)))
	modelm2 = sm.OLS(np.array(memoscore_ln), np.array(sm.add_constant(merged)), missing = "drop")
elif sys.argv[2] == "0":
	#modelm = sm.OLS(np.array(memoscore), np.array(sm.add_constant(merged)))
	modelm2 = sm.OLS(np.array(memoscore), np.array(sm.add_constant(merged)), missing = "drop")

print "checkpoint 3"
if sys.argv[4] == "1":
	modelp2 = sm.OLS(np.array(popscore_ln), np.array(sm.add_constant(merged)), missing = "drop")
	#modelp = sm.OLS(np.array(memoscore_ln), np.array(sm.add_constant(merged)))
elif sys.argv[4] == "0":
	modelp2 = sm.OLS(np.array(popscore), np.array(sm.add_constant(merged)), missing = "drop")
	#modelp = sm.OLS(np.array(memoscore), np.array(sm.add_constant(merged)))
print "checkpoint 3"

resultsm0 = modelm0.fit()
resultsp0 = modelp0.fit()
resultsm1 = modelm1.fit()
resultsp1 = modelp1.fit()
resultsm2 = modelm2.fit()
resultsp2 = modelp2.fit()

print "memorability quadratic model results"
print(resultsm2.summary())
print "popularity quadratic model results"
print(resultsp2.summary())
os.chdir("/mnt/saswork/sh2264/vision/results/")
pickle.dump(resultsm2, open('results_imagenet_'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+sys.argv[4]+'_'+ sys.argv[5] +'.pkl','wb'))
pickle.dump(modelm2, open('model_imagenet_'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+sys.argv[4]+'_'+ sys.argv[5] +'.pkl','wb'))
pickle.dump(modelp2, open('modelsq_imagenet_'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+sys.argv[4]+'_'+ sys.argv[5] +'.pkl','wb'))
pickle.dump(resultsp2, open('resultssq_imagenet_'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+sys.argv[4]+'_'+ sys.argv[5] +'.pkl','wb'))
# write to latex file
print "checkpoint 4: latex file!"
text_file = open('texm_imagenet_'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+sys.argv[4]+'_'+ sys.argv[5] +'.txt', 'w')
text_file.write(summary_col([resultsm0,resultsm1,resultsm2], stars=True, float_format='%0.4f').as_latex())
text_file.close()
text_file = open('texp_imagenet_'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+sys.argv[4]+'_'+ sys.argv[5] +'.txt', 'w')
text_file.write(summary_col([resultsp0,resultsp1,resultsp2], stars=True, float_format='%0.4f').as_latex())
text_file.close()
text_file = open('tex_imagenet_'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+sys.argv[4]+'_'+ sys.argv[5] +'.txt', 'w')
text_file.write(summary_col([resultsm2,resultsp2], stars=True, float_format='%0.4f').as_latex())
text_file.close()

# as a memo: merged.keys()
# variable names: intercept, mean, std, imagenet, category, country, entropy, 
# kl, meansq, stdsq, entropysq, imagenetsq, klsq, mean-entropy, mean-kl, 
# imagenet-entropy, imagenet-kl, imagenet-mean-entropy, imagenet-mean-kl
# std-entropy, std-kl