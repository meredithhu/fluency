#!/usr/bin/env python

import os
from PIL import Image
import numpy as np
import sys
import codecs

input_dir = "/mnt/saswork/sh2264/vision/data"

directories = os.listdir(input_dir)

#category = []
#country = []
#print(category)
tag = []
for folder in directories:
	#Ignoring .DS_Store dir
	if folder == '.DS_Store':
		pass

	else:

		print "folder "+folder

		files = os.listdir(input_dir + '/' + folder)
		os.chdir(input_dir + '/' + folder)

		if "meta.txt" not in files:
			print "no meta.txt found!"
			break

		#mdata = []
		#tag = []

		with codecs.open("meta.txt","r",encoding="utf-8") as txt:
			for line in txt:
				x = line.strip().split("\t")
				if len(x)==3:
					tag.append(x)
				#elif len(x)>3:
				#	mdata.append(x)

		#print category
		#print tag[0]
		#category = category.append([ tuple((x[0],x[1])) for x in tag ])
		#country = country.append([ tuple((x[0],x[2])) for x in tag ])

os.chdir(input_dir)
np.save("/code/category", np.array([ tuple((x[0],x[1])) for x in tag ]))
np.save("/code/country", np.array([ tuple((x[0],x[2])) for x in tag ]))