#!/usr/bin/env python
import os
from PIL import Image
import numpy as np
import sys
import codecs
import pandas as pd
import pickle
import cPickle
from skimage.feature import hog
from skimage import color, exposure


folder = sys.argv[1]

#Directory containing images you wish to convert
input_dir = "/mnt/saswork/sh2264/vision/data"

directories = os.listdir(input_dir)

index = 0

images = os.listdir(input_dir + '/' + folder)
os.chdir(input_dir + '/' + folder)
#index += 1
index2 = 0

for image in images:
	if image == ".DS_Store" or (not image.endswith(".jpeg")):
		continue

	else:
		try:
			im = Image.open(image).convert("RGB") #Opening image

			if im.size[0] == 200:
				size = 195,195
				im.thumbnail(size, Image.ANTIALIAS)

			imhog = color.rgb2gray(im)
			fd, hog_image = hog(imhog, orientations = 8, pixels_per_cell = (8,8), cells_per_block = (1,1))
