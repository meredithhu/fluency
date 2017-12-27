#!/usr/bin/env python

import os
import shutil
from PIL import Image
import numpy as np
import sys
# sources = os.listdir(source)
# for file in sources:
# 	print file
# 	if file.endswith(".txt") and (file.startswith("jc")):
# 		shutil.copy(source+"/"+file,destination+"/"+file)
# 	else:
# 		if len(file)==10 and file.startswith("201"):
# 			subfiles = os.listdir(source+"/"+file)
# 			for subfile in subfiles:
# 				if subfile.endswith(".txt") and ( subfile.startswith("jc") ):
# 					shutil.copy(source+"/"+file+"/"+subfile,destination+"/"+subfile)

folder = sys.argv[1]

#Directory containing images you wish to convert
input_dir = "/mnt/saswork/sh2264/vision/data"
#source = os.getcwd()

if "errors" in os.listdir(input_dir + '/' + folder):
	os.chdir(input_dir + '/' + folder + '/' + 'errors')
	for image in os.listdir(os.getcwd()):
		if image == ".DS_Store" or (not image.endswith(".jpeg")):
			continue
		else:
			if image.endswith(".png") or image.endswith("jpeg") or image.endswith(".jpg"):
				shutil.copy(image, input_dir + '/' + folder + '/' + image)
print "done for folder" + folder		