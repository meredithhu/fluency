#!/usr/bin/env python
import os

from PIL import Image
import numpy as np

#Directory containing images you wish to convert
input_dir = "/mnt/saswork/sh2264/vision/data"
#input_dir = "/Users/sheng/image"

directories = os.listdir(input_dir)

index = 0
index2 = 0

for folder in directories:
	#Ignoring .DS_Store dir
	if folder == '.DS_Store':
		pass

	else:
		print "folder "+folder
		

		images = os.listdir(input_dir + '/' + folder)
		os.chdir(input_dir + '/' + folder)
		index += 1

		for image in images:
			if image == ".DS_Store" or (not image.endswith(".jpeg")):
				continue

			else:
				index2 += 1

				im = Image.open(image).convert("RGB") #Opening image
				im = (np.array(im)) #Converting to numpy array

				try:
					r = im[:,:,0] #Slicing to get R data
					g = im[:,:,1] #Slicing to get G data
					b = im[:,:,2] #Slicing to get B data

					if index2 != 1:
						new_array = np.array([[r] + [g] + [b]], np.uint8) #Creating array with shape (3, 100, 100)
						out = np.append(out, new_array, 0) #Adding new image to array shape of (x, 3, 100, 100) where x is image number

					elif index2 == 1:
						out = np.array([[r] + [g] + [b]], np.uint8) #Creating array with shape (3, 100, 100)

					if index == 1 and index2 == 1:
						index_array = np.array([[image]])

					else:
						#new_index_array = 
						index_array = np.append(index_array, np.array([[image]]), 0)

				except Exception as e:
					print e
					print "Moving image " + image
					if not os.path.exists("errors"):
						os.makedirs("errors")
					os.rename(image, "errors/"+image)

print index

os.chdir("/mnt/saswork/sh2264/vision/code")
np.save('X_train.npy', out) #Saving train image arrays
np.save('X_train_index.npy', index_array) #Saving train labels