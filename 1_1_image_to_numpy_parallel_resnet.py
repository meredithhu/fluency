import os

from PIL import Image
import numpy as np
import sys

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

			#if im.size[0] == 200:
			#	size = 195,195
			#	im.thumbnail(size, Image.ANTIALIAS) # a quality down-sampler
			# 12/09/2017: changed to the flipped, due to ResNet min requirement:
			size = (200, 200)
			if im.size != size:
				im = im.resize(size, Image.BICUBIC) # cubic interpolation for up-sampling


			im = (np.array(im)) #Converting to numpy array

			r = im[:,:,0] #Slicing to get R data
			g = im[:,:,1] #Slicing to get G data
			b = im[:,:,2] #Slicing to get B data

			index2 += 1
			if index2 != 1:
				new_array = np.array([[r] + [g] + [b]], np.uint8) #Creating array with shape (3, 100, 100)
				out = np.append(out, new_array, 0) #Adding new image to array shape of (x, 3, 100, 100) where x is image number
				index_array = np.append(index_array, np.array([[image]]), 0)

			elif index2 == 1:
				out = np.array([[r] + [g] + [b]], np.uint8) #Creating array with shape (3, 100, 100)
				index_array = np.array([[image]])
					

		except Exception as e:
			print e
			try:
				print r.shape
			except:
				pass

			print("Moving image " + image)

			if not os.path.exists("errors"):
				os.makedirs("errors")
			os.rename(image, "errors/"+image)

print "all done"

if 'out' in dir():
	print 'out for folder '+folder+' saved'
	np.save('X_train_resnet_'+folder+'.npy', out) #Saving train image arrays
	np.save('X_train_resnet_index_'+folder+'.npy', index_array) #Saving train labels