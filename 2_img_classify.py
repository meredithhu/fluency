#!/usr/bin/env python
print "system argument 1: number of training epochs"
print "system argument 2: which model"
# added 2 arguments on 12/09/2017
print "system argument 3: transfer or finetune or both"
print "system argument 4: initialize with imagenet or None (random)"

import os
from PIL import Image
import numpy as np
import sys
import codecs
import pandas as pd
import pickle
import cPickle

#from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Merge
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Convolution3D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import MaxPooling3D
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.utils import np_utils
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
#from keras.utils.visualize_util import plot
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
np.random.seed(seed)
os.chdir("/mnt/saswork/sh2264/vision/code")

"""
Transfer learning and fine-tuning functions
"""
def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top, for instance:
    base_model = InceptionV3(weights='imagenet',include_top=False)
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) 
  predictions = Dense(nb_classes, activation='softmax')(x) 
  model = Model(input=base_model.input, output=predictions)
  return model

def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop',    
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

def setup_to_finetune(model):
   """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top 
      layers.
   note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in 
         the inceptionv3 architecture
   Args:
     model: keras model
   """
   for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
      layer.trainable = False
   for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
      layer.trainable = True
   model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),   
                 loss='categorical_crossentropy')

lr=0.0001
def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))
  
  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')
  
  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()

out = np.load('X_train.npy')
index_array = np.load('X_train_index.npy')
ca = np.load("category_crosswalk.npy")
ca = pd.DataFrame({'name': ca[:,1], 'category': ca[:,0], 'index': [int(x) for x in ca[:,2]]})
co = np.load("country_crosswalk.npy")
co = pd.DataFrame({'name': co[:,1], 'country': co[:,0], 'index': [int(x) for x in co[:,2]]})
X_train_name = pd.DataFrame({'name': index_array})

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
X_train = X_train.view('uint8')[:,::4] #rather than:X_train.astype('uint8'), which is more memory efficient and faster: https://stackoverflow.com/questions/1888870/numpy-how-to-convert-an-array-type-quickly
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
### saved to code folder!!! ###
#np.save("X_train_processed.npz",X_train)
#np.save("X_train_name_processed.npz",X_train_name)
#np.save("y_train_processed.npz",y_train) # this is the one-hot vector


train_x = X_train[:118576] # 80%
train_y = y_train[:118576] # 80%
test_x = X_train[118576:] # 20%
test_y = y_train[118576:] # 20%
# 7411*0.8=5929
#train_x = X_train[:5929] # 80%
#train_y = y_train[:5929] # 80%
#test_x = X_train[5929:] # 20%
#test_y = y_train[5929:] # 20%



# what about a deeper model: model1
# intuitions about filter size:
# Very small filter sizes will capture very fine details of the image. 
# On the other hand having a bigger filter size will leave out minute details in the image.
# However conventional kernel size's are 3x3, 5x5 and 7x7. A well known architecture for classification is to use convolution pooling, convolution pooling etc. and some fully connected layers on top. Just start of with a modest number of layers and increase the number while measuring you performance on the test set.

# Create the model

if len(sys.argv[2])<=2:

	if int(sys.argv[2])==1:
		model0 = Sequential()
		model0.add(Convolution2D(32, 3, 3, input_shape=(3, 195, 195), activation='relu', border_mode='same'))
		model0.add(Dropout(0.2))
		model0.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
		model0.add(Dropout(0.2))
		model0.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
		model0.add(Dropout(0.2))
		model0.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Flatten())
		model0.add(Dropout(0.2))
		model0.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
		model0.add(Dropout(0.2))
		model0.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
		model0.add(Dropout(0.2))
		model0.add(Dense(num_classes,activation='softmax'))

		# add handcrafted best performing image features:
		# like SIFT, SURF, HoG, LBP, spatial pyramid
		# https://keras.io/getting-started/sequential-model-guide/#the-merge-layer
	#	mergedmodel = Merge([model0, ??], mode="concat") # or pass a function to mode: mode=lambda x: x[0]-x[1]


	#	model = Sequential()
	#	model.add(mergedmodel)
	#	model.add(Dense(num_classes, activation='softmax'))

	# I think it's easiest to think of the dimensionality as the number of dimensions the filter is repeated along.
	# A typical 2D convolution applied to an RGB image would have a filter shape of (3, filter_height, filter_width), so it combines information from all channels into a 2D output.
	# If you wanted to process each color separately (and equally), you would use a 3D convolution with filter shape (1, filter_height, filter_width).
	# 1D convolution is useful for data with local structure in one dimension, like audio or other time series.

	# two FC layers
	if int(sys.argv[2])==3: # 5 + 3
		model0 = Sequential()
		model0.add(Convolution2D(32, 3, 3, input_shape=(3, 195, 195), activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Dropout(0.2))
		#model0.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
		model0.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Dropout(0.2))
		model0.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
		#model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
		#model0.add(Dropout(0.2))
		model0.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Flatten())
		#model0.add(Dropout(0.2))
		model0.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
		model0.add(Dropout(0.2))
		model0.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
		model0.add(Dropout(0.2))
		model0.add(Dense(num_classes,activation='softmax'))


# three FC layers
# scores = 2.78243842, 0.226757522
	if int(sys.argv[2])==6:# 5 CONVOL + 4 DENSE
		model0 = Sequential()
		model0.add(Convolution2D(32, 3, 3, input_shape=(3, 195, 195), activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Dropout(0.2))
		#model0.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
		model0.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Dropout(0.2))
		model0.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
		#model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
		#model0.add(Dropout(0.2))
		model0.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Flatten())
		#model0.add(Dropout(0.2))
		model0.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
		model0.add(Dropout(0.2))
		model0.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
		model0.add(Dropout(0.2))
		model0.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
		model0.add(Dropout(0.2))
		model0.add(Dense(num_classes,activation='softmax'))


# scores = 2.8070769, 0.2237215
	if int(sys.argv[2])==4: # 5 convol + 3 dense
		model0 = Sequential()
		model0.add(Convolution2D(32, 5, 5, input_shape=(3, 195, 195), activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Dropout(0.2))
		#model0.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
		model0.add(Convolution2D(64, 5, 5, activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Dropout(0.2))
		model0.add(Convolution2D(64, 5, 5, activation='relu', border_mode='same'))
		#model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Convolution2D(128, 5, 5, activation='relu', border_mode='same'))
		#model0.add(Dropout(0.2))
		model0.add(Convolution2D(128, 5, 5, activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Flatten())
		#model0.add(Dropout(0.2))
		model0.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
		model0.add(Dropout(0.2))
		model0.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
		model0.add(Dropout(0.2))
		model0.add(Dense(num_classes,activation='softmax'))


#from keras.layers import Conv2D, Input
### wanted to try ResNet
# input tensor for a 3-channel 256x256 image
#x = Input(shape=(3, 195, 195))
# 3x3 conv with 3 output channels (same as input channels)
#y = Conv2D(3, (5, 5), padding='same')(x)
# this returns x + y.
#z = keras.layers.add([x, y])



# scores = 2.8016526, 0.22601538
	if int(sys.argv[2])==5: # 5 + 3, no max pooling
		model0 = Sequential()
		model0.add(Convolution2D(32, 7, 7, input_shape=(3, 195, 195), activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Dropout(0.2))
		#model0.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
		model0.add(Convolution2D(64, 7, 7, activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Dropout(0.2))
		model0.add(Convolution2D(64, 7, 7, activation='relu', border_mode='same'))
		#model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Convolution2D(128, 7, 7, activation='relu', border_mode='same'))
		#model0.add(Dropout(0.2))
		model0.add(Convolution2D(128, 7, 7, activation='relu', border_mode='same'))
		model0.add(MaxPooling2D(pool_size=(2, 2)))
		model0.add(Flatten())
		#model0.add(Dropout(0.2))
		model0.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
		model0.add(Dropout(0.2))
		model0.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
		model0.add(Dropout(0.2))
		model0.add(Dense(num_classes,activation='softmax'))

# create the model: model2
	if int(sys.argv[2])==2:
		model0 = Sequential()
		model0.add(Convolution2D(32,3,3,input_shape=(3,195,195),border_mode = 'same',activation='relu',W_constraint=maxnorm(3)))
		model0.add(MaxPooling2D(pool_size=(2,2), strides = (1,1)))
		model0.add(Dropout(0.2))
		model0.add(Convolution2D(64,3,3,activation='relu',border_mode='same',W_constraint=maxnorm(3)))
		model0.add(MaxPooling2D(pool_size=(2,2), strides = (1,1)))
		model0.add(Flatten())
		model0.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
		model0.add(Dropout(0.5))
		model0.add(Dense(num_classes,activation='softmax'))

# create the model: model0
	if int(sys.argv[2])==0:
		model0 = Sequential()
		model0.add(Convolution2D(32,3,3,input_shape=(3,195,195),border_mode = 'same',activation='relu',W_constraint=maxnorm(3)))
		#model0.add(MaxPooling2D(pool_size=(2,2), strides = (1,1)))
		model0.add(Dropout(0.2))
		model0.add(Convolution2D(64,3,3,activation='relu',border_mode='same',W_constraint=maxnorm(3)))
		#model0.add(MaxPooling2D(pool_size=(2,2), strides = (1,1)))
		model0.add(Flatten())
		model0.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
		model0.add(Dropout(0.5))
		model0.add(Dense(num_classes,activation='softmax'))


elif len(sys.argv[2])>2:
	if sys.argv[2]=="resnet":
	

		base_model = ResNet50(weights = np.where(sys.argv[4]=="imagenet", "imagenet", None).tolist(), include_top=False) 
		# weight='imagenet' or 'None' meaning random initializations
		#model0.add(Convolution2D(32,3,3,input_shape=(3,195,195),border_mode = 'same',activation='relu',W_constraint=maxnorm(3)))
		#model0.add(MaxPooling2D(pool_size=(2,2), strides = (1,1)))
		#model0.add(Dropout(0.2))
		#model0.add(Convolution2D(64,3,3,activation='relu',border_mode='same',W_constraint=maxnorm(3)))
		#model0.add(MaxPooling2D(pool_size=(2,2), strides = (1,1)))
		#model0.add(Flatten())
		#model0.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
		#model0.add(Dropout(0.5))
		#model0.add(Dense(num_classes,activation='softmax'))
		model0 = add_new_last_layer(base_model, num_classes)

	if sys.argv[2] == "vgg16" or sys.argv[2] == "VGG16": #input 224X224
		base_model = VGG16(weights = np.where(sys.argv[4]=="imagenet", "imagenet", None).tolist(), include_top=False)
		model0 = add_new_last_layer(base_model, num_classes)

	if sys.argv[2] == "vgg19" or sys.argv[2] == "VGG19": #input 224X224
		base_model = VGG19(weights = np.where(sys.argv[4]=="imagenet", "imagenet", None).tolist(), include_top=False)
		model0 = add_new_last_layer(base_model, num_classes)

	if sys.argv[2] == "inception" or sys.argv[2] == "Inception": #input 299X299
		base_model = InceptionV3(weights = np.where(sys.argv[4]=="imagenet", "imagenet", None).tolist(), include_top=False)
		model0 = add_new_last_layer(base_model, num_classes)

	if sys.argv[2] == "xception" or sys.argv[2] == "Xception": #input 299X299
		base_model = Xception(weights = np.where(sys.argv[4]=="imagenet", "imagenet", None).tolist(), include_top=False)
		model0 = add_new_last_layer(base_model, num_classes)




# compile model
print "compiling the model..."
epochs = int(sys.argv[1])
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr = lrate, momentum = 0.9, decay = decay, nesterov = False)
#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
if len(sys.argv[2]) <= 2:
	model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

if sys.argv[3] == "transfer":
	setup_to_transfer_learn(model0, base_model)
elif sys.argv[3] == "finetune":
	setup_to_finetune(model0)
elif sys.argv[3] == "both":
	setup_to_transfer_learn(model0, base_model)
	setup_to_finetune(model0)

# model0 to model for now w/o keras.Merge
model = model0
model.save(args.output_model_file)

print("model summary\n")
print(model.summary())

#datagen.fit(train_x)

#history = model.fit_generator(datagen.flow(train_x,train_y,batch_size=32),
#	samples_per_epoch=len(train_x),nb_epoch=epochs)

history=model.fit(train_x, train_y, validation_data = (test_x, test_y), nb_epoch=epochs, batch_size=32)

# final evaluation of the model
scores = model.evaluate(test_x, test_y, verbose = 0)
#print("Accuracy: %.2f%%" % (scores[1]*100))

features = model.predict(X_train)

#np.save("features_model0.npy",features)
np.save("features_model"+sys.argv[2]+"_"+sys.argv[1]+".npy",features)
#np.save("features_model1.npy",features)
#np.save("features_model2.npy",features)

#np.save("scores_model0.npy",scores)
np.save("scores_model"+sys.argv[2]+"_"+sys.argv[1]+".npy",scores)
#np.save("scores_model1.npy",scores)
#np.save("scores_model2.npy",scores)
print dir(model)
pickle.dump(history, open('history_model'+sys.argv[2]+'_'+sys.argv[1]+'.pkl','wb'))
pickle.dump(model, open('model'+sys.argv[2]+'_'+sys.argv[1]+'.pkl','wb'))
plot(model, to_file='model'+sys.argv[2]+'_'+sys.argv[1]+'.png')

print "done!"