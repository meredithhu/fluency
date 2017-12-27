#!/usr/bin/env python
from __future__ import print_function
import os
os.chdir("/mnt/saswork/sh2264/vision/code/tensorflow/models/tutorials/image/imagenet/deep-learning-models/")
try:
    os.system("rm resnet50.pyc")
    from resnet50 import ResNet50
except Exception as e:
    pass
else:
    pass

print("system argument 1: number of training epochs")
print("system argument 2: which model")
print("system argument 3: transfer or finetune or both")
print("system argument 4: initialize with imagenet or None (random)")

## need to replace log loss in classification with Euclidean loss since now we are dealing with continuous!
## it is easily done by modifying the last Dense() layer using the Lambda function:
from keras.layers import Input, Dense, Lambda
from keras.models import Model
def eucl_dist(inputs):
    x, y = inputs
    return ((x - y)**2).sum(axis=-1)
# demo
# x = Input((32,))
# y1 = Dense(8)(x)
# y2 = Dense(8)(x)
# y = Lambda(eucl_dist, output_shape=(1,))([y1, y2])
# m = Model(x, y)

### since resnet50 might not be imported here....
'''ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
'''

import numpy as np
import warnings
import json
from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from PIL import Image
import sys
import codecs
import pandas as pd
import pickle
import cPickle
import tensorflow as tf
import math
#from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Merge
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Convolution2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.convolutional import Convolution3D
from keras.layers import MaxPooling2D
from keras.layers import MaxPooling3D
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.utils import np_utils
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
#from keras.utils.visualize_util import plot
from keras import backend as K
K.set_image_dim_ordering('th')

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model

### done ####

### import imagenet_utils ###

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def decode_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        results.append(result)
    return results


### done ###

"""
Transfer learning and fine-tuning functions
"""
IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172
def eucl_dist(inputs):
    x, y = inputs
    return ((x - y)**2).sum(axis=-1)
# demo
# x = Input((32,))
# y1 = Dense(8)(x)
# y2 = Dense(8)(x)
# y = Lambda(eucl_dist, output_shape=(1,))([y1, y2])
# m = Model(x, y)
def add_new_last_continuous_layer(base_model):
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
  predictions = Lambda(eucl_dist, output_shape=(1,))(x) #Dense(nb_classes, activation='softmax')(x) 
  model = Model(input=base_model.input, output=predictions)
  return model

def setup_to_transfer_learn_continuous(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop',    
                loss= 'eucl_dist',#'categorical_crossentropy', 
                metrics=['accuracy'])

def setup_to_finetune_continuous(model):
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
                 loss='eucl_dist')#'categorical_crossentropy')

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

from keras.preprocessing import image
#from imagenet_utils import preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import sys

# get the training and test sets
os.chdir("/mnt/saswork/sh2264/vision/code")
X_train_matched = np.load("X_train_scores.npy")
X_train_label = np.load("X_train_scores_label.npy")
# one hot encode outputs
#y_train = np_utils.to_categorical(np.array(y_train))
##y_train = np_utils.to_categorical(np.array(y_train_sample))
y_train_mem = X_train_label[:,3]#.mscore
y_train_pop = X_train_label[:,2]#.pscore

### read in and split data:
msk = np.random.rand(len(y_train_mem)) < 0.8
train_x = X_train_matched[msk] #X_train[:int(math.floor(y_train_mem.shape[0]*0.8))] # 80% # previously 118576
train_y_mem = y_train_mem[msk]# y_train[:int(math.floor(y_train_mem.shape[0]*0.8))] # 80%
train_y_pop = y_train_pop[msk]
test_x = X_train_matched[~msk] # X_train[int(math.floor(y_train.shape[0]*0.8)):] # 20%
test_y_mem = y_train_mem[~msk] #y_train[int(math.floor(y_train.shape[0]*0.8)):] # 20%
test_y_pop = y_train_pop[~msk] #y_train[int(math.floor(y_train.shape[0]*0.8)):] # 20%


if len(sys.argv[2])>2:
    if sys.argv[2]=="resnet":
        base_model = ResNet50(weights = np.where(sys.argv[4].lower()=="imagenet", "imagenet", None).tolist(), include_top=False, input_shape=X_train_matched[0].shape) 
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
        model0 = add_new_last_continuous_layer(base_model)

    if sys.argv[2] == "vgg16" or sys.argv[2] == "VGG16": #input 224X224
        base_model = VGG16(weights = np.where(sys.argv[4].lower()=="imagenet", "imagenet", None).tolist(), include_top=False, input_shape=X_train_matched[0].shape)
        model0 = add_new_last_continuous_layer(base_model)

    if sys.argv[2] == "vgg19" or sys.argv[2] == "VGG19": #input 224X224
        base_model = VGG19(weights = np.where(sys.argv[4].lower()=="imagenet", "imagenet", None).tolist(), include_top=False, input_shape=X_train_matched[0].shape)
        model0 = add_new_last_continuous_layer(base_model)

    if sys.argv[2] == "inception" or sys.argv[2] == "Inception": #input 299X299
        base_model = InceptionV3(weights = np.where(sys.argv[4].lower()=="imagenet", "imagenet", None).tolist(), include_top=False, input_shape=X_train_matched[0].shape)
        model0 = add_new_last_continuous_layer(base_model)

    if sys.argv[2] == "xception" or sys.argv[2] == "Xception": #input 299X299
        base_model = Xception(weights = np.where(sys.argv[4].lower()=="imagenet", "imagenet", None).tolist(), include_top=False, input_shape=X_train_matched[0].shape)
        model0 = add_new_last_continuous_layer(base_model)




# compile model
print("compiling the model...")
epochs = int(sys.argv[1])
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr = lrate, momentum = 0.9, decay = decay, nesterov = False)
#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
if len(sys.argv[2]) <= 2:
    model.compile(loss='eucl_dist', optimizer = sgd, metrics = ['accuracy'])

if sys.argv[3] == "transfer":
    setup_to_transfer_learn_continuous(model0, base_model)
elif sys.argv[3] == "finetune":
    setup_to_finetune_continuous(model0)
elif sys.argv[3] == "both":
    setup_to_transfer_learn_continuous(model0, base_model)
    setup_to_finetune_continuous(model0)

# model0 to model for now w/o keras.Merge
model_mem = model_pop = model0

model0.save(sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+".model")

print("model summary\n")
print(model.summary())

#datagen.fit(train_x)

#history = model.fit_generator(datagen.flow(train_x,train_y,batch_size=32),
#   samples_per_epoch=len(train_x),nb_epoch=epochs)

history_mem = model_mem.fit(train_x, train_y_mem, validation_data = (test_x, test_y_mem), nb_epoch=epochs, batch_size=32)
history_pop = model_pop.fit(train_x, train_y_pop, validation_data = (test_x, test_y_pop), nb_epoch=epochs, batch_size=32)

# final evaluation of the model
scores_mem = model_mem.evaluate(test_x, test_y_mem, verbose = 0)
scores_pop = model_pop.evaluate(test_x, test_y_pop, verbose = 0)
#print("Accuracy: %.2f%%" % (scores[1]*100))

features_mem = model_mem.predict(X_train_matched)
features_pop = model_pop.predict(X_train_matched)

##np.save("features_model0.npy",features)
#np.save("features_model"+sys.argv[2]+"_"+sys.argv[1]+".npy",features)
##np.save("features_model1.npy",features)
##np.save("features_model2.npy",features)

##np.save("scores_model0.npy",scores)
#np.save("scores_model"+sys.argv[2]+"_"+sys.argv[1]+".npy",scores)
##np.save("scores_model1.npy",scores)
##np.save("scores_model2.npy",scores)
#print dir(model)
#pickle.dump(history, open('history_model'+sys.argv[2]+'_'+sys.argv[1]+'.pkl','wb'))
#pickle.dump(model, open('model'+sys.argv[2]+'_'+sys.argv[1]+'.pkl','wb'))
#plot(model, to_file='model'+sys.argv[2]+'_'+sys.argv[1]+'.png')

## commented out the above chunk (double ## means were commented out anyway, single # means it was there to function previously)
# modified the commentted out chunk on 12/10/2017 wheere 3rd & 4th arguments were added into the name string :
np.save("features_model_mem_"+sys.argv[2]+"_"+sys.argv[1]+"_"+sys.argv[3]+"_"+sys.argv[4]+".npy",features_mem)
np.save("scores_model_mem_"+sys.argv[2]+"_"+sys.argv[1]+"_"+sys.argv[3]+"_"+sys.argv[4]+".npy",scores_mem)
print(dir(model_mem))
pickle.dump(history_mem, open('history_model_mem_'+sys.argv[2]+'_'+sys.argv[1]+"_"+sys.argv[3]+"_"+sys.argv[4]+'.pkl','wb'))
pickle.dump(model_mem, open('model_mem_'+sys.argv[2]+'_'+sys.argv[1]+"_"+sys.argv[3]+"_"+sys.argv[4]+'.pkl','wb'))
plot(model_mem, to_file='model_mem_'+sys.argv[2]+'_'+sys.argv[1]+"_"+sys.argv[3]+"_"+sys.argv[4]+'.png')

np.save("features_model_pop_"+sys.argv[2]+"_"+sys.argv[1]+"_"+sys.argv[3]+"_"+sys.argv[4]+".npy",features_pop)
np.save("scores_model_pop_"+sys.argv[2]+"_"+sys.argv[1]+"_"+sys.argv[3]+"_"+sys.argv[4]+".npy",scores_pop)
print(dir(model_pop))
pickle.dump(history_pop, open('history_model_pop_'+sys.argv[2]+'_'+sys.argv[1]+"_"+sys.argv[3]+"_"+sys.argv[4]+'.pkl','wb'))
pickle.dump(model_pop, open('model_pop_'+sys.argv[2]+'_'+sys.argv[1]+"_"+sys.argv[3]+"_"+sys.argv[4]+'.pkl','wb'))
plot(model_pop, to_file='model_pop_'+sys.argv[2]+'_'+sys.argv[1]+"_"+sys.argv[3]+"_"+sys.argv[4]+'.png')


print("done!")