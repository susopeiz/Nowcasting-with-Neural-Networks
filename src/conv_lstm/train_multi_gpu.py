# -*- coding: utf-8 -*-

##############################################################################
# IMPORT MODULES
##############################################################################
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, MaxPool3D,UpSampling2D,Dropout,Flatten,Dense, AveragePooling2D, Convolution2D

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

##############################################################################
# MODEL
##############################################################################

def get_model(input_shape):

  inputs = (None,) + input_shape

  model=Sequential()

  model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                    input_shape=inputs,
                    padding='same', return_sequences=True))

  model.add(BatchNormalization()) 

  model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                    padding='same', return_sequences=True))
  model.add(BatchNormalization())

  model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                    padding='same', return_sequences=True))
  model.add(BatchNormalization())

  model.add(ConvLSTM2D(filters=1, kernel_size=(3, 3),
                    padding='same', return_sequences=False))
  model.add(BatchNormalization())

  model.add(AveragePooling2D(pool_size=(1, 1), strides=None, padding='same'))

  model.add(Convolution2D(1, (3, 3) ,padding='same', activation='relu'))

  model.add(AveragePooling2D(pool_size=(1, 1), strides=None, padding='same'))

  return model