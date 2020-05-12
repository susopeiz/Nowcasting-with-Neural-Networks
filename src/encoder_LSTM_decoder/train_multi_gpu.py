# -*- coding: utf-8 -*-

##############################################################################
# IMPORT MODULES
##############################################################################
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, Dropout, Flatten, TimeDistributed, LSTM, Reshape, BatchNormalization, GlobalAveragePooling2D



##############################################################################
# MODEL
##############################################################################

def get_model(input_shape):

  inputs = (None,) + input_shape

  #Initializing the CNN
  model=Sequential()

  #Adding layers. 
  #First CNV layer
  model.add(TimeDistributed(Conv2D(32,(3,3),activation='relu',padding='same'),input_shape=inputs))
  model.add(BatchNormalization())
  #Pooling
  model.add(TimeDistributed(MaxPool2D(pool_size=(2,2))))

  ##Second CNV layer
  model.add(TimeDistributed(Conv2D(64,(3,3),activation='relu',padding='same')))
  model.add(BatchNormalization())
  ##Pooling
  model.add(TimeDistributed(MaxPool2D(pool_size=(2,2))))

  ##Third CNV layer
  model.add(TimeDistributed(Conv2D(128,(3,3),activation='relu',padding='same')))
  model.add(BatchNormalization())

  #Global Average Pooling
  model.add(TimeDistributed(GlobalAveragePooling2D()))

  #LSTM layer
  model.add(LSTM(256))
  model.add(BatchNormalization())

  model.add(Reshape((4,4,16),input_shape=(None,1024)))


  #DeConvolutions
  model.add(Conv2DTranspose(16,(3,3),activation='relu',padding='same'))
  model.add(BatchNormalization())
  ##Pooling
  model.add(UpSampling2D(size=(2,2)))

  ##Second CNV layer
  model.add(Conv2DTranspose(8,(3,3),activation='relu',padding='same'))
  model.add(BatchNormalization())
  ##Pooling
  model.add(UpSampling2D(size=(2,2)))

  ##Third CNV layer
  model.add(Conv2DTranspose(4,(3,3),activation='relu',padding='same'))
  model.add(BatchNormalization())
  ##Pooling
  model.add(UpSampling2D(size=(2,2)))

  ##Fourth CNV layer
  model.add(Conv2DTranspose(2,(3,3),activation='relu',padding='same'))
  model.add(BatchNormalization())
  ##Pooling
  model.add(UpSampling2D(size=(2,2)))

  ##Second CNV layer
  model.add(Conv2DTranspose(1,(3,3),activation='relu',padding='same'))
  model.add(BatchNormalization())

  ##Pooling
  model.add(UpSampling2D(size=(2,2)))

  #Convolutions
  ##ORDER OF THE INPUT_SHAPE IS NX,NY,Nchannels
  model.add(Conv2DTranspose(1,(3,3),activation='relu',padding='same'))

  return model