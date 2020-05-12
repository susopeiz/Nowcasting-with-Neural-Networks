# -*- coding: utf-8 -*-

##############################################################################
# IMPORT MODULES
##############################################################################
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D,UpSampling2D,Dropout,Flatten,Dense, BatchNormalization


##############################################################################
# MODEL -> ENCODER-DECODER
##############################################################################

def get_model(input_shape):

    #Initializing the CNN
    model=Sequential()

    #Adding layers. 

    #Convolutions
    ##ORDER OF THE INPUT_SHAPE IS NX,NY,Nchannels
    model.add(Conv2D(16,(3,3),input_shape=input_shape,activation='relu',padding='same'))
    model.add(BatchNormalization())
    #Pooling
    model.add(MaxPool2D(pool_size=(2,2)))

    ##Second CNV layer
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    ##Pooling
    model.add(MaxPool2D(pool_size=(2,2)))

    ##Third CNV layer
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    ##Pooling
    #model.add(MaxPool2D(pool_size=(2,2)))


    #DeConvolutions
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    #Pooling
    model.add(UpSampling2D(size=(2,2)))

    ##Second CNV layer
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    ##Pooling
    model.add(UpSampling2D(size=(2,2)))

    #Convolutions
    ##ORDER OF THE INPUT_SHAPE IS NX,NY,Nchannels
    #model.add(Conv2D(1,(3,3),activation='relu'))
    #Pooling
    #model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(1,(3,3),activation='sigmoid',padding='same'))

    return model
