# -*- coding: utf-8 -*-

##############################################################################
# IMPORT MODULES
##############################################################################
from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D



##############################################################################
# MODEL
##############################################################################

def get_model(input_shape):
    print('input_shape {} '.format(input_shape))
    inputs = Input(input_shape)
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    print('conv1 {} '.format(conv1))
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv1)
    print('conv1 {} '.format(conv1))
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    print('pool1 {} '.format(pool1))

    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    print('pool2 {} '.format(pool2))

    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    print('pool3 {} '.format(pool3))

    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    print('pool4 {} '.format(pool4))

    conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv5)
    print('conv5 {} '.format(conv5))

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4], axis=4)
    conv6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv6)
    print('conv6 merged conv4 {} '.format(conv6))

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3], axis=4)
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv7)
    print('conv7 merged conv3 {} '.format(conv7))

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=4)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv8)
    print('conv8 merged conv2 {} '.format(conv8))

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=4)
    conv9 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv9)
    print('conv9 merged conv1 {} '.format(conv9))

    conv10 = Conv3D(1, (1, 1, 16), activation='sigmoid')(conv9)

    print('conv10 {} '.format(conv10))

    model = Model(input=inputs, output=conv10)

    return model
