# -*- coding: utf-8 -*-

##############################################################################
# IMPORT MODULES
##############################################################################
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, BatchNormalization, Input, concatenate



##############################################################################
# MODEL
##############################################################################

# Normalized model
def get_model(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(bn1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    bn2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    bn3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(bn3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    bn4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(bn4)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    bn5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(bn5)
    bn5 = BatchNormalization()(conv5)
    pool5 = MaxPool2D(pool_size=(2, 2))(bn5)

    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool5)
    bn5_2 = BatchNormalization()(conv5_2)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(bn5_2)
    bn5_2 = BatchNormalization()(conv5_2)

    up5_2 = concatenate([UpSampling2D(size=(2, 2))(bn5_2), bn5], axis=3)
    conv6_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(up5_2)
    bn6_2 = BatchNormalization()(conv6_2)
    conv6_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(bn6_2)
    bn6_2 = BatchNormalization()(conv6_2)

    up6 = concatenate([UpSampling2D(size=(2, 2))(bn6_2), bn4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    bn6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(bn6)
    bn6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(bn6), bn3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    bn7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(bn7)
    bn7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(bn7), bn2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    bn8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn8)
    bn8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(bn8), bn1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    bn9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(bn9)
    bn9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(bn9)

    model = Model(input=inputs, output=conv10)

    return model


def get_unet(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model
