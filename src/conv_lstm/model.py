
from keras.layers import AveragePooling2D
from keras.layers import Convolution2D

from keras.models import Sequential
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization


def get_model(xn, yn):
    # Neural network
    model = Sequential()

    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                         input_shape=(None, xn, yn, 1),
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

    model.add(Convolution2D(1, (3, 3), padding='same', activation='relu'))

    model.add(AveragePooling2D(pool_size=(1, 1), strides=None, padding='same'))

    return model
