# IMPORTING MODULES


from netCDF4 import Dataset
from sklearn.model_selection import train_test_split

from keras.callbacks import ReduceLROnPlateau
import os

import numpy as np

from conv_lstm.model import get_model


# SETTING PARAMETERS
# For loading 10 years or 1 year of files, uncomment only one of the two 'dataIn' folders
dataIn = '/content/drive/My Drive/Data/Data_final/LongTrain_10years/clt/'
# .........................................................
## SELECTING REGIONS. To be developed next, forget for now
# A subregion can be selected here

x0 = 0
xn = 100
y0 = 0
yn = 100

file2read = os.listdir(dataIn)[0]
ncfile_r = Dataset(dataIn + file2read)

maps = ncfile_r.variables['clt']  # psl
Nt, Ny0, Nx0 = maps.shape


# DEFINING BATCH DATA GENERATOR
#
# NOTHING TO BE SET OR MODIFIED HERE

def seq2frames(iseq, Nframes):
    # Sequence to frame index
    # Function to be used within train_split_seq to transform sequence index to
    # frame index
    #
    # - iseq: vector of indexes of sequences
    # - Nframes: number of frames to be used as input for the LSTM/CONV3D.
    # Ex: Nframes = 8 means a sequence of 8 consecutive frames

    Nframes_with_test = Nframes + 1

    it0 = iseq * Nframes;
    it = np.array([])
    for i_it in it0:
        it_i = np.arange(i_it, i_it + Nframes_with_test)
        it = np.append(it, it_i)

    return it.astype('int')


def train_split_seq(maps, Nframes, r_train=0.7, r_val=0.2, r_test=0.1, random_state=None):
    # Split data sequences in to sets for training, validation and testing.
    # INPUTS:
    # - maps: netcdf object representing all maps from the set of
    # nc files contained in the specified folder
    #
    # - Nframes: number of frames to be used as input for the LSTM/CONV3D.
    # Ex: Nframes = 8 means a sequence of 8 consecutive frames
    #
    # - r_train: proportion of sequences to be used when training.
    #
    # - r_vali: proportion of sequences to be used when validation.
    #
    # - r_test: proportion of sequences to be used when final testing.
    #
    # - random_state: parameter to be set for repeated results

    # - Nframes: number of frames to be used as input for the LSTM/CONV3D.
    if r_train + r_val + r_test != 1:
        raise Exception('r_train, r_val and r_test should sum up 1')

    Nt, Ny, Nx = maps.shape

    Nseq = np.int(np.floor((Nt - 1) / Nframes))

    iseq = np.arange(Nseq)

    iseq_train = iseq
    iseq_val = iseq
    iseq_test = iseq
    iseq_trainval, iseq_test = train_test_split(iseq, test_size=r_test, random_state=random_state)
    Kval = r_val / (1 - r_test)
    iseq_train, iseq_val = train_test_split(iseq_trainval, test_size=Kval, random_state=random_state)

    # indexes for training
    # expliting of the sample set
    it_train = seq2frames(iseq_train, Nframes)
    it_val = seq2frames(iseq_val, Nframes)
    it_test = seq2frames(iseq_test, Nframes)

    return it_train, it_val, it_test


def batch_generator(maps, x0, xn, y0, yn, Nframes, batch_size=32):
    # Batch data generator.
    # INPUTS:
    # - maps: netcdf object representing all maps from the set of
    # nc files contained in the specified folder
    #
    # - it_train: vector with all map indexex for the training. This parameter is
    # computed with train_split_seq function
    #
    # - it_val: vector with all map indexex for the validation. This parameter is
    # computed with train_split_seq function
    #
    # - Nframes: number of frames to be used as input for the LSTM/CONV3D.
    # Ex: Nframes = 8 means a sequence of 8 consecutive frames
    #
    # - batch_size: number of sequence of frames to be used when training as batch_size
    # Ex: batch_size = 32 means 32 sequences of Nframes.

    while True:

        # _,Ny,Nx = maps.shape

        Nframes_with_test = Nframes + 1

        Lt_train = len(it_train)
        dit_batch = batch_size * Nframes_with_test

        iXY = range(dit_batch)
        iY = range(Nframes, dit_batch, Nframes_with_test)
        iX = ~np.in1d(iXY, iY)

        for it_batch in range(0, Lt_train - dit_batch, dit_batch):
            it_batch_train = it_train[np.arange(it_batch, it_batch + dit_batch)]

            XYtrain = maps[it_batch_train, x0:xn, y0:yn]

            _, Ny, Nx = XYtrain.shape

            Xtrain_i = XYtrain[iX, :, :].reshape(batch_size, Nframes, Ny, Nx)
            Xtrain_i = np.squeeze(Xtrain_i)

            # Ytrain_i = XYtrain[iY,:,:]
            # Ytrain_i = XYtrain[iY,:,:].reshape(batch_size,1,Ny, Nx)
            Ytrain_i = XYtrain[iY, :, :].reshape(batch_size, Ny, Nx)
            Xtrain_i = np.expand_dims(Xtrain_i, axis=Xtrain_i.ndim)
            Ytrain_i = np.expand_dims(Ytrain_i, axis=Ytrain_i.ndim)

            yield (Xtrain_i, Ytrain_i)


def validation_generator(maps, x0, xn, y0, yn, Nframes):
    Nframes_with_test = Nframes + 1

    XYval = maps[it_val, x0:xn, y0:yn]
    Nt_val, Ny, Nx = XYval.shape

    Nseq_val = np.int(Nt_val / Nframes_with_test)

    iXY = range(Nt_val)
    iY = range(Nframes, Nt_val, Nframes_with_test)
    iX = ~np.in1d(iXY, iY)

    Xval = XYval[iX, :, :].reshape(Nseq_val, Nframes, Ny, Nx)
    Xval = np.squeeze(Xval)

    Yval = XYval[iY, :, :].reshape(Nseq_val, Ny, Nx)
    Xval = np.expand_dims(Xval, axis=Xval.ndim)
    Yval = np.expand_dims(Yval, axis=Yval.ndim)

    return (Xval, Yval)


def test_generator(maps, x0, xn, y0, yn, Nframes):
    Nframes_with_test = Nframes + 1

    XYtest = maps[it_test, x0:xn, y0:yn]

    Nt_test, Ny, Nx = XYtest.shape

    Nseq_test = np.int(Nt_test / Nframes_with_test)

    iXY = range(Nt_test)
    iY = range(Nframes, Nt_test, Nframes_with_test)
    iX = ~np.in1d(iXY, iY)

    Xtest = XYtest[iX, :, :].reshape(Nseq_test, Nframes, Ny, Nx)
    Xtest = np.squeeze(Xtest)

    Ytest = XYtest[iY, :, :].reshape(Nseq_test, Ny, Nx)

    Xtest = np.expand_dims(Xtest, axis=Xtest.ndim)
    Ytest = np.expand_dims(Ytest, axis=Ytest.ndim)

    return (Xtest, Ytest)


# BATCH GENERATOR DEFINITION
# ------------------------------------------------------------
# Here is where to set up the parameter of the batch generator

Nframes = 6  # Number of frames within each sequences to be used during training
batch_size = 32  # Number of sequences to included within each batch during training
epochs = 100

r_train = 0.8  # Proportion of all sequences to be used during training
r_val = 0.15  # Proportion of all sequences to be used as validation set
r_test = 0.05  # Proportion of all sequences to be used as final test set

# Number of samples = number of sequences = number maps / number frames
Nseq = np.floor((Nt - 1) / Nframes)

# Number steps per epoch = number of batches = number of samples / batch_size
spe = np.ceil(Nseq / batch_size)

# Using these parameters to divide the date in train, validation and test sets:
if r_train + r_val + r_test != 1:
    raise Exception('r_train, r_val and r_test should sum up 1')
it_train, it_val, it_test = train_split_seq(maps, Nframes=Nframes, r_train=r_train, r_val=r_val, r_test=r_test, random_state=None)

# Creating the batch generator to be used during training
# batch_generator(maps, ilatlim_train, ilonlim_train, it_train, Nframes, batch_size = 32)
BG = batch_generator(maps, x0, xn, y0, yn, Nframes=Nframes, batch_size=batch_size)
# batch_generator(maps, ilatlim_train, ilonlim_train, it_train, x0, xn, y0, yn, Nframes, batch_size = 32):

# Creating the validation data to be used during training
# VG = validation_generator(maps, it_val, Nframes)
VG = validation_generator(maps, x0, xn, y0, yn, Nframes)
XTest, YTest = test_generator(maps, x0, xn, y0, yn, Nframes)
# test_generator(maps, x0, xn, y0, yn, ilatlim_test, ilonlim_test, it_test, Nframes)
print('{} maps in total'.format(Nt))
print('{} / {} / {} maps for training / validation / test'.format(len(it_train), len(it_val), len(it_test)))
print('')
print('{} frames per sequence'.format(Nframes))
print('{} sequences = samples in total'.format(Nseq))
print(
    '{} / {} / {} sequences for training / validation / test'.format(len(it_train) / (Nframes + 1), len(it_val) / (Nframes + 1), len(it_test) / (Nframes + 1)))
print('')
print('{} sequences = samples per batch'.format(batch_size))
print('{} batches per epoch'.format(spe))
print('{} epochs'.format(epochs))

# Neural network
model = get_model(xn, yn)

model.compile(loss='binary_crossentropy', metrics=['mean_squared_error'], optimizer='adam')

model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
history = model.fit_generator(BG, validation_data=VG, epochs=epochs,
                              steps_per_epoch=spe, callbacks=[reduce_lr])

history = model.fit_generator(BG, validation_data=VG, epochs=epochs, steps_per_epoch=spe)