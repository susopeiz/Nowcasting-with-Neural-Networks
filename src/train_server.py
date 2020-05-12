# -*- coding: utf-8 -*-

##############################################################################
# IMPORT MODULES
##############################################################################

import os
import sys
import random
import numpy as np
import keras

from keras.callbacks import Callback, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from netCDF4 import Dataset,MFDataset
from sklearn.model_selection import train_test_split


##############################################################################
# FUNCTIONS
##############################################################################

def help():
  print 'General usage:'
  print '    $> python', __file__, '"model"', '"# of graphic_cards"', '"fit/fit_generator"'
  print
  print 'Examples:'
  print '    $> python', __file__, 'encoder_decoder', '0,1,2,3', 'fit'
  print '    $> python', __file__, 'unet_2d', '4', 'fit_generator'
  print
  print 'Available models:'
  print '    encoder_decoder'
  print '    encoder_lstm_decoder'


def print_error(msg='Unknown'):
  print 'Error:', msg
  print
  help()
  sys.exit()

def seq2frames(iseq, Nframes):
  #Sequence to frame index
  #Function to be used within train_split_seq to transform sequence index to frame index
  #
  # - iseq: vector of indexes of sequences
  # - Nframes: number of frames to be used as input for the LSTM/CONV3D.
  # Ex: Nframes = 8 means a sequence of 8 consecutive frames

  Nframes_with_test = Nframes +1

  it0 = iseq * Nframes;
  it = np.array([])
  for i_it in it0:
    it_i = np.arange(i_it, i_it + Nframes_with_test)
    it = np.append(it, it_i)

  return it.astype('int')


def train_split_seq(maps, Nframes, r_train = 0.7, r_val = 0.2, r_test = 0.1, random_state = None):
  #Split data sequences in to sets for training, validation and testing.
  #INPUTS: 
  # - maps: netcdf object representing all maps from the set of 
  #nc files contained in the specified folder
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
  if r_train + r_val + r_test != 1:
    raise Exception('r_train, r_val and r_test should sum up 1')

  Nt,Ny,Nx = maps.shape

  Nseq = np.int(np.floor((Nt-1) / Nframes))

  iseq = np.arange(Nseq)

  iseq_train = iseq
  iseq_val = iseq
  iseq_test = iseq
  iseq_trainval, iseq_test = train_test_split(iseq, test_size = r_test, random_state = random_state)
  Kval = r_val / (1-r_test)
  iseq_train, iseq_val = train_test_split(iseq_trainval, test_size = Kval, random_state =  random_state)

  it_train = seq2frames(iseq_train, Nframes)
  it_val = seq2frames(iseq_val, Nframes)
  it_test = seq2frames(iseq_test, Nframes)

  return it_train, it_val, it_test


def batch_generator(maps, it_train, Nframes, batch_size = 32):
  #Batch data generator.
  #INPUTS: 
  # - maps: netcdf object representing all maps from the set of
  #nc files contained in the specified folder
  #
  # - it_train: vector with all map indexex for the training. This parameter is
  #computed with train_split_seq function
  #
  # - it_val: vector with all map indexex for the validation. This parameter is
  #computed with train_split_seq function
  #
  # - Nframes: number of frames to be used as input for the LSTM/CONV3D.
  # Ex: Nframes = 8 means a sequence of 8 consecutive frames
  #
  # - batch_size: number of sequence of frames to be used when training as batch_size
  # Ex: batch_size = 32 means 32 sequences of Nframes.

  while True:

    Nframes_with_test = Nframes + 1

    Lt_train = len(it_train)
    dit_batch = batch_size * Nframes_with_test

    iXY = range(dit_batch)
    iY = range(Nframes, dit_batch, Nframes_with_test)
    iX = ~np.in1d(iXY, iY)

    for it_batch in range(0, Lt_train - dit_batch , dit_batch):

      it_batch_train = it_train[np.arange(it_batch, it_batch + dit_batch)]

      #XYtrain = maps[it_batch_train, : , :]
      XYtrain = maps[it_batch_train, xpix_0:xpix_n,ypix_0:ypix_n]

      _,Ny,Nx = XYtrain.shape
      Xtrain_i = XYtrain[iX,:,:].reshape(batch_size, Nframes, Ny, Nx) #ORIGINAL
      #Xtrain_i = XYtrain[iX,:,:].reshape(batch_size, Ny, Nx)
      #Xtrain_i = XYtrain[iX,:,:].reshape(batch_size*Nframes, Ny, Nx)

      Xtrain_i = np.squeeze(Xtrain_i)

      Ytrain_i = XYtrain[iY,:,:]
      #Ytrain_i = XYtrain[iY,:,:].reshape(batch_size, 1, Ny, Nx)

      Xtrain_i = np.expand_dims(Xtrain_i, axis = Xtrain_i.ndim)
      Ytrain_i = np.expand_dims(Ytrain_i, axis = Ytrain_i.ndim)
      #Xtrain_i = np.transpose(Xtrain_i,(0,2,3,4,1))

      yield (Xtrain_i, Ytrain_i)


def validation_generator(maps, it_val, Nframes):

  Nframes_with_test = Nframes + 1

  #XYval = maps[it_val, : , :]
  XYval = maps[it_val, xpix_0:xpix_n,ypix_0:ypix_n]
  Nt_val,Ny,Nx = XYval.shape

  Nseq_val = np.int(Nt_val / Nframes_with_test)

  iXY = range(Nt_val)
  iY = range(Nframes, Nt_val, Nframes_with_test)
  iX = ~np.in1d(iXY, iY)

  Xval = XYval[iX,:,:].reshape(Nseq_val, Nframes, Ny, Nx) #ORIGINAL
  #Xval = XYval[iX,:,:].reshape(Nseq_val, Ny, Nx)
  #Xval = XYval[iX,:,:].reshape(Nseq_val*Nframes, Ny, Nx)
  Xval = np.squeeze(Xval)

  Yval = XYval[iY,:,:]
  #Yval = XYval[iY,:,:].reshape(Nseq_val, 1, Ny, Nx)

  Xval = np.expand_dims(Xval, axis = Xval.ndim)
  Yval = np.expand_dims(Yval, axis = Yval.ndim)
  #Xval = np.transpose(Xval,(0,2,3,4,1))

  return (Xval, Yval)


def test_generator(maps, it_test, Nframes):
  
  Nframes_with_test = Nframes + 1
  
  #XYtest = maps[it_test, : , :]
  XYtest = maps[it_test, xpix_0:xpix_n,ypix_0:ypix_n]
  Nt_test,Ny,Nx = XYtest.shape

  Nseq_test = np.int(Nt_test / Nframes_with_test)

  iXY = range(Nt_test)
  iY = range(Nframes, Nt_test, Nframes_with_test)
  iX = ~np.in1d(iXY, iY)

  Xtest = XYtest[iX,:,:].reshape(Nseq_test, Nframes, Ny, Nx) #ORIGINAL
  #Xtest = XYtest[iX,:,:].reshape(Nseq_test, Ny, Nx)
  #Xtest = XYtest[iX,:,:].reshape(Nseq_test*Nframes, Ny, Nx)
  Xtest = np.squeeze(Xtest)

  Ytest = XYtest[iY,:,:]
  #Ytest = XYtest[iY,:,:,:].reshape(Nseq_test, 1, Ny, Nx)

  Xtest = np.expand_dims(Xtest, axis = Xtest.ndim)
  Ytest = np.expand_dims(Ytest, axis = Ytest.ndim)
  #Xtest = np.transpose(Xtest,(0,2,3,4,1))

  print(Xtest.shape)

  return (Xtest, Ytest)



##############################################################################
# CODE
##############################################################################

print sys.argv[1]
if sys.argv[1] == '-h' or sys.argv[1] == '-help':
  help()
  sys.exit()

# GPU SET UP - bus connection
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# GPU SET UP - visible gpu in this script
os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[2])
num_gpu = str(sys.argv[2]).count(',') + 1


#LOADING DATA
ncfile_r = MFDataset('/home/aidl/git/data/data_final/long_train_10_years/clt/*.nc')
maps = ncfile_r.variables['clt']
Nt, Ny, Nx = maps.shape
print('{} maps ready to be loaded'.format(Nt))


##############################################################################
##############################################################################
##############################################################################

# Parameters to play with.

# BATCH DEFINITION
Nframes = 1 # Number of frames within each sequences to be used during training
batch_size = 128# Number of sequences to included within each batch during training
epochs = 500

r_train = 0.8 #Proportion of all sequences to be used during training
r_val = 0.1 #Proportion of all sequences to be used as validation set
r_test = 0.1 #Proportion of all sequences to be used as final test set

Nt = 30000 # Total number of training frames in case batch generator is NOT used
Nv = 1000 # Total number of validation frames in case batch generator is NOT used
Ntest = 1000 # Total number of test frames

# IMAGE CROP - taking images of 128x128 pixels at upper left corner
xpix_0 = 0
xpix_n = 128
ypix_0 = 0
ypix_n = 128

xpix = xpix_n - xpix_0
ypix = ypix_n - ypix_0

##############################################################################
##############################################################################
##############################################################################


#Number of samples = number of sequences = number maps / number frames
Nseq = np.floor((Nt-1) / Nframes)

#Number steps per epoch = number of batches = number of samples / batch_size
spe = np.ceil(Nseq / batch_size)


if sys.argv[3] == 'fit_generator':
  #Using these parameters to divide the date in train, validation and test sets:
  if r_train + r_val + r_test != 1:
    raise Exception('r_train, r_val and r_test should sum up 1')
  it_train, it_val, it_test = train_split_seq(maps, Nframes = Nframes, r_train = r_train, r_val = r_val, r_test = r_test, random_state = None)

  #Creating the batch generator to be used during training
  BG = batch_generator(maps, it_train, Nframes = Nframes, batch_size = batch_size)

  #Creating the validation data to be used during training
  VG = validation_generator(maps, it_val, Nframes)

  print('{} maps in total'.format(Nt))
  print('{} / {} / {} maps for training / validation / test'.format(len(it_train), len(it_val), len(it_test)))
  print ''
  print('{} frames per sequence'.format(Nframes))
  print('{} sequences = samples in total'.format(Nseq))
  print('{} / {} / {} sequences for training / validation / test'.format(len(it_train)/(Nframes+1), len(it_val)/(Nframes+1), len(it_test)/(Nframes+1)))


if sys.argv[3] == 'fit':
  # This data will be loaded into RAM. model.fit(...) function will be used.

  offset = 1 # to avoid overlap betwen sets -> GOLDEN RULE.

  # Training set 
  Xtrain = maps[0:Nt:2,:128,:128] # Taking 1 frame
  Ytrain = maps[1:Nt+1:2,:128,:128] # Predicting next frame

  # Validation set
  Xval = maps[Nt+offset:Nt+Nv:2,:128,:128]
  Yval = maps[Nt+offset+1:Nt+Nv+1:2,:128,:128]

  # Test set
  Xtest = maps[Nt+Nv+offset:Nt+Nv+Ntest:2,:128,:128]
  Ytest = maps[Nt+Nv+offset+1:Nt+Nv+Ntest+1:2,:128,:128]

  # Shuffle training set
  idisorder_train =np.arange(Nt/2)
  np.random.shuffle(idisorder_train)
  Xtrain = Xtrain[idisorder_train,:,:]
  Ytrain = Ytrain[idisorder_train,:,:]

  # Shuffle validation set
  idisorder_val = np.arange(Nv/2 - offset)
  np.random.shuffle(idisorder_val)
  Xval = Xval[idisorder_val,:,:]
  Yval = Yval[idisorder_val,:,:]

  # Shuffle test set
  idisorder_test = np.arange(Ntest/2 - offset)
  np.random.shuffle(idisorder_test)
  Xtest = Xtest[idisorder_test,:,:]
  Ytest = Ytest[idisorder_test,:,:]

  # Expanding training set
  Xtrain = np.expand_dims(Xtrain, axis = Xtrain.ndim)
  Ytrain = np.expand_dims(Ytrain, axis = Ytrain.ndim)

  # Expanding validation set
  Xval = np.expand_dims(Xval, axis = Xval.ndim)
  Yval = np.expand_dims(Yval, axis = Yval.ndim)

  # Expanding test set
  Xtest = np.expand_dims(Xtest, axis = Xtest.ndim)
  Ytest = np.expand_dims(Ytest, axis = Ytest.ndim)


print ''
print('{} sequences = samples per batch'.format(batch_size))
print('{} batches per epoch'.format(spe))
print('{} epochs'.format(epochs))


# Import chosen model
if sys.argv[1] == 'encoder_decoder':
  from encoder_decoder import get_model

elif sys.argv[1] == 'encoder_lstm_decoder':
  if sys.argv[3] == 'fit_generator':
    print_error('encoder_lstm_decoder with fit is not yet implemented.Use fit_generator option instead')
  from encoder_lstm_decoder import get_model

elif sys.argv[1] == 'conv2d_lstm2d_conv2d':
  from conv2d_lstm2d_conv2d import get_model

elif sys.argv[1] == 'unet_2d':
  from unet_2d import get_model

elif sys.argv[1] == 'unet_3d':
  print 'Warning: This NN is still in development. Errors are expected'
  from unet_3d import get_model

else: 
  print_error('Wrong model name in input parameter:', sys.argv[1])

# get the model
model = get_model((xpix, ypix, Nframes))

if num_gpu >= 2:
  # make model multi gpu when more than 1 gpu
  model = multi_gpu_model(model, gpus = num_gpu)

#compiling the ANN
model.compile(optimizer='adam',loss='binary_crossentropy', metrics = ['mean_squared_error'])

model.summary()

# Setting paths
log_dir = '/home/aidl/tensorboard_log_dir/'
current_config = str(sys.argv[1]) + '_' + str(num_gpu) + '_gpus_' + str(sys.argv[3])

# define TensorBoard path and graphics
tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir+current_config, histogram_freq=0, write_graph=True, write_images=True)

# TRAIN MODEL AND SHOW RESULTS
if sys.argv[3] == 'fit':
  history = model.fit(x = Xtrain, y = Ytrain, epochs = epochs, batch_size = batch_size, validation_data = (Xval, Yval), callbacks=[tbCallBack])
elif sys.argv[3] == 'fit_generator':
  history = model.fit_generator(BG, validation_data = VG, epochs = epochs, steps_per_epoch = spe, use_multiprocessing=True, callbacks=[tbCallBack])
else:
  print_error('Wrong model fit option in parameter input:', sys.argv[3])

# Save model
model.save('results/' + current_config + '_model.h5')
model.save_weights('results/' + current_config + '_weights.h5')

"""
if sys.argv[3] == 'fit_generator':

  Xtest, Ytest = VG  # Test set

  loss_test, metric_test = model.evaluate(Xtest, Ytest)
  print('Mean loss for the test set: {}'.format(loss_test))
  print('Mean metric value for the test set: {}'.format(metric_test))

  #TO CHECK SOME PREDICTIONS
  i = 50
  Xval, Yval = VG
  Ypred = model.predict(Xval[i,:,:,:].reshape(1,Xval.shape[1],Xval.shape[2],1))

  print Ypred

if sys.argv[3] == 'fit':
  """