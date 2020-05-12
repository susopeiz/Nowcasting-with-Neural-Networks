
##IMPORTING MODULES
#-------------------
import os
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset,MFDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse_sk

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D,UpSampling2D,Dropout

##PRELOADING DATA
#--------------------
dataIn = '/content/drive/My Drive/Data/Data_final/Multivariable/' 
def Preloading(dataIn):
    len(os.listdir(dataIn))
    files = sorted(os.listdir(dataIn))
    maps = []
    for k,ifile in enumerate(files):

      ncfile = Dataset(dataIn + ifile)
      var = list(ncfile.variables.keys())[-1]
      maps.append(ncfile.variables[var])
      Nt, Ny, Nx = maps[k].shape

      print('{} ready to be read.'.format(ifile))
      print('{} maps ready to be loaded with dimensions: {}x{}'.format(Nt,Ny,Nx))
      
      return maps, Nt, Ny, Nx
      
## DATA NORMALIZATION
#-----------------------
def NormalizingData(maps, dit = 1000, Nt = 40000):
    maps_max0 = np.empty((4,int(Nt/dit)))
    maps_min0 = np.empty((4,int(Nt/dit)))

    for k,it in enumerate(range(0,Nt,dit)):
      for ic in range(4):
        m = maps[ic][it:it+dit,:,:]

        maps_max0[ic,k] = m.max()
        maps_min0[ic,k] = m.min()

      print('{} / {}'.format(it,Nt))

    maps_max = maps_max0.max(axis=1)
    maps_min = maps_min0.min(axis=1)

    print(maps_max)
    print(maps_min)
    
    return maps_max, maps_min
    
    


## DEFINING BATCH DATA GENERATOR
#---------------------------------

def seq2frames(iseq,Nframes):
  #Sequence to frame index
  #Function to be used within train_split_seq to transform sequence index to
  #frame index
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
  
  
  Nt,_,_ = maps[0].shape
  
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
  

def batch_generator(maps, it_train, Nframes, batch_size = 32, Nchannels = 4, Nx = 128, Ny = 128, maps_max = None, maps_min = None):
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
  
    XYtrain = np.empty( (dit_batch, Ny, Nx, Nchannels) )
  
    for it_batch in range(0, Lt_train - dit_batch , dit_batch):
    
      it_batch_train = it_train[np.arange(it_batch, it_batch + dit_batch)]
      
      for ic in range(Nchannels):
        XYtrain[:, :, :, ic] = ( maps[ic][it_batch_train, :Ny , :Nx] - maps_min[ic] ) / (maps_max[ic] - maps_min[ic])
      
      Xtrain_i = XYtrain[iX,:,:,:].reshape(batch_size, Nframes, Ny, Nx, Nchannels)
      Xtrain_i = np.squeeze(Xtrain_i)
    
      Ytrain_i = XYtrain[iY,:,:,0].reshape(batch_size, Ny, Nx,1)
    
      #Xtrain_i = np.expand_dims(Xtrain_i, axis = Xtrain_i.ndim)
      #Ytrain_i = np.expand_dims(Ytrain_i, axis = Ytrain_i.ndim)
                    
      yield (Xtrain_i, Ytrain_i)
      
      
def validation_generator(maps, it_val, Nframes, Nchannels = 4, Nx = 128, Ny = 128, maps_max = None, maps_min = None):
  
  Nframes_with_test = Nframes + 1
  
  Nt_val = len(it_val)
  
  XYval = np.empty( (Nt_val, Ny, Nx, Nchannels) )
  for ic in range(Nchannels):
      XYval[:, :, :, ic] = ( maps[ic][it_val, :Ny , :Nx] - maps_min[ic] ) / (maps_max[ic] - maps_min[ic])
  
  
  
  Nseq_val = np.int(Nt_val / Nframes_with_test)
  
  iXY = range(Nt_val)
  iY = range(Nframes, Nt_val, Nframes_with_test)
  iX = ~np.in1d(iXY, iY)
  
  
      
  Xval = XYval[iX,:,:,:].reshape(Nseq_val, Nframes, Ny, Nx, Nchannels)
  Xval = np.squeeze(Xval)
  
  Yval = XYval[iY,:,:,0].reshape(Nseq_val, Ny, Nx,1)
  
  
  
  #Xval = np.expand_dims(Xval, axis = Xval.ndim)
  #Yval = np.expand_dims(Yval, axis = Yval.ndim)
  
  return (Xval, Yval)

def test_generator(maps, it_test, Nframes, Nchannels = 4, Nx = 128, Ny = 128, maps_max = None, maps_min = None):
  
  Nframes_with_test = Nframes + 1
  
  Nt_test = len(it_test)
  
  XYtest = np.empty( (Nt_test, Ny, Nx, Nchannels) )
  for ic in range(Nchannels):
      XYtest[:, :, :, ic] = ( maps[ic][it_test, :Ny , :Nx] - maps_min[ic] ) / (maps_max[ic] - maps_min[ic])
  
  
  Nseq_test = np.int(Nt_test / Nframes_with_test)
  
  iXY = range(Nt_test)
  iY = range(Nframes, Nt_test, Nframes_with_test)
  iX = ~np.in1d(iXY, iY)
  
  Xtest = XYtest[iX,:,:].reshape(Nseq_test, Nframes, Ny, Nx, Nchannels)
  Xtest = np.squeeze(Xtest)
  
  Ytest = XYtest[iY,:,:].reshape(Nseq_test, Ny, Nx,1)
  
  #Xtest = np.expand_dims(Xtest, axis = Xtest.ndim)
  #Ytest = np.expand_dims(Ytest, axis = Ytest.ndim)

  return (Xtest, Ytest)    
  
  
#LOADING DATA
#------------------------------------------------------------
#Here is where to set up the parameter of the batch generator
#BATCH GENERATOR DEFINITION
#------------------------------------------------------------
#Here is where to set up the parameter of the batch generator

Nframes = 1 #Number of frames within each sequences to be used during training
batch_size = 32 #Number of sequences to included within each batch during training
epochs = 20
droprate = 0.2

r_train = 0.96 #Proportion of all sequences to be used during training
r_val = 0.02 #Proportion of all sequences to be used as validation set
r_test = 0.02 #Proportion of all sequences to be used as final test set

#Number of samples = number of sequences = number maps / number frames
Nseq = np.floor((Nt-1) / Nframes)

#Number steps per epoch = number of batches = number of samples / batch_size
spe = np.ceil(Nseq / batch_size)

#Using these parameters to divide the date in train, validation and test sets:
if r_train + r_val + r_test != 1:
  raise Exception('r_train, r_val and r_test should sum up 1')
it_train, it_val, it_test = train_split_seq(maps, Nframes = Nframes, r_train = r_train, r_val = r_val, r_test = r_test, random_state = None)

#Creating the batch generator to be used during training
BG = batch_generator(maps, it_train, Nframes = Nframes, batch_size = batch_size, maps_max = maps_max, maps_min = maps_min)

#Creating the validation data to be used during training
VG = validation_generator(maps, it_val, Nframes, maps_max = maps_max, maps_min = maps_min)

#Creating the test data to be used during evaluation
TG = validation_generator(maps, it_test, Nframes, maps_max = maps_max, maps_min = maps_min)

print('{} maps in total'.format(Nt))
print('{} / {} / {} maps for training / validation / test'.format(len(it_train), len(it_val), len(it_test)))
print('')
print('{} frames per sequence'.format(Nframes))
print('{} sequences = samples in total'.format(Nseq))
print('{} / {} / {} sequences for training / validation / test'.format(len(it_train)/(Nframes+1), len(it_val)/(Nframes+1), len(it_test)/(Nframes+1)))
print('')
print('{} sequences = samples per batch'.format(batch_size))
print('{} batches per epoch'.format(spe))
print('{} epochs'.format(epochs))
    
  
    


#-----------------------------------------------------------------
## MODEL DEFINITION
#------------------------------------------------------------------
def model_CNN_channels(input_shape = (128, 128, 4), droprate = 0.2,
                        epochs = 100, generator = BG, validation_data = VG, steps_per_epoch = spe)

    #Initializing the CNN
    model=Sequential()

    #Adding layers. 

    #Convolutions
    ##ORDER OF THE INPUT_SHAPE IS NY,NX,Nchannels
    model.add(Conv2D(16,(3,3),input_shape=input_shape,activation='relu',padding='same'))
    model.add(Dropout(droprate))
    model.add(MaxPool2D(pool_size=(2,2)))

    ##Second CNV layer
    model.add(Conv2D(32,(3,3),activation='relu',padding='valid'))
    model.add(Dropout(droprate))
    model.add(MaxPool2D(pool_size=(2,2)))

    ##Third CNV layer
    model.add(Conv2D(64,(3,3),activation='relu',padding='valid'))
    model.add(Dropout(droprate))
    #model.add(MaxPool2D(pool_size=(2,2)))

    #DeConvolutions
    ##Fourth CNV layer
    model.add(Conv2DTranspose(64,(3,3),activation='relu',padding='valid'))
    model.add(Dropout(droprate))
    model.add(UpSampling2D(size=(2,2)))

    ##Fifth CNV layer
    model.add(Conv2DTranspose(32,(3,3),activation='relu',padding='valid'))
    model.add(Dropout(droprate))
    model.add(UpSampling2D(size=(2,2)))

    ##Sixth CNV layer
    model.add(Conv2DTranspose(1,(3,3),activation='sigmoid',padding='same'))

    #compiling the model
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics = ['mean_squared_error'])
    
    #training the model
    history=model.fit_generator(generator = BG, validation_data = VG, epochs = epochs, steps_per_epoch = spe)

    return model, history
    
    
##TRAINING EVALUATION
#--------------------
def training_curve(history):
    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid()
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.grid()
    plt.title('metric')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.show()


##PREDICTION EVALUATION
#-----------------------
def training_evaluation(TG):
 
    Xtest, Ytest = TG
    mse_benchmark = mse_sk(Ytest.flatten(), Xtest[:,:,:,0].flatten())

    print('Benchmark to beat: {}'.format(mse_test))

    mse_model = mse_sk(Ytest.flatten(), model.predict(Xtest).flatten())
    print('Mean metric value for the predicted test set: {}'.format(mse_model))
    
    return mse_benchmark, mse_model
    