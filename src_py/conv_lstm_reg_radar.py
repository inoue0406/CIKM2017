""" 
Using LSTM for CYKM2017 prediction
ConvLSTM + regression
"""
import pandas
import h5py
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt

os.chdir('C:/home/CIKM2017')

N = 10000
Nt = 2000
nx = 34
ny = 34
nz = 4
nt = 15

# --------------------------
# read data
# radar data
h5file = h5py.File('processed/for_python/radar_train_3d_ds3_alltime.hdf5','r')
x_train =  h5file['MR'].value
x_train =  x_train/np.max(x_train) # regularize to [0-1]
x_train =  x_train.reshape(N, nt, nx, ny, nz)
h5file.close()
# 
h5file = h5py.File('processed/for_python/radar_testA_3d_ds3_alltime.hdf5','r')
x_test =  h5file['MR'].value
x_test =  x_test/np.max(x_test) # regularize to [0-1]
x_test =  x_test.reshape(Nt, nt, nx, ny, nz)
h5file.close()

# gauge obs data
g1 = pandas.read_csv("processed/train/gauge_ts_train.csv")
y_train = g1["rain"].values
g2 = pandas.read_csv("processed/testA/gauge_ts_testA.csv")
y_test = g2["rain"].values

# select only one vertical layer
#x_train_s = x_train[:,:,:,:,[1]] # [] is needed for keeping dimension info
#x_test_s = x_train[:,:,:,:,[1]] 
x_train_s = x_train[:,0:nt:2,:,:,[1]] # [] is needed for keeping dimension info
x_test_s = x_test[:,0:nt:2,:,:,[1]] 

nt2 = 8 # every 2 step

# input and output for Conv-LSTM layear
x_train_in  = x_train_s 
x_test_in   = x_test_s

# --------------------------
# model

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.

seq = Sequential()
seq.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                   input_shape=(nt2, nx, ny, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
# regression layer
seq.add(Flatten())
#seq.add(Dropout(0.5))
seq.add(Dense(1, activation='linear'))


seq.compile(loss='mean_squared_error', optimizer='adadelta')
# print for chk
seq.summary()

#SVG(model_to_dot(seq, show_shapes=True).create(prog='dot', format='svg'))

# ---------------------------------------------------
# training

seq.fit(x_train_in, y_train, batch_size=10,
        epochs=300, validation_split=0.1)

seq.save_weights('res/convlstm/conv_lstm_2d_reg.h5')
seq.load_weights('res/convlstm/conv_lstm_2d_reg.h5')

# test
y_pred = seq.predict(x_test_in, batch_size=32, verbose=0)

# save the results
df = pandas.DataFrame(y_pred)
df.to_csv("res/convlstm/pred_convlstm_reg.csv",
          index=False,header=False,float_format='%7.3f')


