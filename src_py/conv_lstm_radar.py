""" 
Using LSTM for CYKM2017 prediction
"""

import h5py
import os

from keras.models import Sequential
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
x_test =  x_test.reshape(Nt, nx, ny, nz, nt)
h5file.close()

# select only one vertical layer
#x_train_s = x_train[:,:,:,:,[1]] # [] is needed for keeping dimension info
#x_test_s = x_train[:,:,:,:,[1]] 
x_train_s = x_train[:,0:nt:2,:,:,[1]] # [] is needed for keeping dimension info
x_test_s = x_train[:,0:nt:2,:,:,[1]] 

nt2 = 8 # every 2 step

# input and output for Conv-LSTM layear
# "out" is one time-step ahead of "in"
x_train_in  = x_train_s[:,0:(nt2-1),:,:,:] 
x_train_out = x_train_s[:,1:nt2,:,:,:] 
x_test_in   = x_test_s[:,0:(nt2-1),:,:,:] 
x_test_out  = x_test_s[:,1:nt2,:,:,:] #

# --------------------------
# model

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.

seq = Sequential()
seq.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                   input_shape=(None, nx, ny, 1),
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
seq.compile(loss='binary_crossentropy', optimizer='adadelta')

#SVG(model_to_dot(seq, show_shapes=True).create(prog='dot', format='svg'))

# ---------------------------------------------------
# training

seq.fit(x_train_in, x_train_out, batch_size=10,
        epochs=300, validation_split=0.1)

# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions
which = 1004
track = noisy_movies[which][:7, ::, ::, ::]

for j in range(16):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)


# And then compare the predictions
# to the ground truth
track2 = noisy_movies[which][::, ::, ::, ::]
for i in range(15):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Inital trajectory', fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = shifted_movies[which][i - 1, ::, ::, 0]

    plt.imshow(toplot)
    plt.savefig('%i_animate.png' % (i + 1))
