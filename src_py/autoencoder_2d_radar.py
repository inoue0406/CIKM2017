'''
Apply autoencoder for 2d radar echo data.
'''

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import h5py
import os

#----------
os.chdir('C:/home/CIKM2017')

N = 10000
nx = 101
ny = 101

file = h5py.File('processed/train_h5/radar_train_00001.hdf5', 'r') 
data =  file['MR']
file.close()
#---------------
# array size should be (# samples, data length)
x = data[:,:,0,14]
xr = x.reshape(1,nx*ny)
x2 = np.vstack((xr,xr))

x_train = np.zeros((N,nx*ny))

encoding_dim = 32
input_img = Input(shape=(784,))

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input=input_img, output=decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# the data, shuffled and split between train and test sets
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# fitting
autoencoder.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# save
autoencoder.save_weights('autoencoder.h5')
autoencoder.load_weights('autoencoder.h5')

# ------------------------------------
# plotting
# ------------------------------------
import matplotlib.pyplot as plt

# テスト画像を変換
decoded_imgs = autoencoder.predict(x_test)

# 何個表示するか
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # オリジナルのテスト画像を表示
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 変換された画像を表示
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

