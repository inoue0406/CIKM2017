'''
Apply autoencoder for 3d radar echo data.
'''

from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import h5py
import os

#----------
os.chdir('C:/home/CIKM2017')

N = 10000
Nt = 2000
nx = 34
ny = 34
nz = 4

# read data
h5file = h5py.File('processed/for_python/radar_train_3d_ds3_dt3.hdf5','r')
x_train =  h5file['MR'].value
x_train =  x_train/np.max(x_train) # regularize to [0-1]
h5file.close()
# 
h5file = h5py.File('processed/for_python/radar_testA_3d_ds3_dt3.hdf5','r')
x_test =  h5file['MR'].value
x_test =  x_test/np.max(x_test) # regularize to [0-1]
h5file.close()

# change shape
# use only 3th and 5th time step
lyr = 1
x_train = x_train[:,[2,4],:,:,lyr].reshape(N,2*nx*ny)
x_test = x_test[:,[2,4],:,:,lyr].reshape(Nt,2*nx*ny)

# ---------------
input_img = Input(shape=(nx*ny*2,))

encoded = Dense(1024, activation='relu')(input_img)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dense(256, activation='relu')(encoded)

decoded = Dense(512, activation='relu')(encoded)
decoded = Dense(1024, activation='relu')(decoded)
decoded = Dense(nx*ny*2, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_img, outputs=decoded)

#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# fitting
autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# save
autoencoder.save_weights('res/autoencoder_3d_2time/autoencoder.h5')
autoencoder.load_weights('res/autoencoder_3d_2time/autoencoder.h5')

# output middle layer 
intermediate_model = Model(inputs=autoencoder.input, 
                           outputs=autoencoder.layers[3].output)
intermediate_model.compile(optimizer='adadelta', loss='binary_crossentropy')
intermediate_output = intermediate_model.predict(x_train)

fname = 'res/autoencoder_3d_2time/auto_2d_feature256_0626_train_2time_lyr%d.hdf5' % lyr
h5file = h5py.File(fname,'w')
h5file.create_dataset('MR',data= intermediate_output)
h5file.close()

intermediate_output = intermediate_model.predict(x_test)

fname = 'res/autoencoder_3d_2time/auto_2d_feature257_0626_testA_2time_lyr%d.hdf5' % lyr
h5file = h5py.File(fname,'w')
h5file.create_dataset('MR',data= intermediate_output)
h5file.close()

# ------------------------------------
# plotting
# ------------------------------------
import matplotlib.pyplot as plt

# テスト画像を変換
decoded_imgs = autoencoder.predict(x_test)

# 何個表示するか
n = 20
plt.figure(figsize=(40, 8))
for i in range(n):
    # オリジナルのテスト画像を表示
    ax = plt.subplot(2, n, i+1)
    x3d = x_test[i].reshape(2, 34, 34)
    plt.imshow(x3d[0,:,:])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 変換された画像を表示
    ax = plt.subplot(2, n, i+1+n)
    x3d = decoded_imgs[i].reshape(2, 34, 34)
    plt.imshow(x3d[0,:,:])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


