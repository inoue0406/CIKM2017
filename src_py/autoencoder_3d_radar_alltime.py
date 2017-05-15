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
nx = 34
ny = 34
nz = 4
nt = 15

# read data
h5file = h5py.File('processed/for_python/radar_train_3d_ds3_alltime.hdf5','r')
x_train =  h5file['MR'].value
x_train =  x_train/np.max(x_train) # regularize to [0-1]
h5file.close()
# 
h5file = h5py.File('processed/for_python/radar_testA_3d_ds3_alltime.hdf5','r')
x_test =  h5file['MR'].value
x_test =  x_test/np.max(x_test) # regularize to [0-1]
h5file.close()

# ---------------
input_img = Input(shape=(nx*ny*nz,))

encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)

decoded = Dense(256, activation='relu')(encoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(nx*ny*nz, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_img, outputs=decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# fitting
autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# save
autoencoder.save_weights('res/autoencoder_3d_alltime/autoencoder.h5')
autoencoder.load_weights('res/autoencoder_3d_alltime/autoencoder.h5')

# output middle layer 
intermediate_model = Model(inputs=autoencoder.input, 
                           outputs=autoencoder.layers[3].output)
intermediate_model.compile(optimizer='adadelta', loss='binary_crossentropy')
intermediate_output = intermediate_model.predict(x_train)

h5file = h5py.File('res/autoencoder_3d_alltime/auto_3d_feature128_0515_train_alltime.hdf5','w')
h5file.create_dataset('MR',data= intermediate_output)
h5file.close()

intermediate_output = intermediate_model.predict(x_test)

h5file = h5py.File('res/autoencoder_3d_alltime/auto_3d_feature128_0515_testA_alltime.hdf5','w')
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
    x3d = x_test[i].reshape(34, 34, 4)
    plt.imshow(x3d[:,:,3])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 変換された画像を表示
    ax = plt.subplot(2, n, i+1+n)
    x3d = decoded_imgs[i].reshape(34, 34, 4)
    plt.imshow(x3d[:,:,3])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


