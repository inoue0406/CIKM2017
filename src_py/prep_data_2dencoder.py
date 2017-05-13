'''
Apply autoencoder for 2d radar echo data.
'''

import numpy as np
import h5py
import os

#----------
os.chdir('C:/home/CIKM2017')

N = 10000
nx = 101
ny = 101
nz = 4

x_train2d = np.zeros((N,nx*ny),dtype=np.float32)
x_train3d = np.zeros((N,nx*ny*nz),dtype=np.float32)

for n in range(10000):
    fname = 'processed/train_h5/radar_train_%05d.hdf5' % (n+1)
    print(n,fname)
    file = h5py.File(fname, 'r') 
    data =  file['MR']
    #---------------
    # array size should be (# samples, data length)
    x = data[:,:,0,14]
    x3 = data[:,:,:,14]
    x_train2d[n,:] = x.reshape(1,nx*ny)
    x_train3d[n,:] = x3.reshape(1,nx*ny*nz)
    # 
    file.close()

h5file = h5py.File('processed/for_python/radar_train_2d.hdf5','w')
h5file.create_dataset('MR',data= x_train2d)
h5file.close()

h5file = h5py.File('processed/for_python/radar_train_3d.hdf5','w')
h5file.create_dataset('MR',data= x_train3d)
h5file.close()

# test set

N = 2000
x_testA2d = np.zeros((N,nx*ny),dtype=np.float32)
x_testA3d = np.zeros((N,nx*ny*nz),dtype=np.float32)

for n in range(2000):
    fname = 'processed/testA_h5/radar_testA_%05d.hdf5' % (n+1)
    print(n,fname)
    file = h5py.File(fname, 'r') 
    data =  file['MR']
    #---------------
    # array size should be (# samples, data length)
    x = data[:,:,0,14]
    x3 = data[:,:,:,14]
    x_testA2d[n,:] = x.reshape(1,nx*ny)
    x_testA3d[n,:] = x3.reshape(1,nx*ny*nz)
    # 
    file.close()

h5file = h5py.File('processed/for_python/radar_testA_2d.hdf5','w')
h5file.create_dataset('MR',data= x_testA2d)
h5file.close()

h5file = h5py.File('processed/for_python/radar_testA_3d.hdf5','w')
h5file.create_dataset('MR',data= x_testA3d)
h5file.close()


