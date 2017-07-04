'''
Apply autoencoder for 2d radar echo data.
With downsampling
'''

import numpy as np
import h5py
import os

#----------
os.chdir('C:/home/CIKM2017')

N = 10000
#nx = 101
#ny = 101
nx = 34
ny = 34
nz = 4
nt = 15

x_train3d = np.zeros((N*nt,nx*ny*nz),dtype=np.float32)

#n = 1
for n in range(N):
    fname = 'processed/train_h5/radar_train_%05d.hdf5' % (n+1)
    print(n,fname)
    file = h5py.File(fname, 'r') 
    data =  file['MR']
    #---------------
    # array size should be (# samples, data length)
    for t in range(nt):
        #print(n*nt+t)
        x3 = data[:,:,:,t]
        x3 = x3[::3,::3,:] # downsample
        x_train3d[n*nt+t,:] = x3.reshape(1,nx*ny*nz)
    # 
    
    file.close()

h5file = h5py.File('processed/for_python/radar_train_3d_ds3_alltime.hdf5','w')
h5file.create_dataset('MR',data= x_train3d)
h5file.close()

# test set

N = 2000

x_train3d = np.zeros((N*nt,nx*ny*nz),dtype=np.float32)

for n in range(N):
    fname = 'processed/testA_h5/radar_testA_%05d.hdf5' % (n+1)
    print(n,fname)
    file = h5py.File(fname, 'r') 
    data =  file['MR']
    #---------------
    # array size should be (# samples, data length)
    for t in range(nt):
        #print(n*nt+t)
        x3 = data[:,:,:,t]
        x3 = x3[::3,::3,:] # downsample
        x_train3d[n*nt+t,:] = x3.reshape(1,nx*ny*nz)
    # 
    file.close()

h5file = h5py.File('processed/for_python/radar_testA_3d_ds3_alltime.hdf5','w')
h5file.create_dataset('MR',data= x_train3d)
h5file.close()


N = 2000

x_train3d = np.zeros((N*nt,nx*ny*nz),dtype=np.float32)

for n in range(N):
    fname = 'processed/testB_h5/radar_testB_%05d.hdf5' % (n+1)
    print(n,fname)
    file = h5py.File(fname, 'r') 
    data =  file['MR']
    #---------------
    # array size should be (# samples, data length)
    for t in range(nt):
        #print(n*nt+t)
        x3 = data[:,:,:,t]
        x3 = x3[::3,::3,:] # downsample
        x_train3d[n*nt+t,:] = x3.reshape(1,nx*ny*nz)
    # 
    file.close()

h5file = h5py.File('processed/for_python/radar_testB_3d_ds3_alltime.hdf5','w')
h5file.create_dataset('MR',data= x_train3d)
h5file.close()

