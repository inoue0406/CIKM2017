""" 
Undersample temporal resolution 
15 -> 5
"""
import h5py
import os

import numpy as np

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
x_train =  x_train.reshape(N, nt, nx, ny, nz)
h5file.close()
# 
h5file = h5py.File('processed/for_python/radar_testB_3d_ds3_alltime.hdf5','r')
x_test =  h5file['MR'].value
x_test =  x_test.reshape(Nt, nt, nx, ny, nz)
h5file.close()

x_train_s = x_train[:,0:nt:3,:,:,:]
x_test_s = x_test[:,0:nt:3,:,:,:] 

h5file = h5py.File('processed/for_python/radar_train_3d_ds3_dt3.hdf5','w')
h5file.create_dataset('MR',data= x_train_s)
h5file.close()

h5file = h5py.File('processed/for_python/radar_testB_3d_ds3_dt3.hdf5','w')
h5file.create_dataset('MR',data= x_test_s)
h5file.close()
