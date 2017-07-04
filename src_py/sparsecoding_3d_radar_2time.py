'''
Apply autoencoder for 3d radar echo data.
'''

import numpy as np
import h5py
import os

from sklearn.decomposition import sparse_encode
from sklearn.decomposition import MiniBatchDictionaryLearning
#----------
os.chdir('C:/home/CIKM2017')

N = 10000
Nt = 2000
nx = 34
ny = 34
nz = 4

# read data
h5file = h5py.File('processed/for_python/radar_train_3d_ds3.hdf5','r')
x_train =  h5file['MR'].value
x_train =  x_train/np.max(x_train) # regularize to [0-1]
h5file.close()
# 
h5file = h5py.File('processed/for_python/radar_testB_3d_ds3.hdf5','r')
x_test =  h5file['MR'].value
x_test =  x_test/np.max(x_test) # regularize to [0-1]
h5file.close()

# change shape
nt_slct = 100
x_train2 = x_train[0:int(N*nt_slct/100),:]
x_test2 = x_test[0:int(Nt*nt_slct/100),:]

# cut data for testing
#x_train = x_train[0:100,:]

# sparse coding
# 0:29
new_dimensions = 256

mbdl = MiniBatchDictionaryLearning(new_dimensions)
mbdl.fit(x_train2)

coded = sparse_encode(x_train, mbdl.components_)

fname = 'res/sparsecoding/auto_2d_feature%d_0629_train_nt%d.hdf5' % (new_dimensions,nt_slct)
h5file = h5py.File(fname,'w')
h5file.create_dataset('MR',data= coded)
h5file.close()

coded = sparse_encode(x_test, mbdl.components_)

fname = 'res/sparsecoding/auto_2d_feature%d_0629_testB_nt%d.hdf5' % (new_dimensions,nt_slct)
h5file = h5py.File(fname,'w')
h5file.create_dataset('MR',data= coded)
h5file.close()

