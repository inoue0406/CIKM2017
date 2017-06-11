'''
Apply CNN for 3d radar echo data.
With Cross-validation
'''
import pandas
import numpy as np
import h5py
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.utils.vis_utils import plot_model

#----------
os.chdir('C:/home/CIKM2017')

N = 10000
Nt = 2000
nx = 34
ny = 34
nz = 4

# --------------------------
# read data
# radar data
h5file = h5py.File('processed/for_python/radar_train_3d_ds3.hdf5','r')
x_train =  h5file['MR'].value
x_train =  x_train/np.max(x_train) # regularize to [0-1]
x_train =  x_train.reshape(N, nx, ny, nz)
h5file.close()
# 
h5file = h5py.File('processed/for_python/radar_testA_3d_ds3.hdf5','r')
x_test =  h5file['MR'].value
x_test =  x_test/np.max(x_test) # regularize to [0-1]
x_test =  x_test.reshape(Nt, nx, ny, nz)
h5file.close()
# gauge obs data
g1 = pandas.read_csv("processed/train/gauge_ts_train.csv")
y_train = g1["rain"].values
g2 = pandas.read_csv("processed/testA/gauge_ts_testA.csv")
y_test = g2["rain"].values

# plot for chk
#import matplotlib.pyplot as plt
#plt.imshow(x_train[333,:,:,1])

# ---------------
# CNN model

def neural_model1():
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(nx,ny,nz)))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mean_squared_error'])
    return model
 
#plot_model(neural_model1)    

# CV
model = KerasRegressor(build_fn=neural_model1, epochs=100, batch_size=100, verbose=2)
#results = cross_validation.cross_val_score(model, x_train, y_train, cv=5, scoring = "mean_squared_error")
results = cross_val_score(model, x_train, y_train, cv=5, scoring = "mean_squared_error")
print(results)
print("Model1:" + str(results.mean()))

# grid search
model = KerasRegressor(build_fn=neural_model1, verbose=2)
parameters = {'epochs':[10,20,40,60],'batch_size':[20,50,100,200]}
grid = GridSearchCV(model,parameters,n_jobs=1,
                                scoring="neg_mean_absolute_error",cv=5,verbose=2)
grid_result = grid.fit(x_train, y_train)

print(grid_result.best_params_)
grid_result.cv_results_

#----------------------------- 
# Fit model on all the training data
# 
model.fit(x_train, y_train, batch_size=32, epochs=80, verbose=2)

# save
model.save_weights('res/CNN_3d/CNN.h5')
model.load_weights('res/CNN_3d/CNN.h5')

# test
y_pred = model.predict(x_train, batch_size=32, verbose=0)

# save the results
df = pandas.DataFrame(y_pred)
df.to_csv("res/CNN_3d/pred_cnn_3d.csv",
          index=False,header=False,float_format='%7.3f')

