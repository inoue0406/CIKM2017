'''
test code for Convolutional LTSM
after https://github.com/fchollet/keras/pull/1818
'''

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D,Convolution3D
from keras.layers.recurrent_convolutional import LSTMConv2D

