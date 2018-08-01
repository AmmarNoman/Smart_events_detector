from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D,AveragePooling2D, ZeroPadding2D
from keras.utils import np_utils
import numpy as np
import pandas as pd

import config_path as config_p

# fix random seed for reproducibility
seed = 1994
np.random.seed(seed)

path_mean_tr=config_p.b_path+"CNN/mean_CNN_tr.csv"
path_std_tr=config_p.b_path+"CNN/std_CNN_tr.csv"


height = 20 #==d
width = 1
depth = 3*2 #==c
#filters =======
F=[4,8]  #the number of convolution filters to use in each convolution layer
#kernel_size (f,c) =======
f = 4 #the number of rows in each convolution kernel (height)
w = width #the number of columns in each convolution kernel (width)

def prepare_data_cnn(df, C=depth, H=height, W=width):    
    #(n, depth, height, width)
    temp = np.array(df)
    tt = temp.reshape((temp.shape[0], C, H, W))
    #print ("\n",temp.shape," >>> ",tt.shape)
    return tt

def CNN_2D(version="01"):
	# Build Model
	model = Sequential()
	#(n, depth, height, width)
	#print "\nF= ",F
	model.add(Convolution2D(F[0], (f, w), activation='relu', input_shape=(depth,height,width), data_format='channels_first'))
	model.add(AveragePooling2D(pool_size=(2,1), data_format='channels_first'))

	model.add(Convolution2D(F[1], (f, w), activation='relu', data_format='channels_first'))
	#model.add(AveragePooling2D(pool_size=(2,1), data_format='channels_first'))

	model.add(Dropout(0.25))
	model.add(Flatten())

	model.add(Dense(20, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(15, activation='softmax'))

	#print("[INFO] compiling model...")
	model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
	
	#print "\n[INFO] version :: ",version

	#print("\n\n[INFO] loading model...")
	if version=="00":
		#best of 100 epochs in v00
		weights_path = config_p.b_path+"CNN/CNN-04-0.25-v00.hdf5"
	elif version=="01":
		#best of 10k epochs in v01
		weights_path = config_p.b_path+"CNN/CNN-8272-0.34_v01.hdf5"
	
	model.load_weights(weights_path)

	return model

