from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, LSTM

import config_path as config_p


path_mean_tr=config_p.b_path+"RNN/mean_RNN_tr.csv"
path_std_tr=config_p.b_path+"RNN/std_RNN_tr.csv"

def RNN_LSTM(version="03_10k", layer_sizes = [66, 16, 8, 4], dropout_prob = 0.0, n_classes = 15, maxlen = 1):
    # Build Model
    model = Sequential()
    model.add( LSTM(layer_sizes[0], return_sequences=True, input_shape=(maxlen, 66) ,activation='sigmoid') )
    model.add(Dropout(dropout_prob))
    for layer_size in layer_sizes[1:]:
        model.add(LSTM(layer_size, return_sequences=True ,activation='sigmoid'))
        model.add(Dropout(dropout_prob))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    #print("[INFO] compiling LSTM model...")
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    
    #print "\n[INFO] version :: ",version

    #print("\n\n[INFO] loading LSTM model...")
    if version=="03":
        weights_path = config_p.b_path+"RNN/NEW_RNN-916-0.25.hdf5"
    elif version=="03_10k":
        weights_path = config_p.b_path+"RNN/NEW_RNN_v03_10k.h5"
    elif version=="05":
        weights_path = config_p.b_path+"RNN/NEWER_Large_RNN-1843-0.25_v05.hdf5"
    elif version=="05_5k":
        weights_path = config_p.b_path+"RNN/NEWER_Large_RNN_v05_5k.h5"
    
    model.load_weights(weights_path)

    return model