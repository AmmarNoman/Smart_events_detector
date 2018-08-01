from keras.utils import np_utils

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

import config_path as config_p

path_mean_tr=config_p.b_path+"FCN/mean_FCN_tr.csv"
path_std_tr=config_p.b_path+"FCN/std_FCN_tr.csv"

def LSTM_CNN(version="v01"):
    #input Shape (None, 1, 66)
    
    # Build Model
    ip = Input(shape=(1, 66))
    x =  LSTM(8)(ip)
    x = Dropout(0.8)(x)
    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)
    x = concatenate([x, y])
    out = Dense(15, activation='softmax')(x)
    model = Model(ip, out)

    #print("[INFO] compiling LSTM+CNN model...")
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    
    #print "\n[INFO] version :: ",version

    #print("\n\n[INFO] loading LSTM+CNN model...")
    if version=="v01":
        weights_path = config_p.b_path+"FCN/model-03-0.10_v01.hdf5"
    
    model.load_weights(weights_path)

    return model