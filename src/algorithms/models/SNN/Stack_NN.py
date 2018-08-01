from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense

import config_path as config_p

path_mean_tr=config_p.b_path+"RNN/mean_RNN_tr.csv"
path_std_tr=config_p.b_path+"RNN/std_RNN_tr.csv"



def SNN(layer_sizes = [105], dropout_prob = 0.5, n_classes = 15):
    
    # Build Model
    model = Sequential()
    model.add(Dense(layer_sizes[0], input_shape=(layer_sizes[0],), kernel_initializer="uniform"))
    model.add(Dropout(dropout_prob))
    for layer_size in layer_sizes[1:]:
        model.add(Dense(layer_size, kernel_initializer="uniform"))
        model.add(Dropout(dropout_prob))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    #print("[INFO] compiling SNN model...")
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    weights_path = config_p.b_path+"SNN/v03/v03.h5"
    #print("\n\n[INFO] loading SNN model...")
    model.load_weights(weights_path)

    return model