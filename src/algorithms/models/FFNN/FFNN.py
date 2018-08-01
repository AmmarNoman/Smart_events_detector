from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, LSTM

import config_path as config_p


path_mean_tr=config_p.b_path+"RNN/mean_RNN_tr.csv"
path_std_tr=config_p.b_path+"RNN/std_RNN_tr.csv"



def FFNN(layer_sizes = [66, 16, 8, 4], dropout_prob = 0.0, n_classes = 15):
    
    # Build Model
    model = Sequential()
    model.add(Dense(66, input_shape=(layer_sizes[0],), kernel_initializer="uniform", activation="relu"))
    model.add(Dropout(dropout_prob))
    for layer_size in layer_sizes[1:]:
        model.add(Dense(layer_size, kernel_initializer="uniform", activation="relu"))
        model.add(Dropout(dropout_prob))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    #print("[INFO] compiling FFNN model...")
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    weights_path = config_p.b_path+"FFNN/random_train_FFNN-6276-0.31.hdf5"
    #print("\n\n[INFO] loading FFNN model...")
    model.load_weights(weights_path)

    return model