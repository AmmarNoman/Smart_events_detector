from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense

base_path = "/home/gpuadmin/RAMAH/PFE_Youssef/weights-improvement/"

def VNN(layer_sizes = [105], dropout_prob = 0.5, n_classes = 15):
    
    # Build Model
    model = Sequential()
    model.add(Dense(layer_sizes[0], input_shape=(layer_sizes[0],), kernel_initializer="uniform", activation="sigmoid"))
    model.add(Dropout(dropout_prob))
    for layer_size in layer_sizes[1:]:
        model.add(Dense(layer_size, kernel_initializer="uniform", activation="sigmoid"))
        model.add(Dropout(dropout_prob))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    #print("[INFO] compiling VNN model...")
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    weights_path = base_path+"VNN/VNN_model-104-0.16_v01_best.hdf5"
    #print("\n\n[INFO] loading VNN model...")
    model.load_weights(weights_path)

    return model