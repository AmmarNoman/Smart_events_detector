import pandas as pd
import numpy as np
'''from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, LSTM
import matplotlib.pyplot as plt'''

import sys
from os import path

import config_path as config_p

#print path.abspath(".")
sys.path.append( path.dirname( path.abspath(config_p.b_path) ) )

from ..models.predict_models import get_Prediction

# fix random seed for reproducibility
np.random.seed(1994)

#Hyper-parametre
"""
Threshold=0.92
path_input_data=".../workspace/events_gen/data_2014-11-26.csv"
path_input_time=".../workspace/events_gen/time_2014-11-26.csv"
"""

def selectIndex(probs, threshold):
        labels = []
        indexes = []
        ind = 0
        for pred in probs:
            val_max = max(pred)
            if val_max > threshold :
                pred = pred.tolist()
                Y = pred.index(val_max)+1
                if Y!=1:
                    labels.append(Y)
                    indexes.append(ind)
            ind=ind+1
        return labels, indexes


def predict(path_input_data, path_input_time, Y_file, Time_file, model_name="RNN_LSTM", Threshold=0.92):
    #Load test time
    test_time = pd.read_csv(path_input_time, sep=',', header=-1, names=['start', 'end'])
    
    Test_pred = get_Prediction(model_name, path_input_data)
    labels_test, indexes_test = selectIndex(Test_pred, Threshold)

    np.array(labels_test).tofile(Y_file, sep='\n')
    test_time.iloc[indexes_test].to_csv(Time_file, index=False, header=False)


