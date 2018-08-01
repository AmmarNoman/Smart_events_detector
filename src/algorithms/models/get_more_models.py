import pandas as pd
import numpy as np
# fix random seed for reproducibility
seed = 1994
np.random.seed(seed)
#import keras.backend as K

from sklearn.externals import joblib
from keras.models import Model, load_model
import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)


from_path="/home/gpuadmin/RAMAH/workspace/"
base_path = "/home/gpuadmin/RAMAH/PFE_Youssef/weights-improvement/"
batch_s=1000

Models = {
    #Model1 drp = 0.5
    "My-MLSTM-Correct-Shape3_v03":{
        "model_path": base_path+"My-MLSTM-Correct-Shape3_v03/model_weights-533-0.33_My-MLSTM-Correct-Shape3_v03.hdf5",
        "input_shape": (None, 11, 36),
        "vars_selected":[],#X_Acc+Y_Acc+Z_Acc+\
                        #X_Gyro+Y_Gyro+Z_Gyro+\
                        #X_Cf+Y_Cf+Z_Cf+\
                        #Speed+Times,
        "num_el_gyr_acc_mag_cf":36,
        "num_el_speed":36,
        "num_el_time":36,
        "version": "calib",
        "tr_data_path":from_path+"outTrainCFx36.csv"
    },
    #Model2 drp = 0.8, 0.7, 0.6 (up to ep.20000)
    "My-MLSTM-Correct-Shape3_v04":{
        "model_path": base_path+"My-MLSTM-Correct-Shape3_v04/model_weights-14036-0.28_My-MLSTM-Correct-Shape3_v04.hdf5",
        "input_shape": (None, 11, 36),
        "vars_selected":[],#X_Acc+Y_Acc+Z_Acc+\
                        #X_Gyro+Y_Gyro+Z_Gyro+\
                        #X_Cf+Y_Cf+Z_Cf+\
                        #Speed+Times,
        "num_el_gyr_acc_mag_cf":36,
        "num_el_speed":36,
        "num_el_time":36,
        "version": "calib",
        "tr_data_path":from_path+"outTrainCFx36.csv"
    },
    #Model3 My-MLSTM-FFNN4_v01 ep.2
    "My-MLSTM-FFNN4_v01":{
        "model_path": base_path+"My-MLSTM-FFNN4_v01/model_weights-00-0.13_My-MLSTM-FFNN4_v01.hdf5",
        "input_shape": (None, 1, 330),
        "vars_selected":[],#X_Acc+Y_Acc+Z_Acc+\
                        #X_Gyro+Y_Gyro+Z_Gyro+\
                        #X_Cf+Y_Cf+Z_Cf+\
                        #Speed+Times,
        "num_el_gyr_acc_mag_cf":36,
        "num_el_speed":5,
        "num_el_time":1,
        "version": "_",
        "tr_data_path":from_path+"outTrainCF.csv"
    },
    #Model3 My-MLSTM-FFNN4_v01 ep.500
    "My-MLSTM-FFNN4_v01_ep500":{
        "model_path": base_path+"My-MLSTM-FFNN4_v01_ep500/FCN_My-MLSTM-FFNN4_v01_model_ep500.hdf5",
        "input_shape": (None, 1, 330),
        "vars_selected":[],#X_Acc+Y_Acc+Z_Acc+\
                        #X_Gyro+Y_Gyro+Z_Gyro+\
                        #X_Cf+Y_Cf+Z_Cf+\
                        #Speed+Times,
        "num_el_gyr_acc_mag_cf":36,
        "num_el_speed":5,
        "num_el_time":1,
        "version": "_",
        "tr_data_path":from_path+"outTrainCF.csv"
    }
}




def get_data_header(num_el_gyr_acc_mag_cf, num_el_speed, num_el_time):
    Header = ['type']
    
    for axis in ['acc_x_','acc_y_','acc_z_',\
                 'gyr_x_','gyr_y_','gyr_z_',\
                 'mag_x_','mag_y_','mag_z_',\
                 'cf_x_','cf_y_','cf_z_']:
        for nb in range(1,num_el_gyr_acc_mag_cf+1):
            Header.append(axis+str(nb))
    
    for nb in range(1,num_el_speed+1):
        Header.append("spd_"+str(nb))
    
    if num_el_time==1:
        Header.append("time")
    else:
        for nb in range(1,num_el_time+1):
            Header.append("time_"+str(nb))
    
    #XYZ Acc, XYZ Gyro, XYZ Mag, XYZ CF, Speed, Times
    X_Acc =  Header[num_el_gyr_acc_mag_cf*0+1:num_el_gyr_acc_mag_cf*1+1]
    Y_Acc =  Header[num_el_gyr_acc_mag_cf*1+1:num_el_gyr_acc_mag_cf*2+1]
    Z_Acc =  Header[num_el_gyr_acc_mag_cf*2+1:num_el_gyr_acc_mag_cf*3+1]
    X_Gyro = Header[num_el_gyr_acc_mag_cf*3+1:num_el_gyr_acc_mag_cf*4+1]
    Y_Gyro = Header[num_el_gyr_acc_mag_cf*4+1:num_el_gyr_acc_mag_cf*5+1]
    Z_Gyro = Header[num_el_gyr_acc_mag_cf*5+1:num_el_gyr_acc_mag_cf*6+1]
    ############# Mag Not yet ready :: Need to be calibrated ###############
    ##X_Mag = Header[num_el_gyr_acc_mag_cf*6+1:num_el_gyr_acc_mag_cf*7+1] ##
    ##Y_Mag = Header[num_el_gyr_acc_mag_cf*7+1:num_el_gyr_acc_mag_cf*8+1] ##
    ##Z_Mag = Header[num_el_gyr_acc_mag_cf*8+1:num_el_gyr_acc_mag_cf*9+1] ##
    ########################################################################
    X_Cf =  Header[num_el_gyr_acc_mag_cf*9+1:num_el_gyr_acc_mag_cf*10+1]
    Y_Cf =  Header[num_el_gyr_acc_mag_cf*10+1:num_el_gyr_acc_mag_cf*11+1]
    Z_Cf =  Header[num_el_gyr_acc_mag_cf*11+1:num_el_gyr_acc_mag_cf*12+1]
    Speed = Header[num_el_gyr_acc_mag_cf*12+1:num_el_gyr_acc_mag_cf*12+num_el_speed+1]
    Times = Header[num_el_gyr_acc_mag_cf*12+num_el_speed+1:num_el_gyr_acc_mag_cf*12+num_el_speed+num_el_time+1]

    return Header, X_Acc,Y_Acc,Z_Acc, X_Gyro,Y_Gyro,Z_Gyro, X_Cf,Y_Cf,Z_Cf, Speed, Times

def prepare_data(model_name, path_input_data):
    model_info = Models[model_name]
    Header, \
    X_Acc,Y_Acc,Z_Acc, \
    X_Gyro,Y_Gyro,Z_Gyro, \
    X_Cf,Y_Cf,Z_Cf, \
    Speed, Times = get_data_header(model_info["num_el_gyr_acc_mag_cf"], model_info["num_el_speed"], model_info["num_el_time"])
    #vars_selected = model_info["vars_selected"]
    vars_selected = X_Acc+Y_Acc+Z_Acc+ \
                    X_Gyro+Y_Gyro+Z_Gyro+ \
                    X_Cf+Y_Cf+Z_Cf+ \
                    Speed+Times

    mean_std_path = from_path+"mean_std_CFx"+str(model_info["num_el_gyr_acc_mag_cf"])+\
    "x"+str(model_info["num_el_speed"])+"x"+str(model_info["num_el_time"])+".pkl"

    try:
        clf = joblib.load(mean_std_path)
        mean_tr = clf["mean_tr"][vars_selected]
        std_tr = clf["std_tr"][vars_selected]
    except Exception, e:
        print (e)
        train=pd.read_csv(model_info["tr_data_path"], sep=',', header=-1, names=Header)
        X_train = train[Header[1:]]
        mean_tr = X_train.mean()
        std_tr = X_train.std()
        joblib.dump({"mean_tr":mean_tr, "std_tr":std_tr}, mean_std_path)
        mean_tr = mean_tr[vars_selected]
        std_tr = std_tr[vars_selected]
    
    #Load testset
    test=pd.read_csv(path_input_data, sep=',', header=-1, names=Header[1:])
    X_test=test[vars_selected]
    #Normalize data
    X_test = (X_test - mean_tr)/(std_tr+1e-8)
    #Reshape data
    X_test = (np.array(X_test)).reshape(X_test.shape[0], model_info["input_shape"][1], model_info["input_shape"][2])
    print model_name,"<==","X_test shape : ",np.shape(X_test)
    return X_test

def reshape_data(df, shp):
    #shp=[(36,9), (5,1), (1,1)]
    df_x = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        temp =[]
        start = 0
        for dim in shp:  
            for var in range(dim[1]):
                #print start,":",start+dim[0]
                temp.append(list(row[start:start+dim[0]]))
                start = start+dim[0]
        df_x.append(temp)
    return np.array(df_x)

def Pred_Model (model_name, path_input_data, pred_classes=False):
    X_test = prepare_data(model_name, path_input_data)
    model_path = Models[model_name]["model_path"]
    #K.clear_session()
    model = load_model(model_path)

    if pred_classes:
        Test_pred = model.predict(np.array(X_test), batch_size=batch_s)
        Test_pred = Test_pred.argmax(axis=-1)+1
    else:
        #print ("\n\n[INFO]  Predict Proba : ")
        Test_pred = model.predict(np.array(X_test), batch_size=batch_s)

    return Test_pred
