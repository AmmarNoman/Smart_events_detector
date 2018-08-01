import pandas as pd
import numpy as np
from keras.models import Model, load_model
from sklearn.externals import joblib


from RNN import RNN_LSTM as LSTM
from CNN import CNN as CONV2D
from FFNN import FFNN as NN
from SNN import Stack_NN as Stack
from VNN import Voting_NN as Voting

from FCN import LSTM_Conv1D as RCNN

from get_more_models import Pred_Model, Models, from_path, base_path

batch_s = 100000


def Pred_FCN(Header, path_input_data, vs="v01", pred_classes=False):
    mean_rcnn = pd.read_csv(RCNN.path_mean_tr, header=-1, dtype={'0':str,'1':np.float64})
    mean_rcnn = pd.Series(data=np.array(mean_rcnn[1], dtype=np.float64), index=np.array(mean_rcnn[0], dtype=str))
    std_rcnn = pd.read_csv(RCNN.path_std_tr, header=-1, dtype={'0':str,'1':np.float64})
    std_rcnn = pd.Series(data=np.array(std_rcnn[1], dtype=np.float64), index=np.array(std_rcnn[0], dtype=str))

    model = RCNN.LSTM_CNN(version=vs)
    
    vars_selected = Header[1:41]+Header[101:121]+Header[241:246]+Header[246:247]
    #Load testset
    test=pd.read_csv(path_input_data, sep=',', header=-1, names=Header[1:])
    #Normalize data
    X_test=test[vars_selected]
    X_test = (X_test - mean_rcnn)/std_rcnn
    X_test = (np.array(X_test)).reshape(X_test.shape[0], 1, X_test.shape[1])

    
    if pred_classes:
        Test_pred = model.predict(np.array(X_test), batch_size=batch_s)
        Test_pred = Test_pred.argmax(axis=-1)+1
    else:
        #print ("\n\n[INFO]  Predict Proba : ")
        Test_pred = model.predict(np.array(X_test), batch_size=batch_s)
        #print "Test_pred : ",Test_pred.shape
    return Test_pred

def Pred_CNN(Header, path_input_data, vs="01", pred_classes=False):
    mean_ = pd.read_csv(CONV2D.path_mean_tr, header=-1, dtype={'0':str,'1':np.float64})
    mean_ = pd.Series(data=np.array(mean_[1], dtype=np.float64), index=np.array(mean_[0], dtype=str))
    std_ = pd.read_csv(CONV2D.path_std_tr, header=-1, dtype={'0':str,'1':np.float64})
    std_ = pd.Series(data=np.array(std_[1], dtype=np.float64), index=np.array(std_[0], dtype=str))

    model = CONV2D.CNN_2D(version=vs)
    
    vars_selected = Header[1:121]
    #Load testset
    test=pd.read_csv(path_input_data, sep=',', header=-1, names=Header[1:])
    #Normalize data
    X_test=test[vars_selected]
    X_test = (X_test - mean_)/std_
    X_test = CONV2D.prepare_data_cnn(X_test)

    if pred_classes:
        Test_pred = model.predict_classes(np.array(X_test), batch_size=batch_s)+1
    else:
        #print ("\n\n[INFO]  Predict Proba : ")
        Test_pred = model.predict_proba(np.array(X_test), batch_size=batch_s)

    return Test_pred

def Pred_FFNN(Header, path_input_data, vs="00", pred_classes=False):
    mean_ = pd.read_csv(NN.path_mean_tr, header=-1, dtype={'0':str,'1':np.float64})
    mean_ = pd.Series(data=np.array(mean_[1], dtype=np.float64), index=np.array(mean_[0], dtype=str))
    std_ = pd.read_csv(NN.path_std_tr, header=-1, dtype={'0':str,'1':np.float64})
    std_ = pd.Series(data=np.array(std_[1], dtype=np.float64), index=np.array(std_[0], dtype=str))

    model = NN.FFNN()
    
    vars_selected = Header[1:41]+Header[101:121]+Header[241:246]+Header[246:247]
    #Load testset
    test=pd.read_csv(path_input_data, sep=',', header=-1, names=Header[1:])
    #Normalize data
    X_test=test[vars_selected]
    X_test = (X_test - mean_)/std_
    X_test = np.array(X_test)

    if pred_classes:
        Test_pred = model.predict_classes(np.array(X_test), batch_size=batch_s)+1
    else:
        #print ("\n\n[INFO]  Predict Proba : ")
        Test_pred = model.predict_proba(np.array(X_test), batch_size=batch_s)

    return Test_pred


def Pred_RNN(Header, path_input_data, vs="03_10k", layers = [66, 16, 8, 4], pred_classes=False):
    mean_rnn = pd.read_csv(LSTM.path_mean_tr, header=-1, dtype={'0':str,'1':np.float64})
    mean_rnn = pd.Series(data=np.array(mean_rnn[1], dtype=np.float64), index=np.array(mean_rnn[0], dtype=str))
    std_rnn = pd.read_csv(LSTM.path_std_tr, header=-1, dtype={'0':str,'1':np.float64})
    std_rnn = pd.Series(data=np.array(std_rnn[1], dtype=np.float64), index=np.array(std_rnn[0], dtype=str))

    model = LSTM.RNN_LSTM(version=vs, layer_sizes=layers)
    
    vars_selected = Header[1:41]+Header[101:121]+Header[241:246]+Header[246:247]
    #Load testset
    test=pd.read_csv(path_input_data, sep=',', header=-1, names=Header[1:])
    #Normalize data
    X_test=test[vars_selected]
    X_test = (X_test - mean_rnn)/std_rnn
    X_test = (np.array(X_test)).reshape(X_test.shape[0], 1, X_test.shape[1])

    
    if pred_classes:
        Test_pred = model.predict_classes(np.array(X_test), batch_size=batch_s)+1
    else:
        #print ("\n\n[INFO]  Predict Proba : ")
        Test_pred = model.predict_proba(np.array(X_test), batch_size=batch_s)
        Test_pred=Test_pred.reshape((Test_pred.shape[0],Test_pred.shape[2]))

    return Test_pred

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


def get_Prediction(model_name, path_input_data):

    Header, \
    X_Acc,Y_Acc,Z_Acc, \
    X_Gyro,Y_Gyro,Z_Gyro, \
    X_Cf,Y_Cf,Z_Cf, \
    Speed, \
    Times = get_data_header(20, 5, 1)
        
    if model_name=="RNN_LSTM":
        return Pred_RNN(Header, path_input_data, vs="03_10k", layers = [66, 16, 8, 4])


    elif model_name=="CNN_2D":
        return Pred_CNN(Header, path_input_data, vs="01")


    elif model_name=="FFNN":  
        return Pred_FFNN(Header, path_input_data, vs="00")


    elif model_name=="RNN+_LSTM":
        return Pred_RNN(Header, path_input_data, vs="05_5k", layers = [66, 32, 16, 8, 4])
    
    
    elif model_name=="SNN":

        #prob_cnn_v00 = Pred_CNN(Header, path_input_data, vs="00")
        #prob_cnn_v01 = Pred_CNN(Header, path_input_data, vs="01")

        prob_ffnn_v00 = Pred_FFNN(Header, path_input_data, vs="00")

        #prob_rnn_v03 = Pred_RNN(Header, path_input_data, vs="03", layers = [66, 16, 8, 4])
        prob_rnn_v03_ = Pred_RNN(Header, path_input_data, vs="03_10k", layers = [66, 16, 8, 4])
        #prob_rnn_v05 = Pred_RNN(Header, path_input_data, vs="05", layers = [66, 32, 16, 8, 4])
        prob_rnn_v05_ = Pred_RNN(Header, path_input_data, vs="05_5k", layers = [66, 32, 16, 8, 4])

        #df = np.concatenate((prob_cnn_v00, prob_cnn_v01, prob_ffnn_v00, prob_rnn_v03, prob_rnn_v03_, prob_rnn_v05, prob_rnn_v05_),axis=1)
        df = np.concatenate((prob_ffnn_v00, prob_rnn_v03_, prob_rnn_v05_),axis=1)

        model = Stack.SNN(layer_sizes = [45])

        #print ("\n\n[INFO]  Predict Proba : ")
        Test_pred_prob = model.predict_proba(np.array(df), batch_size=batch_s)
        
        return Test_pred_prob
    
    elif model_name=="FCN":
        return Pred_FCN(Header, path_input_data, vs="v01")
    
    elif model_name in Models.keys():
        #"MLTSM"
        return Pred_Model(model_name, path_input_data)
    
    elif model_name=="VNN":
        # from_path="/home/gpuadmin/RAMAH/workspace/"
        # base_path = "/home/gpuadmin/RAMAH/PFE_Youssef/weights-improvement/"
        
        

        def prep_data(train, vars_selected_x36x36x36, vars_selected_x36x5x1, path_input_data, Header):

            #Normalize data
            const = 1e-8
            mean_std_path = from_path+"mean_std_CFx36x36_5x36.pkl"
            try:
                clf = joblib.load(mean_std_path)
                mean_tr_x36x36x36 = clf["mean_tr"][vars_selected_x36x36x36]
                std_tr_x36x36x36 = clf["std_tr"][vars_selected_x36x36x36]
                mean_tr_x36x5x1 = clf["mean_tr"][vars_selected_x36x5x1]
                std_tr_x36x5x1 = clf["std_tr"][vars_selected_x36x5x1]
            except Exception, e:
                print (e)
                X_train = train[Header[1:]]
                mean_tr = X_train.mean()
                std_tr = X_train.std()
                joblib.dump({"mean_tr":mean_tr, "std_tr":std_tr}, mean_std_path)
                mean_tr_x36x36x36 = mean_tr[vars_selected_x36x36x36]
                std_tr_x36x36x36 = std_tr[vars_selected_x36x36x36]
                mean_tr_x36x5x1 = mean_tr[vars_selected_x36x5x1]
                std_tr_x36x5x1 = std_tr[vars_selected_x36x5x1]

            #Load testset
            test=pd.read_csv(path_input_data, sep=',', header=-1, names=Header[1:])
            X_ts_36=test[vars_selected_x36x36x36]
            X_ts_5=test[vars_selected_x36x5x1]
            
            #Normalize data
            X_ts_36 = (X_ts_36 - mean_tr_x36x36x36)/(std_tr_x36x36x36+1e-8)
            X_ts_5 = (X_ts_5 - mean_tr_x36x5x1)/(std_tr_x36x5x1+1e-8)
            #Reshape data
            X_ts_36 = (np.array(X_ts_36)).reshape(X_ts_36.shape[0], 11, 36)
            X_ts_5 = (np.array(X_ts_5)).reshape(X_ts_5.shape[0], 1, X_ts_5.shape[1])
            
            return X_ts_36, X_ts_5

        
        Header, X_Acc,Y_Acc,Z_Acc, X_Gyro,Y_Gyro,Z_Gyro, X_Cf,Y_Cf,Z_Cf, Speed, Times = get_data_header(36, 36, 36)
        
        #Select some sensors like [X_acc Y_acc Z_gyr X_cf Y_cf Z_cf speeds times] Or ...
        vars_selected_x36x36x36 = X_Acc+Y_Acc+\
                        Z_Acc+X_Gyro+Y_Gyro+\
                        Z_Gyro+\
                        X_Cf+Y_Cf+Z_Cf+\
                        Speed+Times

        vars_selected_x36x5x1 = X_Acc+Y_Acc+\
                        Z_Acc+X_Gyro+Y_Gyro+\
                        Z_Gyro+\
                        X_Cf+Y_Cf+Z_Cf+\
                        Speed[1:6]+[Times[-1]]

        train=pd.read_csv(from_path+"outTrainx36x36_5x36.csv", sep=',', header=-1, names=Header)
        X_ts_36, X_ts_5 = prep_data(train, vars_selected_x36x36x36, vars_selected_x36x5x1, path_input_data, Header)


        model_cf_calib_path_FFNN4_v01_ep500 = base_path+"My-MLSTM-FFNN4_v01_ep500/FCN_My-MLSTM-FFNN4_v01_model_ep500.hdf5"
        model_cf_calib_FFNN4_v01_ep500 = load_model(model_cf_calib_path_FFNN4_v01_ep500)
        ts_36x5x1_FFNN4_v01_ep500 = model_cf_calib_FFNN4_v01_ep500.predict(X_ts_5, batch_size=batch_s)

        model_cf_calib_path_FFNN4_v01 = base_path+"My-MLSTM-FFNN4_v01/model_weights-00-0.13_My-MLSTM-FFNN4_v01.hdf5"
        model_cf_calib_FFNN4_v01 = load_model(model_cf_calib_path_FFNN4_v01)
        ts_36x5x1_FFNN4_v01 = model_cf_calib_FFNN4_v01.predict(X_ts_5, batch_size=batch_s)

        model2_cf_calib_path = base_path+"My-MLSTM-Correct-Shape3_v04/model_weights-14036-0.28_My-MLSTM-Correct-Shape3_v04.hdf5"
        model2_cf_calib = load_model(model2_cf_calib_path)
        ts_36x36x36_Shape3_v04 = model2_cf_calib.predict(X_ts_36, batch_size=batch_s)

        df_ts = np.concatenate((ts_36x5x1_FFNN4_v01_ep500, ts_36x5x1_FFNN4_v01, ts_36x36x36_Shape3_v04),axis=1)
        model = Voting.VNN(layer_sizes = [45], dropout_prob = 0.0)
        

        #print ("\n\n[INFO]  Predict Proba : ")
        Test_pred_prob = model.predict_proba(np.array(df_ts), batch_size=batch_s)
        
        return Test_pred_prob
    
    else:
        raise


        
def Pred_c_FCN(Header, path_input_data, vs="v01", pred_classes=False):
    mean_rcnn = pd.read_csv(RCNN.path_mean_tr, header=-1, dtype={'0':str,'1':np.float64})
    mean_rcnn = pd.Series(data=np.array(mean_rcnn[1], dtype=np.float64), index=np.array(mean_rcnn[0], dtype=str))
    std_rcnn = pd.read_csv(RCNN.path_std_tr, header=-1, dtype={'0':str,'1':np.float64})
    std_rcnn = pd.Series(data=np.array(std_rcnn[1], dtype=np.float64), index=np.array(std_rcnn[0], dtype=str))

    model = RCNN.LSTM_CNN(version=vs)
    
    vars_selected = Header[1:41]+Header[101:121]+Header[241:246]+Header[246:247]
    #Load testset
    test=pd.read_csv(path_input_data, sep=',', header=-1, names=Header)
    #Normalize data
    X_test=test[vars_selected]
    Y_test=test[Header[0]]
    X_test = (X_test - mean_rcnn)/std_rcnn
    X_test = (np.array(X_test)).reshape(X_test.shape[0], 1, X_test.shape[1])

    
    if pred_classes:
        Test_pred = model.predict(np.array(X_test), batch_size=batch_s)
        Test_pred = Test_pred.argmax(axis=-1)+1
    else:
        #print ("\n\n[INFO]  Predict Proba : ")
        Test_pred = model.predict(np.array(X_test), batch_size=batch_s)

    return Test_pred, Y_test

def Pred_c_CNN(Header, path_input_data, vs="01", pred_classes=False):
    mean_ = pd.read_csv(CONV2D.path_mean_tr, header=-1, dtype={'0':str,'1':np.float64})
    mean_ = pd.Series(data=np.array(mean_[1], dtype=np.float64), index=np.array(mean_[0], dtype=str))
    std_ = pd.read_csv(CONV2D.path_std_tr, header=-1, dtype={'0':str,'1':np.float64})
    std_ = pd.Series(data=np.array(std_[1], dtype=np.float64), index=np.array(std_[0], dtype=str))

    model = CONV2D.CNN_2D(version=vs)
    
    vars_selected = Header[1:121]
    #Load testset
    test=pd.read_csv(path_input_data, sep=',', header=-1, names=Header)
    #Normalize data
    X_test=test[vars_selected]
    Y_test=test[Header[0]]
    X_test = (X_test - mean_)/std_
    X_test = CONV2D.prepare_data_cnn(X_test)

    if pred_classes:
        Test_pred = model.predict_classes(np.array(X_test), batch_size=batch_s)+1
    else:
        #print ("\n\n[INFO]  Predict Proba : ")
        Test_pred = model.predict_proba(np.array(X_test), batch_size=batch_s)

    return Test_pred, Y_test

def Pred_c_FFNN(Header, path_input_data, vs="00", pred_classes=False):
    mean_ = pd.read_csv(NN.path_mean_tr, header=-1, dtype={'0':str,'1':np.float64})
    mean_ = pd.Series(data=np.array(mean_[1], dtype=np.float64), index=np.array(mean_[0], dtype=str))
    std_ = pd.read_csv(NN.path_std_tr, header=-1, dtype={'0':str,'1':np.float64})
    std_ = pd.Series(data=np.array(std_[1], dtype=np.float64), index=np.array(std_[0], dtype=str))

    model = NN.FFNN()
    
    vars_selected = Header[1:41]+Header[101:121]+Header[241:246]+Header[246:247]
    #Load testset
    test=pd.read_csv(path_input_data, sep=',', header=-1, names=Header)
    #Normalize data
    X_test=test[vars_selected]
    Y_test=test[Header[0]]
    X_test = (X_test - mean_)/std_
    X_test = np.array(X_test)

    if pred_classes:
        Test_pred = model.predict_classes(np.array(X_test), batch_size=batch_s)+1
    else:
        #print ("\n\n[INFO]  Predict Proba : ")
        Test_pred = model.predict_proba(np.array(X_test), batch_size=batch_s)

    return Test_pred, Y_test


def Pred_c_RNN(Header, path_input_data, vs="03_10k", layers = [66, 16, 8, 4], pred_classes=False):
    mean_rnn = pd.read_csv(LSTM.path_mean_tr, header=-1, dtype={'0':str,'1':np.float64})
    mean_rnn = pd.Series(data=np.array(mean_rnn[1], dtype=np.float64), index=np.array(mean_rnn[0], dtype=str))
    std_rnn = pd.read_csv(LSTM.path_std_tr, header=-1, dtype={'0':str,'1':np.float64})
    std_rnn = pd.Series(data=np.array(std_rnn[1], dtype=np.float64), index=np.array(std_rnn[0], dtype=str))

    model = LSTM.RNN_LSTM(version=vs, layer_sizes=layers)
    
    vars_selected = Header[1:41]+Header[101:121]+Header[241:246]+Header[246:247]
    #Load testset
    test=pd.read_csv(path_input_data, sep=',', header=-1, names=Header)
    #Normalize data
    X_test=test[vars_selected]
    Y_test=test[Header[0]]
    X_test = (X_test - mean_rnn)/std_rnn
    X_test = (np.array(X_test)).reshape(X_test.shape[0], 1, X_test.shape[1])

    
    if pred_classes:
        Test_pred = model.predict_classes(np.array(X_test), batch_size=batch_s)+1
    else:
        #print ("\n\n[INFO]  Predict Proba : ")
        Test_pred = model.predict_proba(np.array(X_test), batch_size=batch_s)
        Test_pred=Test_pred.reshape((Test_pred.shape[0],Test_pred.shape[2]))

    return Test_pred, Y_test

        
def get_predict_classes(model_name, path_input_data):
    Header=['type','acc_x_1','acc_x_2','acc_x_3','acc_x_4','acc_x_5','acc_x_6','acc_x_7','acc_x_8','acc_x_9','acc_x_10','acc_x_11','acc_x_12','acc_x_13','acc_x_14','acc_x_15','acc_x_16','acc_x_17','acc_x_18','acc_x_19','acc_x_20','acc_y_1','acc_y_2','acc_y_3','acc_y_4','acc_y_5','acc_y_6','acc_y_7','acc_y_8','acc_y_9','acc_y_10','acc_y_11','acc_y_12','acc_y_13','acc_y_14','acc_y_15','acc_y_16','acc_y_17','acc_y_18','acc_y_19','acc_y_20','acc_z_1','acc_z_2','acc_z_3','acc_z_4','acc_z_5','acc_z_6','acc_z_7','acc_z_8','acc_z_9','acc_z_10','acc_z_11','acc_z_12','acc_z_13','acc_z_14','acc_z_15','acc_z_16','acc_z_17','acc_z_18','acc_z_19','acc_z_20','gyr_x_1','gyr_x_2','gyr_x_3','gyr_x_4','gyr_x_5','gyr_x_6','gyr_x_7','gyr_x_8','gyr_x_9','gyr_x_10','gyr_x_11','gyr_x_12','gyr_x_13','gyr_x_14','gyr_x_15','gyr_x_16','gyr_x_17','gyr_x_18','gyr_x_19','gyr_x_20','gyr_y_1','gyr_y_2','gyr_y_3','gyr_y_4','gyr_y_5','gyr_y_6','gyr_y_7','gyr_y_8','gyr_y_9','gyr_y_10','gyr_y_11','gyr_y_12','gyr_y_13','gyr_y_14','gyr_y_15','gyr_y_16','gyr_y_17','gyr_y_18','gyr_y_19','gyr_y_20','gyr_z_1','gyr_z_2','gyr_z_3','gyr_z_4','gyr_z_5','gyr_z_6','gyr_z_7','gyr_z_8','gyr_z_9','gyr_z_10','gyr_z_11','gyr_z_12','gyr_z_13','gyr_z_14','gyr_z_15','gyr_z_16','gyr_z_17','gyr_z_18','gyr_z_19','gyr_z_20','spd_1','spd_2','spd_3','spd_4','spd_5','time']
        
    if model_name=="RNN_LSTM":
        return Pred_c_RNN(Header, path_input_data, vs="03_10k", layers = [66, 16, 8, 4], pred_classes=True)


    elif model_name=="CNN_2D":
        return Pred_c_CNN(Header, path_input_data, vs="01", pred_classes=True)


    elif model_name=="FFNN":  
        return Pred_c_FFNN(Header, path_input_data, vs="00", pred_classes=True)


    elif model_name=="RNN+_LSTM":
        return Pred_c_RNN(Header, path_input_data, vs="05_5k", layers = [66, 32, 16, 8, 4], pred_classes=True)
    
    
    elif model_name=="SNN":

        prob_ffnn_v00 = Pred_FFNN(Header, path_input_data, vs="00")
        prob_rnn_v03_ = Pred_RNN(Header, path_input_data, vs="03_10k", layers = [66, 16, 8, 4])
        prob_rnn_v05_ = Pred_RNN(Header, path_input_data, vs="05_5k", layers = [66, 32, 16, 8, 4])

        df = np.concatenate((prob_ffnn_v00, prob_rnn_v03_, prob_rnn_v05_),axis=1)

        model = Stack.SNN(layer_sizes = [45])

        test=pd.read_csv(path_input_data, sep=',', header=-1, names=Header)
        Y_test=test[Header[0]]

        #print ("\n\n[INFO]  Predict classes : ")
        Test_pred_classes = model.predict_classes(np.array(df), batch_size=batch_s)+1
        
        return Test_pred_classes, Y_test
    elif model_name=="FCN":
        return Pred_c_FCN(Header, path_input_data, vs="v01", pred_classes=True)
    
    elif model_name in Models.keys():
        #"MLTSM"
        return Pred_Model(model_name, path_input_data, pred_classes=True)
    
    else:
        raise


