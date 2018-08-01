import pandas as pd
import numpy as np
import os
from multiprocessing import Queue
import multiprocessing as mp

from .normalizer.Normalizer import  Normalize
from data_prepaire import prepaire, generate_events
from data_prepaire.data_settings import DataSplit
#from datetime import timedelta
from .test_predict.predict_events import predict_all_events
import config_param as conf
from .helpers.get_map_data import GetLocalData, GetData
from .helpers import data_filter as dt_filter


class Main_plugin_interface:#class Main_plugin:
   

    # ====================================================================== #
    pre_data = None

    def __init__(self, path="#"):
        try:
            if path=="#":
                self.pre_data = self.get_data()
            else:
                self.pre_data = self.get_data(path)
        except Exception, e:
            print ("Failed to get dataset... {}\n".format(e))
    
    def get_data(self, path):
        gd = GetLocalData(path)
        DATA_LINK = gd.map_data
        return DATA_LINK

    def get_data(self):
        gd = GetData()
        DATA_LINK = gd.map_data
        return DATA_LINK

    def normalize_data(self, days):
        normalized_data={}
        DATA_LINK = self.pre_data

        print ("[[ Normalize ]] :: Data length = {}".format(len(DATA_LINK)))
        try:
            self.red.publish('info', 'Getting  PRE-DATA completed successfully')
        except Exception, e:
            print (e)
        
        #print DATA_LINK.columns.values.tolist()
        norm = Normalize()
        Norm_DATA_LINK = DATA_LINK.file_path[DATA_LINK.file_class=="Sensors"]
        for i in Norm_DATA_LINK.index:
            day=DATA_LINK.iloc[i].file_day
            if day in days:
                path = DATA_LINK.iloc[i].file_path
                #Create a smooth & norm data file in the same path as file_path
                smooth_path = norm.Normalizer(path, "--smooth-cf", do_Norm_proc=True)
                DATA_LINK.iloc[i].file_path=smooth_path
                normalized_data[day]=os.path.abspath(smooth_path)

        return normalized_data
        
        #raise
        
    
    def prepaire_cv_parallel(self, prefix_name_file, cv_days=[], normalized=False):
        """ 
            Preparer les donnees des jours en qestions, 
            pour les utiliser apres dans la prentissage 
            ou la validation des modeles. On utilisant 
            les informations correspondantes de chaque 
            Evenement pendant la preparation. 
        """
        prepaired_data={}
        
        print "[[ DataSplit ]] :: Validation days = ",cv_days
        days_to_normalize = cv_days
        
        if len(days_to_normalize)!=0:
            if not normalized:
                self.normalize_data(days_to_normalize)
            DATA_LINK = self.pre_data

            print "[[ Prepaire ]] :: Data length = ", len(DATA_LINK)

            data_sensors = DATA_LINK[DATA_LINK.file_class=="Sensors"]
            data_events = DATA_LINK[DATA_LINK.file_class=="Events"]

            
            validation_files = []
            print "len(data_sensors) = ",len(data_sensors)
            print "len(data_events) = ",len(data_events)

            print "####> Abspath : ",os.path.abspath('.')
            
            def some_work(DATA_LINK, data_sensors, data_events, i, cv_days, q_tr, q_val ):
                ds = DATA_LINK.iloc[i]
                day = ds.file_day
                time = ds.file_time
                path_e = data_events.file_path[data_events.file_day==day][data_events.file_time==time]
                path_s = ds.file_path


                output_name =  "/tmp/out-"+day+".csv"


                if not os.path.exists(conf.project_base_path+"workspace"):
                    os.mkdir(conf.project_base_path+"workspace")

                NUM_EL_GYR_ACC=conf.NUM_EL_GYR_ACC
                NUM_EL_SPEED=conf.NUM_EL_SPEED
                NUM_EL_TIME=conf.NUM_EL_TIME
                time_delta_events_msec=conf.time_delta_events_msec
                #raise
                #os.system('rm ./workspace/out-*.csv')
                
                if day in cv_days:
                    prepaire.Prepaire(path_s, path_e.tolist()[0], output_name, time_delta_events_msec, NUM_EL_GYR_ACC, NUM_EL_SPEED, NUM_EL_TIME)
                    q_val.append(output_name)

            
            with mp.Manager() as manager:
                q_tr = manager.list()
                q_val = manager.list()
                p ={}
                for i in data_sensors.index:
                    p[i] = mp.Process(target=some_work, args=(DATA_LINK, data_sensors, data_events, i, [], cv_days, q_tr, q_val,))
                    p[i].start()
                
                for i in p.keys():
                    p[i].join()
                validation_files = list(q_val)
                    
                print "\n validation_files : ",validation_files

                self.merge_file(conf.project_base_path+"workspace/outTs"+prefix_name_file+".csv", validation_files)
                prepaired_data={
                    "validationset":os.path.abspath(conf.project_base_path+"workspace/outTs"+prefix_name_file+".csv")
                }
                try:
                    for name in validation_files:
                        os.remove(os.path.join(os.path.abspath('.'), name))
                except:
                    print("Error :: Remove tmp files.")
                    print ("Info :: They will be removed anyway ;)")

        return prepaired_data
    
    def prepaire_all_parallel(self, prefix_name_file, train_days=[], cv_days=[], normalized=False):
        """ 
            Preparer les donnees des jours en qestions, 
            pour les utiliser apres dans la prentissage 
            ou la validation des modeles. On utilisant 
            les informations correspondantes de chaque 
            Evenement pendant la preparation. 
        """
        prepaired_data={}
        # dsp = DataSplit()
        # train_days = dsp.data_train_days
        # cv_days = dsp.data_cv_days
        print "[[ DataSplit ]] :: Train days = ",train_days
        print "[[ DataSplit ]] :: Validation days = ",cv_days
        days_to_normalize = train_days + cv_days
        
        if len(days_to_normalize)!=0:
            if not normalized:
                self.normalize_data(days_to_normalize)
            DATA_LINK = self.pre_data

            print "[[ Prepaire ]] :: Data length = ", len(DATA_LINK)

            data_sensors = DATA_LINK[DATA_LINK.file_class=="Sensors"]
            data_events = DATA_LINK[DATA_LINK.file_class=="Events"]

            
            trainning_files = []
            validation_files = []
            print "len(data_sensors) = ",len(data_sensors)
            print "len(data_events) = ",len(data_events)

            print "####> Abspath : ",os.path.abspath('.')
            
            def some_work(DATA_LINK, data_sensors, data_events, i, train_days, cv_days, q_tr, q_val ):
                ds = DATA_LINK.iloc[i]
                day = ds.file_day
                time = ds.file_time
                path_e = data_events.file_path[data_events.file_day==day][data_events.file_time==time]
                path_s = ds.file_path


                output_name =  "/tmp/out-"+day+".csv"


                if not os.path.exists(conf.project_base_path+"workspace"):
                    os.mkdir(conf.project_base_path+"workspace")

                NUM_EL_GYR_ACC=conf.NUM_EL_GYR_ACC
                NUM_EL_SPEED=conf.NUM_EL_SPEED
                NUM_EL_TIME=conf.NUM_EL_TIME
                time_delta_events_msec=conf.time_delta_events_msec
                #raise
                #os.system('rm ./workspace/out-*.csv')

                if day in train_days:
                    prepaire.Prepaire(path_s, path_e.tolist()[0], output_name, time_delta_events_msec, NUM_EL_GYR_ACC, NUM_EL_SPEED, NUM_EL_TIME)
                    q_tr.append(output_name)
                elif day in cv_days:
                    prepaire.Prepaire(path_s, path_e.tolist()[0], output_name, time_delta_events_msec, NUM_EL_GYR_ACC, NUM_EL_SPEED, NUM_EL_TIME)
                    q_val.append(output_name)

            
            with mp.Manager() as manager:
                q_tr = manager.list()
                q_val = manager.list()
                p ={}
                for i in data_sensors.index:
                    p[i] = mp.Process(target=some_work, args=(DATA_LINK, data_sensors, data_events, i, train_days, cv_days, q_tr, q_val,))
                    p[i].start()
                
                for i in p.keys():
                    p[i].join()
                trainning_files = list(q_tr)
                validation_files = list(q_val)
                    
                print "\n trainning_files : ",trainning_files
                print "\n validation_files : ",validation_files

                self.merge_file(conf.project_base_path+"workspace/outTrain"+prefix_name_file+".csv", trainning_files)
                self.merge_file(conf.project_base_path+"workspace/outCv"+prefix_name_file+".csv", validation_files)
                prepaired_data={
                    "trainset":os.path.abspath(conf.project_base_path+"workspace/outTrain"+prefix_name_file+".csv"),
                    "validationset":os.path.abspath(conf.project_base_path+"workspace/outCv"+prefix_name_file+".csv")
                }
                try:
                    for name in trainning_files+validation_files:
                        os.remove(os.path.join(os.path.abspath('.'), name))
                except:
                    print("Error :: Remove tmp files.")
                    print ("Info :: They will be removed anyway ;)")

        return prepaired_data

    def gen(self, windows_size, test_days=[], normalized=False, NUM_EL_GYR_ACC=conf.NUM_EL_GYR_ACC, NUM_EL_SPEED=conf.NUM_EL_SPEED, NUM_EL_TIME=conf.NUM_EL_TIME, time_delta_events_msec=conf.time_delta_events_msec):
        """ 
            Generer des donnees depuis les donnees des Senseurs uniquement, 
            pour les utiliser apres dans le test des modeles pre-traines.  
        """


        generated_events={}
        # dsp = DataSplit()
        # test_days = dsp.data_test_days
        days_to_normalize = test_days
        if len(days_to_normalize)!=0:
            if not normalized:
                self.normalize_data(days_to_normalize)
            DATA_LINK = self.pre_data
            data_sensors = DATA_LINK[DATA_LINK.file_class=="Sensors"]
            

            if not os.path.exists(conf.project_base_path+"workspace/events_gen"):
                os.mkdir(conf.project_base_path+"workspace/events_gen")


            def some_work(DATA_LINK, data_sensors, i, test_days, windows_size, NUM_EL_GYR_ACC, NUM_EL_SPEED, NUM_EL_TIME, time_delta_events_msec, dic_output):
                
                ds = DATA_LINK.iloc[i]
                day = ds.file_day
                inputfile = ds.file_path

                if day in test_days:
                    #print "\nThis file should be normalized : ",inputfile,"\n"
                    h_parm = str(hash(str(windows_size)))
                    h_parm = str(NUM_EL_GYR_ACC)+"X"+str(NUM_EL_SPEED)+"X"+str(NUM_EL_TIME)+"_"+h_parm
                    
                    if not os.path.exists(conf.project_base_path+"workspace/events_gen/"+day):
                        os.mkdir(conf.project_base_path+"workspace/events_gen/"+day) 
                        os.mkdir(conf.project_base_path+"workspace/events_gen/"+day+"/"+h_parm)
                    elif not os.path.exists(conf.project_base_path+"workspace/events_gen/"+day+"/"+h_parm):
                            os.mkdir(conf.project_base_path+"workspace/events_gen/"+day+"/"+h_parm)
                    
                    
                    outfiledata =  conf.project_base_path+"workspace/events_gen/"+day+"/"+h_parm+"/data.csv"
                    outfiletime =  conf.project_base_path+"workspace/events_gen/"+day+"/"+h_parm+"/time.csv"
                    temp={
                        "filedata":os.path.abspath(outfiledata),
                        "filetime":os.path.abspath(outfiletime)
                    }
                    generate_events.gen_events(inputfile, outfiledata, outfiletime, time_delta_events_msec, NUM_EL_GYR_ACC, NUM_EL_SPEED, NUM_EL_TIME, windows_size)
                    dic_output[day]=temp
            
            with mp.Manager() as manager:
                dic_output = manager.dict()
                p ={}
                for i in data_sensors.index:
                    p[i] = mp.Process(target=some_work, args=(DATA_LINK, data_sensors, i, test_days, windows_size, NUM_EL_GYR_ACC, NUM_EL_SPEED, NUM_EL_TIME, time_delta_events_msec, dic_output,))
                    p[i].start()
                
                for i in p.keys():
                    p[i].join()
                generated_events = dict(dic_output)
                print "\n windows size = ",windows_size,"\n",str("="*60)
                print "\n generated events : ",generated_events
                
        return generated_events   

    def predict(self, q=Queue(), events_dates=conf.events_dates_to_test, model_name=conf.model_name):
        events_gen_dir=conf.events_gen_dir
        #####events_dir=dt_filter.get_events_dir(self.pre_data)
        hand_made_events_paths=dt_filter.get_path(self.pre_data, events_dates, ["00:00:00"]*len(events_dates), "Events")

        #events_dates=conf.events_dates_to_test
        #model_name=conf.model_name
        Threshold=conf.Threshold_cluster
        epsilon=conf.epsilon_cluster
        min_samples=conf.min_samples_cluster
        windows_size=conf.windows_size
        NUM_EL_GYR_ACC=conf.NUM_EL_GYR_ACC
        NUM_EL_SPEED=conf.NUM_EL_SPEED
        NUM_EL_TIME=conf.NUM_EL_TIME

        all_diffs = predict_all_events(q, events_gen_dir, hand_made_events_paths, events_dates, model_name, Threshold, epsilon, min_samples, windows_size, NUM_EL_GYR_ACC, NUM_EL_SPEED, NUM_EL_TIME)
        return all_diffs

    def advancePredict(self, q=Queue(), events_dates=conf.events_dates_to_test, model_name=conf.model_name, Threshold=conf.Threshold_cluster, epsilon=conf.epsilon_cluster, min_samples=conf.min_samples_cluster, windows_size=conf.windows_size, NUM_EL_GYR_ACC=conf.NUM_EL_GYR_ACC, NUM_EL_SPEED=conf.NUM_EL_SPEED, NUM_EL_TIME=conf.NUM_EL_TIME):
        events_gen_dir=conf.events_gen_dir
        #####events_dir=dt_filter.get_events_dir(self.pre_data)
        hand_made_events_paths=dt_filter.get_path(self.pre_data, events_dates, ["00:00:00"]*len(events_dates), "Events")

        #events_dates=conf.events_dates_to_test
        #model_name=conf.model_name

        all_diffs = predict_all_events(q, events_gen_dir, hand_made_events_paths, events_dates, model_name, Threshold, epsilon, min_samples, windows_size, NUM_EL_GYR_ACC, NUM_EL_SPEED, NUM_EL_TIME)
        return all_diffs


    #get map of files paths with keys day+time
    def fast_acces(self):
        path_joined = []
        grouped_days_time = []

        DATA_LINK = self.pre_data
        data_days = DATA_LINK.file_day.unique()
        
        for day in data_days:
            data_times = DATA_LINK[DATA_LINK.file_day==day].file_time.unique()
            time_list_temp = []
            for time in data_times[data_times!='00:00:00']:
                selected_data = DATA_LINK[DATA_LINK.file_day==day][DATA_LINK.file_time.apply(lambda x : x in [time, '00:00:00'])]
                if len(selected_data)==5:
                    time_list_temp.append(time)

                    Event_path = selected_data[selected_data.file_class=='Events'][selected_data.file_time=='00:00:00'].file_path.values[0]
                    Sensor_path = selected_data[selected_data.file_class=='Sensors'][selected_data.file_time=='00:00:00'].file_path.values[0]
                    video_path = selected_data[selected_data.file_class=='Video'][selected_data.file_extension=='mp4'].file_path.values[0]
                    subtitle_path = selected_data[selected_data.file_class=='Video'][selected_data.file_extension=='srt'].file_path.values[0]
                    temp = [day+" "+time,[Event_path, Sensor_path, video_path, subtitle_path]]
                    path_joined.append(temp)
            
            grouped_days_time.append([day ,time_list_temp])

        return grouped_days_time, path_joined

    def merge_file(self, outPutFile, interesting_files):
        if len(interesting_files)>0:
            #header_saved = False
            with open(outPutFile,'wb') as fout:
                for filename in interesting_files:
                    with open(filename) as fin:
                        #header = next(fin)
                        #if not header_saved:
                        #    fout.write(header)
                        #    header_saved = True
                        for line in fin:
                            fout.write(line)

    def get_model_names(self):
        return {"models":conf.all_models}






        
