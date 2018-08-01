#Default Param

from datetime import timedelta

NUM_EL_GYR_ACC=20
NUM_EL_SPEED=5
NUM_EL_TIME=1

time_delta_events_msec=timedelta(milliseconds=500)

project_base_path="app/plugins/Smart_events_detector/"

events_gen_dir=project_base_path+"workspace/events_gen"
#events_dir="app/plugins/Smart_events_detector/dataset/Events/InputFolder"

events_dates_to_test=["2014-09-18"] #['2014-11-26','2014-10-25','2014-10-23']

all_test_days=['2014-09-18',   '2014-11-26','2014-10-25','2014-10-23']

all_train_days=['2014-09-17','2014-10-15','2014-10-16','2014-10-17','2014-10-18',\
            '2014-10-24','2014-11-29','2014-11-30','2015-01-25','2014-10-14']

all_cv_days=['2014-09-18','2014-12-16','2014-11-25']



model_name="SNN"
Threshold_cluster=0.85 #0.5
epsilon_cluster=1
min_samples_cluster=1 #5
windows_size=[2.5, 5, 7, 11, 13, 16]
#[2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 12, 14, 16, 20, 25, 30]

all_models=["RNN_LSTM", "CNN_2D", "FFNN", "RNN+_LSTM", "SNN", "FCN"]
