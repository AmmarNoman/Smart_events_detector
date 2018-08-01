import subprocess
import tempfile
import csv
import json
from os import path
from optparse import OptionParser
from compaire_events import compaire_events
from post_process_events import post_process
from post_predict_events import predict
import pandas as pd
from datetime import datetime
import os




#To change ...
def predict_events(events_gen_dir, epsilon, min_samples, event_date, model_name, Threshold, windows_size, NUM_EL_GYR_ACC, NUM_EL_SPEED, NUM_EL_TIME, save_json_file):
    Y_file = tempfile.mktemp()
    Time_file = tempfile.mktemp()
    
    h_parm = str(hash(str(windows_size)))
    h_parm = str(NUM_EL_GYR_ACC)+"X"+str(NUM_EL_SPEED)+"X"+str(NUM_EL_TIME)+"_"+h_parm

    path_input_data = path.join(events_gen_dir, event_date, h_parm, 'data.csv')
    path_input_time = path.join(events_gen_dir, event_date, h_parm, 'time.csv')

    """
    predict events from data.csv & time.csv using your model_name
    """
    predict(path_input_data, path_input_time, Y_file, Time_file, model_name, Threshold)

    if save_json_file:
        p_path = events_gen_dir+"/"+event_date+"/"+h_parm+"/"+str(model_name)
        if not os.path.exists(p_path):
            os.mkdir(p_path)
        tfile_events_json = p_path+"/ep."+str(epsilon)+"_ms."+str(min_samples)+"_th."+str(Threshold)+"_all.json"
    else:
        tfile_events_json = tempfile.mktemp()

    post_process(Y_file, Time_file, tfile_events_json, epsilon, min_samples)

    return tfile_events_json


def merge_diffs(diffs):
    result = {
        'false-negative': 0,
        'correct': 0,
        'wrong': 0,
        'false-positive': 0,
        'correct-percent' : 0.0,
        'correct-percent-no-fn' : 0.0
    }

    for diff in diffs:
        result['false-negative'] += diff['false-negative']['sum']
        result['correct'] += diff['correct']['sum']
        result['wrong'] += diff['wrong']['sum']
        result['false-positive'] += diff['false-positive']['sum']

    result['correct-percent'] = float(result['correct'])/(result['correct'] +
                result['wrong'] + result['false-positive'] +
                result['false-negative'])
    if result['correct']==0:
        result['correct-percent-no-fn']=0.0
    else:
        result['correct-percent-no-fn'] = float(result['correct'])/(result['correct'] +
                result['wrong'] + result['false-positive'])

    return result






'''
    "model_name": "Models that will be used to predict events"
    "events_dates": "Dates of events to be predicted"
    "events_gen_dir": "Directory with generated events for all days"
    "events_dir": "Directory with hand made events"
    "epsilon": "DBScan epsilon"
    "min_samples": "DBScan min samples"
'''
def predict_all_events(q, events_gen_dir, hand_made_events_paths, events_dates, model_name, Threshold, epsilon, min_samples, windows_size, NUM_EL_GYR_ACC, NUM_EL_SPEED, NUM_EL_TIME, save_json_file=True):
    
    #print ('')
    """
    correct_type | wrong_type | false_positive | false_negative = {
        type1 : value,
        type2 : value,
        ...,
        typek : value
    }

    diff = {
        'correct' : {
            'sum' : sum(correct_type.values()),
            'values' : correct_type
        },
        'wrong' : {
            'sum' : sum(wrong_type.values()),
            'values' : wrong_type
        },
        'false-positive' : {
            'sum' : sum(false_positive.values()),
            'values' : false_positive
        },
        'false-negative' : {
            'sum' : sum(false_negative.values()),
            'values' : false_negative
        },
        'correct-percent' : correct_percent,
        'correct-percent-no-fn' : correct_percent_no_fn,
        'path_events_file' : path
    }
    
    stat[i]=[day, 
            diff['correct']['sum'],
            diff['correct']['values']
            diff['wrong']['sum'] + diff['false-positive']['sum'] 
            diff['wrong']['values'],
            diff['false-positive']['values']]

    diffs = {
        day : diff,
    }

    all_diffs = {
        "days" : diffs,
        "summary" : {
            'false-negative': total_FN,
            'correct': total_CORRECT,
            'wrong': total_WRONG,
            'false-positive': total_FP,
            'correct-percent' : 0.0,
            'correct-percent-no-fn' : 0.0
        },
        "params" : {
            "model_name" : model_name,
            "epsilon" : epsilon,
            "min_samples" : min_samples,
            "Threshold" : Threshold
        }
    }


    """

    all_diffs = {}
    diffs = {}
    ind = 0

    for event_date in events_dates:
        events_file = predict_events(events_gen_dir, epsilon, min_samples, event_date, model_name, Threshold, windows_size, NUM_EL_GYR_ACC, NUM_EL_SPEED, NUM_EL_TIME, save_json_file)
        dd = datetime.strptime(event_date, '%Y-%m-%d').date()
        #hand_made_event_file = path.join(events_dir, str(dd.strftime('%Y%m%d')), 'all.json')
        hand_made_event_file = hand_made_events_paths[ind]
        ind = ind + 1
        diff = compaire_events(hand_made_event_file, events_file)
        diff["path_events_file"]=events_file
        diffs[event_date]=diff

        #print (event_date)
        #print (json.dumps(diff))
        #print ('')



    #print ('all')
    #print (json.dumps(merge_diffs(all_diffs)))
    js_r = merge_diffs(diffs.values())

    all_diffs = {
        "days" : diffs,
        "summary" : js_r,
        "params" : {
            "model_name" : model_name,
            "epsilon" : epsilon,
            "min_samples" : min_samples,
            "Threshold" : Threshold
        }

    }
    
    # js_r["epsilon"]=epsilon
    # js_r["min_samples"]=min_samples
    # js_r["Threshold"]=Threshold
    #df_js_r = pd.DataFrame(js_r)
    # ['Threshold','correct','correct-percent','correct-percent-no-fn','epsilon','false-negative','false-positive','min_samples','wrong']

    js_s = all_diffs["summary"]
    js_p = all_diffs["params"]
    js_list = [js_p["Threshold"], js_s["correct"], js_s["correct-percent"], 
               js_s["correct-percent-no-fn"], js_p["epsilon"], js_s["false-negative"], 
               js_s["false-positive"], js_p["min_samples"], js_s["wrong"]]

    try:
        q.put(all_diffs)
    except Exception, e:
        print (e)
    
    return all_diffs

