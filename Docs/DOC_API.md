



## How to use Smart Events Detector API

#### Normalize your Sensor data: {{api/normalize}}
	* Function name **normalize_data(args)** .
	* Input args :
		* arg1 (list): **[days]** from Sensor data day's.
	* Output **Dict** :
		{
			day1 : normalized_data_path,
		}

#### Prepare your Train and Valdition data: {{api/prepare}}
	* Function name **prepaire_all_parallel(args)** .
	* Input args :
		* arg1 (list): **[train_days]** from Sensor & Event data day's.
		* arg2 (list): **[cv_days]** from Sensor & Event data day's.
	* Output **Dict** :
		{
			"trainset" : trainset_path,
			"validationset" : validationset_path
		}

#### Generate your Test data: {{api/generate}}
	* Function name **gen(args)** .
	* Input args :
		* arg1 (list): **[test_days]** from Sensor data day's.
	* Output **Dict** :
		{
			day1 :	{
						"filedata" : testset_path,
						"filetime" : event_time_path
					},
		}

#### Get list of trained model names:  {{api/models}}
	* Function name **get_model_names()** .
	* Output **Dict** :
		{
			"models" :	["model1", "model2",]
		}


#### Predict your Test data: {{api/predict}}
	Detect potential events from generate test data.
	Note : To get full statistics detail, check that you have the corresponding hand made event files.
	* Function name **predict(args)** .
	* Input args :
		* arg1 (list): **[events_dates]** test data day's to predict.
		* arg2 (str): **model_name** Optional.
			* default value : "FFNN".
	* Output **Dict** :
		{
	        "days" : {
		        day1 : {
			        'correct' : {
			            'sum' : sum of correct_values,
			            'values' : {
					        type1 : correct_value,
					        type2 : correct_value,
					        ...,
					        typek : correct_value
					    }
			        },
			        'wrong' : {
			            'sum' : sum of wrong_values,
			            'values' : {
					        type1 : wrong_value,
					        type2 : wrong_value,
					        ...,
					        typek : wrong_value
					    }
			        },
			        'false-positive' : {
			            'sum' : sum of false-positive_values,
			            'values' : {
					        type1 : false-positive_value,
					        type2 : false-positive_value,
					        ...,
					        typek : false-positive_value
					    }
			        },
			        'false-negative' : {
			            'sum' : sum of false-negative_values,
			            'values' : {
					        type1 : false-negative_value,
					        type2 : false-negative_value,
					        ...,
					        typek : false-negative_value
					    }
			        },
			        'correct-percent' : correct_percent,
			        'correct-percent-no-fn' : correct_percent_no_fn,
			        'path_events_file' : path to json file
			    },
		    },
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


#### Advance Predict your Test data: {{api/Advance_predict}}
	Detect potential events from generate test data.
	Note : To get full statistics detail, check that you have the corresponding hand made event files.
	* Function name **advancePredict(args)** .
	* Input args :
		* arg1 (list): **[events_dates]** test data day's to predict.
		* arg2 (str): **model_name** Optional.
			* default value : "FFNN"
		* arg3 (float): 0.0<=**Threshold**<=1.0 Optional.
			* default value : 0.0
		* arg3 (integer): 1<=**epsilon** Optional.
			* default value : 1
		* arg3 (integer): 1<=**min_samples** Optional.
			* default value : 13
	* Output **Dict** :
		{
	        "days" : {
		        day1 : {
			        'correct' : {
			            'sum' : sum of correct_values,
			            'values' : {
					        type1 : correct_value,
					        type2 : correct_value,
					        ...,
					        typek : correct_value
					    }
			        },
			        'wrong' : {
			            'sum' : sum of wrong_values,
			            'values' : {
					        type1 : wrong_value,
					        type2 : wrong_value,
					        ...,
					        typek : wrong_value
					    }
			        },
			        'false-positive' : {
			            'sum' : sum of false-positive_values,
			            'values' : {
					        type1 : false-positive_value,
					        type2 : false-positive_value,
					        ...,
					        typek : false-positive_value
					    }
			        },
			        'false-negative' : {
			            'sum' : sum of false-negative_values,
			            'values' : {
					        type1 : false-negative_value,
					        type2 : false-negative_value,
					        ...,
					        typek : false-negative_value
					    }
			        },
			        'correct-percent' : correct_percent,
			        'correct-percent-no-fn' : correct_percent_no_fn,
			        'path_events_file' : path to json file
			    },
		    },
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