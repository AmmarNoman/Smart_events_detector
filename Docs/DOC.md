## Dataset

#### Use our dataset
	We gathered the original data: it is about 1000 kilometers of video and telemetry data gathered with a phone on the roads.
	You can freely use it to get start with our plugin.

#### Use your own dataset
	In this case you should upload a csv file describe the files distribution in your dataset directory.
	- ie -  Type, Date, Time, Extension, Absolute path
Where **Type** can take 3 values:

* Sensors
	* You need to have a csv data file in the format specified in the "Csv file format" section (Overview).

* Events 
	* If you need to generate **training set** after, then you should create training set with the help of **dashboard**. It means you can record events and save it in json format for further training of model.
	* But it's not necessary if you just want **predict** this data.

* Video
	* Video recorded at the same time of csv file.
	* Subtitles file (srt) for the video. It will only display the data for the period of video.

##### Example:
The following image represents files management of the day "2015-01-25".

![Data set example] (static/img/data_map.png)

	Events,2015-01-25,16:29:16,json,/Users/Ramah/Data/Events/InputFolder/20150125/20150125162916_60.07427_30.34051_1000057.json
	Events,2015-01-25,16:32:38,json,/Users/Ramah/Data/Events/InputFolder/20150125/20150125163238_60.07657_30.33811_1000057.json
	Events,2015-01-25,16:36:00,json,/Users/Ramah/Data/Events/InputFolder/20150125/20150125163600_60.07689_30.33703_1000057.json
	Events,2015-01-25,16:39:23,json,/Users/Ramah/Data/Events/InputFolder/20150125/20150125163923_60.07679_30.33816_1000057.json
	Events,2015-01-25,16:42:45,json,/Users/Ramah/Data/Events/InputFolder/20150125/20150125164245_60.07564_30.34149_1000057.json
	Events,2015-01-25,00:00:00,json,/Users/Ramah/Data/Events/InputFolder/20150125/all.json
	Sensors,2015-01-25,00:00:00,csv,/Users/Ramah/Data/Sensors/2015-01-25_SensorDatafile.csv
	Video,2015-01-25,16:29:16,mp4,/Users/Ramah/Data/Video/20150125/20150125162916_60.07427_30.34051_1000057.mp4
	Video,2015-01-25,16:29:16,srt,/Users/Ramah/Data/Video/20150125/20150125162916_60.07427_30.34051_1000057.srt
	Video,2015-01-25,16:32:38,mp4,/Users/Ramah/Data/Video/20150125/20150125163238_60.07657_30.33811_1000057.mp4
	Video,2015-01-25,16:32:38,srt,/Users/Ramah/Data/Video/20150125/20150125163238_60.07657_30.33811_1000057.srt
	Video,2015-01-25,16:36:00,mp4,/Users/Ramah/Data/Video/20150125/20150125163600_60.07689_30.33703_1000057.mp4
	Video,2015-01-25,16:36:00,srt,/Users/Ramah/Data/Video/20150125/20150125163600_60.07689_30.33703_1000057.srt
	Video,2015-01-25,16:39:23,mp4,/Users/Ramah/Data/Video/20150125/20150125163923_60.07679_30.33816_1000057.mp4
	Video,2015-01-25,16:39:23,srt,/Users/Ramah/Data/Video/20150125/20150125163923_60.07679_30.33816_1000057.srt
	Video,2015-01-25,16:42:45,mp4,/Users/Ramah/Data/Video/20150125/20150125164245_60.07564_30.34149_1000057.mp4
	Video,2015-01-25,16:42:45,srt,/Users/Ramah/Data/Video/20150125/20150125164245_60.07564_30.34149_1000057.srt


## Prepare your dataset for training and validation

#####At this stat we **normalize** your raw data & **generate** csv files from **it** and **hand made event files (json)**.

* **OutTrain.csv** from selected days as training set, thas you can download it, and start your model.
* **OutCv.csv** from selected days as validation set, thas you can download it, and validate your trained model.
	
*Note : we get the raw data from the Sensors directory and json files in the Events Directory*

## Generate Testset

##### **generate** csv files from **raw data**.

## Predict events

##### Predict events from generated csv files (Testing set) in **run** page.




