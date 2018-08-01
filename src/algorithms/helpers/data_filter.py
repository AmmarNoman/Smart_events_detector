import numpy as np
import os

def get_path(df, dates, times, classe):
	df_classe = df[df.file_class==classe]
	paths_date=[]
	for date in dates:
		df_classe_date=df_classe[df_classe.file_day==date]
		#print ("\n++++++++++++\n{}\n".format(df_classe_date))
		for time in times:
			df_classe_date_time=df_classe_date[df_classe_date.file_time==time]
			#print ("df_classe_date_time :\n{}\n\nshape :\n{}\n\npath :\n{}\n\n".format(df_classe_date_time,df_classe_date_time.shape,df_classe_date_time.file_path.values[0]))
			paths_date.append(str(df_classe_date_time.file_path.values[0]))

	return paths_date

# def get_events_dir(df):
# 	df_events=df[df.file_class=="Events"]
# 	dates=np.unique(df_events.file_day.values)[:2]
# 	paths=get_path(df, dates, ["00:00:00"]*2, "Events")

# 	if paths[0]==paths[1]:
# 		paths[0] = os.path.dirname(os.path.dirname(paths[0]))
# 	else:
# 		while paths[0]!=paths[1]:
# 			paths[0] = os.path.dirname(paths[0])
# 			paths[1] = os.path.dirname(paths[1])
	
# 	print("\n\n[END] events_dir_path0 : {}\n\n".format(paths[0]))
# 	return paths[0]