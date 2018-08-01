from interface import Main_plugin_interface
import config_param as conf

class Main_plugin(Main_plugin_interface):
	"""docstring for Main_plugin"""
	def __init__(self):
		Main_plugin_interface.__init__(self)

	def prepaire_all_parallel(self, train_days=[], cv_days=[]):
		prefix_name_file = ""
		return Main_plugin_interface.prepaire_all_parallel(self, prefix_name_file, train_days, cv_days, normalized=False)

	def gen(self, test_days=[]):
		windows_size = conf.windows_size
		return Main_plugin_interface.gen(self, windows_size, test_days, normalized=False, \
			NUM_EL_GYR_ACC=conf.NUM_EL_GYR_ACC, NUM_EL_SPEED=conf.NUM_EL_SPEED,\
			 NUM_EL_TIME=conf.NUM_EL_TIME, time_delta_events_msec=conf.time_delta_events_msec)


