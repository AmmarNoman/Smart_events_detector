# 
import config_path as config_p
class DataSplit:

	data_train_days =[]
	data_cv_days =[]
	data_test_days =[]

	"""docstring for DataSplit
	Split your raw data using days"""

	def __init__(self):
		tr = ""
		with open(config_p.b_path+"train.data", 'r') as b_script:
			tr = b_script.read()
		self.data_train_days = tr.split()

		cv = ""
		with open(config_p.b_path+"validation.data", 'r') as b_script:
			cv = b_script.read()
		self.data_cv_days = cv.split()
		
		ts = ""
		with open(config_p.b_path+"test.data", 'r') as b_script:
			ts = b_script.read()
		self.data_test_days = ts.split()