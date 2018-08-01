import os
import pandas as pd
import numpy as np
from datetime import datetime
import time

class MappingData:
    
    MappedData=None
    
    """docstring for MappingData"""
    def __init__(self, data_path):
        liste_dict = self.deepListDirs(data_path)
        self.MappedData = pd.DataFrame(liste_dict)


    def deepListDirs(self, path):
        liste = []
        items = os.listdir(path)
        for item in items:
            i_path = os.path.join(path, item)
            if not item.startswith('.') and os.path.isdir(i_path):
                if item == 'Events':
                    liste=liste+self.listEvents(i_path)
                if item == 'Sensors':
                    liste=liste+self.listSensors(i_path)
                if item == 'video':
                    liste=liste+self.listVideos(i_path)
        return liste


    def listEvents(self, dir_path):
        find = []
        items = os.listdir(dir_path)
        for item in items:
            i_path = os.path.join(dir_path, item)
            if not item.startswith('.'):
                if os.path.isdir(i_path):
                    find=find+self.listEvents(i_path)
                else:
                    temp = {}
                    temp['file_class']='Events'
                    temp['file_day']=self.get_date_event(i_path)
                    temp['file_time']=self.get_time_event(i_path)
                    temp['file_extension']=self.get_extension(i_path)
                    temp['file_path']=i_path
                    find.append(temp)
        return find


    def listSensors(self, dir_path):
        find = []
        items = os.listdir(dir_path)
        for item in items:
            i_path = os.path.join(dir_path, item)
            if not item.startswith('.'):
                if os.path.isdir(i_path):
                    find=find+self.listSensors(i_path)
                else:
                    temp = {}
                    temp['file_class']='Sensors'
                    temp['file_day']=self.get_date_sensor(i_path)
                    temp['file_time']=self.get_time_sensor(i_path)
                    temp['file_extension']=self.get_extension(i_path)
                    temp['file_path']=i_path
                    find.append(temp)
        return find


    def listVideos(self, dir_path):
        find = []
        items = os.listdir(dir_path)
        for item in items:
            i_path = os.path.join(dir_path, item)
            if not item.startswith('.'):
                if os.path.isdir(i_path):
                    find=find+self.listVideos(i_path)
                else:
                    temp = {}
                    temp['file_class']='Video'
                    temp['file_day']=self.get_date_video(i_path)
                    temp['file_time']=self.get_time_video(i_path)
                    temp['file_extension']=self.get_extension(i_path)
                    temp['file_path']=i_path
                    find.append(temp)
        return find


    def get_date_event(self, path):
        try:
            # Return the date from file name
            return str(datetime.strptime(os.path.basename(path)[:8], '%Y%m%d').date())
        except ValueError:
            try:
                # Return the date from dir name
                return str(datetime.strptime( os.path.basename(os.path.dirname(path)), '%Y%m%d').date())
            except:
                # Return the last modification date of the file
                return str(datetime.fromtimestamp(time.mktime(time.gmtime(os.path.getmtime(path)))).date())
                #return '0000-00-00'


    def get_date_sensor(self, path):
        try:
            # Return the date from file name
            return str(datetime.strptime(os.path.basename(path)[:10], '%Y-%m-%d').date())
        except ValueError:
            try:
                # Return the date from dir name
                return str(datetime.strptime( os.path.basename(os.path.dirname(path)), '%Y%m%d').date())
            except:
                # Return the last modification date of the file
                return str(datetime.fromtimestamp(time.mktime(time.gmtime(os.path.getmtime(path)))).date())
                #return '0000-00-00'


    def get_date_video(self, path):
        try:
            # Return the date from file name
            return str(datetime.strptime(os.path.basename(path)[:8], '%Y%m%d').date())
        
        except ValueError:
            try:
                # Return the date from dir name
                return str(datetime.strptime( os.path.basename(os.path.dirname(path)), '%Y%m%d').date())
            except:
                # Return the last modification date of the file
                return str(datetime.fromtimestamp(time.mktime(time.gmtime(os.path.getmtime(path)))).date())
                #return '0000-00-00'


    def get_time_event(self, path):
        try:
            return str(datetime.strptime(os.path.basename(path)[8:14], '%H%M%S').time())
        except ValueError:
            # Return the last modification time of the file
            #return str(datetime.fromtimestamp(time.mktime(time.gmtime(os.path.getmtime(path)))).time())
            return '00:00:00'


    def get_time_sensor(self, path):
        return '00:00:00'


    def get_time_video(self, path):
        try:
            return str(datetime.strptime(os.path.basename(path)[8:14], '%H%M%S').time())
        except ValueError:
            # Return the last modification time of the file
            #return str(datetime.fromtimestamp(time.mktime(time.gmtime(os.path.getmtime(path)))).time())
            return '00:00:00'


    def get_extension(self, filename):
        return os.path.splitext(filename)[1][1:]

