# This script will find out current orientation of the phone relative
# to the ground and rotate axes. 
# For both accelerometer and gyroscope the same rule must be applied
# That allows the smartphone to be used immediately as a replacement
# for expensive black box Inertial Measurement Units (IMU).

import numpy as np
import os
import sys
import math
import config
import utils
import GlobalsVar
#import gflags
import bisect
from pykalman import KalmanFilter
import pandas as pd
from datetime import datetime


class Normalize:
        

    def read_table(self, file_name):
        table = []
        with open(file_name, "r") as f:
            data = f.readlines()
            for line in data:
                words = line.splitlines()[0].split(';')
                table.append(words)
                
        return table


    def parse_data(self, table, ta, xa, ya, za, tg, xg, yg, zg, tmag, xmag, ymag, zmag, t_geo, speed_geo):
        i = 0
        
        
        while i < len(table):

            if table[i] == ["header","start"]:
                boln = True
                while boln:
                    i=i+1
                    if table[i] == ["header","end"]:
                        boln = False

            if len(table[i]) ==  0:
                continue
            type = table[i][0]
            if type == "1":
                t = float(table[i][2])
                ta.append(t)
                xa.append(float(table[i][3]))
                ya.append(float(table[i][4]))
                za.append(float(table[i][5]))
            elif type == "2":
                t = float(table[i][2])
                tmag.append(t)
                xmag.append(float(table[i][3]))
                ymag.append(float(table[i][4]))
                zmag.append(float(table[i][5]))
            elif type == "4":
                t = float(table[i][2])
                tg.append(t)
                xg.append(float(table[i][3]))
                yg.append(float(table[i][4]))
                zg.append(float(table[i][5]))
            elif type == "geo":
                t_geo.append(float(table[i][2]))
                speed_geo.append(float(table[i][9]))
            
            i=i+1
     

    def filter_proc(self, sensor, axes, result, init_st_mean = [0,0,0]):

        init = 0
        time_index = np.array(sensor[:,2], dtype=float)
        for axe in axes:
            # Construct a Kalman filter
            kf = KalmanFilter(transition_matrices = [1],
                              observation_matrices = [1],
                              initial_state_mean = init_st_mean[init],
                              initial_state_covariance = 1.1,
                              observation_covariance=0.8,
                              transition_covariance=.07)

            
            sensor_axe = pd.Series(sensor[:,axe], index=time_index, dtype=float)
            

            # Use the observed values of the price to get a rolling mean
            state_means, _ = kf.filter(sensor_axe.values)
            state_means = pd.Series(state_means.flatten(), index=sensor_axe.index)
            
            sensor[:,axe] = state_means.values
            init = init + 1

        result.extend(sensor.tolist())

    def filter_data(self, table, ta, xa, ya, za, tg, xg, yg, zg):
        ia = 0
        ig = 0
        i = 0

        aAcc = []
        aGyr = []
        aGeo = []
        result = []

        while i < len(table):
            if table[i] == ["header","start"]:
                boln = True
                while boln:
                    i=i+1
                    if table[i] == ["header","end"]:
                        boln = False

            if len(table[i]) ==  0:
                continue
            type = table[i][0]
            if type == "1":#accelerometer
                table[i][2] = float(ta[ia])
                table[i][3] = float(xa[ia])
                table[i][4] = float(ya[ia])
                table[i][5] = float(za[ia])
                ia = ia + 1
                aAcc.append(table[i])
            elif type == "4":#gyroscope
                table[i][2] = float(tg[ig])
                table[i][3] = float(xg[ig])
                table[i][4] = float(yg[ig])
                table[i][5] = float(zg[ig])
                ig = ig + 1
                aGyr.append(table[i])
            elif type == "geo":#geodata
                aGeo.append(table[i])

            i=i+1

        result = result + aGeo
        
        
        #filter data
        self.filter_proc(np.array(aAcc), [3,4,5], result, [0,0,9.6])
        self.filter_proc(np.array(aGyr), [3,4,5], result, [0,0,0])
        return result



    def smooth_proc(self, sensor, result):
        Flags_smoothRadius = config.smoothRadius 

        for i in xrange(len(sensor)):
            elCopy = []
            for e in sensor[i]:
                elCopy.append(e)
            
            elCopy[3] = 0
            elCopy[4] = 0
            elCopy[5] = 0
            tempCount = 0
            
            il = i-Flags_smoothRadius
            if il < 0:
                il = 0
                
            while il <= i + Flags_smoothRadius and il < len(sensor):
                elCopy[3] += float(sensor[il][3])
                elCopy[4] += float(sensor[il][4])
                elCopy[5] += float(sensor[il][5])
                tempCount = tempCount+1
                il = il + 1
                
            elCopy[3] /= tempCount
            elCopy[4] /= tempCount
            elCopy[5] /= tempCount
                
            result.append(elCopy)


    def smooth_data(self, table, ta, xa, ya, za, tg, xg, yg, zg, tmag, xmag, ymag, zmag, do_cf):
        ia = 0
        im = 0
        ig = 0
        i = 0

        aAcc = []
        aMag = []
        aGyr = []
        aGeo = []
        result = []
        
        while i < len(table):
            if table[i] == ["header","start"]:
                boln = True
                while boln:
                    i=i+1
                    if table[i] == ["header","end"]:
                        boln = False

            if len(table[i]) ==  0:
                continue
            type = table[i][0]
            if type == "1":#accelerometer
                table[i][2] = str(ta[ia])
                table[i][3] = str(xa[ia])
                table[i][4] = str(ya[ia])
                table[i][5] = str(za[ia])
                ia = ia + 1
                aAcc.append(table[i])
            elif type == "2":#magnetic field
                table[i][2] = str(tmag[im])
                table[i][3] = str(xmag[im])
                table[i][4] = str(ymag[im])
                table[i][5] = str(zmag[im])
                im = im + 1
                aMag.append(table[i])
            elif type == "4":#gyroscope
                table[i][2] = str(tg[ig])
                table[i][3] = str(xg[ig])
                table[i][4] = str(yg[ig])
                table[i][5] = str(zg[ig])
                ig = ig + 1
                aGyr.append(table[i])
            elif type == "geo":#geodata
                aGeo.append(table[i])

            i=i+1

        result = result + aGeo
        #smooth data
        self.smooth_proc(aAcc, result)
        self.smooth_proc(aMag, result)
        self.smooth_proc(aGyr, result)

        if do_cf:
            self.ComplementaryFilter(aAcc, aGyr, result)

        return result

    def avg_time(self, str_t0, str_t1):

        datetime_object = datetime.strptime(str_t0, '%H:%M:%S.%f')
        ss_0 = datetime_object.hour*3600+datetime_object.minute*60+datetime_object.second+datetime_object.microsecond/1000000.

        datetime_object = datetime.strptime(str_t1, '%H:%M:%S.%f')
        ss_1 = datetime_object.hour*3600+datetime_object.minute*60+datetime_object.second+datetime_object.microsecond/1000000.

        ss = (ss_0+ss_1)/2.0

        bigmins, secs = divmod(ss, 60)
        hours, mins = divmod(bigmins, 60)

        tt = ""
        st = str(secs).split('.')
        if len(st[0])==1:
            tt = "0"+st[0]+"."+st[1][:3]
        else:
            tt = str(secs)

        tm = ""
        stm = str(int(mins))
        if len(stm)==1:
            tm = "0"+stm
        else:
            tm = stm
            
        th = ""
        sth = str(int(hours))
        if len(sth)==1:
            th = "0"+sth
        else:
            th = sth

        #sss = str(int(hours))+":"+str(int(mins))+":"+tt
        sss = th+":"+tm+":"+tt
        return sss
            
    def ComplementaryFilter(self, aAcc, aGyr, result):
        # rotation angle of the sensor
        # initialize the angles
        # the filtered angles
        last_x_angles = [0.0]
        last_y_angles = [0.0]
        last_z_angles = [0.0]

        

        if len(aAcc)<len(aGyr):
            min_lenght = len(aAcc)
            #aCompf = aAcc[0]
        else:
            min_lenght = len(aGyr)
            #aCompf = aGyr[0]
        

        for i in xrange(min_lenght):

            # Get raw acceleration values
            accel_t = float(aAcc[i][2])
            accel_x = float(aAcc[i][3])
            accel_y = float(aAcc[i][4])
            accel_z = float(aAcc[i][5])
            # Get angle values from accelerometer !!!!
            RADIANS_TO_DEGREES = 180/3.14159
            accel_angle_y = np.arctan(-1*accel_x/np.sqrt(np.power(accel_y,2) + np.power(accel_z,2)))*RADIANS_TO_DEGREES
            accel_angle_x = np.arctan(accel_y/np.sqrt(np.power(accel_x,2) + np.power(accel_z,2)))*RADIANS_TO_DEGREES
            accel_angle_z = 0


            #! Convert gyro values to degrees/sec
            gyro_t = float(aGyr[i][2])
            gyro_x = float(aGyr[i][3])
            gyro_y = float(aGyr[i][4])
            gyro_z = float(aGyr[i][5])
            # Compute the (filtered) gyro angles [" drifting gyro angles "]
            dt = 0.12      #dt=120ms
            gyro_angle_x = gyro_x*dt + last_x_angles[-1]
            gyro_angle_y = gyro_y*dt + last_y_angles[-1]
            gyro_angle_z = gyro_z*dt + last_z_angles[-1]

            # Apply the complementary filter to figure out the change in angle - choice of alpha is
            # estimated now.  Alpha depends on the sampling rate...
            alpha = 0.96
            angle_x = alpha*gyro_angle_x + (1.0 - alpha)*accel_angle_x
            angle_y = alpha*gyro_angle_y + (1.0 - alpha)*accel_angle_y
            angle_z = gyro_angle_z  #Accelerometer doesn't give z-angle

            # saved the latest values
            last_x_angles.append(angle_x)
            last_y_angles.append(angle_y)
            last_z_angles.append(angle_z)

            aCompf = [None]*8
            aCompf[0] = "0"
            aCompf[1] = self.avg_time(aGyr[i][1], aAcc[i][1])
            aCompf[2] = (accel_t+gyro_t)/2.0
            aCompf[3] = angle_x
            aCompf[4] = angle_y
            aCompf[5] = angle_z
            #aCompf[6] = None
            #aCompf[7] = None

            result.append(aCompf)
    
            
    def replace_data(self, table, ta, xa, ya, za, tg, xg, yg, zg, tmag, xmag, ymag, zmag):
        ia = 0
        im = 0
        ig = 0
        i = 0
        
        while i < len(table):
            if table[i] == ["header","start"]:
                boln = True
                while boln:
                    i=i+1
                    if table[i] == ["header","end"]:
                        boln = False

            if len(table[i]) ==  0:
                continue
            type = table[i][0]
            if type == "1":
                table[i][2] = str(ta[ia])
                table[i][3] = str(xa[ia])
                table[i][4] = str(ya[ia])
                table[i][5] = str(za[ia])
                ia = ia + 1
            elif type == "2":
                table[i][2] = str(tmag[im])
                table[i][3] = str(xmag[im])
                table[i][4] = str(ymag[im])
                table[i][5] = str(zmag[im])
                im = im + 1
            elif type == "4":
                table[i][2] = str(tg[ig])
                table[i][3] = str(xg[ig])
                table[i][4] = str(yg[ig])
                table[i][5] = str(zg[ig])
                ig = ig + 1
            i=i+1
                
                
    def write_data(self, file_name, table):
        with open(file_name, "w") as f:
            i = 0
            while i < len(table):
                for j in xrange(len(table[i])):
                    if j == len(table[i])-1 and i!= len(table)-1:
                        string = "\n"
                    else:
                        string = ";"
                    f.write(str(table[i][j])+string)
                i = i + 1
            
            
    def dumb_track_calculation(self, ta, xa, ya, tg, zg, x, y, start = 0):
        x_cur = 0.0
        y_cur = 0.0
        alpha_cur = 0.0
        vx_cur = 0.0
        vy_cur = 0.0
        
        for i in xrange(start):
            x.append(0)
            y.append(0)
        
        j = 0
        for i in xrange(start, len(ta)-1):
            while j < len(zg)-1 and tg[j+1] < ta[i] :
                alpha_cur = alpha_cur + zg[j] * (tg[j+1] - tg[j])
                print alpha_cur, ' ', j, ' ', zg[j], ' ', tg[j + 1], ' ', tg[j]
                j = j + 1
            
            dt = ta[i+1] - ta[i];
            ax_cur = xa[i] * np.cos(alpha_cur) + ya[i] * np.sin(alpha_cur)
            ay_cur = -xa[i] * np.sin(alpha_cur) + ya[i] * np.cos(alpha_cur)
            vx_cur = vx_cur + ax_cur * dt
            vy_cur = vy_cur + ay_cur * dt
            x_cur = x_cur + vx_cur * dt
            y_cur = y_cur + vy_cur * dt
            x.append(x_cur)
            y.append(y_cur)
            
    """
    $> Normalizer file_name [--smooth]
    """        
    def Normalizer(self, path_f, method, do_Norm_proc=True):
        ta, xa, ya, za, tg, xg, yg, zg, t_geo, speed_geo = [], [], [], [], [], [], [], [], [], []
        tmag, xmag, ymag, zmag = [], [], [], []
        x, y, start = [], [], 0
        xa_mean, ya_mean, za_mean = [], [], []
        
        print "Program normalizes sensors values as recording device is always in the same orientation"
        if not os.path.isfile(path_f):
            print "No input file given"
            return None
            
        table = self.read_table(path_f)
        self.parse_data(table, ta, xa, ya, za, tg, xg, yg, zg, tmag, xmag, ymag, zmag, t_geo, speed_geo)
        
        if do_Norm_proc:
            utils.to_mean(ta, xa, xa_mean)
            utils.to_mean(ta, ya, ya_mean)
            utils.to_mean(ta, za, za_mean)
            block_starts = utils.get_block_indices(ta, xa_mean, ya_mean, za_mean)
            print len(block_starts), " block(s) found\n"
            for i in xrange(len(block_starts)):
                start = block_starts[i]
                if i < len(block_starts)-1 :
                    finish = block_starts[i+1]
                else:
                    finish = len(ta)
                
                rot_matrix = utils.get_z_rotation_matrix(start, finish, xa_mean, ya_mean, za_mean)
                utils.rotate_block(start, finish, xa_mean, ya_mean, za_mean, rot_matrix)

                start2 = int ( bisect.bisect_left(tg, ta[block_starts[i]]) )
                if i < len(block_starts)-1 :
                    finish2 = int ( bisect.bisect_left(tg, ta[block_starts[i + 1]]) )
                else:
                    finish2 = len(tg)

                utils.rotate_block(start2, finish2, xg, yg, zg, rot_matrix)
                #A verifier les parametres rot_matrix2
                rot_matrix2 = utils.get_plane_rotation_matrix(start, finish, ta, xa_mean, ya_mean, tg, zg, t_geo, speed_geo)
                
                utils.rotate_block(start, finish, xa_mean, ya_mean, za_mean, rot_matrix2)
                utils.rotate_block(start2, finish2, xg, yg, zg, rot_matrix2)

                utils.rotate_block(start, finish, xa, ya, za, rot_matrix)
                utils.rotate_block(start, finish, xa, ya, za, rot_matrix2)

                start3 = int ( bisect.bisect_left(tmag, ta[block_starts[i]]) )
                if i < len(block_starts)-1 :
                    finish3 = int ( bisect.bisect_left(tmag, ta[block_starts[i + 1]]) )
                else:
                    finish3 = len(tmag)
                utils.rotate_block(start3, finish3, xmag, ymag, zmag, rot_matrix)
                utils.rotate_block(start3, finish3, xmag, ymag, zmag, rot_matrix2)
            
        if method=="--smooth":
            output_filename = os.path.splitext(path_f)[0]+"_smooth.csv"
            print "\nStart smoothing data ...\n"
            smooth_table = self.smooth_data(table, ta, xa, ya, za, tg, xg, yg, zg, tmag, xmag, ymag, zmag, do_cf=False)
            self.write_data(output_filename, smooth_table)
            print "\nSmoothing data completed\n"
        elif method=="--smooth-cf":
            output_filename = os.path.splitext(path_f)[0]+"_smooth_cf.csv"
            print "\nStart smoothing data & CF ...\n"
            smooth_table = self.smooth_data(table, ta, xa, ya, za, tg, xg, yg, zg, tmag, xmag, ymag, zmag, do_cf=True)
            self.write_data(output_filename, smooth_table)
            print "\nSmoothing data & CF completed\n"
        elif method=="--kalman":
            ############
            return None#
            ############
            output_filename = os.path.splitext(path_f)[0]+"_kalman.csv"
            print "\nStart filtering data ...\n"
            filter_table = self.filter_data(table, ta, xa, ya, za, tg, xg, yg, zg)
            self.write_data(output_filename, filter_table)
            print "\nFiltering data completed\n"
        else:
            output_filename = os.path.splitext(path_f)[0]+"_norm.csv"
            print "\nStart normalizing data ...\n"
            #replace_data(table, ta, xa_mean, ya_mean, za_mean, tg, xg, yg, zg)
            self.replace_data(table, ta, xa, ya, za, tg, xg, yg, zg, tmag, xmag, ymag, zmag)
            self.write_data(output_filename, table)
            print "\nNormalizing data completed\n"
        
        return output_filename

