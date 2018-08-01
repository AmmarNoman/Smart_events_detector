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

class Normalize:
        

    def read_table(self, file_name):
        table = []
        with open(file_name, "r") as f:
            data = f.readlines()
            for line in data:
                words = line.splitlines()[0].split(';')
                table.append(words)
                
        return table


    def parse_data(self, table, ta, xa, ya, za, tg, xg, yg, zg, t_geo, speed_geo):
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

    def smooth_data(self, table, ta, xa, ya, za, tg, xg, yg, zg):
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
                table[i][2] = str(ta[ia])
                table[i][3] = str(xa[ia])
                table[i][4] = str(ya[ia])
                table[i][5] = str(za[ia])
                ia = ia + 1
                aAcc.append(table[i])
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
        self.smooth_proc(aGyr, result)
        return result


            
    def replace_data(self, table, ta, xa, ya, za, tg, xg, yg, zg):
        ia = 0
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
    def Normalizer(self, path_f, method):
        ta, xa, ya, za, tg, xg, yg, zg, t_geo, speed_geo = [], [], [], [], [], [], [], [], [], []
        x, y, start = [], [], 0
        xa_mean, ya_mean, za_mean = [], [], []
        
        print "Program normalizes sensors values as recording device is always in the same orientation"
        if not os.path.isfile(path_f):
            print "No input file given"
            return None
            
        table = self.read_table(path_f)
        self.parse_data(table, ta, xa, ya, za, tg, xg, yg, zg, t_geo, speed_geo)
        if method=="--smooth":
            output_filename = os.path.splitext(path_f)[0]+"_smooth.csv"
        elif method=="--kalman":
            output_filename = os.path.splitext(path_f)[0]+"_kalman.csv"
        else:
            output_filename = os.path.splitext(path_f)[0]+"_norm.csv"
        
        utils.to_mean(ta, xa, xa_mean)
        utils.to_mean(ta, ya, ya_mean)
        utils.to_mean(ta, za, za_mean)
        block_starts = utils.get_block_indices(ta, xa_mean, ya_mean, za_mean)
        #print len(ta),'  ',len(xa),'  ',len(ya),'  ',len(za)
        print len(block_starts), " block(s) found\n"
        for i in xrange(len(block_starts)):
            start = block_starts[i]
            if i < len(block_starts)-1 :
                finish = block_starts[i+1]
            else:
                finish = len(ta)
            #print "start = ",start
            #print "finish = ",finish
            rot_matrix = utils.get_z_rotation_matrix(start, finish, xa_mean, ya_mean, za_mean)
            #print "rot_matrix = \n", rot_matrix
            #print "\nxa_mean:\n",xa_mean
            #print "\nya_mean:\n",ya_mean
            #print "\nza_mean:\n",za_mean
            #print "\nrotate_block\n"
            utils.rotate_block(start, finish, xa_mean, ya_mean, za_mean, rot_matrix)
            #print "\nxa_mean:\n",xa_mean
            #print "\nya_mean:\n",ya_mean
            #print "\nza_mean:\n",za_mean

            start2 = int ( bisect.bisect_left(tg, ta[block_starts[i]]) )
            if i < len(block_starts)-1 :
                finish2 = int ( bisect.bisect_left(tg, ta[block_starts[i + 1]]) )
            else:
                finish2 = len(tg)

            #print "\nxg:\n",xg
            #print "\nyg:\n",yg
            #print "\nzg:\n",zg
            #print "\nrotate_block\n"
            utils.rotate_block(start2, finish2, xg, yg, zg, rot_matrix)
            #print "\nxg:\n",xg
            #print "\nyg:\n",yg
            #print "\nzg:\n",zg
            #print "\nta:\n",ta
            #print "\ntg:\n",tg
            #A verifier les parametres rot_matrix2
            rot_matrix2 = utils.get_plane_rotation_matrix(start, finish, ta, xa_mean, ya_mean, tg, zg, t_geo, speed_geo)
            utils.rotate_block(start, finish, xa_mean, ya_mean, za_mean, rot_matrix2)
            utils.rotate_block(start2, finish2, xg, yg, zg, rot_matrix2)

            utils.rotate_block(start, finish, xa, ya, za, rot_matrix)
            utils.rotate_block(start, finish, xa, ya, za, rot_matrix2)
        
        if method=="--smooth":
            print "\nStart smoothing data ...\n"
            smooth_table = self.smooth_data(table, ta, xa, ya, za, tg, xg, yg, zg)
            self.write_data(output_filename, smooth_table)
            print "\nSmoothing data completed\n"
        elif method=="--kalman":
            print "\nStart filtering data ...\n"
            filter_table = self.filter_data(table, ta, xa, ya, za, tg, xg, yg, zg)
            self.write_data(output_filename, filter_table)
            print "\nFiltering data completed\n"
        else:
            print "\nStart normalizing data ...\n"
            #replace_data(table, ta, xa_mean, ya_mean, za_mean, tg, xg, yg, zg)
            self.replace_data(table, ta, xa, ya, za, tg, xg, yg, zg)
            self.write_data(output_filename, table)
            print "\nNormalizing data completed\n"
        
        return output_filename