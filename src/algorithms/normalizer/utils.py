import bisect
import math
import numpy as np
import sys

import GlobalsVar
import config

Flags_Percent = config.sm_range_part
Flags_Radius = config.sm_radius

Flags_threshold = config.block_diff_thres
Flags_time_thres = config.block_time_thres
Flags_adjacent = config.adjacent

Flags_part = config.z_range_part

Flags_speed_thres = config.speed_detection_thres

#Mean is calculated by throwing aside boundary values and taking regular mean value of the rest.
def quantile_mean(time, t, x):
    global Flags_Radius
    global Flags_Percent
    radius =  Flags_Radius   
    percent = Flags_Percent
    
    
    low_emp = bisect.bisect_left(t, time - radius)
    up_emp = bisect.bisect_right(t, time + radius)
    x2 = x[low_emp: up_emp]
    """
    print "\nlow_emp = ",low_emp
    print "up_emp = ",up_emp
    print "x2 = ", x2
    """
    x2.sort()

    n = len(x2)
    low = int (math.floor(n * (0.5 * (1 - percent))))
    up =  int (math.ceil(n * (0.5 * (1 + percent))))
    
    if (low == up) :
        return 0;
    
    if (up - low == 1) :
        return x2[low]
    
    current = 0.0
    for i in range(low,up):
        current += x2[i]
    
    return current / (up - low)

    
#Mean function is applied to each element in array.
def to_mean(t, x, res):
    for i in xrange(len(t)):
        res.append(quantile_mean(t[i], t, x))

"""
ta=[42029.69643622685, 42029.69643761574, 42029.69643899305, 42029.696440405096, 42029.696441782406]
xa=[9.348, 9.041, 8.926, 8.811, 10.152]
ya=[-1.799, -1.991, -1.876, -2.144, -2.527]
za=[-1.301, -1.838, -1.34, -1.263, -1.109]
"""


def difference(i, j, x, y, z):
    return np.square(x[j] - x[i]) + np.square(y[j] - y[i]) + np.square(z[j] - z[i])


#Returns division of the given array on different blocks.
#New block is started if the current difference is bigger than the given threshold.
#'adjacent' parameter determines used algorithm for which points to use to count the current difference.
def get_block_indices(t, x, y, z):
    global Flags_threshold
    global Flags_time_thres
    global Flags_adjacent

    threshold = Flags_threshold
    time_thres = Flags_time_thres
    adjacent = Flags_adjacent

    result = []
    result.append(0)
    last = 0
    for i in xrange(1, len(x)) :
        diff = difference(last, i, x, y, z)
        if diff >= threshold or (t[i] - t[i-1]) > time_thres :
            result.append(i)
            last = i
        elif adjacent :
            last = i
        
    return result

def get_rotation_matrix(nx, ny, nz, cos_phi, sin_phi):
    m00 = cos_phi + (1.0 - cos_phi) * nx * nx
    m01 = (1.0 - cos_phi) * nx * ny - sin_phi * nz
    m02 = (1.0 - cos_phi) * nx * nz + sin_phi * ny
    m10 = (1.0 - cos_phi) * ny * nx + sin_phi * nz
    m11 = cos_phi + (1.0 - cos_phi) * ny * ny
    m12 = (1.0 - cos_phi) * ny * nz - sin_phi * nx
    m20 = (1.0 - cos_phi) * nz * nx - sin_phi * ny
    m21 = (1.0 - cos_phi) * nz * ny + sin_phi * nx
    m22 = cos_phi + (1.0 - cos_phi) * nz * nz
    m = np.matrix([[m00,m01,m02],[m10,m11,m12],[m20,m21,m22]], dtype=float)
    
    return m

class Vector3d:
     x, y, z, len_squared = 0.0, 0.0, 0.0, 0.0

     def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.len_squared = (x * x + y * y + z * z)

     def __lt__(self,Vector3d):
        return self.len_squared < Vector3d.len_squared

     def __gt__(self,Vector3d):
        return self.len_squared > Vector3d.len_squared

     def __ge__(self,Vector3d):
        return self.len_squared >= Vector3d.len_squared

     def __le__(self,Vector3d):
        return self.len_squared <= Vector3d.len_squared


#Returns a rotation matrix which changes coordinates so that the Z axis points to the ground.
def get_z_rotation_matrix(start, end, x, y, z):
    global Flags_part
    part = Flags_part

    assert (start < end)

    #list of class Vector3d
    acc = []
    for i in xrange(start,end):
        acc.append(Vector3d(x[i], y[i], z[i]))
    
    acc.sort()
    ###print "start=",start,"   end=",end

    xx = 0.0
    yy = 0.0 
    zz = 0.0
    start2 = int (len(acc) * (1 - part) * 0.5)
    end2 = int (len(acc) * (1 + part) * 0.5)
    
    if start2 == end2 :
        ###print start2," = ",end2
        start2 = start
        end2 = end
    ###print "len(acc) = ",len(acc),"   range(",start2,",",end2,")"
    for i in xrange(start2,end2):
        xx += acc[i].x
        yy += acc[i].y
        zz += acc[i].z

    xx /= end2 - start2
    yy /= end2 - start2
    zz /= end2 - start2

    len1 = np.sqrt(np.square(xx) + np.square(yy) + np.square(zz))
    len2 = np.sqrt(np.square(xx) + np.square(yy))

    if (len1 < GlobalsVar.EPSILON) or (len2 < GlobalsVar.EPSILON):
        print >> sys.stderr, "Small len1 in get_z_rotation_matrix"
        return get_rotation_matrix(0, 0, 1, 1, 0)
    

    nx = yy / len2
    ny = -xx / len2
    nz = 0.0
    cos_phi = zz / len1
    sin_phi = np.sqrt(1 - np.square(cos_phi))
    #print "\nnx = ",nx, "\nny = ",ny, "\nnz = ",nz, "\ncos_phi = ",cos_phi, "\nsin_phi = ",sin_phi
    #print "get_rotation_matrix(nx, ny, nz, cos_phi, sin_phi) = \n",get_rotation_matrix(nx, ny, nz, cos_phi, sin_phi)
    return get_rotation_matrix(nx, ny, nz, cos_phi, sin_phi)

#Returns a rotation matrix which changes coordinates so that the X axis points forward (currently may point backward).
def get_plane_rotation_matrix(start, end, t, x, y, tg, zg, t_geo, speed_geo):
    #print "start = ",start, "\n end = ",end, "\n t = ",len(t), "\n x = ",len(x), "\n y = ",len(y), "\n tg = ",len(tg), "\n zg = ",len(zg), "\n t_geo = ",len(t_geo), "\n speed_geo = ",len(speed_geo)
    global Flags_speed_thres
    speed_thres = Flags_speed_thres

    assert (start < end)
    xx = 0.0
    yy = 0.0
    c = 0.0
    for i in xrange(start,end):
        #print "t[",i,"] = ",t[i]
        #print "zg[350:355] = ",zg[350:355]
        #print "bisect.bisect_left(zg, t[",i,"]) = ", bisect.bisect_left(zg, t[i])
        #print zg
        index_zg = bisect.bisect_left(zg, t[i])
        if index_zg==len(zg):
            #a random numbre -416621723
            coeff = 1.0 / (1.0 + np.abs(-416621723))
        else:
            coeff = 1.0 / (1.0 + np.abs(zg[ index_zg ]))

        speed_index = int (bisect.bisect_left(t_geo, t[i]))
        if (speed_index+1 >= len(speed_geo)) or (t_geo[speed_index+1]-t_geo[speed_index] > speed_thres) or (speed_geo[speed_index+1] >= speed_geo[speed_index]):
            xx += x[i] * coeff
            yy += y[i] * coeff
        else :
            xx -= x[i] * coeff
            yy -= y[i] * coeff
        
        c += coeff
    
    xx /= c
    yy /= c
    leng = np.sqrt(np.square(xx) + np.square(yy))
    if leng < GlobalsVar.EPSILON:
        return get_rotation_matrix(0, 0, 1, 1, 0)
    
    xx /= leng
    yy /= leng

    #print "\nget_rotation_matrix(0, 0, 1, ",xx,", ",yy,") = \n", get_rotation_matrix(0, 0, 1, xx, yy)
    return get_rotation_matrix(0, 0, 1, xx, yy)


#Applies rotation to given vectors of coordinates.
def rotate_block(start, end, x, y, z, m):
    for i in xrange(start,end):
        x2 = m[0,0] * x[i] + m[0,1] * y[i] + m[0,2] * z[i]
        y2 = m[1,0] * x[i] + m[1,1] * y[i] + m[1,2] * z[i]
        z2 = m[2,0] * x[i] + m[2,1] * y[i] + m[2,2] * z[i]
        x[i] = x2
        y[i] = y2
        z[i] = z2
    
