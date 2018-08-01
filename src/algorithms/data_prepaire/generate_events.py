#!/usr/bin/python
# Copyright (c) 2017, Smart Events Detector Project
# All rights reserved.

# This script generate a potential events from raw data, using windows of multi size?
# (see TRIAL_INTERVALS_SEC variable)
import numpy as np
# fix random seed for reproducibility
seed = 1994
np.random.seed(seed)

import csv
import sys
import datetime
from datetime import timedelta
from scipy.interpolate import interp1d

CF_KEY = '0'
ACC_KEY = '1'
MAG_KEY = '2'
GYR_KEY = '4'
CF_KEY_INT = 0
ACC_KEY_INT = 1
MAG_KEY_INT = 2
GYR_KEY_INT = 4
GEO_KEY = 'geo'
TIME_KEY = 'time'
TYPE_KEY = 'type'
DIRECTION_KEY = 'direction'
TYPE_INDX = 0
TIME_INDX = 1
X_INDX = 3
Y_INDX = 4
Z_INDX = 5
SPD_INDX = 9
SpeedCoefficient = 3.6
EVENT_START = 'start'
EVENT_END = 'end'

EVENT_TYPE_IDLE = -1
EVENT_DIR_LEFT = 0

TIME_DELTA_IDLE_START_END = timedelta(seconds=10)
TIME_DELTA_EVENTS = timedelta(milliseconds=500)
IDLE_TO_EVENTS_RELATION = 1

TRIAL_INTERVALS_SEC = [2.5, 5, 7, 11, 13, 16]
TRIAL_STEP_SEC = 0.5

TIME_FORMAT = '%H:%M:%S.%f'

num_elements_gyr_acc = 20
num_elements_speed = 5
num_elements_duration = 1

time_delta_events = TIME_DELTA_EVENTS


def bsearch(arr, searchValue, left, right, extr=lambda v: v):
    if searchValue < extr(arr[0]):
        raise ValueError('search value is less than left value')

    if right < left:
        return right

    mid = (left + right) / 2
    if searchValue > extr(arr[mid]):
        return bsearch(arr, searchValue, mid + 1, right, extr)
    elif searchValue < extr(arr[mid]):
        return bsearch(arr, searchValue, left, mid - 1, extr)
    else:
        return mid


def parse_row(row):
    if row[TYPE_INDX] == ACC_KEY or row[TYPE_INDX] == GYR_KEY or row[TYPE_INDX] == MAG_KEY or row[TYPE_INDX] == CF_KEY:
        row[TYPE_INDX] = int(row[TYPE_INDX])
        row[X_INDX] = float(row[X_INDX])
        row[Y_INDX] = float(row[Y_INDX])
        row[Z_INDX] = float(row[Z_INDX])
        row[TIME_INDX] = datetime.datetime.strptime(row[TIME_INDX], TIME_FORMAT)
        return row

    if row[TYPE_INDX] == GEO_KEY:
        row[SPD_INDX] = float(row[SPD_INDX]) * SpeedCoefficient
        row[TIME_INDX] = datetime.datetime.strptime(row[TIME_INDX], TIME_FORMAT)
        return row

    return None


def load_data(inputfile):
    with open(inputfile, 'rb') as input:
        csvreader = csv.reader(input, delimiter=';')

        data = []
        num_errors = 0

        for row in csvreader:
            try:
                parsed = parse_row(row)
                if parsed is not None:
                    data.append(parsed)
            except ValueError:
                num_errors += 1

        if num_errors > 0:
            print 'Errors occured while parsing is ' + str(num_errors)

        return data


def sort_data_by_time(data):
    return sorted(data, key=lambda row: row[TIME_INDX])


def duration_deploye(duration, lenght):
    data_time = []
    cte = 0
    step = duration/float(lenght)
    for t in xrange(lenght):
        data_time.append(step+cte)
        cte = cte + step
    return data_time

def get_data_for_interval(data, start_indx, end_indx, event_type, event_dir):
    if start_indx < 0 or end_indx >= len(data):
        raise IndexError('index out of range')

    result = {
        ACC_KEY: {
            'x': [],
            'y': [],
            'z': []
        },
        MAG_KEY: {
            'x': [],
            'y': [],
            'z': []
        },
        GYR_KEY: {
            'x': [],
            'y': [],
            'z': []
        },
        CF_KEY: {
            'x': [],
            'y': [],
            'z': []
        },
        GEO_KEY: {
            'spd': []
        },
        TYPE_KEY: event_type,
        DIRECTION_KEY: event_dir
    }

    for i in xrange(start_indx, end_indx + 1):
        row = data[i]
        if row[TYPE_INDX] == ACC_KEY_INT:
            result[ACC_KEY]['x'].append(row[X_INDX])
            result[ACC_KEY]['y'].append(row[Y_INDX])
            result[ACC_KEY]['z'].append(row[Z_INDX])
        if row[TYPE_INDX] == MAG_KEY_INT:
            result[MAG_KEY]['x'].append(row[X_INDX])
            result[MAG_KEY]['y'].append(row[Y_INDX])
            result[MAG_KEY]['z'].append(row[Z_INDX])
        if row[TYPE_INDX] == GYR_KEY_INT:
            result[GYR_KEY]['x'].append(row[X_INDX])
            result[GYR_KEY]['y'].append(row[Y_INDX])
            result[GYR_KEY]['z'].append(row[Z_INDX])
        if row[TYPE_INDX] == CF_KEY_INT:
            result[CF_KEY]['x'].append(row[X_INDX])
            result[CF_KEY]['y'].append(row[Y_INDX])
            result[CF_KEY]['z'].append(row[Z_INDX])
        if row[TYPE_INDX] == GEO_KEY:
            result[GEO_KEY]['spd'].append(row[SPD_INDX])

    result[ACC_KEY]['x'] = interpolate_array(result[ACC_KEY]['x'], num_elements_gyr_acc)
    result[ACC_KEY]['y'] = interpolate_array(result[ACC_KEY]['y'], num_elements_gyr_acc)
    result[ACC_KEY]['z'] = interpolate_array(result[ACC_KEY]['z'], num_elements_gyr_acc)

    result[MAG_KEY]['x'] = interpolate_array(result[MAG_KEY]['x'], num_elements_gyr_acc)
    result[MAG_KEY]['y'] = interpolate_array(result[MAG_KEY]['y'], num_elements_gyr_acc)
    result[MAG_KEY]['z'] = interpolate_array(result[MAG_KEY]['z'], num_elements_gyr_acc)

    result[GYR_KEY]['x'] = interpolate_array(result[GYR_KEY]['x'], num_elements_gyr_acc)
    result[GYR_KEY]['y'] = interpolate_array(result[GYR_KEY]['y'], num_elements_gyr_acc)
    result[GYR_KEY]['z'] = interpolate_array(result[GYR_KEY]['z'], num_elements_gyr_acc)

    result[CF_KEY]['x'] = interpolate_array(result[CF_KEY]['x'], num_elements_gyr_acc)
    result[CF_KEY]['y'] = interpolate_array(result[CF_KEY]['y'], num_elements_gyr_acc)
    result[CF_KEY]['z'] = interpolate_array(result[CF_KEY]['z'], num_elements_gyr_acc)

    result[GEO_KEY]['spd'] = interpolate_array(result[GEO_KEY]['spd'], num_elements_speed)

    duration = data[end_indx][TIME_INDX] - data[start_indx][TIME_INDX]
    d_time = duration.seconds*1000 + duration.microseconds/1000
    #result[TIME_KEY] = [d_time]
    result[TIME_KEY] = duration_deploye(d_time, num_elements_duration)

    return result


def interpolate_array(arr, num_el):
    if num_el <= 0:
        raise ValueError('num_el cannot be 0 or less')

    if len(arr) == 0:
        arr = [0, 0]
    elif len(arr) == 1:
        arr = [arr[0], arr[0]]

    one_el_len = float(len(arr) - 1)/(num_el - 1)

    x = [i for i in xrange(len(arr))]
    y = arr

    # due to rounding issues we need to prolong array by one element
    x += [x[-1] + 1]
    y += [arr[-1]]

    intrpl = interp1d(x, y)
    try:
        result = [intrpl(i * one_el_len)[()] for i in xrange(num_el)]
    except ValueError:
        raise

    return result


def write_one_row(data, datawriter, timewriter, start_date, end_date):
    row = data[ACC_KEY]['x'] + data[ACC_KEY]['y'] + data[ACC_KEY]['z'] + \
        data[GYR_KEY]['x'] + data[GYR_KEY]['y'] + data[GYR_KEY]['z'] + \
        data[MAG_KEY]['x'] + data[MAG_KEY]['y'] + data[MAG_KEY]['z'] + \
        data[CF_KEY]['x'] + data[CF_KEY]['y'] + data[CF_KEY]['z'] + \
        data[GEO_KEY]['spd'] + data[TIME_KEY]

    date_arr = [start_date.strftime(TIME_FORMAT), end_date.strftime(TIME_FORMAT)]

    datawriter.writerow(['{0:.3f}'.format(x) for x in row])
    timewriter.writerow(date_arr)


def write_data(data, outfiledata, outfiletime, windows_size):
    last_date = None

    datawriter = csv.writer(outfiledata, delimiter=',', quoting=csv.QUOTE_NONE)
    timewriter = csv.writer(outfiletime, delimiter=',', quoting=csv.QUOTE_NONE)

    for start_indx in xrange(0, len(data)):
        start_date = data[start_indx][TIME_INDX]

        if last_date != None:
            time_diff = start_date - last_date
            if time_diff.total_seconds() < TRIAL_STEP_SEC:
                continue

        last_date = start_date
        for interval in windows_size:
            msec = int((interval-int(interval))*1000)
            end_date = start_date + \
                timedelta(seconds=int(interval), milliseconds=msec)
            end_indx = bsearch(data, end_date, 0, len(data) - 1,
                                lambda row: row[TIME_INDX])

            data_for_interval = get_data_for_interval(data, start_indx, end_indx, EVENT_TYPE_IDLE, EVENT_DIR_LEFT)
            write_one_row(data_for_interval, datawriter, timewriter, start_date, end_date)

        sys.stdout.write("\r%d of %d" % (start_indx, len(data)) )
        sys.stdout.flush()

def gen_events(inputfile, outfiledata, outfiletime, time_delta_events_msec, NUM_EL_GYR_ACC, NUM_EL_SPEED, NUM_EL_TIME, windows_size=TRIAL_INTERVALS_SEC):

    global num_elements_speed
    global num_elements_gyr_acc
    global time_delta_events
    global num_elements_duration

    time_delta_events = time_delta_events_msec
    num_elements_gyr_acc = NUM_EL_GYR_ACC
    num_elements_speed = NUM_EL_SPEED
    num_elements_duration = NUM_EL_TIME


    data = load_data(inputfile)

    data = sort_data_by_time(data)

    with open(outfiledata, 'wb') as out_f_data:
        with open(outfiletime, 'wb') as out_f_time:
            write_data(data, out_f_data, out_f_time, windows_size)
