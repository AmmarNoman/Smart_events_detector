#!/usr/bin/python
import csv
import numpy as np
import datetime
import json
from sklearn.cluster import DBSCAN
from optparse import OptionParser

TIME_FORMAT_LONG = '%H:%M:%S.%f'
TIME_FORMAT_SHORT = '%H:%M:%S'
SECONDS_IN_DAY = 24 * 60 * 60


def get_options():
    parser = OptionParser()
    parser.add_option("-d", "--data-file", dest="datafile",
                    help="Csv file with predicted event id and probability")
    parser.add_option("-t", "--time-file", dest="timefile",
                    help="Csv file with time of events")
    parser.add_option("-o", "--out-file", dest="outfile",
                    help="The name of output file")
    parser.add_option("--epsilon", dest="epsilon", type="int", help="DBScan epsilon")
    parser.add_option("--min-samples", dest="min_samples", type="int", help="DBScan min samples")

    (options, args) = parser.parse_args()

    if not options.datafile or not options.timefile or not options.outfile \
            or not options.epsilon or not options.min_samples:
        raise KeyError('Not all required options specified')

    return options


def load_data(datafile, timefile):
    data = []
    time = []

    with open(datafile, 'rb') as input:
        csvreader = csv.reader(input, delimiter=',')

        for row in csvreader:
            data.append([int(row[0])])

    with open(timefile, 'rb') as input:
        csvreader = csv.reader(input, delimiter=',')

        for row in csvreader:
            def toSeconds(date):
                return date.hour * 60 * 60 + date.minute * 60 + \
                        date.second + float(date.microsecond) / 1000000

            start_time = datetime.datetime.strptime(row[0], TIME_FORMAT_LONG)
            row[0] = toSeconds(start_time)

            end_time = datetime.datetime.strptime(row[1], TIME_FORMAT_LONG)
            row[1] = toSeconds(end_time)

            time.append(row)

    return data, time


def export_matrix_as_events(matrix, file):
    dt = datetime.datetime(2006, 11, 21)

    events = []
    for row in matrix:
        event = {}
        event['type'] = int(row[0]/2 - 1)
        event['direction'] = int(row[0] % 2)

        start_date = dt + datetime.timedelta(seconds=round(row[1]))
        end_date = dt + datetime.timedelta(seconds=round(row[2]))

        event['start'] = start_date.strftime(TIME_FORMAT_SHORT)
        event['end'] = end_date.strftime(TIME_FORMAT_SHORT)

        events.append(event)

    json.dump(events, file, sort_keys=False, indent=4)


def post_process(datafile, timefile, outfile, epsilon, min_samples):
    data, time = load_data(datafile, timefile)
    #data <==> Y
    #time <==> Y index
    data = np.matrix(data)
    time = np.matrix(time)
    
    all = np.concatenate((data, time), axis=1)
    
    if all.shape[1]!=0:
        res = DBSCAN(eps=epsilon,min_samples=min_samples).\
            fit_predict(all[:, 1] + all[:, 0] * SECONDS_IN_DAY)

        num_clusters = max(res) + 1
        result = np.zeros(shape=(num_clusters, all.shape[1]))

        for cl_id in xrange(0, num_clusters):
            cluster = all[res == cl_id]
            max_index = np.argmax(cluster[:, 1])
            result[cl_id, :] = cluster[max_index, :]

        result = result[result[:, 2].argsort(), :]
        with open(outfile, 'wb') as output:
            export_matrix_as_events(result, output)
    else:
        print("\nNo events have been detected!!\n*********************************\n")
        events = []
        with open(outfile, 'wb') as output:
            json.dump(events, output, sort_keys=False, indent=4)
    
    


def main():
    options = get_options()

    post_process(options.datafile, options.timefile, options.outfile,
                 options.epsilon, options.min_samples)


if __name__ == '__main__':
    main()
