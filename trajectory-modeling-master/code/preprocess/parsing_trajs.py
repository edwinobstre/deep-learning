"""
Rui Chen
Oct. 2018

    =============================================
    Month: 4/5/6, each containing data of 30 days
    =============================================

    data 04-01
    2018-04-01 00:00:00/02-23:59:50/52/54/56
    file00:  [1522540800, 1522627190]
    file89:  [1522540800, 1522627192]
    file125: [1522540802, 1522627196]
    file127: [1522540800, 1522627197]
    =============================================
    data 04-02
    2018-04-02 00:00:01-23:59:46/53
    file00:  [1522627201, 1522713586]
    file65:  [1522627201, 1522713593]
    =============================================
    data 05-01
    2018-05-01 00:00:00-23:59:59
    file00:  [1525132800, 1525219199]
    =============================================
"""

"""
=========================================================
    #utim = int(utim)
    #dateArray = datetime.datetime.utcfromtimestamp(utim)
    ## otherStyletime == "2013-10-10 23:40:00
    #timUTC = dateArray.strftime("%Y-%m-%d %H:%M:%S")
    #timLst.append(timUTC)
==========================================================
"""
from datetime import datetime

import pandas as pd
import numpy as np
from bisect import bisect_left
import csv

# Case one
NUM = 10
# Case two
num_path = 10
pts_per_path = 120
# range - square length
# Notice, no more than 0.02
lgh = 0.005

class DataParsing:
    def __init__(self, csv_data, row, col):
        self.data_matrix = csv_data
        self.data_row = row
        self.data_col = col
        self.diff_dist = 0.005
        self.posLst = 0
        self.utcLst = 0

        """ 
            AABB represents the Axis-aligned minimum bounding box
            The whole covering area.   
            [left, bottom right, top]
            [min_lon, min_lat, max_lon, max_lat] 
        """
        self.AABB = []

    def utm_paring(self, tim_long):
        # otherStyletime == "2013-10-10 23:40:00
        # timUTC = dateArray.strftime("%Y-%m-%d %H:%M:%S")
        self.tim_utc = datetime.utcfromtimestamp(tim_long)
        self.tim_hours = self.tim_utc.hour + \
                         self.tim_utc.minute / 60. + \
                         self.tim_utc.second / 3600.

        return self.tim_hours

    def trajectory_filter(self, x, y):
        # implicitly delete those with only one point/position
        if abs(max(x) - min(x)) > self.diff_dist and \
                abs(max(y) - min(y)) > self.diff_dist:
            return True

    def data_process(self):
        # caid//latitude//longitude//horizontal_accuracy//id_type
        # //utc_timestamp//geo_hash
        posLst = []
        tem_pos = []
        xpos, ypos= [], []
        utcLst, utmp= [], []
        dict_caid = {}

        flag = 0
        lat_all_list, lon_all_list = [], []
        for it in range(row):
            # row by row
            trac_row = self.data_matrix.iloc[it]
            caid, latd, lngd, hacc, ityp, uint, geoP = trac_row
            timh = self.utm_paring(uint)

            """ store all coordinates to get the AABB box """
            lat_all_list.append(latd)
            lon_all_list.append(lngd)

            if caid in dict_caid.keys():
                xpos.append(lngd)
                ypos.append(latd)
                utmp.append(timh)
            else:
                # exception when input the first row
                if it == 0:
                    xpos.append(lngd)
                    ypos.append(latd)
                    utmp.append(timh)
                else:
                    # only keep trajectories with expanded area
                    if self.trajectory_filter(xpos, ypos):
                        # tem_pos = [xpos, ypos] is not good for
                        # later rank selecting
                        tem_pos = xpos + ypos
                        posLst.append(tem_pos)
                        utcLst.append(utmp)
                    # tem_pos = []
                    xpos = []
                    ypos = []
                    utmp = []

                    xpos.append(lngd)
                    ypos.append(latd)
                    utmp.append(timh)

                dict_caid[caid] = flag
                flag = flag + 1

        self.posLst = posLst
        self.utcLst = utcLst

        """ get the AABB box """
        self.AABB = [min(lon_all_list), max(lat_all_list), max(lon_all_list), min(lat_all_list)]
        print(self.AABB)

        return posLst, utcLst


def output_list_to_gis(outfile, trajs, tims):

    id_ = 1
    with open(outfile, "w") as f:
        siz = len(trajs)
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['section_id', 'traj_id', 'time', 'lon_start', 'lat_start', 'lon_end', 'lat_end', 'color_idx'])

        fid = 1
        for iter in np.arange(siz):
            nodes_ = len(trajs[iter]) // 2
            x_trajs = trajs[iter][:nodes_]
            y_trajs = trajs[iter][nodes_:]
            minutes = tims[iter]

            for it in np.arange(nodes_-1):
                ''' future usage for color assignment '''
                clr_idx = minutes[it] // 2

                content = [id_, fid, round(minutes[it+1]-minutes[it], 5),
                           round(x_trajs[it], 5), round(y_trajs[it], 5),
                           round(x_trajs[it+1], 5), round(y_trajs[it+1], 5),
                           round(clr_idx)]
                writer.writerow(content)
                id_ = id_ + 1

            fid = fid + 1

def output_list_to_RNN(out_data, out_map_node, out_map_edge, trajs):
    id_ = 1
    siz = len(trajs)

    nodeDicts = {}
    cnts = 0
    for iter in np.arange(siz):
        nodes_ = len(trajs[iter]) // 2
        x_trajs = trajs[iter][:nodes_]
        y_trajs = trajs[iter][nodes_:]
        for it in np.arange(nodes_ - 1):
            if cnts not in nodeDicts.keys():
                nodeDicts[cnts] = [x_trajs[it], y_trajs[it]]
                cnts += 1


    outfile_data_and_edge(out_data, out_map_edge, trajs, nodeDicts)
    outfile_map_node(out_map_node, nodeDicts)
    return 0

def outfile_map_node(filename, nodeDicts):
    file = open(filename, 'w')
    for k, v in nodeDicts.items():
        file.write(str(k) + '\t' + str(v[0]) + '\t' + str(v[1]) + '\t\n')

def outfile_data_and_edge(datafile, edgefile, trajs, nodeDicts):
    datafile = open(datafile, 'w')
    edgefile = open(edgefile, 'w')

    # divide each trajectory to 10 sections from 1-10
    # 5 from 11-50
    # 2 from 51-100
    edgeID = []
    trajID = []
    for iter in np.arange(100):
        nodes_ = len(trajs[iter]) // 2
        x_trajs = trajs[iter][:nodes_]
        y_trajs = trajs[iter][nodes_:]
        y_trajs = trajs[iter][nodes_:]
        if iter < 9:
            sects = 10
            for it in np.arange(sects):
                sectID = nodes_ // sects
                x_coords = x_trajs[it:it+sectID]
                y_coords = y_trajs[it:it+sectID]
                startNode = find_nodeID(nodeDicts, [x_trajs[it * sectID], y_trajs[it * sectID]])
                endNode = find_nodeID(nodeDicts, [x_trajs[(it + 1) + sectID], y_trajs[(it + 1) + sectID]])
                edgefile.write(edgeID)
                edgeID.append(it)

                edgefile.write(startNode)
                edgefile.write(endNode)
                edgefile.write(endNode - startNode + 1)
                # edgefile.write(x_coords[it], y_coords[it] for i in np.arange())

            trajID.append(edgeID)
        elif it < 49:
            a = 0
        else:
            a = 0


def find_nodeID(dicts, value):
    for ikey, ivalue in dicts.items():
        if ivalue == value:
            return ikey

def output_list_to_csv(outfile, params):

    with open(outfile, "w") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(params)


if __name__ == '__main__':

    csv_data = pd.read_csv(
        '../../data/subset/part-00000-tid-8141827983377314098-d791bf0d-7d99-4c04-a5e7-73de919ac77f-27221-c000.csv')
    row, col = csv_data.shape

    data_model = DataParsing(csv_data, row, col)
    posLst, utcLst = data_model.data_process()

    # traj_all_order = sorted(posLst, key=len, reverse=True)
    traj_all_order = sorted(posLst, key=len)
    time_all_order = sorted(utcLst, key=len)

    # ranking from maximum node to minimum
    traj_all_order.reverse()
    time_all_order.reverse()

    name_file = '../../data/processed/gis-trajs-tims.csv'
    output_list_to_gis(name_file, traj_all_order, time_all_order)
    
    name_file = '../../data/processed/trajs-order.csv'
    output_list_to_csv(name_file, traj_all_order)
    
    name_file = '../../data/processed/time-order.csv'
    output_list_to_csv(name_file, time_all_order)
"""
    out_data = '../RNN-TrajModel/data-RNN/trajs-whole/data/drive.txt'
    out_map_edge = '../RNN-TrajModel/data-RNN/trajs-whole/map/edgeOSM.txt'
    out_map_node = '../RNN-TrajModel/data-RNN/trajs-whole/map/nodeOSM.txt'
    # output_list_to_RNN(out_data, out_map_node, out_map_edge, traj_all_order)
"""
