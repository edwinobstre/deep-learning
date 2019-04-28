#!/usr/bin/env python
# coding: utf-8

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy import spatial
import xml.etree.ElementTree as ET
import  csv
import pickle
from geopy.distance import geodesic
from geopy.distance import great_circle
from geopy.distance import vincenty
from map_to_graph import OSMDict
import queue


def get_graph_of_trajs(nodes, height, width):
    lat, lon = nodes[-1, 0], nodes[-1, 1]

    top = lat + height / 2.0
    bottom = lat - height / 2.0
    left = lon - width / 2.0
    right = lon + width / 2.0

    dot_list = []
    line_list = []

    # handling node first
    for i in range(len(nodes)-1, 0, -1):
        if bottom <= nodes[i, 0] and nodes[i, 0] <= top and left <= nodes[i, 1] and nodes[i, 1] <= right:
            dot_list.append([(nodes[i, 0] - bottom) / (top - bottom), (nodes[i, 1] - left) / (right - left)])
        else:
            break

    line_list = line_list + [(dot_list[i], dot_list[i+1]) for i in range(len(dot_list)-1)]

    return dot_list, line_list

#TODO write by yulei
def get_graph_of_trajs_output(nodes, height, width):
    lat, lon = nodes[0, 0], nodes[0, 1]

    top = lat + height / 2.0
    bottom = lat - height / 2.0
    left = lon - width / 2.0
    right = lon + width / 2.0

    dot_list = []

    # handling node first
    for i in range(1, len(nodes)):
        if bottom <= nodes[i, 0] and nodes[i, 0] <= top and left <= nodes[i, 1] and nodes[i, 1] <= right:
            dot_list.append([(nodes[i, 0] - bottom) / (top - bottom), (nodes[i, 1] - left) / (right - left)])
        else:
            break
    return dot_list

def node_from_id(prev_graph_locs, nodes_dict):
    siz = len(prev_graph_locs)
    nodes = []
    for i in range(siz):
        inode = nodes_dict[prev_graph_locs[i]]
        nodes.append([inode[0], inode[1]])

    return np.array(nodes)

#TODO write by yulei
def find_kth_neighbor(k, adjacent_mat, start, index, explore):
    """
    :param k:   search depth
    :param adjacent_mat:    adjacent_mat for node
    :param start:    current_node id
    :param index:   current depth
    :param explore:     nodes passed
    :return:
    """
    explored = explore
    res = []
    q = queue.Queue()
    q.put(start)
    while not q.empty():
        if(index == k+1):
            break
        temp = []
        length = q.qsize()
        for i in range(length):
            current_node = q.get()
            temp.append(current_node)
            explored.add(current_node)
            neighbors = adjacent_mat[current_node]
            for neighbor in neighbors:
                if not (neighbor in explored):
                    q.put(neighbor)
        index = index + 1
        res.append(temp)
    return res





def traj_info_output(node_dict, infile, mode='node_by_node', type_dist='euclid', idx_traj=1):
    """  structure: 1,2,4,6,8,12,7,23

    :param node_dict:   dict of graph nodes
    :param infile:      original trajectory file
    :param outfile:     output trajectory file in form of graph nodes
    :param mode:        'node_by_node'->discretized by nodes;
                        'node_by_edge'->disc.. by ways (NO consider now)
    :type_dist:         'euclid' calculate euclidean distance;
                        'geo' calculate geospatial distance.
    :return:            traj_discretized, traj_origin
    """
    trajs = pd.read_csv(infile, delimiter='\t')
    # num_trajs = max(trajs['traj_id']) - 100
    num_trajs = 1
    
    ''' dictionary to dataframe '''
    nodeArray = pd.DataFrame.from_dict(node_dict, orient='index', columns=['lat', 'lon'])

    '''dictionary for the id_pick_traj '''
    idx = trajs.index[trajs['traj_id'] == idx_traj]
    geo = pd.DataFrame({
        'lat': trajs['lat_start'][idx],
        'lon': trajs['lon_start'][idx]
    })

    geo_pnts = geo.to_numpy()
    dic_pnts = nodeArray[['lat', 'lon']]

    ''' discretization '''
    if mode == 'node_by_node':
        traj_discretized = []
        ''' loop all trajs '''
        for id in range(num_trajs):
            ''' for each traj calc distance to get the nearest point from graph '''
            if type_dist == 'euclid':
                dic_pnts = dic_pnts.to_numpy()
                result = spatial.distance.cdist(geo_pnts, dic_pnts)
                idx = result.argmin(axis=1)

                # get the nearest point from the graph
                traj_discretized = nodeArray.index[idx]
                continue

            if type_dist == 'geo':
                # ''' build-in function: too-slow '''
                # query = list(zip(geo.lat, geo.lon))
                # nodeArray['coord'] = list(zip(nodeArray.lat, nodeArray.lon))
                # mat_ = pd.DataFrame(np.zeros((geo.shape[0], nodeArray.shape[0])),
                #                     index=geo.index, columns=nodeArray.index)
                #
                # def get_distance(mat_):
                #     geo_coord = query[mat_.name]
                #     print(mat_.name)
                #     return nodeArray['coord'].apply(vincenty, args=(geo_coord,), ellipsoid='WGS-84')
                #
                # result = mat_.apply(get_distance, axis=1)

                ''' for-loop function '''
                result = []
                for j in range(geo.shape[0]):
                    dic_pnts['lat_diff'] = geo.loc[geo.index[j], 'lat'] - dic_pnts['lat']
                    dic_pnts['lon_diff'] = geo.loc[geo.index[j], 'lon'] - dic_pnts['lon']
                    idx_diff = dic_pnts.loc[(dic_pnts['lat_diff'].abs() < 0.003) & (dic_pnts['lon_diff'].abs() < 0.003)]
                    comp_pnts = list(zip(idx_diff.lat, idx_diff.lon))

                    idx_temp = 0
                    dist_min = 999999
                    query = (geo.loc[geo.index[j], 'lat'], geo.loc[geo.index[j], 'lon'])
                    for i in range(len(comp_pnts)):
                        dist = geodesic(query, comp_pnts[i])
                        if dist < dist_min:
                            dist_min = dist
                            idx_temp = i

                    traj_discretized.append(idx_diff.index[idx_temp])

    return traj_discretized, geo_pnts

# TODO adjust mapped node
def check_current_node(iter, node_dict_ways, way_dict_nodes, trajs_disc):
    inode_prev = trajs_disc[iter-1]
    way_prev = node_dict_ways[inode_prev]

    inode_curr = trajs_disc[iter]
    way_curr = node_dict_ways[inode_curr]

    if trajs_disc[iter+1] is not None:
        inode_next1 = trajs_disc[iter+1]
        way_next1 = node_dict_ways[inode_next1]
        if len(way_next1) == 1:
            if way_prev == way_next1 and way_curr == way_prev:
                return inode_curr
            if way_prev == way_next1 and way_curr != way_prev:
                return inode_next1
    elif trajs_disc[iter+2] is not None:
        inode_next2 = trajs_disc[iter+2]
        way_next2 = node_dict_ways[inode_next2]
        if len(way_next2) == 1:
            if way_prev == way_next2 and way_curr == way_prev:
                return inode_curr
            if way_prev == way_next2 and way_curr != way_prev:
                return inode_next2
    elif trajs_disc[iter+3] is not None:
        inode_next3 = trajs_disc[iter+3]
        way_next3 = node_dict_ways[inode_next3]
        if len(way_next3) == 1:
            if way_prev == way_next3 and way_curr == way_prev:
                return inode_curr
            if way_prev == way_next3 and way_curr != way_prev:
                return inode_next3

    return inode_curr



def plot_graph_slidewindow(osm_dict, nodes_dict, trajs_disc, trajs_coord, height, width):

    adjacent_mat = osm_dict.get_adjacent_dict()
    node_dict_ways = osm_dict.get_nodeways_dict()
    way_dict_nodes = osm_dict.get_waynodes_dict()

    for iter in range(42, trajs_coord.shape[0] - 5): # pick nodes every 3-(changeable) nodes

        # Step One : get the current section: current node and previous ones
        prev_graph_locs = trajs_disc[:iter + 1]

        spatial_locs = node_from_id(prev_graph_locs, nodes_dict)
        dots_section, lines_section = get_graph_of_trajs(spatial_locs, height, width)

        # Step Two : check and get current node from the trajectory
        inode = trajs_disc[iter]

        # ouput filter
        label = trajs_disc[iter+1]
        explored = set()
        k = 3
        for i in range(len(prev_graph_locs)-2,len(prev_graph_locs)-k-2, -1):
            explored.add(prev_graph_locs[i])

        # get the potential output nodes id and locs.
        res = find_kth_neighbor(k, adjacent_mat, inode, 0, explored)
        potential_output_id = []
        size = []
        for i in range(0, len(res)):
            size.append(len(res[i]))
            for j in range(len(res[i])):
                potential_output_id.append(res[i][j])

        # only plot sections with more than 4 nodes
        if len(dots_section) < 4:
            continue

        # only plot in k layers label
        if label not in potential_output_id:
            continue

        potential_output_id.append(label)

        # Step Three : get and plot background dots and lines
        dots_window, lines_window, = osm_dict.plot_graph_from_node(inode, height, width)
        for i in range(len(lines_window,)):
            plt.plot((lines_window[i][0][1], lines_window[i][1][1]),
                     (lines_window[i][0][0], lines_window[i][1][0]), color='0.75') # 'g-'

        dots_window = np.array(dots_window)
        plt.plot(dots_window[:, 1], dots_window[:, 0], 'bo', alpha=.3, ms=5)

        # Step Four : draw graph node of section (on the top of the background)
        for i in range(len(lines_section)):
            plt.plot((lines_section[i][0][1], lines_section[i][1][1]),
                     (lines_section[i][0][0], lines_section[i][1][0]), 'ko-', linewidth=2)

        dots_section = np.array(dots_section)
        plt.plot(dots_section[:, 1], dots_section[:, 0], 'ks-', alpha=.3, ms=3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # Step Five : plot real-traj section (read) leading to this node
        prev_locs = trajs_coord[:iter + 1]
        dots_real, lines_real = get_graph_of_trajs(prev_locs, height, width)
        for i in range(len(lines_real)):
            plt.plot((lines_real[i][0][1], lines_real[i][1][1]),
                     (lines_real[i][0][0], lines_real[i][1][0]), 'ro-', linewidth=1)

        dots_real = np.array(dots_real)
        plt.plot(dots_real[:, 1], dots_real[:, 0], 'rs')


        # no borader and axis
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('../../data/output_img_euclid/x_/' + str(iter) + '.png', bbox_inches='tight', pad_inches=0)
        # plt.title(iter)
        # plt.show()
        plt.clf()
        plt.close()

        # Step Six : get and plot true label and make a explored set
        plt.figure()

        start = []
        temp = 0
        for i in range(len(size)):
            temp = temp + size[i]
            start.append(temp)
        spatial_output_locs = node_from_id(potential_output_id, nodes_dict)
        dots_output_section = get_graph_of_trajs_output(spatial_output_locs, height, width)
        # only plot sections with existing nodes

        # no borader and axis
        dots_output_section = np.array(dots_output_section)

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        for i in range(1,len(start)):
            if i == 1:
                plt.plot(dots_output_section[start[i-1]-1:start[i]-1, 1], dots_output_section[start[i-1]-1:start[i]-1, 0],
                         'ro', ms=15)
            elif i == 2:
                plt.plot(dots_output_section[start[i - 1]-1:start[i]-1, 1], dots_output_section[start[i - 1]-1:start[i]-1, 0],
                         'yo', ms=10)
            elif i == 3:
                    plt.plot(dots_output_section[start[i - 1]-1:start[i]-1, 1],
                             dots_output_section[start[i - 1]-1:start[i]-1, 0],
                             'bo', ms=8)
        plt.plot(dots_output_section[-1, 1], dots_output_section[-1:, 0], 'ko', ms=5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig('../../data/output_img_euclid/label/' + str(iter) + '.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        print(str(iter))

def main():
    # original traj_gis and openstreetmap_graph file
    gisfile = '../../data/processed/gis-trajs-tims.csv'
    graphfile = '../../data/OSM_output/graph_info_dump'

    # load graph info from pickle (especially, node info)
    osm_dict = pickle.load(open(graphfile, 'rb'))
    osm_dict.build_all()
    nodes_dict = osm_dict.get_node_dict()

    # set sliding window size
    height = 0.008
    width = 0.008
    # discretize trajectories into graph path
    trajs_disc, trajs_coord = traj_info_output(nodes_dict, gisfile, 'node_by_node', 'euclid', 1) # euclid
    # plot final figure
    plot_graph_slidewindow(osm_dict, nodes_dict, trajs_disc, trajs_coord, height, width)


if __name__ == "__main__":
    main()



