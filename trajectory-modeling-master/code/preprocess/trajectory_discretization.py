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

from map_to_graph import OSMDict

def plot_graph_of_trajs(nodes, height, width):
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

def node_from_id(prev_graph_locs, nodes_dict):
    siz = len(prev_graph_locs)
    nodes = []
    for i in range(siz):
        inode = nodes_dict[prev_graph_locs[i]]
        nodes.append([inode[0], inode[1]])

    return np.array(nodes)

def traj_info_output(node_dict, infile, outfile, mode='node_by_node', id_pick_traj=1):
    """  structure: 1,2,4,6,8,12,7,23

    :param node_dict:   dict of graph nodes
    :param infile:      original trajectory file
    :param outfile:     output trajectory file in form of graph nodes
    :param mode:        'node_by_node'->discretized by nodes;
                        'node_by_edge'->disc.. by ways (NO consider now)
    :return:            no-return
    """
    trajs = pd.read_csv(infile, delimiter='\t')
    # num_trajs = max(trajs['traj_id']) - 100
    num_trajs = 1
    
    DIST_DELTA = 0.02
    DIST_MIN = 9999
    
    ''' dictionary to list '''
    nodeArray = []
    for key, value in node_dict.items():
        temp = [key, value[0], value[1]]
        nodeArray.append(temp)
    nodeArray = np.array(nodeArray)

    '''dictionary for the id_pick_traj '''
    idx = trajs.index[trajs['traj_id'] == id_pick_traj].tolist()
    lat, lon = trajs['lat_start'][idx], trajs['lon_start'][idx]
    traj_coords = np.array([lat, lon]).T
    # print(traj_coords.shape) # (1053, 2)
    
    ''' discretization '''
    if mode == 'node_by_node':
        traj_discretized = []
        with open(outfile, 'w') as f:
            ''' loop all trajs '''
            for id in range(num_trajs):
                ''' for each traj '''
                idx = trajs.index[trajs['traj_id'] == id+1].tolist()

                # one location on the traj
                lat, lon = trajs['lat_start'][idx], trajs['lon_start'][idx]
                geo_pnts = np.array([lat, lon]).T

                # calc distance to get the nearest point from graph
                dic_pnts = nodeArray[:, 1:].astype(np.float)
                # TODO euclidean distance to spatial distance
                result = spatial.distance.cdist(geo_pnts, dic_pnts)
                idx = result.argmin(1)

                # get the nearest point from the graph
                nodes_disc = nodeArray[idx, 0].tolist()

                traj_discretized.append(nodes_disc)

    # print(len(traj_discretized[0])) # 1053

    return traj_discretized, traj_coords


def main():
    # ## Dictionary Generation

    # original traj_gis and openstreetmap_graph file
    gisfile = '../../data/processed/gis-trajs-tims.csv'
    graphfile = '../../data/OSM_output/graph_info_dump'

    # load graph info from pickle (especially, node info)
    osm_dict = pickle.load(open(graphfile, 'rb'))
    osm_dict.build_all()
    nodes_dict = osm_dict.get_node_dict()
    # output final descretized trajs
    trajfile = '../../data/processed/traj_vector.txt'


    # discretize trajectories into graph path
    trajs_disc, trajs_coord = traj_info_output(nodes_dict, gisfile, trajfile, mode='node_by_node', id_pick_traj=1)

    # plot the map
    height = 0.008
    width = 0.008
    for iter in range(trajs_coord.shape[0] - 5):
        iter = iter + 4
        # 1. pick a node from the trajectory
        # for the first trajectory, pick nodes every n-(changeable) nodes
        inode = trajs_disc[0][iter]
        dots, lines = osm_dict.plot_graph_from_node(inode, height, width)

        # 2. plot the map around the node # implemented
        #
        for i in range(len(lines)):
            plt.plot((lines[i][0][0], lines[i][1][0]), (lines[i][0][1], lines[i][1][1]), 'g-')

        dots = np.array(dots)
        plt.plot(dots[:, 0], dots[:, 1], 'bs')

        # 3.1 plot the trajectory (read) leading to this node
        # all locations before and including the current one
        prev_locs = trajs_coord[:iter+1]
        dots2, lines2 = plot_graph_of_trajs(prev_locs, height, width)

        for i in range(len(lines2)):
            plt.plot((lines2[i][0][0], lines2[i][1][0]), (lines2[i][0][1], lines2[i][1][1]), 'ro-', linewidth=2)

        dots2 = np.array(dots2)
        plt.plot(dots2[:, 0], dots2[:, 1], 'rs')


        # 3.2 plot the trajectory (map to graph) leading to this node
        prev_graph_locs = trajs_disc[0][:iter+1]
        spatial_locs = node_from_id(prev_graph_locs, nodes_dict)
        dots3, lines3 = plot_graph_of_trajs(spatial_locs, height, width)

        for i in range(len(lines3)):
            plt.plot((lines3[i][0][0], lines3[i][1][0]), (lines3[i][0][1], lines3[i][1][1]), 'ro-', linewidth=2)

        dots3 = np.array(dots3)
        plt.plot(dots3[:, 0], dots3[:, 1], 'ko-')

        plt.show()

        # 4. save the image


if __name__ == "__main__":
    main()



