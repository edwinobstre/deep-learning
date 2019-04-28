#!/usr/bin/env python
# coding: utf-8

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import scipy
import xml.etree.ElementTree as ET
import  csv

''' parsing open street map and extract node and way information '''
class OSMDict(object):
    
    def __init__(self, xmlfile):

        self.node_num = 0
        self.way_num = 0

        self.node_loc_dict = {} # store all graph nodes
        self.node_way_dict = {} # store all graph nodes

        self.way_dict = {} # store all ways(one way contains several nodes)
        self.graph_mat = {} # store graph information (nodes and adjacent ones)
        
        # create element tree object for xml file parsing
        tree = ET.parse(xmlfile) 
        root = tree.getroot() 
        self.root = root
        
    def insert_way_to_node(self, node, way_id):
        """ func to add way_id to an existing node
            node-{way1, way2, way3, ...}

        :param node:    node as the key 
        :param way_id:  dict-value
        :return:        dict
        """
        if node in self.node_way_dict.keys():
            self.node_way_dict[node].append(way_id)
        else:
            self.node_way_dict[node] = []
            self.node_way_dict[node].append(way_id)

        return
            
    def get_node_dict(self):
        return self.node_loc_dict

    def get_adjacent_dict(self):
        return self.graph_mat

    def get_nodeways_dict(self):
        return self.node_way_dict

    def get_waynodes_dict(self):
        return self.way_dict

    def node_graph_num(self):
        """ func to return the total number of nodes on the graph

        :return:    number-of-nodes
        """
        num_node = len(self.node_loc_dict) 

        return num_node 
    
    def way_graph_num(self):
        """ func to return total number of ways from the map/graph

        :return:    number-of-ways
        """
        way_num = len(self.way_dict)
        return way_num 

    def build_all(self):
        """build all three dictionaries"""

        self.build_node_dict()
        self.build_way_dict()
        self.build_graph()

        return 
    
    def build_node_dict(self):
        """create dict for node {node_id: lat, lon}

        :return: 
        """
        for node in self.root.findall("node"):
            id = node.get('id')
            lat = float(node.get('lat'))
            lon = float(node.get('lon'))
            self.node_loc_dict[id] = [lat, lon]
        
        return

            
    def build_way_dict(self):
        """create dict for way {way_id: pnt_id_1, pnt_id_2....},
            check in between if a node has this way as a value in itself dict

        :return:
        """
        for edge in self.root.findall("way"):
            wayid = edge.get('id')
            node_seq = []
            
            for subedge in edge.findall('nd'):
                nd = subedge.get('ref')
                node_seq.append(nd)
                self.insert_way_to_node(nd, wayid)
            self.way_dict[wayid] = node_seq
        
        return
    
    def build_graph(self):
        """ generate graph matrix:  adjacent matrix
        :return: 
        """
        for iway in self.root.findall("way"):

            # get all nodes from a way to a list
            node_list = []
            for inode in iway.findall('nd'):
                nd = inode.get('ref')
                node_list.append(nd)
            

            # convert the chain from a way to graph edges
            for i, node in enumerate(node_list):
                
                # if  key not in dictionary: create new key
                if node not in self.graph_mat.keys():
                    self.graph_mat[node] = []

                if i < len(node_list) - 1:
                    if (node_list[i+1] not in self.graph_mat[node]):
                        self.graph_mat[node].append(node_list[i+1])
 #                   else: 
 #                       print('node duplicate %s found in neighbors of %s (1st if):' % (node_list[i+1], node))

                if i > 0:
                    if (node_list[i-1] not in self.graph_mat[node]):
                        self.graph_mat[node].append(node_list[i-1])
 #                   else:
 #                       print('node duplicate %s in neighbors of %s (2nd if):' % (node_list[i-1], node))

        return


    def plot_graph_in_window(self, topleft, bottomright):
        """ Draw nodes-edges of graph in a window.

        :param topleft: gps location of the top left corner
        :param bottomright:     bottom right corner
        :return:        an matrix, which is an image 
        """
       
        # to be implemented

        return None

    def plot_graph_from_node_label(self, node, height, width, order=3):
        lat, lon = self.node_loc_dict[node]

        top = lat + height / 2.0
        bottom = lat - height / 2.0
        left = lon - width / 2.0
        right = lon + width / 2.0

        def in_range(loc):
            flag = bottom <= loc[0] and loc[0] <= top and left <= loc[1] and loc[1] <= right
            return flag

        def loc_to_pos(loc):
            pos = [(loc[0] - bottom) / (top - bottom), (loc[1] - left) / (right - left)]
            return pos

            # traversal the graph within the window

        buff = [node]

        # store dots and lines to be plotted
        dot_list = []
        iterator = 0

        order_monitor = 0
        while iterator < len(buff):

            front = buff[iterator]
            iterator = iterator + 1

            # expand the fronterior
            neighbors = self.graph_mat[front]
            candidates = []

            for nei in neighbors:
                if in_range(self.node_loc_dict[nei]) and nei not in buff:
                    candidates.append(nei)

            buff = buff + candidates
            order_monitor += 1

            if order_monitor == order:
                break

        # handling this node
        for i in range(len(buff)):
            front = buff[i]
            front_loc = self.node_loc_dict[front]

            front_pos = loc_to_pos(front_loc)
            dot_list.append(front_pos)

        print('len(buff), ', len(dot_list))

        return dot_list[1:]

    def plot_graph_from_node(self, node, height, width):
        """ Draw nodes-edges of graph in a window.

        :param node: node id at the center of the plot
        :param height: height of the plot window 
        :param width:  width of the plot window 
        :return: 
            dot_list: a list of positions of dots to be plotted
            line_list: a list of tuples, each of which contains the two ends of a line section to be plotted
        """

        lat, lon = self.node_loc_dict[node] 
        
        top = lat + height / 2.0
        bottom = lat - height / 2.0
        left = lon - width / 2.0
        right = lon + width / 2.0

        def in_range(loc): 
            flag = bottom <= loc[0] and loc[0] <= top and left <= loc[1] and loc[1] <= right 
            return flag 
        
        def loc_to_pos(loc):
            pos = [(loc[0] - bottom) / (top - bottom), (loc[1] - left) / (right - left)]
            return pos

        
        # traversal the graph within the window

        buff = [node]

        # store dots and lines to be plotted
        dot_list = []
        line_list = []
        iterator = 0
        
        while iterator < len(buff):
            
            front = buff[iterator]
            iterator = iterator + 1

            if front == '61462919':
                a=1
            if front == '61462950':
                a = 1

            # expand the fronterior
            neighbors = self.graph_mat[front]
            candidates = []

            for nei in neighbors:
                if in_range(self.node_loc_dict[nei]) and nei not in buff:
                    candidates.append(nei)

            # for nei in neighbors:
            #     if in_range(self.node_loc_dict[nei]):
            #         if nei not in buff:
            #             candidates.append(nei)
            #
            #         if nei in buff and buff.index(nei) < iterator-1:
            #             candidates.append(nei)

            buff = buff + candidates

            # handling this node 
            front_loc = self.node_loc_dict[front]

#           print('location:', front_loc, ', and buffer size:', len(buff))
            front_pos = loc_to_pos(front_loc)
            dot_list.append(front_pos)

            nei_pos = [loc_to_pos(self.node_loc_dict[nd]) for nd in neighbors]
            line_list = line_list + [(front_pos, single_nei_pos) for single_nei_pos in nei_pos]

        return dot_list, line_list 


if __name__ == "__main__":
   
    datapath = '../../data/OSM_output/'

    osm_dict = OSMDict(datapath + 'interpreter')
    
    osm_dict.build_all()
      
    node = '2931780651' 
    height = 0.002
    width = 0.002
    dots, lines = osm_dict.plot_graph_from_node(node, height, width)

#    print(dots)
#    print(lines)


    # pickle dump the object into a file and read it back later

