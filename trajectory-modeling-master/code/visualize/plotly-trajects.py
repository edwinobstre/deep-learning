from datetime import datetime

import pandas as pd
import numpy as np
from bisect import bisect_left
import csv


import os
import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


import plotly.graph_objs as go

import plotly.plotly as py
import plotly.figure_factory as ff

mapbox_access_token = 'pk.eyJ1Ijoib2JiaWUiLCJhIjoiY2pvNGQycWV3MTZqNDNrcWNvNWE2dHY0eSJ9.eUl3y7PFLpyIw51XjFYxhg'

# Case one
NUM = 10
# Case two
num_path = 10
pts_per_path = 120
# range - square length
# Notice, no more than 0.02
lgh = 0.005


class DrawTrajectories:
    def __init__(self, time_lating, positions, siz_node):
        self.tim_lasting = time_lating
        self.pos_to_draw = positions
        self.siz_nodes = siz_node
        self.tim_diff = 0.3
        self.lon_diff = 0.0015
        self.lat_diff = 0.0015
        self.lat_filter_diff = 0.004
        self.lon_filter_diff = 0.004

    def plt_all(self): # tim, pos2draw, siz
        # draw separate figures when cited
        plt.figure()
        ax = plt.axes(projection=ccrs.Mercator())  # PlateCarree
        # ax.coastlines(resolution='10m')
        ax.coastlines()
        # ax.stock_img()

        for i in range(self.siz_nodes):
            num = len(self.pos_to_draw[i]) // 2
            plt.plot(self.pos_to_draw[i][:len(self.pos_to_draw[i]) // 2],
                     self.pos_to_draw[i][len(self.pos_to_draw[i]) // 2:],
                     marker='.', alpha=0.5,
                     transform=ccrs.PlateCarree(),
                     label="t={0}".format(num)
                     )

        leg = ax.legend(loc="upper left", bbox_to_anchor=[0, 1],
                        ncol=2, shadow=True, title="points number per path", fancybox=True)
        # leg.get_title().set_color("red")

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = False
        gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'gray'}
        gl.xlabel_style = {'color': 'red', 'weight': 'bold'}

        plt.show(block=False)

    def plt_all_combined(self):
        plt.figure()

        for i in range(self.siz_nodes):
            num = len(self.tim_lasting[i])
            plt.subplot(self.siz_nodes, 1, i + 1)
            plt.plot(self.tim_lasting[i], self.pos_to_draw[i][len(self.pos_to_draw[i]) // 2:],
                     marker='.', alpha=0.5,
                     label="t={0}".format(num),
                     )
            plt.title("t={0}".format(num))
            plt.subplots_adjust(hspace=1)
            plt.tight_layout()
            # plt.ylim(42.2, 42.6)
            plt.xlim(0, 24)

        plt.figure()
        # plt.ylim(42, 43)
        # plt.xlim(0, 180)
        for i in range(self.siz_nodes):
            num = len(self.tim_lasting[i])
            plt.subplot(self.siz_nodes, 1, i + 1)
            plt.plot(self.tim_lasting[i], self.pos_to_draw[i][:len(self.pos_to_draw[i]) // 2],
                     marker='.', alpha=0.5,
                     label="t={0}".format(num),
                     )
            plt.title("t={0}".format(num))
            plt.subplots_adjust(hspace=1)
            plt.tight_layout()

        plt.show(block=False)

    def traj_filter(self, i_lon, i_lat):
        max_lon = max(i_lon)
        min_lon = min(i_lon)
        max_lat = max(i_lat)
        min_lat = min(i_lat)

        siz_value = len(i_lat)

        if ((max_lat - min_lat < self.lat_filter_diff) and
           (max_lon - min_lon < self.lon_filter_diff) and
           (siz_value > 10)):
            return True
        else:
            return False



    def itraj_section_dist(self, lon, lat, clr):
        lon_within_time = np.array(lon)
        lat_within_time = np.array(lat)

        lon_diff = lon_within_time[1:] - lon_within_time[:-1]
        lat_diff = lat_within_time[1:] - lat_within_time[:-1]
        idx_lon = np.argwhere(np.abs(lon_diff) > self.lon_diff)
        idx_lat = np.argwhere(np.abs(lat_diff) > self.lat_diff)

        idx = np.concatenate((idx_lon, idx_lat))

        idx = np.unique(np.sort(idx.transpose()))

        beg_flag = 0
        time_sect = []
        time_all_sects = []
        siz_iter = idx.size
        for it in np.arange(siz_iter + 1):

            if it == siz_iter:
                i_lon = lon[beg_flag:]
                i_lat = lat[beg_flag:]

                if self.traj_filter(i_lon, i_lat):
                    continue

                time_sect = [dict(
                    type='scattermapbox',
                    lon=i_lon,
                    lat=i_lat,
                    mode='lines',
                    line=dict(
                        width=1.5,
                        color=clr,
                    ),
                    # opacity=0.8,
                )
                ]
                time_all_sects = time_all_sects + time_sect
                continue

            end_flag = np.asscalar(idx[it])

            if beg_flag == end_flag:
                beg_flag = beg_flag + 1
                continue

            i_lon = lon[beg_flag:end_flag + 1]
            i_lat = lat[beg_flag:end_flag + 1]

            if self.traj_filter(i_lon, i_lat):
                beg_flag = end_flag + 1
                continue

            time_sect = [dict(
                type='scattermapbox',
                lon=i_lon,
                lat=i_lat,
                mode='lines',
                line=dict(
                    width=1.5,
                    color=clr,
                ),
                # opacity=0.8,
            )
            ]
            beg_flag = end_flag + 1

            time_all_sects = time_all_sects + time_sect

        return time_all_sects




    def itraj_section_time(self, lon, lat, tim, clr):

        tim = np.array(tim)
        tim_diff = tim[1:] - tim[:-1]
        idx_sect = np.argwhere(tim_diff > self.tim_diff)
        siz_iter = idx_sect.size

        beg_flag = 0
        itraj_sect = []
        traj_all_sects = []
        for it in np.arange(siz_iter + 1):

            if it == siz_iter:
                i_lon = lon[beg_flag:]
                i_lat = lat[beg_flag:]
                tim_test = tim[beg_flag:]
                itraj_sect = self.itraj_section_dist(i_lon, i_lat, clr)
                # itraj_sect = [dict(
                #     type='scattermapbox',
                #     lon=i_lon,
                #     lat=i_lat,
                #     mode='lines',
                #     line=dict(
                #         width=1,
                #         color=clr,
                #     ),
                #     opacity=0.8,
                # )
                # ]
                traj_all_sects = traj_all_sects + itraj_sect
                continue

            end_flag = np.asscalar(idx_sect[it])
            tim_test = tim[beg_flag:end_flag]

            if end_flag - beg_flag > self.tim_diff:
                beg_flag = beg_flag + 1

            if beg_flag == end_flag:
                beg_flag = beg_flag + 1
                continue

            i_lon = lon[beg_flag:end_flag + 1]
            i_lat = lat[beg_flag:end_flag + 1]
            itraj_sect = self.itraj_section_dist(i_lon, i_lat, clr)

            # itraj_sect = [dict(
            #     type='scattermapbox',
            #     lon=i_lon,
            #     lat=i_lat,
            #     mode='lines',
            #     line=dict(
            #         width=1,
            #         color=clr,
            #     ),
            #     opacity=0.8,
            # )
            # ]
            beg_flag = end_flag + 1

            traj_all_sects = traj_all_sects + itraj_sect

        return traj_all_sects

    def itrajs_time_distance(self, lon_, lat_, tim_, clr_grad):

        time_series = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
        traj_add = []

        min_tim = np.floor(np.min(tim_)).astype(int)
        max_tim = np.ceil(np.max(tim_)).astype(int)
        min_tim_idx = np.argwhere(min_tim >= time_series)[-1]
        min_tim_start = np.array(time_series)[min_tim_idx]
        max_tim_idx = np.argwhere(max_tim <= time_series)[0]
        max_tim_end = np.array(time_series)[max_tim_idx.astype(int)]

        iter_num = max_tim_idx - min_tim_idx


#########################################################################################
        #
        # time_series = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
        # traj_add = []
        #
        # min_tim = np.floor(np.min(tim_)).astype(int)
        # max_tim = np.ceil(np.max(tim_)).astype(int)
        # min_tim_idx = np.argwhere(min_tim >= time_series)[-1]
        # min_tim_start = np.array(time_series)[min_tim_idx]
        # max_tim_idx = np.argwhere(max_tim <= time_series)[0]
        # max_tim_end = np.array(time_series)[max_tim_idx.astype(int)]
        #
        # iter_num = max_tim_idx - min_tim_idx
        #
        # beg_flag = []
        #
        # beg_flag = 631
        #
        # end_flag = 781
        # i_lon = lon_[beg_flag:end_flag + 1]
        # i_lat = lat_[beg_flag:end_flag + 1]
        # i_tim = tim_[beg_flag:end_flag + 1]
        #
        # itraj_section = self.itraj_section_time(i_lon, i_lat, i_tim, 'rgb(255,0,0)')

        beg_flag = 0
        # beg_flag = np.asscalar(np.argwhere(tim_ >= min_tim_start)[0])
        for it in np.arange(iter_num):
            end_flag = np.asscalar(np.argwhere(tim_ <= min_tim_start + 2 * (it + 1))[-1])
            i_lon = lon_[beg_flag:end_flag + 1]
            i_lat = lat_[beg_flag:end_flag + 1]
            i_tim = tim_[beg_flag:end_flag + 1]
            clr_sect = clr_grad[np.asscalar(min_tim_idx) + it]

            if len(i_tim) <= 1:
                beg_flag = end_flag + 1

                # beg_flag = np.asscalar(np.argwhere(tim_ >= min_tim_start + 2 * it)[0])
                # beg_flag = np.asscalar(np.argwhere(tim_ <= min_tim_start + 2 * (it + 1))[0])
                continue
            itraj_section = self.itraj_section_time(i_lon, i_lat, i_tim, clr_sect)

            # itraj_section = [dict(
            #     type='scattermapbox',
            #     lon=lon_[beg_flag:end_flag],
            #     lat=lat_[beg_flag:end_flag],
            #     mode='lines',
            #     line=dict(
            #         width=1,
            #         color=clr_sect,
            #     ),
            #     opacity=0.8,
            # )
            # ]
            # notice that list[a:b] cannot get value for b; to connect sections
            beg_flag = end_flag + 1
            traj_add = traj_add + itraj_section

        return traj_add


    def plt_top_plotly(self, num_trajectories):

        clr_grad = ['rgb(0,0,255)', 'rgb(137,137,255)', 'rgb(196,196,255)',
                    'rgb(255,177,177)', 'rgb(255,98,98)', 'rgb(255,0,0)',
                    'rgb(255,0,0)', 'rgb(255,98,98)', 'rgb(255,177,177)',
                    'rgb(196,196,255)', 'rgb(137,137,255)', 'rgb(0,0,255)']

        trajs_list = []
        trajs_each = []
        joint_list = []

        lat_avg = 0
        lon_avg = 0

        # partition one trajectory into 12 sections with gradient colors
        for i in range(num_trajectories):
            # num = len(self.tim_lasting[-i - 1])

            siz_coords = len(self.pos_to_draw[-i - 1]) // 2
            lon_ = self.pos_to_draw[-i - 1][:siz_coords]
            lat_ = self.pos_to_draw[-i - 1][siz_coords:]
            tim_ = self.tim_lasting[-i - 1]

            # opacity_idx = siz_coords / (len(self.pos_to_draw[-1]) // 2)

            lat_avg = np.average(lat_)
            lon_avg = np.average(lon_)


            # call data process for each trajectory
            trajs_each = self.itrajs_time_distance(lon_, lat_, tim_, clr_grad)
            joint_list = joint_list + trajs_each

        ## one whole trajectory
        # for i in range(num_trajectories):
        #     # num = len(self.tim_lasting[-i - 1])
        #
        #     siz_coords = len(self.pos_to_draw[-i - 1]) // 2
        #     lon_ = self.pos_to_draw[-i - 1][:siz_coords]
        #     lat_ = self.pos_to_draw[-i - 1][siz_coords:]
        #
        #     # opacity_idx = siz_coords / (len(self.pos_to_draw[-1]) // 2)
        #
        #     lat_avg = np.average(lat_)
        #     lon_avg = np.average(lon_)
        #     trajs_list.append(dict(
        #         type='scattermapbox',
        #         lon=lon_,
        #         lat=lat_,
        #         mode='lines',
        #         line=dict(
        #             width=1,
        #             color=clr_grad[i]
        #         ),
        #         opacity=0.8,
        #         )
        #     )
        layout = go.Layout(
            autosize=True,
            hovermode='closest',
            mapbox=dict(
                accesstoken=mapbox_access_token,
                bearing=0,
                center=dict(
                    lat=lat_avg,
                    lon=lon_avg
                ),
                style='dark',
                pitch=0,
                zoom=10,
            ),
        )

        fig = dict(data=joint_list, layout=layout)
        py.iplot(fig, filename='top-45-dark-with-section')

    def plt_top_chosen_combined(self, num_trajectories):

        fig, ax = plt.subplots()
        for i in range(num_trajectories):
            num = len(self.tim_lasting[-i-1])

            plt.plot(self.pos_to_draw[-i-1][:len(self.pos_to_draw[-i-1]) // 2],
                     self.pos_to_draw[-i-1][len(self.pos_to_draw[-i-1]) // 2:],
                     marker='.', alpha=0.5,
                     label="t={0}".format(num),
                     )
            plt.title("t={0}".format(num))
            plt.subplots_adjust(hspace=1)
            plt.tight_layout()
            # plt.ylim(42.2, 42.6)
            # plt.xlim(0, 24)
        leg = ax.legend(loc="upper left", bbox_to_anchor=[0, 1],
                            ncol=2, shadow=True, title="points number per path", fancybox=True)

        fig, ax = plt.subplots()
        for i in range(num_trajectories):
            num = len(self.tim_lasting[-i-1])
            # plt.subplot(self.siz_nodes, 1, num_trajectories)
            plt.plot(self.tim_lasting[-i-1],
                     self.pos_to_draw[-i-1][len(self.pos_to_draw[-i-1]) // 2:],
                     marker='.', alpha=0.5,
                     label="t={0}".format(num),
                     )
            plt.title("t={0}".format(num))
            plt.subplots_adjust(hspace=1)
            plt.tight_layout()
            plt.xlim(0, 24)
        leg = ax.legend(loc="upper left", bbox_to_anchor=[0, 1],
                        ncol=2, shadow=True, title="points number per path", fancybox=True)

        plt.show(block=False)

def read_csv(filename):
    
    rows = []
    with open(filename, 'r') as csvfile:
        for row in csvreader:
            rows.append(row)

    return rows


if __name__ == '__main__':
    name_file = '../../data/processed/trajs-order.csv'
    traj_all_order = read_csv(name_file, traj_all_order)

    name_file = '../data/processed/time-order.csv'
    time_all_order = read_csv()


    """
        Case one
        show the top NUM path
        dict vector seems to be meaningless at this point
    """

    # NUM = len(posLst)
    NUM = 45
    traj_top_order = traj_all_order[-NUM:]
    time_top_order = time_all_order[-NUM:]
    draw_top_trajs = DrawTrajectories(time_top_order, traj_top_order, NUM)
    # draw_top_trajs.plt_all_combined()
    # draw_top_trajs.plt_top_chosen_combined(2)
    draw_top_trajs.plt_top_plotly(NUM)
'''
    """
        Case two
        user choose the upper number (no more than) of paths to show, which has specific
        point number (or range), that is, if specified number is not found, find those
        10 values bigger than it recursively
    """
    list_len_sublst = [len(it) // 2 for it in traj_all_order]
    # len_sublst = list(map(len, traj_all_order))
    idx = [i for i, val in enumerate(list_len_sublst) if val == pts_per_path]

    if not idx:
        pntfirst = bisect_left(list_len_sublst, pts_per_path)
        near_list = traj_all_order[pntfirst - num_path:pntfirst]
        time_list = time_all_order[pntfirst - num_path:pntfirst]
        draw_less_trajs = DrawTrajectories(time_top_order, traj_top_order, NUM)
        # draw_less_trajs.plt_all()

    else:
        fix_list = [traj_all_order[i] for i in idx]
        fix_time = [time_all_order[i] for i in idx]
        draw_chosen_trajs = DrawTrajectories(fix_time, fix_list, len(idx))
        # draw_chosen_trajs.plt_all()


    plt.show()
'''
