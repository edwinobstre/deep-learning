import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.plotly as py

mapbox_access_token = 'pk.eyJ1Ijoib2JiaWUiLCJhIjoiY2pvNGQycWV3MTZqNDNrcWNvNWE2dHY0eSJ9.eUl3y7PFLpyIw51XjFYxhg'


class Draw_GIS_Trajectories:
    def __init__(self, gis_trajs, num):
        self.traj_data = gis_trajs
        self.num_draw_threshold = num
        self.tim_diff = 0.3
        self.lon_diff = 0.0015
        self.lat_diff = 0.0015
        self.lat_filter_diff = 0.004
        self.lon_filter_diff = 0.004
        self.clrs = ['rgb(0,0,255)', 'rgb(137,137,255)', 'rgb(196,196,255)',
                     'rgb(255,177,177)', 'rgb(255,98,98)', 'rgb(255,0,0)',
                     'rgb(255,0,0)', 'rgb(255,98,98)', 'rgb(255,177,177)',
                     'rgb(196,196,255)', 'rgb(137,137,255)', 'rgb(0,0,255)']
        self.clr_temp = 'rgb(0,0,255)'
        self.label_nodes = []

    def traj_valid_filter(self, section_line):
        tim_diff = section_line[2]
        lon_diff = abs(section_line[5] - section_line[3])
        lat_diff = abs(section_line[6] - section_line[4])
        if tim_diff <= self.tim_diff and \
            lon_diff <= self.lon_diff and lat_diff <= self.lat_diff:
            return True

    def plt_top_plotly(self):

        joint_list = []
        sect_each = []
        # partition one trajectory into 12 sections with gradient colors
        for i in range(self.num_draw_threshold):
            ''' trajs_id begins from 0 '''
            i = i + 1
            trajs = self.traj_data[self.traj_data['traj_id'] == i].values
            self.label_nodes.append(str(trajs.shape[0]))

            for j in range(trajs.shape[0]):
                if self.traj_valid_filter(trajs[j]):
                    sect_each = [dict(
                        type='scattermapbox',
                        lon=[trajs[j][3], trajs[j][5]],
                        lat=[trajs[j][4], trajs[j][6]],
                        mode='lines',
                        line=dict(
                            width=1.5,
                            color=self.clrs[int(trajs[j][7])],
                        ),
                        # opacity=0.8,
                    )
                    ]
                    joint_list = joint_list + sect_each

            self.plot_by_plotly(joint_list)

    def plot_by_plotly(self, trajs_list):
        lon_avg = np.mean(self.traj_data.iloc[:, 3])
        lat_avg = np.mean(self.traj_data.iloc[:, 4])
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
        fig = dict(data=trajs_list, layout=layout)
        filename = 'Top-' + str(self.num_draw_threshold) + 'trajectories-dark-theme'
        py.iplot(fig, filename=filename)


def main():
    data_file = '../../data/processed/gis-trajs-tims.csv'
    gis_trajs = pd.read_csv(data_file, delimiter='\t')

    NUM = 1

    draw_top_trajs = Draw_GIS_Trajectories(gis_trajs, NUM)
    draw_top_trajs.plt_top_plotly()


if __name__ == '__main__':
    main()