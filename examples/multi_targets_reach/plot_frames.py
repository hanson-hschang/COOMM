"""
Created on Oct. 12, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import pickle
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from matplotlib import rc
rc('text', usetex=True)

from coomm.frames import RodFrame, StrainFrame


class Frame(RodFrame, StrainFrame):
    def __init__(self, file_dict, fig_dict, gs_dict, **kwargs):
        StrainFrame.__init__(self, file_dict, fig_dict, gs_dict, **kwargs)
        RodFrame.__init__(self, file_dict, fig_dict, gs_dict, **kwargs)
        
    def reset(self,):
        StrainFrame.reset(self,)
        RodFrame.reset(self,)
        
    def set_ref_configuration(self, position, shear, kappa):
        StrainFrame.set_ref_configuration(self, shear, kappa)
        L0 = RodFrame.set_ref_configuration(self, position)
        return L0

    def plot_ref_configuration(self,):
        StrainFrame.plot_ref_configuration(self)
        RodFrame.plot_ref_configuration(self)

    def set_labels(self, time=None):
        StrainFrame.set_labels(self,)
        RodFrame.set_labels(self, time)

    @classmethod
    def get_frame(cls, filename):
        file_dict = defaultdict(list)
        file_dict['check_folder_flag'] = True
        file_dict['folder_name'] = filename+"_frames"
        file_dict['figure_name'] = "frame{:04d}.png"

        fig_dict=defaultdict(list)
        fig_dict['figsize']= (18, 9)

        gs_dict = defaultdict(list)
        gs_dict['ncols'] = 6
        gs_dict['nrows'] = 3
        gs_dict['width_ratios'] = np.ones(gs_dict['ncols'])
        gs_dict['height_ratios'] = np.ones(gs_dict['nrows'])
        
        
        axes_strain_info =dict(
            axes_kappa_indices=[
                [0, 5],
                [1, 5],
                [2, 5]
            ],
            axes_shear_indices=[
                [0, 4],
                [1, 4],
                [2, 4]
            ]
        )
        ax_main_info = dict(
            indices=[slice(0,3), slice(0,3)],
            planner_flag=False
        )

        return cls(
            file_dict=file_dict,
            fig_dict=fig_dict,
            gs_dict=gs_dict,
            axes_strain_info=axes_strain_info,
            ax_main_info=ax_main_info
        )

def main(filename):

    with open(filename+"_data.pickle", "rb") as f:
        data = pickle.load(f)
        rod_data = data['systems'][0]
        spheres_data = data['systems'][1]
        recording_fps = data['recording_fps']

    with open(filename+"_systems.pickle", "rb") as f:
        data = pickle.load(f)
        rod = data['systems'][0]
    
    frame = Frame.get_frame(filename=filename)

    L0 = frame.set_ref_configuration(
        position=rod_data["position"][0],
        shear=rod_data['sigma'][0]+np.array([0, 0, 1])[:, None],
        kappa=rod_data['kappa'][0],
    )

    print("Plotting frames ...")
    for k in tqdm(range(len(rod_data["time"]))):
    # for k in tqdm([-1]):

        frame.reset()
        
        frame.plot_rod(
            position=rod_data["position"][0],
            director=rod_data["director"][0],
            radius=rod_data["radius"][0],
            color='orange',
        )

        ax_main = frame.plot_rod(
            position=rod_data["position"][k],
            director=rod_data["director"][k],
            radius=rod_data["radius"][k]
        )

        for sphere_data in spheres_data:
            ax_main.scatter(
                sphere_data["position"][k][0, 0]/L0,
                sphere_data["position"][k][1, 0]/L0,
                sphere_data["position"][k][2, 0]/L0,
                color='grey'
            )

        axes_shear, axes_curvature = frame.plot_strain(
            shear=rod_data['sigma'][k]+np.array([0, 0, 1])[:, None],
            kappa=rod_data['kappa'][k]
        )

        frame.plot_strain(
            shear=rod_data['sigma'][0]+np.array([0, 0, 1])[:, None],
            kappa=rod_data['kappa'][0],
            color='orange'
        )

        frame.set_ax_main_lim(
            x_lim=[-1.1, 1.1],
            y_lim=[-1.1, 1.1],
            z_lim=[-1.1, 1.1]
        )
        frame.set_axes_strain_lim()

        frame.set_labels(rod_data["time"][k])
        frame.save(show=False)

    frame.movie(
        frame_rate=recording_fps,
        movie_name=filename+"_movie"
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot simulation result in frames.'
    )
    parser.add_argument(
        '--filename', type=str, default='simulation',
        help='a str: data file name',
    )
    args = parser.parse_args()
    main(filename=args.filename)
