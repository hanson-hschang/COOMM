"""
Created on Nov. 11, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import pickle
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rc
rc('text', usetex=True)

import sys
sys.path.append("../")       # include ActuationModel directory

from coomm.frames import (
    RodFrame,
    TransverseMuscleFrame,
    LongitudinalMuscleFrame,
    ObliqueMuscleFrame,
    MuscleFrameBase
)
from coomm.frames.muscle_frame import (
    muscle_length_color,
    muscle_force_weight_color,
    muscles_color
)

class Frame(RodFrame, TransverseMuscleFrame, LongitudinalMuscleFrame, ObliqueMuscleFrame):
    def __init__(self, file_dict, fig_dict, gs_dict, **kwargs):
        TransverseMuscleFrame.__init__(self, file_dict, fig_dict, gs_dict, **kwargs)
        LongitudinalMuscleFrame.__init__(self, file_dict, fig_dict, gs_dict, **kwargs)
        ObliqueMuscleFrame.__init__(self, file_dict, fig_dict, gs_dict, **kwargs)
        RodFrame.__init__(self, file_dict, fig_dict, gs_dict, **kwargs)
        
    def reset(self,):
        MuscleFrameBase.reset(self,)
        TransverseMuscleFrame.reset(self,)
        LongitudinalMuscleFrame.reset(self,)
        ObliqueMuscleFrame.reset(self,)
        RodFrame.reset(self,)
        
    def set_ref_configuration(self, position, shear, kappa):
        L0 = RodFrame.set_ref_configuration(self, position)
        return L0

    def plot_ref_configuration(self,):
        RodFrame.plot_ref_configuration(self)

    def set_labels(self, time=None):
        RodFrame.set_labels(self, time)
        self.axes_TM['muscle'][0].set_title(
            'TM',color=muscles_color['TM'],
            fontsize=self.fontsize
        )
        self.axes_LM['muscle'][0].set_title(
            r'LM$_{1^+}$',color=muscles_color['LM'],
            fontsize=self.fontsize
        )
        self.axes_LM['muscle'][1].set_title(
            r'LM$_{2^+}$',color=muscles_color['LM'],
            fontsize=self.fontsize
        )
        self.axes_LM['muscle'][2].set_title(
            r'LM$_{1^-}$', color=muscles_color['LM'],
            fontsize=self.fontsize
        )
        self.axes_LM['muscle'][3].set_title(
            r'LM$_{2^-}$', color=muscles_color['LM'],
            fontsize=self.fontsize
        )
        self.axes_OM['muscle'][0].set_ylabel(
            r'OM$_+$', color=muscles_color['OM'],
            fontsize=self.fontsize
        )
        self.axes_OM['muscle'][1].set_ylabel(
            r'OM$_-$', color=muscles_color['OM'],
            fontsize=self.fontsize
        )
        
        legend_elements = [
            Line2D([0], [0], color=muscle_length_color, label='muscle length $\ell^{m}$'),
            Line2D([0], [0], color=muscle_force_weight_color, label='weighted force $f_\ell$'),
        ]
        self.ax_muscles.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.03),
            ncol=2,
            fontsize=self.fontsize
        )

    def plot_muscle_activation(
        self, group_number, muscle_group_type,
        muscle_s_activation, muscle_activation, **kwargs
    ):
        axes_muscle = getattr(
            self, "axes_"+muscle_group_type
        )
        kwargs.setdefault("color", muscles_color[muscle_group_type])
        
        return MuscleFrameBase.plot_muscle_activation(
            axes_muscle['activation'][group_number],
            muscle_s_activation, muscle_activation,
            **kwargs
        )
    
    def plot_muscle_length(
        self, group_number, muscle_type,  muscle_number,
        muscle_s, muscle_length
    ):
        axes_muscle = getattr(
            self, "axes_"+muscle_type
        )
        return MuscleFrameBase.plot_muscle_length(
            axes_muscle['length'][group_number][muscle_number],
            muscle_s, muscle_length
        )

    def plot_force_length_weight(
        self, group_number, muscle_type, muscle_number,
        muscle_s, force_length_weight
    ):
        axes_muscle = getattr(
            self, "axes_"+muscle_type
        )
        return MuscleFrameBase.plot_force_weight(
            axes_muscle['force_weight'][group_number][muscle_number],
            muscle_s, force_length_weight
        )
        

    @classmethod
    def get_frame(cls, filename):
        file_dict = defaultdict(list)
        file_dict['check_folder_flag'] = True
        file_dict['folder_name'] = filename+"_frames_muscle"
        file_dict['figure_name'] = "frame{:04d}.png"

        fig_dict=defaultdict(list)
        fig_dict['figsize']= (20, 8)

        gs_dict = defaultdict(list)
        gs_dict['ncols'] = 10
        gs_dict['nrows'] = 4
        gs_dict['width_ratios'] = np.ones(gs_dict['ncols'])
        gs_dict['height_ratios'] = np.ones(gs_dict['nrows'])
        
        ax_muscles_info = dict(
            indices=[slice(0, 4), slice(5, 10)]
        )

        ax_muscle_x0_offset = -0.0125
        ax_muscle_y0_offset = -0.019
        ax_muscle_p0_offset = [ax_muscle_x0_offset, ax_muscle_y0_offset]
        ax_muscle_x1_offset = 0.008
        ax_muscle_y1_offset = 0.012

        axes_TM_info = dict(
            indices=[
                dict(
                    activation=[0, 5],
                    others=[[1, 5]],
                    muscle=[slice(0, 2), 5]
                ),
            ],
            yticklabels_showflag=[
                dict(
                    activation=True,
                    length=[True,],
                    force_weight=[False,]
                ),
            ],
            axes_muscle_offset=[
                [ax_muscle_p0_offset, [0, ax_muscle_y1_offset]]
            ],
        )

        axes_LM_info = dict(
            indices=[
                dict(
                    activation=[0, 6],
                    others=[[1, 6]],
                    muscle=[slice(0, 2), 6]
                ),
                dict(
                    activation=[0, 7],
                    others=[[1, 7]],
                    muscle=[slice(0, 2), 7]
                ),
                dict(
                    activation=[0, 8],
                    others=[[1, 8]],
                    muscle=[slice(0, 2), 8]
                ),
                dict(
                    activation=[0, 9],
                    others=[[1, 9]],
                    muscle=[slice(0, 2), 9]
                ),
            ],
            yticklabels_showflag=[
                dict(
                    activation=True,
                    length=[False,],
                    force_weight=[False,]
                ),
                dict(
                    activation=True,
                    length=[False,],
                    force_weight=[False,]
                ),
                dict(
                    activation=True,
                    length=[False,],
                    force_weight=[False,]
                ),
                dict(
                    activation=True,
                    length=[False,],
                    force_weight=[True,]
                ),
            ],
            axes_muscle_offset=[
                [ax_muscle_p0_offset, [0, ax_muscle_y1_offset]],
                [ax_muscle_p0_offset, [0, ax_muscle_y1_offset]],
                [ax_muscle_p0_offset, [0, ax_muscle_y1_offset]],
                [ax_muscle_p0_offset, [ax_muscle_x1_offset, ax_muscle_y1_offset]]
            ],
        )

        axes_OM_info = dict(
            indices=[
                dict(
                    activation=[2, 5],
                    others=[
                        [2, 6],
                        [2, 7],
                        [2, 8],
                        [2, 9],
                    ],
                    muscle=[2, slice(5, 10)]
                ),
                dict(
                    activation=[3, 5],
                    others=[
                        [3, 6],
                        [3, 7],
                        [3, 8],
                        [3, 9],
                    ],
                    muscle=[3, slice(5, 10)]
                ),
            ],
            yticklabels_showflag=[
                dict(
                    activation=True,
                    length=[True, False, False, False,],
                    force_weight=[False, False, False, True]
                ),
                dict(
                    activation=True,
                    length=[True, False, False, False,],
                    force_weight=[False, False, False, True]
                ),
            ],
            axes_muscle_offset=[
                [ax_muscle_p0_offset, [ax_muscle_x1_offset, ax_muscle_y1_offset]],
                [ax_muscle_p0_offset, [ax_muscle_x1_offset, ax_muscle_y1_offset]],
            ],
        )

        ax_main_info = dict(
            indices=[slice(0,4), slice(0,4)],
            planner_flag=False
        )

        return cls(
            file_dict=file_dict,
            fig_dict=fig_dict,
            gs_dict=gs_dict,
            ax_muscles_info=ax_muscles_info,
            axes_TM_info=axes_TM_info,
            axes_LM_info=axes_LM_info,
            axes_OM_info=axes_OM_info,
            ax_main_info=ax_main_info,
        )

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(-height_z, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

def main(filename):

    with open(filename+"_data.pickle", "rb") as f:
        data = pickle.load(f)
        rod_data = data['systems'][0]
        sphere_data = data['systems'][1]
        muscle_groups_data = data['muscle_groups']
        algo_data = data['algo']
        recording_fps = data['recording_fps']

    with open(filename+"_systems.pickle", "rb") as f:
        data = pickle.load(f)
        rod = data['systems'][0]
        muscle_groups = data['muscle_groups']
    
    frame = Frame.get_frame(filename=filename)

    L0 = frame.set_ref_configuration(
        position=rod_data["position"][0],
        shear=rod_data['sigma'][0]+np.array([0, 0, 1])[:, None],
        kappa=rod_data['kappa'][0],
    )

    print("Plotting frames ...")
    for k in tqdm(range(len(rod_data["time"]))):
        frame.reset()

        frame.plot_rod(
            position=rod_data["position"][0],
            director=rod_data["director"][0],
            radius=rod_data["radius"][0],
            color='orange',
            alpha=0.3
        )

        ax_main = frame.plot_rod(
            position=rod_data["position"][k],
            director=rod_data["director"][k],
            radius=rod_data["radius"][k]
        )

        ax_main.scatter(
            sphere_data["position"][k][0, 0]/L0,
            sphere_data["position"][k][1, 0]/L0,
            sphere_data["position"][k][2, 0]/L0,
            color='grey'
        )

        for muscle_group_data in muscle_groups_data:
            muscle_info = muscle_group_data['muscle_group_info'][k].split('_')
            group_number, muscle_group_type = (
                int(muscle_info[0]), muscle_info[1]
            )

            algo_activation = algo_data["activations"][k][group_number]

            if group_number > 0:
                group_number -= 1
            if group_number > 3:
                group_number -= 4

            frame.plot_muscle_activation(
                group_number, muscle_group_type, 
                muscle_group_data["s_activation"][k],
                muscle_group_data["activation"][k]
            )

            frame.plot_muscle_activation(
                group_number, muscle_group_type, 
                muscle_group_data["s_activation"][k],
                algo_activation,
                color='black',
                fill=False
            )
   
            for muscle_data in muscle_group_data['muscles']:
                muscle_info = muscle_data['muscle_info'][k].split('_')
                muscle_number, muscle_type = (
                    int(muscle_info[0]), muscle_info[1]
                )
                frame.plot_muscle_length(
                    group_number, muscle_type, muscle_number,
                    muscle_data["s_activation"][k], muscle_data["muscle_normalized_length"][k]
                )
                frame.plot_force_length_weight(
                    group_number, muscle_type, muscle_number,
                    muscle_data["s_activation"][k], muscle_data["force_length_weight"][k]
                )

        frame.set_ax_main_lim(
            x_lim=[-1.1, 1.1],
            y_lim=[-1.1, 1.1],
            z_lim=[-1.1, 1.1]
        )

        frame.set_labels(rod_data["time"][k])
        frame.save()

    frame.movie(
        frame_rate=recording_fps,
        movie_name=filename+"_muscle_movie"
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot simulation result in filename_frames_muscle.'
    )
    parser.add_argument(
        '--filename', type=str, default='simulation',
        help='a str: data file name',
    )
    args = parser.parse_args()
    main(filename=args.filename)
