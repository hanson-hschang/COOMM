__doc__ = """
Muscle frame imple
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

from coomm.frames.frame import FrameBase
from coomm.frames.frame_tools import (
    default_colors,
    default_label_fontsize,
    change_box_to_arrow_axes,
    change_box_to_only_x_line_ax,
    change_box_to_only_y_arrow_ax,
    add_y_ticks
)

from coomm._rendering_tool import (
    process_position, process_director
)

# TODO: plot muscles in the ax_main, which needs 
# process_position and process_director from frame_tools

# TODO: Maybe combine default color/parameters into class?
muscle_length_color = default_colors['tab:brown']
muscle_force_weight_color = default_colors['tab:blue']

muscles_color = dict(
    TM=default_colors['tab:green'],
    LM=default_colors['tab:red'],
    OM=default_colors['tab:cyan'],
)


class MuscleFrameBase(FrameBase):
    """MuscleFrameBase.
    """

    def __init__(self, file_dict, fig_dict, gs_dict, **kwargs):
        """__init__.

        Parameters
        ----------
        file_dict :
        fig_dict :
        gs_dict :
        """
        FrameBase.__init__( # Try to use super
            self,
            file_dict=file_dict,
            fig_dict=fig_dict,
            gs_dict=gs_dict
        )

        self.fontsize = kwargs.get("fontsize", default_label_fontsize)
        self.offset = kwargs.get("offset", np.zeros(3))
        self.rotation = kwargs.get("rotation", np.identity(3))
        MuscleFrameBase.set_ref_setting( # TODO:.. This part look weird
            self,
            reference_total_length=kwargs.get("reference_total_length", 1),
            n_elems=kwargs.get("n_elems", 100)
        )

        self.ax_muscles_info = kwargs["ax_muscles_info"]

    def set_n_elems(self, n_elems):
        """set_n_elems.

        Parameters
        ----------
        n_elems :
        """
        self.n_elems = n_elems
        self.s = np.linspace(0, 1, self.n_elems+1)

    def set_ref_setting(self, reference_total_length, n_elems):
        """set_ref_setting.

        Parameters
        ----------
        reference_total_length :
        n_elems :
        """
        self.reference_total_length = reference_total_length
        MuscleFrameBase.set_n_elems(self, n_elems)

    def ax_muscles_reset(self):
        """ax_muscles_reset.
        """
        ax = self.fig.add_subplot(
            self.gs[
                self.ax_muscles_info['indices'][0],
                self.ax_muscles_info['indices'][1]
            ]
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        self.ax_muscles = ax
        return self.ax_muscles

    def reset(self, axes_muscle_info=None):
        """reset.

        Parameters
        ----------
        axes_muscle_info :
        """
        FrameBase.reset(self,)
        if axes_muscle_info is None:
            return self.ax_muscles_reset()

        axes_muscle = []
        axes_muscle_activation = []
        axes_muscle_length = []
        axes_muscle_force_weight = []

        for i, axes_muscle_indices_i in enumerate(axes_muscle_info['indices']):
            muscle_group_yticklabels_showflag = axes_muscle_info['yticklabels_showflag'][i]
            ax = self.fig.add_subplot(
                self.gs[
                    axes_muscle_indices_i['activation'][0],
                    axes_muscle_indices_i['activation'][1]
                ],
                xlim=[-0.05, 1.15],
                ylim=[-0.1, 1.1],
            )
            ax = change_box_to_arrow_axes(
                self.fig, ax,
                xaxis_ypos=-0.05,
                x_offset=[0.05, -0.05]
            )
            ax.tick_params(axis='x', direction='in', length=4, width=1, pad=0)
            ax.tick_params(axis='y', direction='in', length=4, width=1, pad=1)
            if muscle_group_yticklabels_showflag['activation'] is not True:
                ax.set_yticklabels([])
            axes_muscle_activation.append(ax)

            axes_muscle_length.append([])
            axes_muscle_force_weight.append([])
            for j, axes_muscle_indices_i_j in enumerate(axes_muscle_indices_i['others']):
                ax = self.fig.add_subplot(
                    self.gs[
                        axes_muscle_indices_i_j[0],
                        axes_muscle_indices_i_j[1]
                    ],
                    xlim=[-0.05, 1.15],
                    ylim=[-0.2, 2.2],
                )
                ax = change_box_to_only_x_line_ax(
                    self.fig, ax,
                    xaxis_ypos=-0.1,
                    x_offset=[0.05, -0.15]
                )
                ax = change_box_to_only_y_arrow_ax(
                    self.fig, ax,
                    yaxis_xpos=0.0,
                    y_offset=[0.1, 0],
                    color=muscle_length_color
                )
                ax.tick_params(axis='x', direction='in', length=4, width=1, pad=0)
                ax.tick_params(
                    axis='y', direction='in', length=4, width=1, pad=1,
                    colors=muscle_length_color
                )
                if muscle_group_yticklabels_showflag['length'][j] is not True:
                    ax.set_yticklabels([])
                axes_muscle_length[-1].append(ax)

                ax = ax.twinx()
                ax.set_ylim([-0.1, 1.1])
                ax = change_box_to_only_y_arrow_ax(
                    self.fig, ax,
                    yaxis_xpos=1.0,
                    y_offset=[0.05, 0],
                    color=muscle_force_weight_color
                )
                ax.tick_params(
                    axis='y', direction='in', length=0, width=0, pad=-8,
                    labelcolor=muscle_force_weight_color
                )
                add_y_ticks(
                    ax=ax, yticks=[0, 0.5, 1], ticks_xpos=1,
                    length=0.04, linewidth=1,
                    color=muscle_force_weight_color
                )
                if muscle_group_yticklabels_showflag['force_weight'][j] is not True:
                    ax.set_yticklabels([])
                axes_muscle_force_weight[-1].append(ax)
            
            ax = self.fig.add_subplot(
                self.gs[
                    axes_muscle_indices_i['muscle'][0],
                    axes_muscle_indices_i['muscle'][1]
                ]
            )
            
            ax_muscle_offset = axes_muscle_info['axes_muscle_offset'][i]
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.patch.set_facecolor('grey')
            ax.patch.set_alpha(0.3)
            bbox = ax.get_position()
            bbox.x0 += ax_muscle_offset[0][0]
            bbox.y0 += ax_muscle_offset[0][1]
            bbox.x1 += ax_muscle_offset[1][0]
            bbox.y1 += ax_muscle_offset[1][1]
            ax.set_position(bbox)
            axes_muscle.append(ax)

        return dict(
            muscle=axes_muscle,
            activation=axes_muscle_activation,
            length=axes_muscle_length,
            force_weight=axes_muscle_force_weight
        )

    @classmethod
    def plot_muscle_activation(cls, ax, s, activation, **kwargs):
        """plot_muscle_activation.

        Parameters
        ----------
        ax :
        s :
        activation :
        """
        alpha = kwargs.get("alpha", 1)
        if kwargs.get('fill', True):
            ax.fill_between(
                s, activation,
                color=kwargs["color"],
                alpha=alpha,
            )
        else:
            ax.plot(
                s, activation,
                color=kwargs["color"],
                alpha=alpha,
            )
        return ax

    @classmethod
    def plot_muscle_length(cls, ax, s, muscle_length, **kwargs):
        ax.plot(
            s, muscle_length,
            color=kwargs.get("color", muscle_length_color)
        )
        return ax

    @classmethod
    def plot_force_weight(cls, ax, s, force_weight, **kwargs):
        ax.plot(
            s, force_weight,
            color=kwargs.get("color", muscle_force_weight_color)
        )
        return ax


class TransverseMuscleFrame(MuscleFrameBase):
    """TransverseMuscleFrame.
    """

    def __init__(self, file_dict, fig_dict, gs_dict, **kwargs):
        """__init__.

        Parameters
        ----------
        file_dict :
        fig_dict :
        gs_dict :
        """
        MuscleFrameBase.__init__(
            self,
            file_dict=file_dict,
            fig_dict=fig_dict,
            gs_dict=gs_dict,
            **kwargs
        )

        self.axes_TM_info = kwargs["axes_TM_info"]
    
    def reset(self,):
        """reset.
        """
        self.axes_TM = MuscleFrameBase.reset(self, self.axes_TM_info)

class LongitudinalMuscleFrame(MuscleFrameBase):
    """LongitudinalMuscleFrame.
    """

    def __init__(self, file_dict, fig_dict, gs_dict, **kwargs):
        """__init__.

        Parameters
        ----------
        file_dict :
        fig_dict :
        gs_dict :
        """
        MuscleFrameBase.__init__(
            self,
            file_dict=file_dict,
            fig_dict=fig_dict,
            gs_dict=gs_dict,
            **kwargs
        )

        self.axes_LM_info = kwargs["axes_LM_info"]
    
    def reset(self,):
        """reset.
        """
        self.axes_LM = MuscleFrameBase.reset(self, self.axes_LM_info)

class ObliqueMuscleFrame(MuscleFrameBase):
    """ObliqueMuscleFrame.
    """

    def __init__(self, file_dict, fig_dict, gs_dict, **kwargs):
        MuscleFrameBase.__init__(
            self,
            file_dict=file_dict,
            fig_dict=fig_dict,
            gs_dict=gs_dict,
            **kwargs
        )

        self.axes_OM_info = kwargs["axes_OM_info"]
    
    def reset(self,):
        """__init__.

        Parameters
        ----------
        file_dict :
        fig_dict :
        gs_dict :
        """
        self.axes_OM = MuscleFrameBase.reset(self, self.axes_OM_info)
