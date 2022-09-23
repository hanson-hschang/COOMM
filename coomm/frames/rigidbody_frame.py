__doc__ = """
Rigid-body frame implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from coomm.frames.frame import FrameBase
from coomm.frames.frame_tools import (
    base_colors,
    default_label_fontsize,
)

from coomm._rendering_tool import (
    process_position, process_director
)

# TODO: Combine into class maybe
rigidbody_color = base_colors['y']

class RigidbodyFrame(FrameBase):
    """RigidbodyFrame.
    """

    def __init__(self, file_dict, fig_dict, gs_dict, **kwargs):
        """__init__.

        Parameters
        ----------
        file_dict :
        fig_dict :
        gs_dict :
        """
        FrameBase.__init__(
            self,
            file_dict=file_dict,
            fig_dict=fig_dict,
            gs_dict=gs_dict
        )
        
        self.ax_main_info = kwargs["ax_main_info"]
        self.ax_main_indices = self.ax_main_info["indices"]
        self.ax_main_3d_flag = not self.ax_main_info.get("planner_flag", True)
        self.plot_rod = (
            self.plot_rod3d if self.ax_main_3d_flag else self.plot_rod2d
        )

        self.fontsize = kwargs.get("fontsize", default_label_fontsize)
        self.rigidbody_color = kwargs.get("rigidbody_color", rigidbody_color)
        self.offset = kwargs.get("offset", np.zeros(3))
        self.rotation = kwargs.get("rotation", np.identity(3))
        self.reference_total_length = 1
        self.reference_configuration_flag = False
        RigidbodyFrame.set_n_elems(self, kwargs.get("n_elems", 1))

    def set_n_elems(self, n_elems):
        """set_n_elems.

        Parameters
        ----------
        n_elems :
        """
        self.n_elems = n_elems

    def reset(self,):
        """reset.
        """
        FrameBase.reset(self,)
        
        if self.ax_main_3d_flag:
            self.ax_main = self.fig.add_subplot(
                self.gs[
                    self.ax_main_indices[0],
                    self.ax_main_indices[1]
                ],
                projection='3d'
            )
        else:
            self.ax_main = self.fig.add_subplot(
                self.gs[
                    self.ax_main_indices[0],
                    self.ax_main_indices[1]
                ]
            )

    def plot_rigidybody2d(self, position, director, radius, color=None):
        """plot_rigidybody2d.

        Parameters
        ----------
        position :
        director :
        radius :
        color :
        """
        return self.ax_main

    def plot_rigidybody3d(self, position, director, radius, color=None):
        """plot_rigidybody3d.

        Parameters
        ----------
        position :
        director :
        radius :
        color :
        """
        return self.ax_main

    def set_ax_main_lim(
        self, 
        x_lim=[-1.1, 1.1], # FIXME: Never use mutable object for default
        y_lim=[-1.1, 1.1],
        z_lim=[-1.1, 1.1]
    ):
        """set_ax_main_lim.

        Parameters
        ----------
        x_lim :
        y_lim :
        z_lim :
        """
        self.ax_main.set_xlim(x_lim)
        self.ax_main.set_ylim(y_lim)
        if self.ax_main_3d_flag:
            self.ax_main.set_zlim(z_lim)

    def set_labels(self, time=None):
        """set_labels.

        Parameters
        ----------
        time :
        """
        if time is not None:
            self.ax_main.set_title(
                "time={:.2f} [sec]".format(time), 
                fontsize=self.fontsize
            )
        
        self.ax_main.legend()
