__doc__ = """
Rod frame implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from comm.frames.frame import FrameBase
from comm.frames.frame_tools import (
    base_colors, 
    default_label_fontsize,
)
from comm._rendering_tool import (
    process_position, process_director
)

# TODO: Maybe combine into class
rod_color = base_colors['m']
algo_rod_color = base_colors['g']

class RodFrame(FrameBase):
    """RodFrame.
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
        self.rod_color = kwargs.get("rod_color", rod_color)
        self.offset = kwargs.get("offset", np.zeros(3))
        self.rotation = kwargs.get("rotation", np.identity(3))
        self.reference_total_length = 1
        self.reference_configuration_flag = False
        RodFrame.set_n_elems(self, kwargs.get("n_elems", 100))

    def set_n_elems(self, n_elems):
        """set_n_elems.

        Parameters
        ----------
        n_elems :
        """
        self.n_elems = n_elems
        self.s = np.linspace(0, 1, self.n_elems+1)

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
        
        if self.reference_configuration_flag:
            RodFrame.plot_ref_configuration(self)

    def set_ref_configuration(self, position):
        """set_ref_configuration.

        Parameters
        ----------
        position :
        """
        self.reference_configuration_flag = True
        self.reference_position = position.copy()
        
        reference_length = np.linalg.norm(
            position[:, 1:]-position[:, :-1], axis=0
        )
        self.reference_total_length = reference_length.sum()
        RodFrame.set_n_elems(self, reference_length.shape[0])
        return self.reference_total_length

    def plot_ref_configuration(self,):
        """plot_ref_configuration.
        """
        line_position = process_position(
            self.reference_position,
            self.offset, self.rotation
        ) / self.reference_total_length
        
        if self.ax_main_3d_flag:
            self.ax_main.plot(
                line_position[0], line_position[1], line_position[2],
                color="grey", linestyle="--"
            )
        else:
            self.ax_main.plot(
                line_position[0], line_position[1],
                color="grey", linestyle="--"
            )

    def calculate_line_position(self, position, director, radius):
        """calculate_line_position.

        Parameters
        ----------
        position :
        director :
        radius :
        """
        line_center = process_position(
            position, self.offset, self.rotation
        ) / self.reference_total_length
        line_position = process_position(
            (position[:, :-1] + position[:, 1:])/2,
            self.offset, self.rotation
        )
        line_director = process_director(director, self.rotation)
        line_up = (
            (line_position + line_director[1, :, :] * radius) 
            / self.reference_total_length
        )
        line_down = (
            (line_position - line_director[1, :, :] * radius) 
            / self.reference_total_length
        )
        line_left = (
            (line_position + line_director[0, :, :] * radius)
            / self.reference_total_length
        )
        line_right = (
            (line_position - line_director[0, :, :] * radius)
            / self.reference_total_length
        )
        return line_center, [line_up, line_right, line_down, line_left]

    def plot_rod2d(self, position, director, radius, **kwargs):
        """plot_rod2d.

        Parameters
        ----------
        position :
        director :
        radius :
        """
        color = kwargs.get("color", self.rod_color)
        alpha = kwargs.get("alpha", 1)
        line_center, lines = self.calculate_line_position(
            position, director, radius
        )
        self.ax_main.plot(
            line_center[0], line_center[1], 
            color=color,
            alpha=alpha,
            linestyle="--"
        )
        self.ax_main.plot(
            lines[0][0], lines[0][1],
            color=color,
            alpha=alpha,
        )
        self.ax_main.plot(
            lines[2][0], lines[2][1],
            color=color,
            alpha=alpha,
        )
        # self.ax_main.plot(
        #     lines[1][0], lines[1][1],
        #     color=color,
        #     alpha=alpha,
        # )
        # self.ax_main.plot(
        #     lines[3][0], lines[3][1],
        #     color=color,
        #     alpha=alpha,
        # )
        self.ax_main.plot(
            [lines[0][0, -1], line_center[0, -1], lines[2][0, -1]],
            [lines[0][1, -1], line_center[1, -1], lines[2][1, -1]],
            color=color,
            alpha=alpha,
            label='sim'
        )
        # self.ax_main.plot(
        #     [lines[1][0, -1], line_center[0, -1], lines[3][0, -1]],
        #     [lines[1][1, -1], line_center[1, -1], lines[3][1, -1]],
        #     color=color,
        #     alpha=alpha,
        # )
        return self.ax_main

    def plot_rod3d(self, position, director, radius, **kwargs):
        """plot_rod3d.

        Parameters
        ----------
        position :
        director :
        radius :
        """
        color = kwargs.get("color", self.rod_color)
        alpha = kwargs.get("alpha", 1)
        line_center, lines = self.calculate_line_position(
            position, director, radius
        )
        self.ax_main.plot(
            line_center[0], line_center[1], line_center[2],
            color=color,
            alpha=alpha,
            linestyle="--"
        )
        for line in lines:    
            self.ax_main.plot(
                line[0], line[1], line[2],
                color=color,
                alpha=alpha,
            ) 
        self.ax_main.plot(
            [lines[0][0, -1], line_center[0, -1], lines[2][0, -1]],
            [lines[0][1, -1], line_center[1, -1], lines[2][1, -1]],
            [lines[0][2, -1], line_center[2, -1], lines[2][2, -1]],
            color=color,
            alpha=alpha,
            label='sim'
        )
        self.ax_main.plot(
            [lines[1][0, -1], line_center[0, -1], lines[3][0, -1]],
            [lines[1][1, -1], line_center[1, -1], lines[3][1, -1]],
            [lines[1][2, -1], line_center[2, -1], lines[3][2, -1]],
            color=color,
            alpha=alpha,
        )
        return self.ax_main

    def set_ax_main_lim(
        self, 
        x_lim=[-1.1, 1.1], # FIXME
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
        
        # self.ax_main.legend()

class StrainFrame(FrameBase):
    """StrainFrame.
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

        self.axes_strain_info = kwargs["axes_strain_info"]
        self.axes_kappa_indices = self.axes_strain_info["axes_kappa_indices"]
        self.axes_shear_indices = self.axes_strain_info["axes_shear_indices"]

        self.fontsize = kwargs.get("fontsize", default_label_fontsize)
        self.rod_color = kwargs.get("rod_color", rod_color)
        
        self.reference_configuration_flag = False
        StrainFrame.set_n_elems(self, kwargs.get("n_elems", 100))

    def set_n_elems(self, n_elems):
        """set_n_elems.

        Parameters
        ----------
        n_elems :
        """
        self.n_elems = n_elems
        s = np.linspace(0, 1, self.n_elems+1)
        self.s_shear = (s[:-1] + s[1:])/2
        self.s_kappa = s[1:-1].copy()

    def reset(self,):
        """reset.
        """
        FrameBase.reset(self,)
        
        self.axes_kappa = []
        self.axes_shear = []

        for i in range(3):
            self.axes_kappa.append(
                self.fig.add_subplot(
                    self.gs[
                        self.axes_kappa_indices[i][0],
                        self.axes_kappa_indices[i][1]
                    ],
                    xlim=[-0.1, 1.1]
                )
            )
            self.axes_shear.append(
                self.fig.add_subplot(
                    self.gs[
                        self.axes_shear_indices[i][0],
                        self.axes_shear_indices[i][1]
                    ],
                    xlim=[-0.1, 1.1]
                )
            )        
        
        if self.reference_configuration_flag:
            StrainFrame.plot_ref_configuration(self,)

    def set_ref_configuration(self, shear, kappa):
        """set_ref_configuration.

        Parameters
        ----------
        shear :
        kappa :
        """
        self.reference_configuration_flag = True
        self.reference_shear = shear.copy()
        self.reference_kappa = kappa.copy()
        StrainFrame.set_n_elems(self, self.reference_shear.shape[1])
        return 

    def plot_ref_configuration(self,):
        """plot_ref_configuration.
        """
        for index_i in range(3):
            self.axes_shear[index_i].plot(
                self.s_shear,
                self.reference_shear[index_i],
                color="grey",
                linestyle="--"
            )
            self.axes_kappa[index_i].plot(
                self.s_kappa,
                self.reference_kappa[index_i],
                color="grey",
                linestyle="--"
            )

    def plot_strain(self, shear, kappa, color=None):
        """plot_strain.

        Parameters
        ----------
        shear :
        kappa :
        color :
        """
        for index_i in range(3):
            self.axes_shear[index_i].plot(
                self.s_shear,
                shear[index_i],
                color=self.rod_color if color is None else color
            )
            self.axes_kappa[index_i].plot(
                self.s_kappa,
                kappa[index_i],
                color=self.rod_color if color is None else color
            )
        return self.axes_shear, self.axes_kappa

    def set_axes_strain_lim(
        self, 
        axes_shear_lim = [ # FIXME
            [-0.11, 0.11],
            [-0.11, 0.11],
            [-0.1, 2.1]
        ],
        axes_kappa_lim = [
            [-110, 110],
            [-110, 110],
            [-110, 110],
        ]
    ):
        """set_axes_strain_lim.

        Parameters
        ----------
        axes_shear_lim :
        axes_kappa_lim :
        """
        for index_i in range(3):
            shear_mean = np.average(axes_shear_lim[index_i]) if index_i != 2 else 1
            shear_log = np.floor(
                np.log10(axes_shear_lim[index_i][1] - shear_mean)
            )
            kappa_mean = np.average(axes_kappa_lim[index_i])
            kappa_log = np.floor(
                np.log10(axes_kappa_lim[index_i][1] - kappa_mean)
            )
            
            self.axes_shear[index_i].set_ylim(axes_shear_lim[index_i])
            self.axes_kappa[index_i].set_ylim(axes_kappa_lim[index_i])
            self.axes_shear[index_i].ticklabel_format(
                axis='y', scilimits=(shear_log, shear_log),
                useOffset=shear_mean
            )
            self.axes_kappa[index_i].ticklabel_format(
                axis='y', scilimits=(kappa_log, kappa_log),
                useOffset=kappa_mean
            )

    def set_labels(self,):
        """set_labels.
        """
        for index_i in range(3):
            self.axes_shear[index_i].set_ylabel(
                "d$_{}$".format(index_i+1),
                fontsize=self.fontsize,
                rotation=0,
                labelpad=15.0
            )
            self.axes_kappa[index_i].yaxis.set_label_position("left")

        self.axes_shear[0].set_title("shear", fontsize=self.fontsize)
        ylim = self.axes_shear[2].get_ylim()
        ylim_mean = np.average(ylim)
        position = 0.9 * (ylim[1] - ylim_mean) + ylim_mean
        self.axes_shear[2].text(
            0, position, 'stretch', 
            fontsize=self.fontsize, 
            ha='left', va='top'
        )
        self.axes_shear[2].set_xlabel("$s$", fontsize=self.fontsize)
        
        self.axes_kappa[0].set_title("curvature", fontsize=self.fontsize)
        ylim = self.axes_kappa[2].get_ylim()
        ylim_mean = np.average(ylim)
        position = 0.9 * (ylim[1] - ylim_mean) + ylim_mean
        self.axes_kappa[2].text(
            1, position, 'twist',
            fontsize=self.fontsize,
            ha='right', va='top'
        )
        self.axes_kappa[2].set_xlabel("$s$", fontsize=self.fontsize)
