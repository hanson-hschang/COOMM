__doc__ = """
Frame base class
"""

import os
import matplotlib.pyplot as plt
from matplotlib import gridspec

from coomm._rendering_tool import check_folder

class FrameBase:
    """FrameBase.
    """

    def __init__(self, file_dict, fig_dict, gs_dict):
        """__init__.

        Parameters
        ----------
        file_dict :
        fig_dict :
        gs_dict :
        """
        self.figure_name = file_dict["figure_name"]
        self.folder_name = file_dict.get("folder_name", None)
        self.frame_count = 0
        self.fig_dict = fig_dict
        self.gs_dict = gs_dict
        self.fig = None
        self.gs = None

        if file_dict.get("check_folder_flag", True):
            check_folder(self.folder_name)
            file_dict['check_folder_flag'] = False

    def reset(self):
        """reset.
        """
        if self.fig is None:
            self.fig = plt.figure(**self.fig_dict)
            self.gs = gridspec.GridSpec(
                figure=self.fig,
                **self.gs_dict
            )

    def show(self,):
        """show.
        """
        plt.show()

    def save(self, show=False, frame_count=None):
        """save.

        Parameters
        ----------
        show :
        frame_count :
        """
        if self.folder_name is None:
            self.fig.savefig(self.figure_name)
        else:
            frame_count = (
                self.frame_count if frame_count is None else frame_count
            )
            self.fig.savefig(
                self.folder_name + "/" +
                self.figure_name.format(frame_count)
            )
            self.frame_count += 1
        if show:
            self.show()
        else:
            plt.close(self.fig)
        self.fig = None
        self.gs = None

    def movie(self, frame_rate, movie_name, start_number=0):
        """movie.

        Parameters
        ----------
        frame_rate :
        movie_name :
        start_number :
        """
        print("Creating movie:", movie_name+".mov")
        cmd = "ffmpeg -r {}".format(frame_rate)
        figure_name = self.figure_name.replace("{:", "%")
        figure_name = figure_name.replace("}", "")
        cmd += " -start_number {}".format(start_number)
        cmd += " -i " + self.folder_name + "/" + figure_name
        cmd += " -b:v 90M -c:v libx264 -pix_fmt yuv420p -f mov"
        cmd += " -y " + movie_name + ".mov"
        os.system(cmd)

# TODO: What is this??
# -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
