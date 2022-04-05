"""
Created on Dec. 21, 2021
@author: Heng-Sheng (Hanson) Chang
"""

from comm.povray import POVRAYBase
from comm.povray.muscles.muscle import POVRAYSlenderMuscle

class POVRAYLongitudinalMuscle(POVRAYBase, POVRAYSlenderMuscle):
    def __init__(self, **kwargs):
        POVRAYBase.__init__(self, **kwargs)
        POVRAYSlenderMuscle.__init__(self, **kwargs)
        self.color_string = self.to_color_string(self.muscle_color)
        self.muscle_label = "// longitudinal muscle data\n"
