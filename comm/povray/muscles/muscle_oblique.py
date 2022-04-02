"""
Created on Dec. 23, 2021
@author: Heng-Sheng (Hanson) Chang
"""

from comm.povray import POVRAYBase
from comm.povray.muscles.muscle import POVRAYSlenderMuscle

class POVRAYObliqueMuscle(POVRAYBase, POVRAYSlenderMuscle):
    def __init__(self, **kwargs):
        POVRAYBase.__init__(self, **kwargs)
        POVRAYSlenderMuscle.__init__(self, **kwargs)
        self.color_string = self.to_color_string(self.muscle_color)
        self.muscle_label = "// oblique muscle data\n"
