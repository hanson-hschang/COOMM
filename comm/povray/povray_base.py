"""
Created on Dec. 17, 2021
@author: Heng-Sheng (Hanson) Chang
"""

from numba import njit
import numpy as np

from comm._rendering_tool import (
    process_position, process_director
)

class POVRAYBase:
    def __init__(self, **kwargs):
        self.color_string = self.to_color_string(kwargs.get("color", [0.45, 0.39, 1.0]))
        self.scale = kwargs.get("scale", 16)
        self.rotation_matrix = kwargs.get("rotation_matrix", np.eye(3))
        self.offset = kwargs.get("offset", np.zeros(3))
    
    @staticmethod
    def alpha_to_transmit(alpha):
        return 1-alpha

    def to_color_string(self, color):
        string = color if isinstance(color, str) else (
            "<%f, %f, %f>" % (color[0], color[1], color[2])
        )
        return string

    def adapted_position(self, position):
        return process_position(position, self.offset, self.rotation_matrix)

    def adapted_director(self, director):
        return process_director(director, self.rotation_matrix)
        
