"""
Created on Jan. 11, 2022
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from comm.povray.povray_base import POVRAYBase

class POVRAYCylinder(POVRAYBase):
    def __init__(self, **kwargs):
        POVRAYBase.__init__(self, **kwargs)

    def write_to(self, file, position_data, director_data, height_data, radius_data, alpha):
        position = np.zeros((3, 2))
        position[:, 0] = position_data[:, 0].copy() - np.array([0, 0, 0.012])
        position[:, 1] = position[:, 0] + director_data[2, :, 0]*height_data/4
        position = self.adapted_position(position)
        radius = radius_data

        string = "// cylinder data\n"
        string += "cylinder{\n"
        
        string += "\t<%f, %f, %f>, " % (position[0, 0], position[1, 0], position[2, 0])
        string += "<%f, %f, %f>, " % (position[0, 1], position[1, 1], position[2, 1])
        string += "%f\n" % radius
        string += "\n\ttexture{\n"
        string += "\t\tpigment{ color rgb" + self.color_string + " transmit %f }\n" % self.alpha_to_transmit(alpha)
        string += "\t\tfinish{ phong 1 }\n\t}\n"
        string += "\tscale<1, 1, 1>*%f\n}\n\n" % self.scale

        file.writelines(string)
