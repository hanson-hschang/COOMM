"""
Created on Jan. 11, 2022
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from comm.povray.povray_base import POVRAYBase

class POVRAYSphere(POVRAYBase):
    def __init__(self, **kwargs):
        POVRAYBase.__init__(self, **kwargs)

    def write_to(self, file, position_data, radius_data, alpha):
        position = self.adapted_position(
            position_data[:, :]
        )
        radius = radius_data

        string = "// ball data\n"
        string += "sphere{\n"
        
        string += ("\t<%f, %f, %f>, %f" % (position[0, 0], position[1, 0], position[2, 0], radius))
        string += "\n\ttexture{\n"
        string += "\t\tpigment{ color rgb" + self.color_string + " transmit %f }\n" % self.alpha_to_transmit(alpha)
        string += "\t\tfinish{ phong 1 }\n\t}\n"
        string += "\tscale<1, 1, 1>*%f\n}\n\n" % self.scale

        file.writelines(string)
