"""
Created on Feb. 21, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from coomm.povray.povray_base import POVRAYBase

class POVRAYRod(POVRAYBase):
    def __init__(self, **kwargs):
        POVRAYBase.__init__(self, **kwargs)

    def write_to(self, file, position_data, radius_data, alpha):
        position = self.adapted_position(
            (position_data[:, :-1]+position_data[:, 1:])/2 
        )
        radius = radius_data
        n_elements = radius.shape[0]

        string = "// rod data\n"
        string += "sphere_sweep{\n\tb_spline %d" % n_elements
        for n in range(n_elements):
            string += (",\n\t<%f, %f, %f>, %f" % (position[0, n], position[1, n], position[2, n], radius[n]))
        string += "\n\ttexture{\n"
        string += "\t\tpigment{ color rgb" + self.color_string + " transmit %f }\n" % self.alpha_to_transmit(alpha)
        string += "\t\tfinish{ phong 1 }\n\t}\n"
        string += "\tscale<1, 1, 1>*%f\n}\n\n" % self.scale

        file.writelines(string)
