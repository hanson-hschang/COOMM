"""
Created on Feb. 21, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from comm._rod_tool import _material_to_lab

class POVRAYMuscle:
    def __init__(self, muscle_color, activation_color):
        self.muscle_color = np.array(muscle_color)
        self.activation_color = np.array(activation_color)

class POVRAYSlenderMuscle(POVRAYMuscle):
    def __init__(self, **kwargs):
        muscle_color = kwargs["muscle_color"]
        POVRAYMuscle.__init__(self, muscle_color, kwargs.get("activation_color", muscle_color))
        self.muscle_label = "// muscle data\n"

    def write_to(self, file, position_data, director_data, muscle_position_data, radius_data, muscle_activation=None, alpha=1.0):
        position = (position_data[:, :-1]+position_data[:, 1:])/2 
        position += _material_to_lab(director_data, muscle_position_data)
        position = self.adapted_position(position)
        radius = radius_data.copy()
        n_elements = position.shape[1]

        if muscle_activation is None:
            muscle_activation = np.ones(n_elements)

        start_index = 0
        end_index = n_elements - 2

        string = self.muscle_label
        string += "union{\n"

        for n in range(start_index, end_index-1):
            # string += "\tsphere{\n"
            # string += "\t\t<%f, %f, %f>, %f\n" % (
            #     position[0, n],
            #     position[1, n],
            #     position[2, n],
            #     radius[n]
            # )
            # string += "\t\ttexture{\n"
            # string += "\t\t\tpigment{ color rgb " + self.color_string 
            # string += " transmit %f }\n" % self.alpha_to_transmit(alpha*muscle_activation[n])
            # string += "\t\t\tfinish{ phong 1 }\n"
            # string += "\t\t}\n"
            # string += "\t\tscale<1, 1, 1>*%f\n" % self.scale
            # string += "\t}\n"

            string += "\tcone{\n"
            string += "\t\t<%f, %f, %f>, %f,\n" % (
                position[0, n], position[1, n], position[2, n], radius[n]
            )
            string += "\t\t<%f, %f, %f>, %f\n" % (
                position[0, n+1], position[1, n+1], position[2, n+1], radius[n+1]
            )
            string += "\t\ttexture{\n"
            string += "\t\t\tpigment{ color rgb " + self.color_string 
            string += " transmit %f }\n" % self.alpha_to_transmit(
                alpha*(muscle_activation[n]+muscle_activation[n+1])/2
            )
            string += "\t\t\tfinish{ phong 1 }\n"
            string += "\t\t}\n"
            string += "\t\tscale<1, 1, 1>*%f\n" % self.scale
            string += "\t}\n"

            # string += "\tsphere{\n"
            # string += "\t\t<%f, %f, %f>, %f\n" % (
            #     position[0, n+1],
            #     position[1, n+1],
            #     position[2, n+1],
            #     radius[n+1]
            # )
            # string += "\t\ttexture{\n"
            # string += "\t\t\tpigment{ color rgb " + self.color_string 
            # string += " transmit %f }\n" % self.alpha_to_transmit(alpha*muscle_activation[n+1])
            # string += "\t\t\tfinish{ phong 1 }\n"
            # string += "\t\t}\n"
            # string += "\t\tscale<1, 1, 1>*%f\n" % self.scale
            # string += "\t}\n"

        string += "}\n\n"

        file.writelines(string)

class POVRAYRingMuscle(POVRAYMuscle):
    def __init__(self, **kwargs):
        muscle_color = kwargs["muscle_color"]
        POVRAYMuscle.__init__(self, muscle_color, kwargs.get("activation_color", muscle_color))
        self.n_muscle_nodes = kwargs.get("n_muscle_nodes", 20)
        self.muscle_label = "// muscle data\n"
    
    def write_to(self, file, position_data, director_data, muscle_position_data, radius_data, muscle_activation=None, alpha=1.0):
        position = (position_data[:, :-1]+position_data[:, 1:])/2
        position_difference = position_data[:, :-1] - position_data[:, 1:]
        
        position += _material_to_lab(director_data, muscle_position_data)
        position = self.adapted_position(position)
        n_elements = position.shape[1]

        major_radius = radius_data.copy()
        minor_radius = np.linalg.norm(position_difference, axis=0)
        
        for n in range(n_elements):
            minor_radius[n] = np.min([major_radius[n]/2, minor_radius[n]])
        

        if muscle_activation is None:
            muscle_activation = np.ones(n_elements)

        string = self.muscle_label
        for n in range(n_elements-1):
            string += "sphere_sweep{\n"
            string += "\tb_spline %d" % (self.n_muscle_nodes+1)
            for nn in range(self.n_muscle_nodes+1):
                theta = nn/self.n_muscle_nodes*2*np.pi
                position_muscle = position[:, n] + (major_radius[n]-minor_radius[n]) * (
                    np.cos(theta)*director_data[0, :, n]+np.sin(theta)*director_data[1, :, n]
                )
                string += ",\n\t<%f, %f, %f>, %f" % (
                    position_muscle[0], position_muscle[1], position_muscle[2], minor_radius[n]
                )
            string += "\n\ttexture{\n"
            string += "\t\tpigment{ color rgb " + self.color_string 
            string += " transmit %f }\n" % self.alpha_to_transmit(alpha*muscle_activation[n])
            string += "\t\tfinish{ phong 1 }\n"
            string += "\t}\n"
            string += "\tscale<1, 1, 1>*%f\n" % self.scale
            string += "}\n"

        string += "\n"

        file.writelines(string)
