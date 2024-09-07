"""
Created on Sep. 23, 2021
@author: Heng-Sheng (Hanson) Chang
"""

from collections import defaultdict
import numpy as np

import elastica as el
from coomm.callback_func import SphereCallBack

from examples.set_arm_environment import ArmEnvironment

class Environment(ArmEnvironment):
    
    def get_data(self):
        return [self.rod_parameters_dict, self.sphere_parameters]

    def setup(self):
        self.set_arm()
        self.set_target()

    def set_target(self):
        self.spheres: list[el.Sphere] = []
        self.sphere_parameters: list[defaultdict] = []
        sphere_positions = [
            np.array([0.01, 0.15, 0.06]),
            np.array([0.02, 0.02, 0.02]),
        ]
        angle = np.pi * 0.75
        sphere_directors = [
            np.array([[ 0, 0, 1],
                      [ 0, 1, 0],
                      [-1, 0, 0]]),
            np.array([[ 0, 0, 1],
                      [ np.cos(angle), np.sin(angle), 0],
                      [-np.sin(angle), np.cos(angle), 0]])
        ]
        
        """ Set up a sphere object """
        for position, director in zip(sphere_positions, sphere_directors):
            sphere = el.Sphere(
                center=position,
                base_radius=0.006,
                density=1000
            )
            sphere.director_collection[:, :, 0] = director
            self.simulator.append(sphere)
            sphere_parameters_dict = defaultdict(list)
            self.simulator.collect_diagnostics(sphere).using(
                SphereCallBack,
                step_skip=self.step_skip,
                callback_params=sphere_parameters_dict
            )

            self.spheres.append(sphere)
            self.sphere_parameters.append(sphere_parameters_dict)
        
            """ Set up boundary conditions """
            self.simulator.constrain(sphere).using(
                el.OneEndFixedRod,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,)
            )

        # target_radius = 0.006
        # self.sphere = el.Sphere(
        #     center=np.array([0.01, 0.15, 0.06]),
        #     base_radius=target_radius,
        #     density=1000
        # )
        # self.sphere.director_collection[:, :, 0] = np.array(
        #     [[ 0, 0, 1],
        #      [ 0, 1, 0],
        #      [-1, 0, 0]]
        # )
        # self.simulator.append(self.sphere)
        # self.sphere_parameters_dict = defaultdict(list)
        # self.simulator.collect_diagnostics(self.sphere).using(
        #     SphereCallBack,
        #     step_skip=self.step_skip,
        #     callback_params=self.sphere_parameters_dict
        # )
        
        # """ Set up boundary conditions """
        # self.simulator.constrain(self.shearable_rod).using(
        #     el.OneEndFixedRod,
        #     constrained_position_idx=(0,),
        #     constrained_director_idx=(0,)
        # )