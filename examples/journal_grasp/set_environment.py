"""
Created on Sep. 23, 2021
@author: Heng-Sheng (Hanson) Chang
"""

from collections import defaultdict
import numpy as np

import elastica as el
from coomm.callback_func import CylinderCallBack

import sys
sys.path.append("../")          # include examples directory
from set_arm_environment import ArmEnvironment

class Environment(ArmEnvironment):

    def get_systems(self,):
        return [self.shearable_rod, self.cylinder]
    
    def get_data(self,):
        return [self.rod_parameters_dict, self.cylinder_parameters_dict]

    def setup(self):

        self.set_arm()

        """ Set up a cylinder object """
        self.cylinder = el.Cylinder(
            start=np.array([0.06, -0.04, -0.18]),
            # start=np.array([0.04, 0, -0.22]),
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            base_length=0.4,
            base_radius=0.02,
            density=1750
        )

        self.simulator.append(self.cylinder)
        
        self.cylinder_parameters_dict = defaultdict(list)
        self.simulator.collect_diagnostics(self.cylinder).using(
            CylinderCallBack,
            step_skip=self.step_skip,
            callback_params=self.cylinder_parameters_dict
        )

        """ Set up boundary and contact conditions """
        self.simulator.constrain(self.cylinder).using(
            el.OneEndFixedRod,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,)
        )

        self.simulator.connect(self.shearable_rod, self.cylinder).using(
            el.ExternalContact, 1e2, 0.1
        )
