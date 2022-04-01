"""
Created on Nov. 16, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np

from objects.object import Object
from objects.target import Target

class Point(Object):
    def __init__(self, position, director, n_elements, cost_weight):
        Object.__init__(self, n_elements, cost_weight)
        self.position = position.copy()
        self.director = director.copy()

    def update_pose_from_sphere(self, sphere):
        self.update_position(sphere.position_collection[:, 0])
        self.update_director(sphere.director_collection[:, :, 0])

    def update_position(self, position):
        self.position = position.copy()

    def update_director(self, director):
        self.director = director.copy()

    @classmethod
    def get_point_from_sphere(cls, sphere, n_elements, cost_weight):
        return Point(
            sphere.position_collection[:, 0].copy(),
            sphere.director_collection[:, :, 0].copy(),
            n_elements, cost_weight
        )

    def calculate_continuous_cost_gradient_wrt_position(self, **kwargs):
        pass
    
    def calculate_continuous_cost_gradient_wrt_director(self, **kwargs):
        pass
    
    def calculate_discrete_cost_gradient_wrt_position(self, **kwargs):
        pass
    
    def calculate_discrete_cost_gradient_wrt_director(self, **kwargs):
        pass

class PointTarget(Point, Target):
    def __init__(self, position, director, n_elements, cost_weight, target_cost_weight, **kwargs):
        Point.__init__(self, position, director, n_elements, cost_weight)
        Target.__init__(self, target_cost_weight)
        self.director_cost_flag = kwargs.get('director_cost_flag', False)

    @classmethod
    def get_point_target_from_sphere(cls, sphere, n_elements, cost_weight, target_cost_weight, **kwargs):
        return PointTarget(
            sphere.position_collection[:, 0].copy(),
            sphere.director_collection[:, :, 0].copy(),
            n_elements, cost_weight, target_cost_weight, **kwargs
        )
    
    def calculate_continuous_cost_gradient_wrt_position(self, **kwargs):
        pass
    
    def calculate_continuous_cost_gradient_wrt_director(self, **kwargs):
        pass
    
    def calculate_discrete_cost_gradient_wrt_position(self, **kwargs):
        position = 0.5*(kwargs['position'][:, -1]+kwargs['position'][:, -2])
        self.cost_gradient.discrete.wrt_position[:, -1] = (
            self.target_cost_weight['position'] * (position-self.position)
        )
    
    def calculate_discrete_cost_gradient_wrt_director(self, **kwargs):
        director = kwargs['director'][:, :, -1]
        vector = np.zeros(3)
        skew_symmetric_matrix = director @ self.director.T - self.director @ director.T
        vector[0] = skew_symmetric_matrix[1, 2]
        vector[1] = -skew_symmetric_matrix[0, 2]
        vector[2] = skew_symmetric_matrix[0, 1]
        self.cost_gradient.discrete.wrt_director[:, -1] = (
            self.target_cost_weight['director'] * director.T @ vector
        )
