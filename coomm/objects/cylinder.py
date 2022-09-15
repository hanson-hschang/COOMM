__doc__ = """
Cylinder object implementation
"""

import numpy as np

from coomm.objects.object import Object
from coomm.objects.target import Target

class Cylinder(Object):
    """Cylinder.
    """

    def __init__(self, position, director, radius, length, n_elements, cost_weight):
        """__init__.

        Parameters
        ----------
        position :
            position
        director :
            director
        radius :
            radius
        length :
            length
        n_elements :
            n_elements
        cost_weight :
            cost_weight
        """
        Object.__init__(self, n_elements, cost_weight)
        self.position = position.copy()
        self.director = director.copy()
        self.radius = radius
        self.length = length

    def update_pose_from_sphere(self, sphere):
        """update_pose_from_sphere.

        Parameters
        ----------
        sphere :
            sphere
        """
        self.update_position(sphere.position_collection[:, 0])
        self.update_director(sphere.director_collection[:, :, 0])

    def update_position(self, position):
        """update_position.

        Parameters
        ----------
        position :
            position
        """
        self.position = position.copy()

    def update_director(self, director):
        """update_director.

        Parameters
        ----------
        director :
            director
        """
        self.director = director.copy()

    @classmethod
    def get_cylinder(cls, cylinder, n_elements, cost_weight):
        """get_cylinder.

        Parameters
        ----------
        cylinder :
            cylinder
        n_elements :
            n_elements
        cost_weight :
            cost_weight
        """
        return Cylinder(
            cylinder.position_collection[:, 0].copy(),
            cylinder.director_collection[:, :, 0].copy(),
            n_elements, cost_weight
        )

    def calculate_continuous_cost_gradient_wrt_position(self, **kwargs):
        """calculate_continuous_cost_gradient_wrt_position.
        """
        position = 0.5*(kwargs['position'][:, :-1]+kwargs['position'][:, 1:])
        radius = kwargs['radius']
        position_diff = position-self.position[:, None]
        vertical_dist = np.einsum("ij, i -> j", position_diff, self.director[2, :])
        horizontal_position_diff = position_diff - vertical_dist * self.director[2, :, None]
        position_dist = np.linalg.norm(horizontal_position_diff, axis=0)
        adjust_distance_ratio = (position_dist-(radius+self.radius))/position_dist
        adjust_distance_ratio[adjust_distance_ratio>0] = 0
        adjust_distance_ratio[vertical_dist>self.length/2] = 0
        adjust_distance_ratio[-vertical_dist>self.length/2] = 0
        self.cost_gradient.continuous.wrt_position[:, :] = (
            self.cost_weight['position'] * horizontal_position_diff * adjust_distance_ratio
        )

    def calculate_continuous_cost_gradient_wrt_director(self, **kwargs):
        """calculate_continuous_cost_gradient_wrt_director.
        """
        pass
    
    def calculate_discrete_cost_gradient_wrt_position(self, **kwargs):
        """calculate_discrete_cost_gradient_wrt_position.
        """
        pass
    
    def calculate_discrete_cost_gradient_wrt_director(self, **kwargs):
        """calculate_discrete_cost_gradient_wrt_director.
        """
        pass

class CylinderTarget(Cylinder, Target):
    """CylinderTarget.
    """

    def __init__(self, position, director, radius, length, n_elements, cost_weight, target_cost_weight, **kwargs):
        """__init__.

        Parameters
        ----------
        position :
        director :
        radius :
        length :
        n_elements :
        cost_weight :
        target_cost_weight :
        """
        Cylinder.__init__(self, position, director, radius, length, n_elements, cost_weight)
        Target.__init__(self, target_cost_weight)
        self.director_cost_flag = kwargs.get('director_cost_flag', False)

    @classmethod
    def get_cylinder(cls, cylinder, n_elements, cost_weight, target_cost_weight, **kwargs):
        """get_cylinder.

        Parameters
        ----------
        cylinder :
            cylinder
        n_elements :
            n_elements
        cost_weight :
            cost_weight
        target_cost_weight :
            target_cost_weight
        """
        return CylinderTarget(
            cylinder.position_collection[:, 0].copy(),
            cylinder.director_collection[:, :, 0].copy(),
            cylinder.radius,
            cylinder.length,
            n_elements, cost_weight, target_cost_weight, **kwargs
        )
    
    def calculate_continuous_cost_gradient_wrt_position(self, **kwargs):
        """calculate_continuous_cost_gradient_wrt_position.
        """
        Cylinder.calculate_continuous_cost_gradient_wrt_position(self, **kwargs)
        position = 0.5*(kwargs['position'][:, :-1]+kwargs['position'][:, 1:])
        radius = kwargs['radius']
        position_diff = position-self.position[:, None]
        
        # vertical_dist = np.einsum("ij, i -> j", position_diff, self.director[2, :])
        # horizontal_position_diff = position_diff - vertical_dist * self.director[2, :, None]
        # position_dist = np.linalg.norm(horizontal_position_diff, axis=0)
        # adjust_distance_ratio = (position_dist-(radius+self.radius))/position_dist
        # adjust_distance_ratio[adjust_distance_ratio<0] = 0
        # self.cost_gradient.continuous.wrt_position[:, :] += (
        #     self.cost_weight['position'] * horizontal_position_diff * adjust_distance_ratio
        # )

        position_dist = np.linalg.norm(position_diff, axis=0)
        adjust_distance_ratio = (position_dist-(radius+self.radius))/position_dist
        
        adjust_distance_ratio[adjust_distance_ratio<0] = 0
        self.cost_gradient.continuous.wrt_position[:, :] += (
            self.target_cost_weight['position'] * position_diff * adjust_distance_ratio
        )
    
    def calculate_continuous_cost_gradient_wrt_director(self, **kwargs):
        """calculate_continuous_cost_gradient_wrt_director.
        """
        position = 0.5*(kwargs['position'][:, :-1]+kwargs['position'][:, 1:])
        director = kwargs['director'][:, :, :]
        n_elems = director.shape[2]
        vector = np.zeros((3, n_elems))
        coefficient = np.zeros(n_elems)
        # for n in range(n_elems):
        #     skew_symmetric_matrix = director[:, :, n] @ self.director.T - self.director @ director[:, :, n].T
        #     vector[0, n] = skew_symmetric_matrix[1, 2]
        #     vector[1, n] = -skew_symmetric_matrix[0, 2]
        #     vector[2, n] = skew_symmetric_matrix[0, 1]
        #     self.cost_gradient.continuous.wrt_director[:, n] = (
        #         self.target_cost_weight['director'][n] * director[:, :, n].T @ vector[:, n]
        #     )
        
        for n in range(n_elems):
            vector[1, n] = - np.dot(director[2, :, n], self.director[2, :])
            vector[2, n] = np.dot(director[1, :, n], self.director[2, :])
            coefficient[n] = np.dot(director[0, :, n], self.director[2, :])
            self.cost_gradient.continuous.wrt_director[:, n] = (
                self.target_cost_weight['director'][n] * coefficient[n] * director[:, :, n].T @ vector[:, n]
            )

        vector = np.zeros((3, n_elems))
        coefficient = np.zeros(n_elems)
        for n in range(n_elems):
            position_diff = self.position - position[:, n]
            position_diff = position_diff / np.linalg.norm(position_diff)
            vector[1, n] = - np.dot(director[2, :, n], position_diff)
            vector[2, n] = np.dot(director[1, :, n], position_diff)
            coefficient[n] = min(np.dot(self.director[0, :], position_diff), 0)
            self.cost_gradient.continuous.wrt_director[:, n] += (
                self.target_cost_weight['director'][n] * coefficient[n] * director[:, :, n].T @ vector[:, n]
            )

    def calculate_discrete_cost_gradient_wrt_position(self, **kwargs):
        """calculate_discrete_cost_gradient_wrt_position.
        """
        # position = 0.5*(kwargs['position'][:, -1]+kwargs['position'][:, -2])
        # self.cost_gradient.discrete.wrt_position[:, -1] = (
        #     self.target_cost_weight['position'] * (position-self.position)
        # )
        pass
    
    def calculate_discrete_cost_gradient_wrt_director(self, **kwargs):
        """calculate_discrete_cost_gradient_wrt_director.
        """
        # director = kwargs['director'][:, :, -1]
        # vector = np.zeros(3)
        # skew_symmetric_matrix = director @ self.director.T - self.director @ director.T
        # vector[0] = skew_symmetric_matrix[1, 2]
        # vector[1] = -skew_symmetric_matrix[0, 2]
        # vector[2] = skew_symmetric_matrix[0, 1]
        # self.cost_gradient.discrete.wrt_director[:, -1] = (
        #     self.target_cost_weight['director'] * director.T @ vector
        # )
        pass
