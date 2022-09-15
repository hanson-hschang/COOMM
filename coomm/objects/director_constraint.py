__doc__ = """
Director constraint object implementation
"""

import numpy as np

from coomm.objects.object import Object
from coomm.objects.target import Target

class DirectorConstraint(Object, Target):
    """DirectorConstraint.
    """

    def __init__(self, director, n_elements, cost_weight, target_cost_weight, **kwargs):
        """__init__.

        Parameters
        ----------
        director :
            director
        n_elements :
            n_elements
        cost_weight :
            cost_weight
        target_cost_weight :
            target_cost_weight
        kwargs :
            kwargs
        """
        Object.__init__(self, n_elements, cost_weight)
        Target.__init__(self, target_cost_weight)
        self.director = director.copy()
        
        # self.director_flags = [True, True, True]
        # for i in range(len(self.director_flags)):
        #     if self.director[i, 0] is None:
        #         self.director_flags[i] = False

    def update_constraint(self, target_director, director):
        """update_constraint.

        Parameters
        ----------
        target_director :
            target_director
        director :
            director
        """
        self.update_director(target_director, director)

    # def update_position(self, position):
    #     self.position = position.copy()

    def update_director(self, target_director, director):
        """update_director.

        Parameters
        ----------
        target_director :
            target_director
        director :
            director
        """
        # for i, director_flag in enumerate(self.director_flags):
        #     self.director[i, :] = (
        #         target_director[i, :].copy() if director_flag else director[i, :].copy()
        #     ) 
        pass

    # @classmethod
    # def get_point_from_sphere(cls, sphere, n_elements, cost_weight):
    #     return Point(
    #         sphere.position_collection[:, 0].copy(),
    #         sphere.director_collection[:, :, 0].copy(),
    #         n_elements, cost_weight
    #     )

    def calculate_continuous_cost_gradient_wrt_position(self, **kwargs):
        """calculate_continuous_cost_gradient_wrt_position.

        Parameters
        ----------
        kwargs :
            kwargs
        """
        pass
    
    def calculate_continuous_cost_gradient_wrt_director(self, **kwargs):
        """calculate_continuous_cost_gradient_wrt_director.

        Parameters
        ----------
        kwargs :
            kwargs
        """
        director = kwargs['director'].copy()
        vector = np.zeros((3, self.n_elements))
        skew_symmetric_matrix = (
            np.einsum('ijn,kjn->ikn', director, self.director) - 
            np.einsum('ijn,kjn->ikn', self.director, director)
        )
        vector[0, :] = skew_symmetric_matrix[1, 2, :]
        vector[1, :] = -skew_symmetric_matrix[0, 2, :]
        vector[2, :] = skew_symmetric_matrix[0, 1, :]
        self.cost_gradient.discrete.wrt_director[:, :] = (
            self.target_cost_weight['director'] * np.einsum('jik,jk->ik', director, vector)
        )
        # print("director")
        # print(director)
        # print("self.director")
        # print(self.director)
        # print("vector")
        # print(vector)
    
    def calculate_discrete_cost_gradient_wrt_position(self, **kwargs):
        """calculate_discrete_cost_gradient_wrt_position.

        Parameters
        ----------
        kwargs :
            kwargs
        """
        pass
    
    def calculate_discrete_cost_gradient_wrt_director(self, **kwargs):
        """calculate_discrete_cost_gradient_wrt_director.

        Parameters
        ----------
        kwargs :
            kwargs
        """
        pass

# TODO: What is this? Do we want to include or remove?
# class PointTarget(Point, Target):
#     def __init__(self, position, director, n_elements, cost_weight, target_cost_weight, **kwargs):
#         Point.__init__(self, position, director, n_elements, cost_weight)
#         Target.__init__(self, target_cost_weight)
#         self.director_cost_flag = kwargs.get('director_cost_flag', False)

#     @classmethod
#     def get_point_target_from_sphere(cls, sphere, n_elements, cost_weight, target_cost_weight, **kwargs):
#         return PointTarget(
#             sphere.position_collection[:, 0].copy(),
#             sphere.director_collection[:, :, 0].copy(),
#             n_elements, cost_weight, target_cost_weight, **kwargs
#         )
    
#     def calculate_continuous_cost_gradient_wrt_position(self, **kwargs):
#         pass
    
#     def calculate_continuous_cost_gradient_wrt_director(self, **kwargs):
#         pass
    
#     def calculate_discrete_cost_gradient_wrt_position(self, **kwargs):
#         position = 0.5*(kwargs['position'][:, -1]+kwargs['position'][:, -2])
#         self.cost_gradient.discrete.wrt_position[:, -1] = (
#             self.target_cost_weight['position'] * (position-self.position)
#         )
    
#     def calculate_discrete_cost_gradient_wrt_director(self, **kwargs):
#         director = kwargs['director'][:, :, -1]
#         vector = np.zeros(3)
#         skew_symmetric_matrix = director @ self.director.T - self.director @ director.T
#         vector[0] = skew_symmetric_matrix[1, 2]
#         vector[1] = -skew_symmetric_matrix[0, 2]
#         vector[2] = skew_symmetric_matrix[0, 1]
#         self.cost_gradient.discrete.wrt_director[:, -1] = (
#             self.target_cost_weight['director'] * director.T @ vector
#         )
