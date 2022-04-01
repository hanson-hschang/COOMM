"""
Created on Oct. 15, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from numba import njit

from actuations.muscles.muscle import MuscleForce

class TransverseMuscle(MuscleForce):
    def __init__(
        self,
        rest_muscle_area,
        max_muscle_stress,
        **kwargs
    ):
        n_elem = rest_muscle_area.shape[0]
        kwargs.setdefault("type_name", "TM")
        MuscleForce.__init__(
            self,
            [[0], [0], [0]] * np.ones((3, n_elem)),
            rest_muscle_area,
            -max_muscle_stress * np.ones(n_elem),
            **kwargs
        )

    @staticmethod
    @njit(cache=True)
    def calculate_muscle_length(muscle_length, muscle_strain):
        blocksize = muscle_length.shape[0]
        for i in range(blocksize):
            muscle_length[i] = 1 / (
                muscle_strain[0, i]**2 + 
                muscle_strain[1, i]**2 + 
                muscle_strain[2, i]**2
            )**0.25
