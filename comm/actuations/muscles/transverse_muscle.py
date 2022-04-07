__doc__ = """
Transverse muscle implementation
"""

import numpy as np
from numba import njit

from comm.actuations.muscles.muscle import MuscleForce

class TransverseMuscle(MuscleForce):
    """TransverseMuscle.
    """

    def __init__(
        self,
        rest_muscle_area,
        max_muscle_stress,
        **kwargs
    ):
        """__init__.

        Parameters
        ----------
        rest_muscle_area :
        max_muscle_stress :
        """
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
