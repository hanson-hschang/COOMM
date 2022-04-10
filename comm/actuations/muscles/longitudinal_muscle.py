__doc__ = """
Longitudinal muscle model definition.
"""

import numpy as np
from numba import njit

from comm.actuations.muscles.muscle import MuscleForce

class LongitudinalMuscle(MuscleForce):
    """LongitudinalMuscle
    """

    def __init__(
        self,
        muscle_init_angle: float,
        ratio_muscle_position,
        rest_muscle_area,
        max_muscle_stress: float,
        **kwargs
    ):
        """
        Initialize longitudinal muscle model.

        Parameters
        ----------
        muscle_init_angle : float
        ratio_muscle_position :
        rest_muscle_area :
        max_muscle_stress : float
        """
        n_elem = rest_muscle_area.shape[0]
        ratio_muscle_position = ratio_muscle_position * \
            np.array([np.cos(muscle_init_angle), np.sin(muscle_init_angle), 0])
        ratio_muscle_position = np.repeat(ratio_muscle_position[:,None], n_elem, axis=1)
        max_muscle_stress = max_muscle_stress * np.ones(n_elem),
        super().__init__(
            ratio_muscle_position,
            rest_muscle_area,
            max_muscle_stress,
            type_name="LM",
            **kwargs
        )
