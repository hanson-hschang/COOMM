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
        muscle_init_angle,
        ratio_muscle_position,
        rest_muscle_area,
        max_muscle_stress,
        **kwargs
    ):
        """
        Initialize longitudinal muscle model.

        Parameters
        ----------
        muscle_init_angle :
        ratio_muscle_position :
        rest_muscle_area :
        max_muscle_stress :
        """
        n_elem = rest_muscle_area.shape[0]
        kwargs.setdefault("type_name", "LM")
        MuscleForce.__init__(
            self,
            ratio_muscle_position * (
                [[np.cos(muscle_init_angle)], 
                 [np.sin(muscle_init_angle)],
                 [0]] * 
                np.ones((3, n_elem))
            ),
            rest_muscle_area,
            max_muscle_stress * np.ones(n_elem),
            **kwargs
        )
