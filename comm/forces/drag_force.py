__doc__ = """
Force module
"""

import numpy as np
from numba import njit

from elastica._calculus import quadrature_kernel
from elastica.external_forces import inplace_addition

from elastica.external_forces import NoForces

from comm._rod_tool import _lab_to_material, _material_to_lab, average2D


class DragForce(NoForces):
    """DragForce.
    """

    def __init__(self, rho_environment,
        c_per, c_tan, system,
        step_skip: int, callback_params: dict
    ):
        """__init__.

        Parameters
        ----------
        rho_environment :
        c_per :
        c_tan :
        system :
        step_skip : int
        callback_params : dict
        """

        self.rho_environment = rho_environment
        self.c_per = c_per
        self.c_tan = c_tan

        self.scale_per = 0.5 * self.rho_environment * c_per
        self.scale_tan = 0.5 * self.rho_environment * c_tan

        self.velocity_material_frame = np.zeros((3, system.n_elems))
        self.drag_force_material_frame = np.zeros((3, system.n_elems))
        self.drag_force = np.zeros((3, system.n_elems+1))

        self.step = 0
        self.every = step_skip
        self.callback_params = callback_params
    
    def apply_torques(self, system, time: np.float = 0.0):
        """apply_torques.

        Parameters
        ----------
        system :
        time : np.float
        """
        Pa = 2 * system.radius * system.lengths
        Sa = Pa * np.pi
        self.calculate_drag_force(
            self.scale_per*Pa, self.scale_tan*Sa,
            system.director_collection, system.velocity_collection,
            self.velocity_material_frame,
            self.drag_force_material_frame, self.drag_force,
        )
        inplace_addition(system.external_forces, self.drag_force)
        self.callback()

    def callback(self):
        """callback.
        """
        if self.step % self.every == 0:
            self.callback_func()
        self.step += 1

    def callback_func(self):
        """callback_func.
        """
        self.callback_params['velocity_material_frame'].append(
            self.velocity_material_frame.copy()
        )
        self.callback_params['drag_froce_material_frame'].append(
            self.drag_force_material_frame.copy()
        )

    @staticmethod
    @njit(cache=True)
    def calculate_drag_force(
        scale_per, scale_tan,
        director, velocity,
        velocity_material_frame,
        drag_force_material_frame, drag_force
        ):
        velocity_material_frame[:, :] = (
            _lab_to_material(director, average2D(velocity))
            )
        square_velocity_with_direction = (
            np.abs(velocity_material_frame) * velocity_material_frame
            )
        drag_force_material_frame[:2, :] = (
            - scale_per * square_velocity_with_direction[:2, :]
            )
        drag_force_material_frame[2, :] = (
            - scale_tan * square_velocity_with_direction[2, :]
            )
        drag_force[:, :] = (
            quadrature_kernel(
                _material_to_lab(director, drag_force_material_frame)
                )
            )
