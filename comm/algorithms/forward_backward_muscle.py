__doc__ = """
Forward Backward Muscle model implementation.
"""

from os import stat
import numpy as np
from numba import njit

from tqdm import tqdm

from elastica._linalg import _batch_matvec, _batch_cross
from elastica._calculus import quadrature_kernel
from elastica.external_forces import inplace_addition

from comm.algorithms.forward_backward import ForwardBackward
from comm._rod_tool import (
    inverse,
    _lab_to_material,
    _material_to_lab,
    sigma_to_shear,
    average2D,
    calculate_dilatation
)

class ForwardBackwardMuscle(ForwardBackward):
    """ForwardBackwardMuscle.
    """

    def __init__(self, rod, muscles, algo_config, **kwargs):
        """__init__.

        Parameters
        ----------
        rod :
        muscles :
        algo_config :
        """
        ForwardBackward.__init__(self, rod, algo_config, **kwargs)
        self.activation_diff_tolerance = self.config['activation_diff_tolerance']
        self.muscles = muscles
        self.s_activations = []
        self.activations = []
        self.prev_activations = []
        for muscle in self.muscles:
            self.s_activations.append(muscle.s_activation.copy())
            self.activations.append(muscle.activation.copy())
            self.prev_activations.append(np.full(muscle.activation.shape, np.inf))

    def save_to_prev_activations(self, activations):
        """save_to_prev_activations.

        Parameters
        ----------
        activations :
        """
        for i in range(len(self.prev_activations)):
            self.prev_activations[i] = activations[i].copy()

    def update(self, iteration):
        """update.

        Parameters
        ----------
        iteration :

        Returns
        -------
        """
        self.save_to_prev_activations(self.activations)

        # find the equlibrium for the current muscle activations
        self.find_equilibrium_strain(
            self.static_rod.sigma, self.static_rod.kappa,
            self.static_rod.shear_matrix, self.static_rod.bend_matrix,
            self.static_rod.dilatation, self.static_rod.voronoi_dilatation,
            *self.calculate_total_muscle_forces_couples()
        )

        # forward path
        self.static_rod.update_from_strain(
            self.static_rod.sigma, self.static_rod.kappa
        )

        # update cost-related terms in objects
        self.objects(
            position=self.static_rod.position_collection,
            director=self.static_rod.director_collection,
            radius=self.static_rod.radius
        )

        # backward path
        self.discrete_cost_gradient_condition()
        self.continuous_cost_gradient_condition()

        self.costate_backward_evolution(
            self.static_rod.rest_lengths, 
            self.static_rod.director_collection, 
            self.static_rod.sigma,
            self.costate.internal_force_discrete_jump,
            self.costate.internal_couple_discrete_jump,
            self.costate.internal_force_derivative,
            self.costate.internal_couple_derivative,
            self.costate.internal_force, self.costate.internal_couple
        )
        
        # update activations
        self.update_activations(
            self.find_target_activations()
        )

        # check if the updated activations are similar with previous ones
        self.done = self.check_activations_difference()

        return ForwardBackward.update(self, iteration)

    def calculate_total_muscle_forces_couples(self):
        """calculate_total_muscle_forces_couples.

        Returns
        -------
        muscle_forces:
        muscle_couples:
        """

        muscle_forces = np.zeros(self.static_rod.sigma.shape)
        muscle_couples = np.zeros(self.static_rod.kappa.shape)

        for muscle, activation in zip(self.muscles, self.activations):
            muscle.apply_activation(activation)
            muscle(self.static_rod)
            inplace_addition(muscle_forces, muscle.internal_force)
            inplace_addition(muscle_couples, muscle.internal_couple)

        return muscle_forces, muscle_couples

    @staticmethod
    @njit(cache=True)
    def find_equilibrium_strain(
        sigma, kappa,
        shear_matrix, bend_matrix,
        dilatation, voronoi_dilatation,
        muscle_forces, muscle_couples
    ):
        kappa[:, :] = - _batch_matvec(
            inverse(bend_matrix/voronoi_dilatation**3), muscle_couples
        )
        sigma[:, :] = - _batch_matvec(
            inverse(shear_matrix/dilatation), muscle_forces
        )

    def discrete_cost_gradient_condition(self,):
        self.costate.internal_force_discrete_jump[:, -1] = (
            - self.objects.cost_gradient.discrete.wrt_position[:, -1]
        )
        # print(self.costate.internal_force_discrete_jump[:, :])
        self.costate.internal_couple_discrete_jump[:, -1] = (
            - self.objects.cost_gradient.discrete.wrt_director[:, -1]
        )

    def continuous_cost_gradient_condition(self,):
        self.costate.internal_force_derivative[:, :] = (
            self.objects.cost_gradient.continuous.wrt_position
        )
        self.costate.internal_couple_derivative[:, :] = (
            self.objects.cost_gradient.continuous.wrt_director
        )

    # This method is not completed. It only compute tip jump rather than any jump at any location
    @staticmethod
    @njit(cache=True)
    def costate_backward_evolution(
        rest_lengths, director, sigma,
        internal_force_lab_frame_at_tip,            # This name needs to be changed to internal_force_lab_frame_jump or something similar
        internal_couple_lab_frame_at_tip,           # This name needs to be changed
        internal_force_lab_frame_derivative,
        internal_couple_lab_frame_derivative,
        internal_force, internal_couple
    ):
        blocksize = rest_lengths.shape[0]
        internal_force_lab_frame = np.zeros((3, blocksize))
        internal_couple_lab_frame = np.zeros((3, blocksize))
        shear = sigma_to_shear(sigma)
        dilatation, _ = calculate_dilatation(shear)

        # TODO: The next line is incomplete
        internal_force_lab_frame[:, -1] = internal_force_lab_frame_at_tip[:, -1]
        
        force_derivative = average2D(internal_force_lab_frame_derivative)

        # n_s = f
        for k in range(blocksize-1):
            internal_force_lab_frame[:, -1-k-1] = internal_force_lab_frame[:, -1-k] - (
                force_derivative[:, -1-k] * rest_lengths[-1-k] * dilatation[-1-k]
            )
        internal_force[:, :] = _lab_to_material(director, internal_force_lab_frame)

        internal_couple_lab_frame[:, -1] = internal_couple_lab_frame_at_tip[:, -1]

        # m_s = -r_s x n + c
        internal_couple_lab_frame_derivative[:, :] -= _batch_cross(
            _material_to_lab(director, shear), internal_force_lab_frame
        )
        couple_derivative = average2D(internal_couple_lab_frame_derivative)
        for k in range(blocksize-1):
            internal_couple_lab_frame[:, -1-k-1] = internal_couple_lab_frame[:, -1-k] - (
                couple_derivative[:, -1-k] * rest_lengths[-1-k] * dilatation[-1-k]
            )
        internal_couple[:, :] = average2D(
            _lab_to_material(director, internal_couple_lab_frame)
        )

    def find_target_activations(self):
        """find_target_activations.
        """
        target_activations = []
        for muscle in self.muscles:
            muscle.apply_activation(np.ones(muscle.activation.shape))
            muscle(self.static_rod)
            target_activations.append(
                self.calculate_target_activation(
                    self.costate.internal_force,
                    self.costate.internal_couple,
                    self.static_rod.dilatation,
                    self.static_rod.voronoi_dilatation,
                    self.static_rod.shear_matrix,
                    self.static_rod.bend_matrix,
                    muscle.internal_force, muscle.internal_couple
                )
            )
        return target_activations

    @staticmethod
    @njit(cache=True)
    def calculate_target_activation(
        internal_force, internal_couple,
        dilatation, voronoi_dilatation,
        shear_matrix, bend_matrix,
        muscle_internal_force, muscle_internal_couple
    ):
        blocksize = internal_force.shape[1]
        target_activation = np.zeros(blocksize)
        temp_shear = _batch_matvec(
            inverse(shear_matrix/dilatation),
            muscle_internal_force
        )
        temp_kappa = _batch_matvec(
            inverse(bend_matrix/voronoi_dilatation),
            muscle_internal_couple
        )
        temp_force_innerproduct = np.zeros(blocksize)
        temp_couple_innerproduct = np.zeros(blocksize+1)
        for k in range(blocksize-1):
            for i in range(3):
                temp_force_innerproduct[k] += internal_force[i, k] * temp_shear[i, k]
                temp_couple_innerproduct[k+1] += internal_couple[i, k+1] * temp_kappa[i, k+1]
        temp_force_innerproduct[-1] += internal_force[i, -1] * temp_shear[i, -1]
        for k in range(blocksize):
            target_activation[k] = -temp_force_innerproduct[k] - 0.5*(
                temp_couple_innerproduct[k] + temp_couple_innerproduct[k+1]
            )
        return target_activation

    def update_activations(self, target_activations):
        """update_activations.

        Parameters
        ----------
        target_activations :
        """
        for activation, target_activation, in zip(self.activations, target_activations):
            activation[:] -= self.stepsize * (activation - target_activation)
            activation[:] = np.clip(activation, 0, 1)

    def check_activations_difference(self):
        """check_activations_difference

        Returns
        -------
        """
        norm = 0
        for prev_activation, activation in zip(self.prev_activations, self.activations):
            norm += np.sum((activation-prev_activation)**2)/activation.shape[0]
        norm /= len(self.activations)
        return True if norm < self.activation_diff_tolerance else False
