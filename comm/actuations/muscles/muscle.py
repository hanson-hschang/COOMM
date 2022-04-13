__doc__ = """
Muscle base class implementation.
"""

from typing import Union

from collections import defaultdict
import numpy as np
from numba import njit

from elastica._linalg import _batch_cross
from elastica._calculus import quadrature_kernel
from elastica.external_forces import inplace_addition
from comm._rod_tool import average2D, difference2D, sigma_to_shear

from comm.actuations.actuation import (
    _force_induced_couple,
    _internal_to_external_load,
    ContinuousActuation,
    ApplyActuations,
)


@njit(cache=True)
def unit_weight(input):
    return np.ones(input.shape)


@njit(cache=True)
def force_length_weight_guassian(muscle_length, sigma=0.25):
    force_weight = np.exp(-0.5 * ((muscle_length - 1) / sigma) ** 2)
    return force_weight


# force-length curve (x) = 3.06 x^3 - 13.64 x^2 + 18.01 x - 6.44
@njit(cache=True)
def force_length_weight_poly(
    muscle_length, f_l_coefficients=np.array([-6.44, 18.01, -13.64, 3.06])
):
    degree = f_l_coefficients.shape[0]

    blocksize = muscle_length.shape[0]
    force_weight = np.zeros(blocksize)
    for i in range(blocksize):
        for power in range(degree):
            force_weight[i] += f_l_coefficients[power] * (muscle_length[i] ** power)
        force_weight[i] = (
            0 if (force_weight[i] < 0) or (muscle_length[i] > 2) else force_weight[i]
        )
    return force_weight


class MuscleInfo:
    def __init__(self, type_name, index, **kwargs):
        self.type_name = type_name
        self.index = index

    def __str__(
        self,
    ):
        return "{}_".format(self.index) + self.type_name


class Muscle(ContinuousActuation, MuscleInfo):
    """Muscle base class"""

    def __init__(
        self,
        ratio_muscle_position,
        rest_muscle_area,
        type_name: str = "muscle",
        index: int = 0,
        **kwargs,
    ):
        """__init__.

        Parameters
        ----------
        ratio_muscle_position :
        rest_muscle_area :
        """
        self.n_elements = rest_muscle_area.shape[0]
        self.s = np.linspace(0, 1, self.n_elements + 1)
        ContinuousActuation.__init__(self, self.n_elements)
        self.muscle_normalized_length = np.zeros(self.n_elements)
        self.muscle_rest_length = np.ones(self.n_elements)
        self.muscle_length = np.zeros(self.n_elements)
        self.muscle_tangent = np.zeros((3, self.n_elements))
        self.muscle_strain = np.zeros((3, self.n_elements))
        self.muscle_position = np.zeros((3, self.n_elements))
        self.ratio_muscle_position = ratio_muscle_position.copy()
        self.rest_muscle_area = rest_muscle_area.copy()
        self.muscle_area = self.rest_muscle_area.copy()
        MuscleInfo.__init__(self, type_name=type_name, index=index)

    def __call__(self, system):
        self.calculate_muscle_area(
            self.rest_muscle_area, self.muscle_area, system.dilatation
        )
        self.calculate_muscle_position(
            self.muscle_position, system.radius, self.ratio_muscle_position
        )
        self.calculate_muscle_strain(
            self.muscle_strain,
            self.muscle_position,
            system.sigma,
            system.kappa,
            system.rest_voronoi_lengths,
            system.voronoi_dilatation,
        )
        self.calculate_muscle_tangent(self.muscle_tangent, self.muscle_strain)

    def set_current_length_as_rest_length(self, system):
        """set_current_length_as_rest_length.

        Parameters
        ----------
        system :
        """
        Muscle.__call__(self, system)
        self.calculate_muscle_length(self.muscle_length, self.muscle_strain)
        self.muscle_rest_length[:] = self.muscle_length

    @staticmethod
    @njit(cache=True)
    def calculate_muscle_area(rest_muscle_area, muscle_area, dilatation):
        muscle_area[:] = rest_muscle_area / dilatation

    @staticmethod
    @njit(cache=True)
    def calculate_muscle_position(muscle_position, radius, ratio_muscle_position):
        muscle_position[:, :] = radius * ratio_muscle_position

    @staticmethod
    @njit(cache=True)
    def calculate_muscle_strain(
        muscle_strain,
        off_center_displacement,
        sigma,
        kappa,
        rest_voronoi_lengths,
        voronoi_dilatation,
    ):
        shear = sigma_to_shear(sigma)
        muscle_position_derivative = difference2D(off_center_displacement) / (
            rest_voronoi_lengths * voronoi_dilatation
        )
        muscle_strain[:, :] = shear + quadrature_kernel(
            _batch_cross(kappa, average2D(off_center_displacement))
            + muscle_position_derivative
        )

    @staticmethod
    @njit(cache=True)
    def calculate_muscle_tangent(muscle_tangent, muscle_strain):
        blocksize = muscle_strain.shape[1]
        for i in range(blocksize):
            muscle_tangent[:, i] = muscle_strain[:, i] / np.sqrt(
                muscle_strain[0, i] ** 2
                + muscle_strain[1, i] ** 2
                + muscle_strain[2, i] ** 2
            )

    @staticmethod
    @njit(cache=True)
    def calculate_muscle_length(muscle_length, muscle_strain):
        blocksize = muscle_length.shape[0]
        for i in range(blocksize):
            muscle_length[i] = np.sqrt(
                muscle_strain[0, i] ** 2
                + muscle_strain[1, i] ** 2
                + muscle_strain[2, i] ** 2
            )

    @staticmethod
    @njit(cache=True)
    def calculate_muscle_normalized_length(
        muscle_normalized_length, muscle_length, muscle_rest_length
    ):
        muscle_normalized_length[:] = muscle_length / muscle_rest_length


class MuscleForce(Muscle):
    """MuscleForce"""

    def __init__(
        self,
        ratio_muscle_position,
        rest_muscle_area,
        max_muscle_stress: Union[float, np.array],
        **kwargs,
    ):
        """
        Muscle force class implementation

        Parameters
        ----------
        ratio_muscle_position :
        rest_muscle_area :
        max_muscle_stress : Union[float, np.array]
        """
        super().__init__(
            ratio_muscle_position=ratio_muscle_position,
            rest_muscle_area=rest_muscle_area,
            muscle_type="muscle_force",
            **kwargs,
        )
        self.activation = np.zeros(self.n_elements)
        self.s_activation = (self.s[:-1] + self.s[1:]) / 2
        if isinstance(max_muscle_stress, float):
            self.max_muscle_stress = max_muscle_stress
        elif isinstance(max_muscle_stress, np.ndarray):
            self.max_muscle_stress = max_muscle_stress.copy()
        else:
            raise TypeError(
                f"{max_muscle_stress=} must be either float or np.ndarray. "
            )
        self.muscle_force = np.zeros(self.n_elements)
        self.s_force = 0.5 * (self.s[:-1] + self.s[1:])
        self.force_length_weight = kwargs.get("force_length_weight", unit_weight)

    def __call__(self, system):
        Muscle.__call__(self, system)
        self.calculate_muscle_length(self.muscle_length, self.muscle_strain)
        self.calculate_muscle_normalized_length(
            self.muscle_normalized_length, self.muscle_length, self.muscle_rest_length
        )
        self.calculate_muscle_force(
            self.muscle_force,
            self.get_activation(),
            self.max_muscle_stress,
            self.force_length_weight(self.muscle_normalized_length),
            self.muscle_area,
        )
        self.calculate_force_and_couple(
            self.muscle_force,
            self.muscle_tangent,
            self.muscle_position,
            self.internal_force,
            self.internal_couple,
            self.external_force,
            self.external_couple,
            system.director_collection,
            system.kappa,
            system.tangents,
            system.rest_lengths,
            system.rest_voronoi_lengths,
            system.dilatation,
            system.voronoi_dilatation,
        )

    @staticmethod
    @njit(cache=True)
    def calculate_muscle_force(
        muscle_force, muscle_activation, max_muscle_stress, weight, muscle_area
    ):
        muscle_force[:] = (muscle_activation * max_muscle_stress * weight) * muscle_area

    @staticmethod
    @njit(cache=True)
    def calculate_force_and_couple(
        muscle_force,
        muscle_tangent,
        muscle_position,
        internal_force,
        internal_couple,
        external_force,
        external_couple,
        director_collection,
        kappa,
        tangents,
        rest_lengths,
        rest_voronoi_lengths,
        dilatation,
        voronoi_dilatation,
    ):
        internal_force[:, :] = muscle_force * muscle_tangent
        _force_induced_couple(internal_force, muscle_position, internal_couple)
        _internal_to_external_load(
            director_collection,
            kappa,
            tangents,
            rest_lengths,
            rest_voronoi_lengths,
            dilatation,
            voronoi_dilatation,
            internal_force,
            internal_couple,
            external_force,
            external_couple,
        )

    def apply_activation(self, activation):
        self.set_activation(self.activation, activation)

    @staticmethod
    @njit(cache=True)
    def set_activation(muscle_activation, activation):
        muscle_activation[:] = activation

    def get_activation(self):
        return self.activation


class MuscleGroup(ContinuousActuation, MuscleInfo):
    """MuscleGroup."""

    def __init__(self, muscles, **kwargs):
        """__init__.

        Parameters
        ----------
        muscles :
        """
        ContinuousActuation.__init__(self, muscles[0].n_elements)
        MuscleInfo.__init__(
            self,
            type_name=kwargs.get("type_name", "muscle_group"),
            index=kwargs.get("index", 0),
        )
        self.muscles = muscles
        for m, muscle in enumerate(self.muscles):
            muscle.index = m
        self.activation = np.zeros(self.muscles[0].activation.shape)
        self.s_activation = self.muscles[0].s_activation.copy()

    def __call__(self, system):
        self.reset_actuation()
        for muscle in self.muscles:
            muscle(system)
            inplace_addition(self.internal_force, muscle.internal_force)
            inplace_addition(self.external_force, muscle.external_force)
            inplace_addition(self.internal_couple, muscle.internal_couple)
            inplace_addition(self.external_couple, muscle.external_couple)

    def set_current_length_as_rest_length(self, system):
        """set_current_length_as_rest_length.

        Parameters
        ----------
        system :
        """
        for muscle in self.muscles:
            muscle.set_current_length_as_rest_length(system)

    def apply_activation(self, activation):
        """apply_activation.

        Parameters
        ----------
        activation :
        """
        self.set_activation(self.activation, activation)
        for muscle in self.muscles:
            muscle.set_activation(muscle.activation, self.activation)

    @staticmethod
    @njit(cache=True)
    def set_activation(muscle_activation, activation):
        muscle_activation[:] = activation

    def get_activation(self):
        return self.activation


class ApplyMuscles(ApplyActuations):
    """ApplyMuscles."""

    def __init__(self, muscles, step_skip: int, callback_params_list: list):
        """__init__.

        Parameters
        ----------
        muscles :
        step_skip : int
        callback_params_list : list
        """
        ApplyActuations.__init__(self, muscles, step_skip, callback_params_list)
        for m, muscle in enumerate(muscles):
            muscle.index = m

    def callback_func(self, muscles, callback_params_list):
        """callback_func.

        Parameters
        ----------
        muscles :
        callback_params_list :
        """
        for muscle, callback_params in zip(muscles, callback_params_list):
            callback_params["muscle_info"].append(str(muscle))
            callback_params["s_activation"].append(muscle.s_activation.copy())
            callback_params["activation"].append(muscle.activation.copy())
            callback_params["muscle_length"].append(muscle.muscle_length.copy())
            callback_params["muscle_normalized_length"].append(
                muscle.muscle_normalized_length.copy()
            )
            callback_params["force_length_weight"].append(
                muscle.force_length_weight(muscle.muscle_normalized_length).copy()
            )
            callback_params["muscle_position"].append(muscle.muscle_position.copy())
            callback_params["internal_force"].append(muscle.internal_force.copy())
            callback_params["internal_couple"].append(muscle.internal_couple.copy())
            callback_params["external_force"].append(muscle.external_force.copy())
            callback_params["external_couple"].append(muscle.external_couple.copy())


class ApplyMuscleGroups(ApplyMuscles):
    """ApplyMuscleGroups."""

    def __init__(self, muscle_groups, step_skip: int, callback_params_list: list):
        """__init__.

        Parameters
        ----------
        muscle_groups :
        step_skip : int
        callback_params_list : list
        """
        ApplyMuscles.__init__(self, muscle_groups, step_skip, callback_params_list)
        for muscle_group, callback_params in zip(
            muscle_groups, self.callback_params_list
        ):
            callback_params["muscles"] = [
                defaultdict(list) for _ in muscle_group.muscles
            ]

    def callback_func(self, muscle_groups, callback_params_list):
        """callback_func.

        Parameters
        ----------
        muscle_groups :
        callback_params_list :
        """
        for muscle_group, callback_params in zip(muscle_groups, callback_params_list):
            callback_params["muscle_group_info"].append(str(muscle_group))
            callback_params["s_activation"].append(muscle_group.s_activation.copy())
            callback_params["activation"].append(muscle_group.activation.copy())
            callback_params["internal_force"].append(muscle_group.internal_force.copy())
            callback_params["internal_couple"].append(
                muscle_group.internal_couple.copy()
            )
            callback_params["external_force"].append(muscle_group.external_force.copy())
            callback_params["external_couple"].append(
                muscle_group.external_couple.copy()
            )
            ApplyMuscles.callback_func(
                self, muscle_group.muscles, callback_params["muscles"]
            )
