from __future__ import annotations

from typing import Callable, Self

import numpy as np
from numba import njit

import elastica as ea
from elastica._linalg import _batch_cross
from elastica._calculus import quadrature_kernel

from .muscles.muscle import MuscleInfo
from .._rod_tool import average2D, difference2D

from coomm.actuations.actuation import (
    _force_induced_couple,
    _internal_to_external_load,
)


class ApplyMuscleActuations(ea.NoForces):
    """
    Apply batched muscle actuation directly onto a rod.

    Example
    -------
    >>> rod: ea.CosseratRod
    >>> simulator: ea.Simulator
    >>> activation = np.zeros(rod.n_elems)
    >>> batch_muscle = (
    ...     BatchMuscle(num_elements=rod.n_elems)
    ...     .add_transverse_muscle(
    ...         rest_muscle_area=tm_area, max_muscle_stress=15_000, activation=activation
    ...     )
    ...     .add_longitudinal_muscle(
    ...         muscle_init_angle=0.0,
    ...         ratio_muscle_position=lm_ratio,
    ...         rest_muscle_area=lm_area,
    ...         max_muscle_stress=10_000,
    ...         activation=activation.copy(),
    ...     )
    ...     .configure(force_length_weight=force_length_weight_poly)
    ...     .blocking(rod)
    ... )
    >>> simulator.add_forcing_to(rod).using(ApplyMuscleActuations, batch_muscle=batch_muscle)
    """

    def __init__(self, batch_muscle: BatchMuscle):
        super().__init__()
        self.batch_muscle = batch_muscle

    def apply_torques(self, system, time: np.float64 = 0.0):
        self.batch_muscle.forward(system)


class BatchMuscle(MuscleInfo):
    """
    Batched muscle actuation for TM / LM / OM fibers on a single rod.

    Muscle types:
    - TM: Transverse muscle
    - LM: Longitudinal muscle
    - OM: Oblique muscle
    """

    def __init__(
        self,
        num_elements: int,
        index: int = 0,
        type_name: str = "batch_muscle",
    ) -> None:
        MuscleInfo.__init__(self, type_name=type_name, index=index)

        self.num_elements = num_elements
        self.force_length_weight: Callable | None = None

        self._ratio_positions: list[np.ndarray] = []
        self._rest_areas: list[np.ndarray] = []
        self._max_stresses: list[float | np.ndarray] = []
        self._activations: list[np.ndarray] = []

        self._blocked = False
        self.n_muscles: int

    def configure(self, *, force_length_weight: Callable | None = None) -> Self:
        """Configure batch options before ``blocking``."""
        if force_length_weight is not None:
            self.force_length_weight = force_length_weight
        return self

    def _register_muscle(
        self,
        ratio_muscle_position: np.ndarray,
        rest_muscle_area: np.ndarray,
        max_muscle_stress: float,
        activation: np.ndarray,
    ) -> None:
        if self._blocked:
            raise RuntimeError("Cannot add muscles after blocking().")

        n = self.num_elements
        if ratio_muscle_position.shape != (3, n):
            raise ValueError(
                f"{ratio_muscle_position.shape=} must match (3, {n}) for num_elements."
            )
        if rest_muscle_area.shape != (n,):
            raise ValueError(f"{rest_muscle_area.shape=} must match ({n},) for num_elements.")

        self._ratio_positions.append(ratio_muscle_position)
        self._rest_areas.append(rest_muscle_area)
        self._max_stresses.append(max_muscle_stress)
        self._activations.append(activation)

    def add_transverse_muscle(
        self,
        rest_muscle_area: np.ndarray,
        max_muscle_stress: float,
        activation: np.ndarray,
    ) -> Self:
        self._register_muscle(
            ratio_muscle_position=np.zeros((3, self.num_elements)),
            rest_muscle_area=rest_muscle_area,
            max_muscle_stress=-max_muscle_stress,
            activation=activation,
        )
        return self

    def add_longitudinal_muscle(
        self,
        muscle_init_angle: float,
        ratio_muscle_position: float | np.ndarray,
        rest_muscle_area: np.ndarray,
        max_muscle_stress: float,
        activation: np.ndarray,
    ) -> Self:
        ratio = ratio_muscle_position * np.array(
            [[np.cos(muscle_init_angle)], [np.sin(muscle_init_angle)], [0.0]]
        )
        self._register_muscle(
            ratio_muscle_position=np.broadcast_to(ratio, (3, self.num_elements)),
            rest_muscle_area=rest_muscle_area,
            max_muscle_stress=max_muscle_stress,
            activation=activation,
        )
        return self

    def add_oblique_muscle(
        self,
        muscle_init_angle: float,
        ratio_muscle_position: float | np.ndarray,
        rotation_number: float,
        rest_muscle_area: np.ndarray,
        max_muscle_stress: float,
        activation: np.ndarray,
    ) -> Self:
        s = np.linspace(0.0, 1.0, self.num_elements + 1)
        s_muscle_position = 0.5 * (s[:-1] + s[1:])
        phase = muscle_init_angle + 2.0 * np.pi * rotation_number * s_muscle_position
        ratio = ratio_muscle_position * np.array(
            [np.cos(phase), np.sin(phase), np.zeros(self.num_elements)]
        )
        self._register_muscle(
            ratio_muscle_position=ratio,
            rest_muscle_area=rest_muscle_area,
            max_muscle_stress=max_muscle_stress,
            activation=activation,
        )
        return self

    def blocking(self, system: ea.rod.RodBase) -> Self:
        """Finalize muscle registration, allocate buffers, and set rest lengths."""
        if self._blocked:
            raise RuntimeError("Muscle is already blocked: cannot block again.")
        if not self._ratio_positions:
            raise RuntimeError("No muscles registered. Add muscles before blocking().")

        self.n_muscles = len(self._ratio_positions)

        self.s = np.linspace(0.0, 1.0, self.num_elements + 1)
        self.s_activation = 0.5 * (self.s[:-1] + self.s[1:])

        self.ratio_muscle_position = np.stack(self._ratio_positions, axis=0)
        self.rest_muscle_area = np.stack(self._rest_areas, axis=0)

        self.max_muscle_stress = np.zeros((self.n_muscles, self.num_elements))
        for m, stress in enumerate(self._max_stresses):
            self.max_muscle_stress[m, :] = stress

        # Contiguous layout per muscle (23, n): normalized_length, rest_length, length,
        # tangent[3], strain[3], position[3], ratio_position[3], rest_area,
        # internal_force[3], internal_couple[3] (last column unused), muscle_force
        n = self.num_elements
        self._block = np.zeros((self.n_muscles * 23, n))
        self._muscle_state = self._block.reshape(self.n_muscles, 23, n)

        self._muscle_state[:, 1, :] = 1.0
        self._muscle_state[:, 12:15, :] = self.ratio_muscle_position
        self._muscle_state[:, 15, :] = self.rest_muscle_area

        self.muscle_normalized_length = self._muscle_state[:, 0, :]
        self.muscle_rest_length = self._muscle_state[:, 1, :]
        self.muscle_length = self._muscle_state[:, 2, :]
        self.muscle_tangent = self._muscle_state[:, 3:6, :]
        self.muscle_strain = self._muscle_state[:, 6:9, :]
        self.muscle_position = self._muscle_state[:, 9:12, :]
        self.ratio_muscle_position = self._muscle_state[:, 12:15, :]
        self.rest_muscle_area = self._muscle_state[:, 15, :]
        self.internal_force = self._muscle_state[:, 16:19, :]
        self.internal_couple = self._muscle_state[:, 19:22, :-1]
        self.muscle_force = self._muscle_state[:, 22, :]

        self._blocked = True

        self._set_current_length_as_rest_length(system)

        return self

    def forward(self, system: ea.rod.RodBase) -> None:
        """Update muscle kinematics and emplace loads on the rod."""
        _nb_batch_update_muscle_strain_and_geometry(
            system.radius,
            self.ratio_muscle_position,
            system.sigma,
            system.kappa,
            system.rest_voronoi_lengths,
            system.voronoi_dilatation,
            self.muscle_position,
            self.muscle_strain,
            self.muscle_tangent,
            self.muscle_length,
            self.muscle_normalized_length,
            self.muscle_rest_length,
        )
        if self.force_length_weight is None:
            weights = np.ones_like(self.muscle_normalized_length)
        else:
            weights = self.force_length_weight(self.muscle_normalized_length)
        _nb_batch_calculate_muscle_actuation(
            self.muscle_force,
            self._activations,
            self.max_muscle_stress,
            weights,
            self.rest_muscle_area,
            system.dilatation,
            self.muscle_tangent,
            self.muscle_position,
            self.internal_force,
            self.internal_couple,
            system.external_forces,
            system.external_torques,
            system.director_collection,
            system.kappa,
            system.tangents,
            system.rest_lengths,
            system.rest_voronoi_lengths,
            system.dilatation,
        )

    def _set_current_length_as_rest_length(self, system: ea.rod.RodBase) -> None:
        """Set each muscle's rest length from the current rod geometry."""
        _nb_batch_update_muscle_strain_and_geometry(
            system.radius,
            self.ratio_muscle_position,
            system.sigma,
            system.kappa,
            system.rest_voronoi_lengths,
            system.voronoi_dilatation,
            self.muscle_position,
            self.muscle_strain,
            self.muscle_tangent,
            self.muscle_length,
            self.muscle_normalized_length,
            self.muscle_rest_length,
        )
        self.muscle_rest_length[:] = self.muscle_length

    # FIXME: For now, activation is completely handled externally.
    # No memory copy is needed
    # def apply_activation(self, muscle_index: int, activation: float | np.ndarray) -> None:
    #     """Set activation for one muscle."""
    #     self.activation[muscle_index, :] = activation

    # def set_activations(self, activations: np.ndarray) -> None:
    #     """Set all muscle activations. Shape: ``(n_muscles, num_elements)``."""
    #     self.activation[:, :] = activations


@njit(cache=True)
def _nb_batch_update_muscle_strain_and_geometry(
    rod_radius,
    ratio_muscle_position,
    rod_sigma,
    rod_kappa,
    rod_rest_voronoi_lengths,
    rod_voronoi_dilatation,
    muscle_position,
    muscle_strain,
    muscle_tangent,
    muscle_length,
    muscle_normalized_length,
    muscle_rest_length,
):
    n_muscles = ratio_muscle_position.shape[0]
    n_elements = ratio_muscle_position.shape[2]

    for m in range(n_muscles):
        muscle_position[m, :, :] = rod_radius * ratio_muscle_position[m, :, :]

        muscle_position_derivative = difference2D(muscle_position[m]) / (
            rod_rest_voronoi_lengths * rod_voronoi_dilatation
        )

        muscle_strain[m, :, :] = (
            quadrature_kernel(
                _batch_cross(rod_kappa, average2D(muscle_position[m])) + muscle_position_derivative
            )
            + rod_sigma
        )
        muscle_strain[m, 2, :] += 1.0

        if m == 0:
            _nb_transverse_muscle_length(muscle_length[m], muscle_strain[m])
        else:
            _nb_muscle_length(muscle_length[m], muscle_strain[m])

        for i in range(n_elements):
            muscle_tangent[m, :, i] = muscle_strain[m, :, i] / muscle_length[m, i]
            muscle_normalized_length[m, i] = muscle_length[m, i] / muscle_rest_length[m, i]


@njit(cache=True)
def _nb_batch_calculate_muscle_actuation(
    muscle_force,
    muscle_activation,
    max_muscle_stress,
    weight,
    rest_muscle_area,
    dilatation,
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
    dilatation_field,
):
    n_muscles = muscle_force.shape[0]
    n_elements = muscle_force.shape[1]

    for m in range(n_muscles):
        for i in range(n_elements):
            muscle_force[m, i] = (
                muscle_activation[m][0]
                * max_muscle_stress[m, i]
                * weight[m, i]
                * rest_muscle_area[m, i]
                / dilatation[i]
            )

        internal_force[m, :, :] = muscle_force[m] * muscle_tangent[m]
        _force_induced_couple(internal_force[m], muscle_position[m], internal_couple[m])

        _internal_to_external_load(
            director_collection,
            kappa,
            tangents,
            rest_lengths,
            rest_voronoi_lengths,
            dilatation_field,
            internal_force[m],
            internal_couple[m],
            external_force,
            external_couple,
        )


@njit(cache=True)
def _nb_muscle_length(muscle_length: np.ndarray, muscle_strain: np.ndarray):
    blocksize = muscle_length.shape[0]
    for i in range(blocksize):
        muscle_length[i] = np.sqrt(
            muscle_strain[0, i] ** 2 + muscle_strain[1, i] ** 2 + muscle_strain[2, i] ** 2
        )


@njit(cache=True)
def _nb_transverse_muscle_length(muscle_length: np.ndarray, muscle_strain: np.ndarray):
    """calculate_muscle_length.

    Parameters
    ----------
    muscle_length : np.ndarray
        shape: (n_element)
    muscle_strain : np.ndarray
        shape: (3, n_element)
    """
    blocksize = muscle_length.shape[0]
    for i in range(blocksize):
        # fmt: off
        muscle_length[i] = (
            1 / (muscle_strain[0, i] ** 2 +  \
                    muscle_strain[1, i] ** 2 +  \
                    muscle_strain[2, i] ** 2) ** 0.25 \
        )
        # fmt: on
