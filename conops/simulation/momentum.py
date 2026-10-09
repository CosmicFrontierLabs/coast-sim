from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..common.vector import quaternion_to_rotation_matrix

EARTH_GRAVITATIONAL_PARAMETER_KM3_S2 = 398600.4418


def _finite_vector3(value: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _finite_matrix3(value: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return matrix


def gravity_gradient_torque_body(
    position_eci_km: npt.ArrayLike,
    attitude_quaternion_eci_to_body: npt.ArrayLike,
    inertia_tensor_body_kg_m2: npt.ArrayLike,
    *,
    gravitational_parameter_km3_s2: float = EARTH_GRAVITATIONAL_PARAMETER_KM3_S2,
) -> npt.NDArray[np.float64]:
    """Return central-body gravity-gradient torque in body coordinates.

    The position is measured from the central body to the spacecraft. The
    attitude quaternion uses COAST's scalar-first ECI-to-body convention.
    """

    position = _finite_vector3(position_eci_km, name="position_eci_km")
    radius_km = float(np.linalg.norm(position))
    if radius_km <= 0.0:
        raise ValueError("position_eci_km must have nonzero magnitude")
    if not np.isfinite(gravitational_parameter_km3_s2) or (
        gravitational_parameter_km3_s2 <= 0.0
    ):
        raise ValueError("gravitational parameter must be finite and positive")

    inertia = _finite_matrix3(
        inertia_tensor_body_kg_m2, name="inertia_tensor_body_kg_m2"
    )
    rotation_eci_to_body = quaternion_to_rotation_matrix(
        attitude_quaternion_eci_to_body
    )
    radial_body = rotation_eci_to_body @ (position / radius_km)
    return np.asarray(
        3.0
        * gravitational_parameter_km3_s2
        / radius_km**3
        * np.cross(radial_body, inertia @ radial_body),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class MomentumSample:
    """Instantaneous disturbance torque and accumulated stored momentum."""

    gravity_gradient_torque_body_n_m: tuple[float, float, float]
    stored_momentum_body_n_m_s: tuple[float, float, float]
    stored_momentum_norm_n_m_s: float


class StoredMomentumTracker:
    """Integrate momentum stored while rejecting gravity-gradient torque.

    Momentum is integrated in ECI so changing spacecraft attitude only changes
    its body-frame components; it does not create spurious stored momentum.
    This is a sampled integrator, not a trajectory interpolator. Callers must
    supply sufficiently fine executed-attitude and orbit samples, and lower
    ``max_sample_interval_s`` for fast motion or convergence studies.
    """

    def __init__(
        self,
        inertia_tensor_body_kg_m2: npt.ArrayLike,
        initial_momentum_body_n_m_s: npt.ArrayLike = (0.0, 0.0, 0.0),
        *,
        max_sample_interval_s: float = 10.0,
    ) -> None:
        if (
            not np.isfinite(max_sample_interval_s)
            or not 0.0 < max_sample_interval_s <= 10.0
        ):
            raise ValueError(
                "max_sample_interval_s must be finite, positive, and <= 10 s"
            )
        self.max_sample_interval_s = float(max_sample_interval_s)
        self.inertia_tensor_body_kg_m2 = _finite_matrix3(
            inertia_tensor_body_kg_m2, name="inertia_tensor_body_kg_m2"
        )
        self.initial_momentum_body_n_m_s = _finite_vector3(
            initial_momentum_body_n_m_s, name="initial_momentum_body_n_m_s"
        )
        self.reset()

    def reset(self) -> None:
        """Restore the configured initial state for a new simulation run."""
        self._momentum_eci_n_m_s: npt.NDArray[np.float64] | None = None
        self._previous_torque_eci_n_m: npt.NDArray[np.float64] | None = None
        self._previous_utime: float | None = None

    def validate_sample_interval(self, elapsed_s: float) -> None:
        """Reject undersampling without changing integration state."""
        if not np.isfinite(elapsed_s) or elapsed_s <= 0.0:
            raise ValueError("momentum sample interval must be finite and positive")
        if elapsed_s > self.max_sample_interval_s * (1.0 + 1e-12):
            raise ValueError(
                f"stored momentum sample interval {elapsed_s:g} s exceeds "
                f"{self.max_sample_interval_s:g} s; supply finer executed-attitude "
                "and ephemeris samples"
            )

    def update(
        self,
        *,
        utime: float,
        position_eci_km: npt.ArrayLike,
        attitude_quaternion_eci_to_body: npt.ArrayLike,
    ) -> MomentumSample:
        """Advance to one attitude sample using trapezoidal torque integration."""
        if not np.isfinite(utime):
            raise ValueError("utime must be finite")

        elapsed_s = (
            0.0 if self._previous_utime is None else float(utime - self._previous_utime)
        )
        if elapsed_s < 0.0:
            raise ValueError("momentum samples must have nondecreasing timestamps")
        # Subtracting Unix timestamps can round a nominally valid gap upward.
        time_resolution = float(np.spacing(abs(utime))) * 2.0
        if elapsed_s > self.max_sample_interval_s + time_resolution:
            self.validate_sample_interval(elapsed_s)

        rotation_eci_to_body = quaternion_to_rotation_matrix(
            attitude_quaternion_eci_to_body
        )
        torque_body = gravity_gradient_torque_body(
            position_eci_km,
            attitude_quaternion_eci_to_body,
            self.inertia_tensor_body_kg_m2,
        )
        torque_eci = rotation_eci_to_body.T @ torque_body

        if self._previous_utime is None:
            self._momentum_eci_n_m_s = (
                rotation_eci_to_body.T @ self.initial_momentum_body_n_m_s
            )
        else:
            assert self._momentum_eci_n_m_s is not None
            assert self._previous_torque_eci_n_m is not None
            self._momentum_eci_n_m_s += (
                0.5 * (self._previous_torque_eci_n_m + torque_eci) * elapsed_s
            )

        self._previous_utime = float(utime)
        self._previous_torque_eci_n_m = torque_eci
        assert self._momentum_eci_n_m_s is not None
        momentum_body = rotation_eci_to_body @ self._momentum_eci_n_m_s

        return MomentumSample(
            gravity_gradient_torque_body_n_m=(
                float(torque_body[0]),
                float(torque_body[1]),
                float(torque_body[2]),
            ),
            stored_momentum_body_n_m_s=(
                float(momentum_body[0]),
                float(momentum_body[1]),
                float(momentum_body[2]),
            ),
            stored_momentum_norm_n_m_s=float(np.linalg.norm(self._momentum_eci_n_m_s)),
        )
