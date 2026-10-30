"""Stored momentum from the environmental torques.

Reaction wheels absorb the angular impulse of every torque the environment
puts on the spacecraft: gravity gradient from the Earth's field across the
body, solar radiation pressure on its surfaces, aerodynamic drag at low
altitude. Each torque is a :class:`TorqueSource` evaluated in the body frame
at a sample of the orbit and attitude; :class:`StoredMomentumTracker`
integrates their sum in inertial coordinates, so changing attitude alone
never creates momentum, and reports the stored vector in the body frame
with each source's torque.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from ..common.geodesy import EARTH_ROTATION_RATE_RAD_S
from ..common.vector import quaternion_to_rotation_matrix
from .atmosphere import AtmosphereModel, atmosphere_from_config
from .geomagnetic import GeomagneticField, field_from_name

if TYPE_CHECKING:
    import rust_ephem

    from ..config import MissionConfig
    from ..config.surfaces import SurfaceModel

EARTH_GRAVITATIONAL_PARAMETER_KM3_S2 = 398600.4418
SOLAR_PRESSURE_N_M2 = 4.56e-6
"""Solar radiation pressure at one astronomical unit."""
ASTRONOMICAL_UNIT_KM = 149597870.7


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


def solar_radiation_pressure_torque_body(
    sun_body_km: npt.ArrayLike,
    surfaces: SurfaceModel,
    *,
    sunlit: bool = True,
) -> npt.NDArray[np.float64]:
    """Return the torque of sunlight on the spacecraft's surfaces, in body
    coordinates, about the surface model's centre of mass.

    ``sun_body_km`` is the vector from the spacecraft to the Sun in the body
    frame; its length scales the pressure from one astronomical unit. Each
    plate the Sun shines on takes the force of what it absorbs, reflects
    specularly and reflects diffusely; plates facing away take none, and
    nothing does in eclipse.
    """
    if not sunlit:
        return np.zeros(3, dtype=np.float64)
    sun = _finite_vector3(sun_body_km, name="sun_body_km")
    distance_km = float(np.linalg.norm(sun))
    if distance_km <= 0.0:
        raise ValueError("sun_body_km must have nonzero magnitude")
    direction = sun / distance_km
    pressure = SOLAR_PRESSURE_N_M2 * (ASTRONOMICAL_UNIT_KM / distance_km) ** 2
    center_of_mass = np.asarray(surfaces.center_of_mass_m, dtype=np.float64)
    torque = np.zeros(3, dtype=np.float64)
    for surface in surfaces.surfaces:
        normal = np.asarray(surface.normal, dtype=np.float64)
        cosine = float(normal @ direction)
        if cosine <= 0.0:
            continue
        specular = surface.specular_reflectivity
        diffuse = surface.diffuse_reflectivity
        force = (
            -pressure
            * surface.area_m2
            * cosine
            * (
                (1.0 - specular) * direction
                + 2.0 * (specular * cosine + diffuse / 3.0) * normal
            )
        )
        arm = np.asarray(surface.center_m, dtype=np.float64) - center_of_mass
        torque += np.cross(arm, force)
    return torque


def aerodynamic_drag_torque_body(
    flow_body_km_s: npt.ArrayLike,
    density_kg_m3: float,
    surfaces: SurfaceModel,
) -> npt.NDArray[np.float64]:
    """Return the torque of the atmosphere on the spacecraft's surfaces, in
    body coordinates, about the surface model's centre of mass.

    ``flow_body_km_s`` is the spacecraft's velocity relative to the
    atmosphere in the body frame, so the flow comes from that direction.
    Each plate facing the flow takes the free-molecular drag of its
    projected area along the flow, with its own drag coefficient; plates
    facing away take none. Self-shadowing is not modelled.
    """
    flow = _finite_vector3(flow_body_km_s, name="flow_body_km_s")
    speed_m_s = float(np.linalg.norm(flow)) * 1000.0
    if speed_m_s <= 0.0 or density_kg_m3 <= 0.0:
        return np.zeros(3, dtype=np.float64)
    if not np.isfinite(density_kg_m3):
        raise ValueError("density must be finite")
    direction = flow / float(np.linalg.norm(flow))
    pressure = 0.5 * density_kg_m3 * speed_m_s**2
    center_of_mass = np.asarray(surfaces.center_of_mass_m, dtype=np.float64)
    torque = np.zeros(3, dtype=np.float64)
    for surface in surfaces.surfaces:
        normal = np.asarray(surface.normal, dtype=np.float64)
        cosine = float(normal @ direction)
        if cosine <= 0.0:
            continue
        force = (
            -pressure * surface.drag_coefficient * surface.area_m2 * cosine * direction
        )
        arm = np.asarray(surface.center_m, dtype=np.float64) - center_of_mass
        torque += np.cross(arm, force)
    return torque


@dataclass(frozen=True)
class OrbitSample:
    """The orbit, Sun and attitude at one instant, as the torque sources
    see them: the ECI position, the vector from the spacecraft to the Sun
    (None where unknown), whether the spacecraft is sunlit, and the rotation
    from ECI into the body frame."""

    utime: float
    position_eci_km: npt.NDArray[np.float64]
    rotation_eci_to_body: npt.NDArray[np.float64]
    sun_eci_km: npt.NDArray[np.float64] | None = None
    sunlit: bool = True
    velocity_eci_km_s: npt.NDArray[np.float64] | None = None
    position_ecef_km: npt.NDArray[np.float64] | None = None
    """The Earth-fixed position, for the geomagnetic field; None where unknown."""
    stored_momentum_body_n_m_s: npt.NDArray[np.float64] | None = None
    """The momentum stored as the sample begins, in the body frame, for the
    sources that act on it."""

    def to_body(self, vector_eci: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(
            self.rotation_eci_to_body @ np.asarray(vector_eci), dtype=np.float64
        )

    def geomagnetic_field_body_t(
        self, field: GeomagneticField
    ) -> npt.NDArray[np.float64] | None:
        """The field at the sample in the body frame; None without an
        Earth-fixed position."""
        if self.position_ecef_km is None:
            return None
        return self.to_body(
            field.field_eci_t(self.position_eci_km, self.position_ecef_km, self.utime)
        )

    def flow_eci_km_s(self) -> npt.NDArray[np.float64] | None:
        """The spacecraft's velocity relative to the atmosphere, which
        rotates with the Earth, in inertial coordinates; None without the
        orbital velocity."""
        if self.velocity_eci_km_s is None:
            return None
        rotation = np.array([0.0, 0.0, EARTH_ROTATION_RATE_RAD_S], dtype=np.float64)
        return np.asarray(
            self.velocity_eci_km_s - np.cross(rotation, self.position_eci_km),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class OrbitTables:
    """An ephemeris as the arrays a torque bound is taken over: the sample
    times, the ECI positions, and where the ephemeris has them the ECI
    velocities, the Earth-fixed positions and the Sun's ECI positions."""

    times: npt.NDArray[np.float64]
    positions_eci_km: npt.NDArray[np.float64]
    velocities_eci_km_s: npt.NDArray[np.float64] | None = None
    positions_ecef_km: npt.NDArray[np.float64] | None = None
    sun_positions_eci_km: npt.NDArray[np.float64] | None = None

    @classmethod
    def from_ephem(cls, ephem: rust_ephem.Ephemeris) -> OrbitTables:
        """The tables of ``ephem``; a test double may lack all but the
        positions."""
        from ..common.ephemeris import position_vectors, velocity_vectors

        times = np.array(
            [
                float(
                    (t - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
                )
                if isinstance(t, np.datetime64)
                else t.timestamp()
                for t in ephem.timestamp
            ],
            dtype=np.float64,
        )
        positions = np.asarray(position_vectors(ephem, "gcrs"), dtype=np.float64)
        sun = getattr(getattr(ephem, "sun_pv", None), "position", None)
        ecef = getattr(getattr(ephem, "itrs_pv", None), "position", None)
        velocities = velocity_vectors(ephem)
        return cls(
            times,
            positions,
            velocities,
            np.asarray(ecef, dtype=np.float64) if ecef is not None else None,
            np.asarray(sun, dtype=np.float64) if sun is not None else None,
        )

    def min_radius_km(self) -> float:
        """The least distance from the Earth's centre along the orbit,
        inside each linearly interpolated segment, not only at the samples."""
        first, delta = (
            self.positions_eci_km[:-1],
            np.diff(self.positions_eci_km, axis=0),
        )
        if len(first) == 0:
            return float(np.linalg.norm(self.positions_eci_km[0]))
        length2 = np.sum(delta * delta, axis=1)
        fraction = np.clip(
            -np.sum(first * delta, axis=1) / np.where(length2 > 0, length2, 1), 0, 1
        )
        return float(np.linalg.norm(first + fraction[:, None] * delta, axis=1).min())

    def max_flow_km_s(self) -> float:
        """The greatest speed relative to the atmosphere along the orbit;
        zero without velocities."""
        if self.velocities_eci_km_s is None:
            return 0.0
        rotation = np.array([0.0, 0.0, EARTH_ROTATION_RATE_RAD_S], dtype=np.float64)
        flow = self.velocities_eci_km_s - np.cross(rotation, self.positions_eci_km)
        return float(np.linalg.norm(flow, axis=1).max())

    def max_field_t(self, field: GeomagneticField) -> float:
        """The strongest geomagnetic field along the orbit; zero without
        Earth-fixed positions, where the sources see no field either."""
        if self.positions_ecef_km is None:
            return 0.0
        return max(
            float(np.linalg.norm(field.field_eci_t(position, ecef, float(utime))))
            for utime, position, ecef in zip(
                self.times, self.positions_eci_km, self.positions_ecef_km, strict=True
            )
        )

    def min_sun_distance_km(self) -> float:
        """The least distance to the Sun along the orbit; one astronomical
        unit without the Sun's positions."""
        if self.sun_positions_eci_km is None:
            return ASTRONOMICAL_UNIT_KM
        return float(
            np.linalg.norm(
                self.sun_positions_eci_km - self.positions_eci_km, axis=1
            ).min()
        )


class TorqueSource(Protocol):
    """One environmental torque, evaluated in the body frame, with a bound
    on its magnitude along an orbit."""

    name: str

    def torque_body_n_m(self, sample: OrbitSample) -> npt.NDArray[np.float64]: ...

    def torque_bound_n_m(self, orbit: OrbitTables) -> float: ...


def _surface_lever_m2_m(surfaces: SurfaceModel, weights: Sequence[float]) -> float:
    """Σ weight × area × lever arm over the plates, for a force bound."""
    center_of_mass = np.asarray(surfaces.center_of_mass_m, dtype=np.float64)
    return float(
        sum(
            weight
            * surface.area_m2
            * float(
                np.linalg.norm(
                    np.asarray(surface.center_m, dtype=np.float64) - center_of_mass
                )
            )
            for weight, surface in zip(weights, surfaces.surfaces, strict=True)
        )
    )


class GravityGradientTorque:
    """Central-body gravity gradient on the inertia tensor."""

    name = "gravity_gradient"

    def __init__(self, inertia_tensor_body_kg_m2: npt.ArrayLike) -> None:
        self.inertia_tensor_body_kg_m2 = _finite_matrix3(
            inertia_tensor_body_kg_m2, name="inertia_tensor_body_kg_m2"
        )

    def torque_body_n_m(self, sample: OrbitSample) -> npt.NDArray[np.float64]:
        position = sample.position_eci_km
        radius_km = float(np.linalg.norm(position))
        if radius_km <= 0.0:
            raise ValueError("position_eci_km must have nonzero magnitude")
        radial_body = sample.rotation_eci_to_body @ (position / radius_km)
        return np.asarray(
            3.0
            * EARTH_GRAVITATIONAL_PARAMETER_KM3_S2
            / radius_km**3
            * np.cross(radial_body, self.inertia_tensor_body_kg_m2 @ radial_body),
            dtype=np.float64,
        )

    def torque_bound_n_m(self, orbit: OrbitTables) -> float:
        """At most 3μ/(2r³) times the spread of the principal moments, at
        the orbit's lowest point."""
        radius_km = orbit.min_radius_km()
        if radius_km <= 0.0:
            return float("inf")
        moments = np.linalg.eigvalsh(self.inertia_tensor_body_kg_m2)
        return float(
            1.5
            * EARTH_GRAVITATIONAL_PARAMETER_KM3_S2
            * (moments[-1] - moments[0])
            / radius_km**3
        )


class SolarRadiationPressureTorque:
    """Sunlight on the spacecraft's surfaces; nothing where the Sun vector
    is unknown or the spacecraft is in eclipse."""

    name = "solar_radiation_pressure"

    def __init__(self, surfaces: SurfaceModel) -> None:
        self.surfaces = surfaces

    def torque_body_n_m(self, sample: OrbitSample) -> npt.NDArray[np.float64]:
        if sample.sun_eci_km is None:
            return np.zeros(3, dtype=np.float64)
        return solar_radiation_pressure_torque_body(
            sample.to_body(sample.sun_eci_km), self.surfaces, sunlit=sample.sunlit
        )

    def torque_bound_n_m(self, orbit: OrbitTables) -> float:
        """Every plate lit face-on at the orbit's nearest approach to the
        Sun: the force on a plate is at most the pressure times its area
        times 1 + ρs + 2ρd/3."""
        pressure = (
            SOLAR_PRESSURE_N_M2
            * (ASTRONOMICAL_UNIT_KM / orbit.min_sun_distance_km()) ** 2
        )
        return pressure * _surface_lever_m2_m(
            self.surfaces,
            [
                1.0 + s.specular_reflectivity + 2.0 * s.diffuse_reflectivity / 3.0
                for s in self.surfaces.surfaces
            ],
        )


class AerodynamicDragTorque:
    """The atmosphere on the spacecraft's surfaces, at the density an
    atmosphere model gives; nothing where the orbital velocity is unknown."""

    name = "aerodynamic_drag"

    def __init__(self, surfaces: SurfaceModel, atmosphere: AtmosphereModel) -> None:
        self.surfaces = surfaces
        self.atmosphere = atmosphere
        self.last_density_kg_m3: float | None = None
        """The density at the last sample, for telemetry."""

    def torque_body_n_m(self, sample: OrbitSample) -> npt.NDArray[np.float64]:
        flow = sample.flow_eci_km_s()
        if flow is None:
            self.last_density_kg_m3 = None
            return np.zeros(3, dtype=np.float64)
        density = self.atmosphere.density_kg_m3(
            sample.utime, sample.position_eci_km, sample.position_ecef_km
        )
        self.last_density_kg_m3 = density
        return aerodynamic_drag_torque_body(
            sample.to_body(flow), density, self.surfaces
        )

    def torque_bound_n_m(self, orbit: OrbitTables) -> float:
        """Every plate facing the flow at the orbit's densest point and
        greatest speed."""
        speed_m_s = orbit.max_flow_km_s() * 1000.0
        if speed_m_s <= 0.0:
            return 0.0
        ecefs = (
            orbit.positions_ecef_km
            if orbit.positions_ecef_km is not None
            else [None] * len(orbit.times)
        )
        density = max(
            self.atmosphere.density_kg_m3(float(utime), position, ecef)
            for utime, position, ecef in zip(
                orbit.times, orbit.positions_eci_km, ecefs, strict=True
            )
        )
        return (
            0.5
            * density
            * speed_m_s**2
            * _surface_lever_m2_m(
                self.surfaces, [s.drag_coefficient for s in self.surfaces.surfaces]
            )
        )


class MagnetorquerUnloading:
    """Magnetorquers unloading the wheels against the geomagnetic field.

    The torquers can only turn the spacecraft about axes perpendicular to
    the field, so they take out the stored momentum's component across the
    field and leave the component along it. They aim to remove the stored
    momentum over ``time_constant_s``, commanding the dipole that gives that
    torque, capped at ``max_dipole_a_m2``: at the cap the unloading torque is
    the cap times the field, whatever the momentum. A negative torque source,
    reported as the others are.
    """

    name = "magnetorquer_unloading"

    def __init__(
        self,
        field: GeomagneticField,
        max_dipole_a_m2: float,
        time_constant_s: float,
    ) -> None:
        if not np.isfinite(max_dipole_a_m2) or max_dipole_a_m2 <= 0.0:
            raise ValueError("max_dipole_a_m2 must be finite and positive")
        if not np.isfinite(time_constant_s) or time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be finite and positive")
        self.field = field
        self.max_dipole_a_m2 = float(max_dipole_a_m2)
        self.time_constant_s = float(time_constant_s)

    def dipole_body_a_m2(
        self,
        momentum_body_n_m_s: npt.NDArray[np.float64],
        field_body_t: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """The dipole commanded against ``momentum_body_n_m_s`` in ``field_body_t``."""
        field_squared = float(field_body_t @ field_body_t)
        if field_squared <= 0.0:
            return np.zeros(3, dtype=np.float64)
        wanted = -momentum_body_n_m_s / self.time_constant_s
        # m x B = wanted has the solution m = (B x wanted) / |B|^2 for the
        # part of ``wanted`` across the field; the part along it is unreachable.
        dipole = np.cross(field_body_t, wanted) / field_squared
        magnitude = float(np.linalg.norm(dipole))
        if magnitude > self.max_dipole_a_m2:
            dipole *= self.max_dipole_a_m2 / magnitude
        return np.asarray(dipole, dtype=np.float64)

    def torque_body_n_m(self, sample: OrbitSample) -> npt.NDArray[np.float64]:
        field = sample.geomagnetic_field_body_t(self.field)
        momentum = sample.stored_momentum_body_n_m_s
        if field is None or momentum is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(np.cross(self.dipole_body_a_m2(momentum, field), field))

    def torque_bound_n_m(self, orbit: OrbitTables) -> float:
        """The greatest dipole across the strongest field along the orbit."""
        return self.max_dipole_a_m2 * orbit.max_field_t(self.field)


class ResidualDipoleTorque:
    """The torque of the geomagnetic field on the spacecraft's residual
    magnetic dipole, fixed in the body."""

    name = "residual_dipole"

    def __init__(
        self, field: GeomagneticField, dipole_body_a_m2: npt.ArrayLike
    ) -> None:
        self.field = field
        self.dipole_body_a_m2 = _finite_vector3(
            dipole_body_a_m2, name="dipole_body_a_m2"
        )

    def torque_body_n_m(self, sample: OrbitSample) -> npt.NDArray[np.float64]:
        field = sample.geomagnetic_field_body_t(self.field)
        if field is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(np.cross(self.dipole_body_a_m2, field), dtype=np.float64)

    def torque_bound_n_m(self, orbit: OrbitTables) -> float:
        """The dipole across the strongest field along the orbit."""
        return float(np.linalg.norm(self.dipole_body_a_m2)) * orbit.max_field_t(
            self.field
        )


@dataclass(frozen=True)
class MomentumSample:
    """Instantaneous disturbance torques and accumulated stored momentum."""

    gravity_gradient_torque_body_n_m: tuple[float, float, float]
    stored_momentum_body_n_m_s: tuple[float, float, float]
    stored_momentum_norm_n_m_s: float
    torques_body_n_m: dict[str, tuple[float, float, float]]
    """Each source's torque in body coordinates, by source name."""
    torque_body_n_m: tuple[float, float, float]
    """The sources' total torque in body coordinates."""
    geomagnetic_field_body_t: tuple[float, float, float] | None = None
    """The geomagnetic field in body coordinates where a source uses one."""
    atmosphere_density_kg_m3: float | None = None
    """The atmosphere's density at the sample where drag is tracked."""
    actuator_momentum_body_n_m_s: tuple[float, float, float] | None = None
    """What the wheels hold in body coordinates: the stored momentum less the
    body's own, I·ω, while it turns; None where no angular velocity was given."""
    actuator_momentum_norm_n_m_s: float | None = None
    """The norm of the wheels' momentum; None where no angular velocity was given."""


def _as_tuple(vector: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    return (float(vector[0]), float(vector[1]), float(vector[2]))


@dataclass(frozen=True)
class MomentumState:
    """A tracker's integration state, to continue from later: the stored
    momentum in inertial coordinates, the torque at the last sample and
    when it was, each None before the first sample."""

    momentum_eci_n_m_s: tuple[float, float, float] | None
    previous_torque_eci_n_m: tuple[float, float, float] | None
    previous_utime: float | None


class StoredMomentumTracker:
    """Integrate the momentum stored while rejecting the environmental torques.

    Momentum is integrated in ECI so changing spacecraft attitude only changes
    its body-frame components; it does not create spurious stored momentum.
    This is a sampled integrator, not a trajectory interpolator: the torque
    is taken as linear between samples, so callers sample a slew finely
    enough that the attitude moves only a few degrees between samples. A held
    attitude needs no more than the orbit's resolution, since the torques
    then vary only with the orbit.

    ``sources`` are the torques integrated; by default the gravity gradient
    on the inertia tensor alone, or none with ``gravity_gradient_enabled``
    off. ``max_sample_interval_s`` is an optional guard on the interval
    between samples, for a caller that certifies motion against a capacity
    at a fixed cadence; None, the default, samples at whatever resolution
    the caller has.
    """

    def __init__(
        self,
        inertia_tensor_body_kg_m2: npt.ArrayLike,
        initial_momentum_body_n_m_s: npt.ArrayLike = (0.0, 0.0, 0.0),
        *,
        sources: Sequence[TorqueSource] | None = None,
        max_sample_interval_s: float | None = None,
        gravity_gradient_enabled: bool = True,
    ) -> None:
        if max_sample_interval_s is not None and (
            not np.isfinite(max_sample_interval_s)
            or not 0.0 < max_sample_interval_s <= 10.0
        ):
            raise ValueError(
                "max_sample_interval_s must be finite, positive, and <= 10 s"
            )
        self.max_sample_interval_s = (
            float(max_sample_interval_s) if max_sample_interval_s is not None else None
        )
        self.inertia_tensor_body_kg_m2 = _finite_matrix3(
            inertia_tensor_body_kg_m2, name="inertia_tensor_body_kg_m2"
        )
        self.initial_momentum_body_n_m_s = _finite_vector3(
            initial_momentum_body_n_m_s, name="initial_momentum_body_n_m_s"
        )
        if sources is not None:
            self.sources: list[TorqueSource] = list(sources)
        elif gravity_gradient_enabled:
            self.sources = [GravityGradientTorque(self.inertia_tensor_body_kg_m2)]
        else:
            self.sources = []
        self.reset()

    @property
    def gravity_gradient_enabled(self) -> bool:
        """Whether the gravity gradient is among the sources integrated."""
        return any(isinstance(source, GravityGradientTorque) for source in self.sources)

    def reset(self) -> None:
        """Restore the configured initial state for a new simulation run."""
        self._momentum_eci_n_m_s: npt.NDArray[np.float64] | None = None
        self._previous_torque_eci_n_m: npt.NDArray[np.float64] | None = None
        self._previous_utime: float | None = None
        self._actuator_norm_n_m_s: float | None = None

    def copy(self) -> StoredMomentumTracker:
        """Independent speculative state; immutable inputs may be shared."""
        result = copy(self)
        for name in ("_momentum_eci_n_m_s", "_previous_torque_eci_n_m"):
            value = getattr(self, name)
            setattr(result, name, None if value is None else value.copy())
        return result

    def assign(self, other: StoredMomentumTracker) -> None:
        """Take ``other``'s integration state, as a live tracker does from a
        forecast committed on a copy of it."""
        for name in ("_momentum_eci_n_m_s", "_previous_torque_eci_n_m"):
            value = getattr(other, name)
            setattr(self, name, None if value is None else value.copy())
        self._previous_utime = other._previous_utime
        self._actuator_norm_n_m_s = other._actuator_norm_n_m_s

    def validate_sample_interval(self, elapsed_s: float) -> None:
        """Reject undersampling without changing integration state, where a
        sample interval guard is set."""
        if not np.isfinite(elapsed_s) or elapsed_s <= 0.0:
            raise ValueError("momentum sample interval must be finite and positive")
        if (
            self.max_sample_interval_s is not None
            and elapsed_s > self.max_sample_interval_s * (1.0 + 1e-12)
        ):
            raise ValueError(
                f"stored momentum sample interval {elapsed_s:g} s exceeds "
                f"{self.max_sample_interval_s:g} s; supply finer executed-attitude "
                "and ephemeris samples"
            )

    @property
    def state(self) -> MomentumState:
        """The integration state now, hashable, for :meth:`restore`."""
        return MomentumState(
            momentum_eci_n_m_s=(
                _as_tuple(self._momentum_eci_n_m_s)
                if self._momentum_eci_n_m_s is not None
                else None
            ),
            previous_torque_eci_n_m=(
                _as_tuple(self._previous_torque_eci_n_m)
                if self._previous_torque_eci_n_m is not None
                else None
            ),
            previous_utime=self._previous_utime,
        )

    def restore(self, state: MomentumState) -> None:
        """Continue from ``state``, as taken from :attr:`state`."""
        self._momentum_eci_n_m_s = (
            np.array(state.momentum_eci_n_m_s, dtype=np.float64)
            if state.momentum_eci_n_m_s is not None
            else None
        )
        self._previous_torque_eci_n_m = (
            np.array(state.previous_torque_eci_n_m, dtype=np.float64)
            if state.previous_torque_eci_n_m is not None
            else None
        )
        self._previous_utime = state.previous_utime
        self._actuator_norm_n_m_s = None

    def unload(self, residual_fraction: float = 0.0) -> None:
        """Take the stored momentum out of the wheels, as a thruster dump
        does, leaving ``residual_fraction`` of it along the same direction."""
        if not 0.0 <= residual_fraction < 1.0:
            raise ValueError("residual_fraction must be in [0, 1)")
        if self._momentum_eci_n_m_s is not None:
            self._momentum_eci_n_m_s *= residual_fraction
        self._actuator_norm_n_m_s = None

    @property
    def stored_momentum_norm_n_m_s(self) -> float:
        """The stored momentum's norm now."""
        if self._momentum_eci_n_m_s is None:
            return float(np.linalg.norm(self.initial_momentum_body_n_m_s))
        return float(np.linalg.norm(self._momentum_eci_n_m_s))

    @property
    def actuator_momentum_norm_n_m_s(self) -> float:
        """The norm of what the wheels hold at the last sample: the stored
        momentum less the body's own while it turns, where the sample gave
        the angular velocity; the stored momentum's norm otherwise."""
        if self._actuator_norm_n_m_s is None:
            return self.stored_momentum_norm_n_m_s
        return self._actuator_norm_n_m_s

    @property
    def momentum_eci_n_m_s(self) -> npt.NDArray[np.float64] | None:
        """The stored momentum in inertial coordinates, once a sample has
        set it; None before the first sample."""
        if self._momentum_eci_n_m_s is None:
            return None
        return self._momentum_eci_n_m_s.copy()

    def start_from_eci(self, momentum_eci_n_m_s: npt.ArrayLike) -> None:
        """Begin from a stored momentum given in inertial coordinates, as a
        forecast does from the momentum a run has when its horizon starts."""
        self.reset()
        self._momentum_eci_n_m_s = _finite_vector3(
            momentum_eci_n_m_s, name="momentum_eci_n_m_s"
        ).copy()

    def update(
        self,
        *,
        utime: float,
        position_eci_km: npt.ArrayLike,
        attitude_quaternion_eci_to_body: npt.ArrayLike,
        sun_eci_km: npt.ArrayLike | None = None,
        sunlit: bool = True,
        velocity_eci_km_s: npt.ArrayLike | None = None,
        position_ecef_km: npt.ArrayLike | None = None,
        angular_velocity_body_rad_s: npt.ArrayLike | None = None,
        torque_eci_n_m: npt.ArrayLike | None = None,
    ) -> MomentumSample:
        """Advance to one attitude sample using trapezoidal torque integration.

        ``sun_eci_km`` is the vector from the spacecraft to the Sun, ``sunlit``
        whether the Sun shines on it, and ``position_ecef_km`` the Earth-fixed
        position for the geomagnetic field; a source that needs them
        contributes nothing without them. ``angular_velocity_body_rad_s`` is
        the body's rate: the momentum tracked is the whole spacecraft's, and
        the wheels hold it less the body's own, I·ω, reported as the
        actuator momentum; without a rate the wheels hold it all.
        ``torque_eci_n_m`` stands in for the sources' torque where a
        trajectory forecast evaluates a partial integration cell with the
        same linearly interpolated torque as its full cell.
        """
        if not np.isfinite(utime):
            raise ValueError("utime must be finite")
        body_momentum = (
            np.zeros(3, dtype=np.float64)
            if angular_velocity_body_rad_s is None
            else self.inertia_tensor_body_kg_m2
            @ _finite_vector3(angular_velocity_body_rad_s, name="angular velocity")
        )

        elapsed_s = (
            0.0 if self._previous_utime is None else float(utime - self._previous_utime)
        )
        if elapsed_s < 0.0:
            raise ValueError("momentum samples must have nondecreasing timestamps")
        if self.max_sample_interval_s is not None:
            # Subtracting Unix timestamps can round a nominally valid gap upward.
            time_resolution = float(np.spacing(abs(utime))) * 2.0
            if elapsed_s > self.max_sample_interval_s + time_resolution:
                self.validate_sample_interval(elapsed_s)

        rotation_eci_to_body = quaternion_to_rotation_matrix(
            attitude_quaternion_eci_to_body
        )
        body_momentum = (
            np.zeros(3, dtype=np.float64)
            if angular_velocity_body_rad_s is None
            else self.inertia_tensor_body_kg_m2
            @ _finite_vector3(angular_velocity_body_rad_s, name="angular velocity")
        )
        if self._momentum_eci_n_m_s is None:
            momentum_before = self.initial_momentum_body_n_m_s.copy()
        else:
            momentum_before = rotation_eci_to_body @ self._momentum_eci_n_m_s
        sample = OrbitSample(
            utime=float(utime),
            position_eci_km=_finite_vector3(position_eci_km, name="position_eci_km"),
            rotation_eci_to_body=rotation_eci_to_body,
            sun_eci_km=(
                _finite_vector3(sun_eci_km, name="sun_eci_km")
                if sun_eci_km is not None
                else None
            ),
            sunlit=bool(sunlit),
            velocity_eci_km_s=(
                _finite_vector3(velocity_eci_km_s, name="velocity_eci_km_s")
                if velocity_eci_km_s is not None
                else None
            ),
            position_ecef_km=(
                _finite_vector3(position_ecef_km, name="position_ecef_km")
                if position_ecef_km is not None
                else None
            ),
            stored_momentum_body_n_m_s=np.asarray(momentum_before, dtype=np.float64),
        )
        torques: dict[str, tuple[float, float, float]] = {}
        torque_body = np.zeros(3, dtype=np.float64)
        for source in self.sources:
            contribution = np.asarray(source.torque_body_n_m(sample), dtype=np.float64)
            torques[source.name] = _as_tuple(contribution)
            torque_body += contribution
        torque_eci = rotation_eci_to_body.T @ torque_body
        if torque_eci_n_m is not None:
            torque_eci = _finite_vector3(torque_eci_n_m, name="torque_eci_n_m")
            torque_body = rotation_eci_to_body @ torque_eci

        if self._previous_utime is None:
            if self._momentum_eci_n_m_s is None:
                # The configured initial momentum is the wheels'; a body
                # already turning carries its own besides.
                self._momentum_eci_n_m_s = rotation_eci_to_body.T @ (
                    self.initial_momentum_body_n_m_s + body_momentum
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
        actuator: npt.NDArray[np.float64] | None = None
        self._actuator_norm_n_m_s = None
        if angular_velocity_body_rad_s is not None:
            actuator = momentum_body - body_momentum
            self._actuator_norm_n_m_s = float(np.linalg.norm(actuator))

        field_body: tuple[float, float, float] | None = None
        density: float | None = None
        for source in self.sources:
            if field_body is None and isinstance(
                source, (MagnetorquerUnloading, ResidualDipoleTorque)
            ):
                field = sample.geomagnetic_field_body_t(source.field)
                field_body = _as_tuple(field) if field is not None else None
            if isinstance(source, AerodynamicDragTorque):
                density = source.last_density_kg_m3
        return MomentumSample(
            gravity_gradient_torque_body_n_m=(
                _as_tuple(torque_body)
                if torque_eci_n_m is not None
                else torques.get(GravityGradientTorque.name, (0.0, 0.0, 0.0))
            ),
            stored_momentum_body_n_m_s=_as_tuple(momentum_body),
            stored_momentum_norm_n_m_s=float(np.linalg.norm(self._momentum_eci_n_m_s)),
            torques_body_n_m=torques,
            torque_body_n_m=_as_tuple(torque_body),
            geomagnetic_field_body_t=field_body,
            atmosphere_density_kg_m3=density,
            actuator_momentum_body_n_m_s=(
                _as_tuple(actuator) if actuator is not None else None
            ),
            actuator_momentum_norm_n_m_s=self._actuator_norm_n_m_s,
        )


def build_torque_sources(config: MissionConfig) -> list[TorqueSource]:
    """The torque sources a configuration's stored-momentum tracking enables,
    in the order they are reported: gravity gradient, solar radiation
    pressure and aerodynamic drag on the configured or placeholder surfaces,
    the residual dipole and the magnetorquers' unloading, the last two in
    the configured geomagnetic field. Empty where tracking is off."""
    from ..config.surfaces import guess_surfaces

    momentum_config = config.spacecraft_bus.attitude_control.stored_momentum
    inertia = config.spacecraft_bus.inertia_tensor_body_kg_m2
    sources: list[TorqueSource] = []
    if not momentum_config.enabled:
        return sources
    if inertia is None:
        raise ValueError(
            "inertia_tensor_body_kg_m2 is required when stored-momentum "
            "tracking is enabled"
        )
    if momentum_config.gravity_gradient_enabled:
        sources.append(GravityGradientTorque(inertia))
    surfaces = config.spacecraft_bus.surfaces
    if surfaces is None and (
        momentum_config.solar_radiation_pressure_enabled
        or momentum_config.aerodynamic_drag_enabled
    ):
        surfaces = guess_surfaces(config.solar_panel, config.payload)
    if momentum_config.solar_radiation_pressure_enabled:
        assert surfaces is not None
        sources.append(SolarRadiationPressureTorque(surfaces))
    if momentum_config.aerodynamic_drag_enabled:
        assert surfaces is not None
        sources.append(
            AerodynamicDragTorque(
                surfaces, atmosphere_from_config(momentum_config.atmosphere)
            )
        )
    field = field_from_name(momentum_config.geomagnetic_field)
    if momentum_config.residual_dipole_body_a_m2 is not None:
        sources.append(
            ResidualDipoleTorque(field, momentum_config.residual_dipole_body_a_m2)
        )
    if momentum_config.magnetorquer_max_dipole_a_m2 is not None:
        sources.append(
            MagnetorquerUnloading(
                field,
                momentum_config.magnetorquer_max_dipole_a_m2,
                momentum_config.magnetorquer_time_constant_s,
            )
        )
    return sources


def tracker_from_config(config: MissionConfig) -> StoredMomentumTracker | None:
    """A tracker for a configuration's stored-momentum tracking, or None
    where it is off."""
    momentum_config = config.spacecraft_bus.attitude_control.stored_momentum
    if not momentum_config.enabled:
        return None
    sources = build_torque_sources(config)
    inertia = config.spacecraft_bus.inertia_tensor_body_kg_m2
    assert inertia is not None
    return StoredMomentumTracker(
        inertia_tensor_body_kg_m2=inertia,
        initial_momentum_body_n_m_s=momentum_config.initial_momentum_body_n_m_s,
        sources=sources,
    )


def slew_sample_interval_s(config: MissionConfig) -> float:
    """How often a slew is sampled for the tracker: the configured attitude
    step at the fastest configured slew rate, so the attitude moves no more
    than that step between samples. A coupled-axis ellipsoid cannot exceed its
    fastest principal rate."""
    attitude_control = config.spacecraft_bus.attitude_control
    fastest = (
        max(attitude_control.max_slew_rate_body)
        if attitude_control.max_slew_rate_body is not None
        else attitude_control.max_slew_rate
    )
    if not np.isfinite(fastest) or fastest <= 0.0:
        raise ValueError(
            "momentum tracking requires a finite positive maximum slew rate"
        )
    return float(attitude_control.stored_momentum.max_attitude_step_deg / fastest)
