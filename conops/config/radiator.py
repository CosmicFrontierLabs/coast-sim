"""Radiator configuration and thermal exposure modelling.

This module mirrors the star tracker pattern for body-mounted radiators:
- Per-radiator orientation in spacecraft body frame
- Optional hard constraints evaluated in radiator-local frame
- Exposure tracking to Sun and Earth over orbit
- Aggregate metrics for scheduler penalties and telemetry
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import numpy.typing as npt
import rust_ephem
from pydantic import Field, field_validator
from rust_ephem.constraints import ConstraintConfig

from ..common import dtutcfromtimestamp, scbodyvector
from ..common.vector import radec2vec, rotvec, vec2radec, vecnorm
from ._base import ConfigModel
from .constraint import Constraint


class RadiatorOrientation(ConfigModel):
    """Orientation of a radiator normal in spacecraft body frame."""

    normal: tuple[float, float, float] = Field(
        default=(0.0, 1.0, 0.0),
        description="Radiator outward normal as unit vector in body frame",
    )

    @field_validator("normal")
    @classmethod
    def validate_unit_vector(
        cls, v: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        magnitude = np.sqrt(sum(x**2 for x in v))
        if magnitude < 0.99 or magnitude > 1.01:
            raise ValueError(f"Radiator normal must be unit vector. Got {magnitude}")
        return v

    def to_rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Build a local-frame basis matrix from radiator normal."""
        bore = np.array(self.normal, dtype=np.float64)

        if abs(bore[2]) < 0.99:
            second = np.cross(np.array([0.0, 0.0, 1.0]), bore)
        else:
            second = np.cross(np.array([1.0, 0.0, 0.0]), bore)

        second = vecnorm(second)
        third = vecnorm(np.cross(bore, second))
        return np.column_stack([bore, second, third])

    def transform_pointing(
        self, ra_deg: float, dec_deg: float, roll_deg: float = 0.0
    ) -> tuple[float, float]:
        """Transform spacecraft pointing to radiator-local frame."""
        ra_rad = np.deg2rad(ra_deg)
        dec_rad = np.deg2rad(dec_deg)
        v_sc = radec2vec(ra_rad, dec_rad)

        roll_rad = np.deg2rad(roll_deg)
        v_body = rotvec(1, roll_rad, v_sc)

        rot_matrix = self.to_rotation_matrix()
        v_local = rot_matrix.T @ v_body

        ra_local_rad, dec_local_rad = vec2radec(v_local)
        return np.rad2deg(ra_local_rad), np.rad2deg(dec_local_rad)


class Radiator(ConfigModel):
    """Configuration for a single body-mounted radiator panel."""

    _eclipse_constraint = rust_ephem.EclipseConstraint()

    name: str = "Radiator"
    width_m: float = Field(default=1.0, gt=0.0)
    height_m: float = Field(default=1.0, gt=0.0)
    orientation: RadiatorOrientation = Field(default_factory=RadiatorOrientation)
    subsystem: str = Field(
        default="spacecraft_bus",
        description="Subsystem served by this radiator (e.g. payload, bus)",
    )

    efficiency: float = Field(default=0.9, ge=0.0, le=1.0)
    emissivity: float = Field(default=0.85, ge=0.0, le=1.0)
    dissipation_coefficient_w_per_m2: float = Field(
        default=220.0,
        ge=0.0,
        description="Nominal heat rejection coefficient used for first-order thermal model",
    )
    sun_loading_factor: float = Field(default=0.7, ge=0.0)
    earth_loading_factor: float = Field(default=0.3, ge=0.0)

    hard_constraint: Constraint | None = None

    @property
    def area_m2(self) -> float:
        return self.width_m * self.height_m

    def set_ephem(self, ephem: rust_ephem.Ephemeris) -> None:
        if self.hard_constraint is not None:
            self.hard_constraint.ephem = ephem

    def in_hard_constraint(
        self, ra_deg: float, dec_deg: float, utime: float, roll_deg: float = 0.0
    ) -> bool:
        if self.hard_constraint is None:
            return False
        ra_local, dec_local = self.orientation.transform_pointing(
            ra_deg, dec_deg, roll_deg
        )
        return self.hard_constraint.in_constraint(ra_local, dec_local, utime)

    def exposure_factors(
        self,
        ra_deg: float,
        dec_deg: float,
        utime: float,
        ephem: rust_ephem.Ephemeris,
        roll_deg: float = 0.0,
    ) -> tuple[float, float]:
        """Return (sun_exposure, earth_exposure) as geometric fractions in [0, 1]."""
        idx = ephem.index(dtutcfromtimestamp(utime))

        # Inertial unit vectors from spacecraft to Sun/Earth centers.
        sun_vec = radec2vec(
            np.deg2rad(ephem.sun_ra_deg[idx]),
            np.deg2rad(ephem.sun_dec_deg[idx]),
        )
        earth_vec = radec2vec(
            np.deg2rad(ephem.earth_ra_deg[idx]),
            np.deg2rad(ephem.earth_dec_deg[idx]),
        )

        sun_body = scbodyvector(
            np.deg2rad(ra_deg), np.deg2rad(dec_deg), np.deg2rad(roll_deg), sun_vec
        )
        earth_body = scbodyvector(
            np.deg2rad(ra_deg), np.deg2rad(dec_deg), np.deg2rad(roll_deg), earth_vec
        )

        normal = np.asarray(self.orientation.normal, dtype=np.float64)

        sun_norm = np.linalg.norm(sun_body)
        earth_norm = np.linalg.norm(earth_body)

        sun_exposure = 0.0
        if sun_norm > 0:
            sun_unit = sun_body / sun_norm
            sun_exposure = max(0.0, float(np.dot(normal, sun_unit)))

            in_eclipse = self._eclipse_constraint.in_constraint(
                ephemeris=ephem,
                target_ra=0.0,
                target_dec=0.0,
                time=dtutcfromtimestamp(utime),
            )
            if in_eclipse:
                sun_exposure = 0.0

        earth_exposure = 0.0
        if earth_norm > 0:
            earth_unit = earth_body / earth_norm
            earth_exposure = max(0.0, float(np.dot(normal, earth_unit)))

        return sun_exposure, earth_exposure

    def heat_dissipation_w(self, sun_exposure: float, earth_exposure: float) -> float:
        """Compute first-order radiator heat shedding capability in Watts."""
        loading = (
            self.sun_loading_factor * sun_exposure
            + self.earth_loading_factor * earth_exposure
        )
        attenuation = max(0.0, 1.0 - loading)
        return (
            self.area_m2
            * self.emissivity
            * self.efficiency
            * self.dissipation_coefficient_w_per_m2
            * attenuation
        )


class RadiatorConfiguration(ConfigModel):
    """Configuration for spacecraft radiator subsystem."""

    radiators: list[Radiator] = Field(default_factory=list)

    @staticmethod
    def _normal_to_euler_deg(
        normal: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        x, y, z = vecnorm(np.asarray(normal, dtype=np.float64))
        yaw_deg = float(np.rad2deg(np.arctan2(y, x)))
        pitch_deg = float(np.rad2deg(np.arctan2(z, np.hypot(x, y))))
        roll_deg = 0.0
        return roll_deg, pitch_deg, yaw_deg

    @cached_property
    def radiator_hard_constraint(self) -> ConstraintConfig | None:
        combined: ConstraintConfig | None = None

        for rad in self.radiators:
            if rad.hard_constraint is None:
                continue
            base_constraint = rad.hard_constraint.constraint
            if base_constraint is None:
                continue

            roll_deg, pitch_deg, yaw_deg = self._normal_to_euler_deg(
                rad.orientation.normal
            )
            offset_constraint = base_constraint.boresight_offset(
                roll_deg=roll_deg,
                pitch_deg=pitch_deg,
                yaw_deg=yaw_deg,
            )

            if combined is None:
                combined = offset_constraint
            else:
                combined = combined | offset_constraint

        return combined

    def set_ephem(self, ephem: rust_ephem.Ephemeris) -> None:
        for rad in self.radiators:
            rad.set_ephem(ephem)

    def num_radiators(self) -> int:
        return len(self.radiators)

    def radiators_violating_hard_constraints(
        self, ra_deg: float, dec_deg: float, utime: float, roll_deg: float = 0.0
    ) -> int:
        count = 0
        for rad in self.radiators:
            if rad.in_hard_constraint(ra_deg, dec_deg, utime, roll_deg):
                count += 1
        return count

    def is_pointing_valid(
        self,
        ra_deg: float,
        dec_deg: float,
        utime: float,
        roll_deg: float = 0.0,
    ) -> bool:
        if len(self.radiators) == 0:
            return True
        return (
            self.radiators_violating_hard_constraints(ra_deg, dec_deg, utime, roll_deg)
            == 0
        )

    def get_radiator_by_name(self, name: str) -> Radiator | None:
        for rad in self.radiators:
            if rad.name == name:
                return rad
        return None

    def exposure_metrics(
        self,
        ra_deg: float,
        dec_deg: float,
        utime: float,
        ephem: rust_ephem.Ephemeris,
        roll_deg: float = 0.0,
    ) -> dict[str, float | list[dict[str, float | str | bool]]]:
        """Aggregate Sun/Earth exposure and heat dissipation metrics for all radiators."""
        if not self.radiators:
            return {
                "sun_exposure": 0.0,
                "earth_exposure": 0.0,
                "heat_dissipation_w": 0.0,
                "per_radiator": [],
            }

        area_total = sum(rad.area_m2 for rad in self.radiators)
        if area_total <= 0:
            area_total = 1.0

        sun_weighted = 0.0
        earth_weighted = 0.0
        heat_total = 0.0
        per_radiator: list[dict[str, float | str | bool]] = []

        for rad in self.radiators:
            sun_exp, earth_exp = rad.exposure_factors(
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                utime=utime,
                ephem=ephem,
                roll_deg=roll_deg,
            )
            heat_w = rad.heat_dissipation_w(sun_exp, earth_exp)

            sun_weighted += sun_exp * rad.area_m2
            earth_weighted += earth_exp * rad.area_m2
            heat_total += heat_w

            per_radiator.append(
                {
                    "name": rad.name,
                    "subsystem": rad.subsystem,
                    "area_m2": rad.area_m2,
                    "sun_exposure": sun_exp,
                    "earth_exposure": earth_exp,
                    "heat_dissipation_w": heat_w,
                    "hard_violation": rad.in_hard_constraint(
                        ra_deg, dec_deg, utime, roll_deg
                    ),
                }
            )

        return {
            "sun_exposure": sun_weighted / area_total,
            "earth_exposure": earth_weighted / area_total,
            "heat_dissipation_w": heat_total,
            "per_radiator": per_radiator,
        }


class DefaultRadiatorConfiguration(RadiatorConfiguration):
    """Default radiator configuration with no explicit radiators configured."""

    radiators: list[Radiator] = Field(default_factory=list)
