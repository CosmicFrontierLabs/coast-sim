"""Radiator configuration and thermal exposure modelling.

This module mirrors the star tracker pattern for body-mounted radiators:
- Per-radiator orientation in spacecraft body frame
- Optional hard constraints evaluated in radiator-local frame
- Exposure tracking to Sun and Earth over orbit
- Aggregate metrics for scheduler penalties and telemetry
"""

from __future__ import annotations

from functools import cached_property
from typing import cast

import numpy as np
import numpy.typing as npt
import rust_ephem
from pydantic import Field, field_validator
from rust_ephem.constraints import ConstraintConfig

from ..common import dtutcfromtimestamp, scbodyvector
from ..common.vector import normal_to_euler_deg, radec2vec
from ._base import ConfigModel
from .constraint import Constraint

STEFAN_BOLTZMANN_W_PER_M2_K4 = 5.670374419e-8
_ECLIPSE_CONSTRAINT = rust_ephem.EclipseConstraint()


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


class Radiator(ConfigModel):
    """Configuration for a single body-mounted radiator panel."""

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
        description="Upper bound on emitted thermal flux used for calibration/backward compatibility",
    )
    absorptivity: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Effective absorptivity for incoming radiative flux",
    )
    radiator_temperature_k: float = Field(
        default=300.0,
        gt=0.0,
        description="Representative radiator surface temperature in Kelvin",
    )
    sink_temperature_k: float = Field(
        default=3.0,
        ge=0.0,
        description="Background sink temperature in Kelvin",
    )
    solar_constant_w_per_m2: float = Field(
        default=1361.0,
        ge=0.0,
        description="Solar constant used for absorbed Sun flux term (W/m^2)",
    )
    earth_ir_flux_w_per_m2: float = Field(
        default=237.0,
        ge=0.0,
        description="Effective Earth IR flux term (W/m^2)",
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

    @cached_property
    def _hard_constraint_with_offset(self) -> ConstraintConfig | None:
        if self.hard_constraint is None:
            return None
        base_constraint = self.hard_constraint.constraint
        if base_constraint is None:
            return None

        roll_deg, pitch_deg, yaw_deg = normal_to_euler_deg(self.orientation.normal)
        return base_constraint.boresight_offset(
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
        )

    def in_hard_constraint(
        self, ra_deg: float, dec_deg: float, utime: float, roll_deg: float = 0.0
    ) -> bool:
        if self.hard_constraint is None:
            return False

        offset_constraint = self._hard_constraint_with_offset
        if offset_constraint is None:
            return self.hard_constraint.in_constraint(
                ra_deg,
                dec_deg,
                utime,
                target_roll=roll_deg,
            )

        ephem = self.hard_constraint.ephem
        assert ephem is not None, "Ephemeris must be set to use in_hard_constraint"
        return cast(
            bool,
            offset_constraint.in_constraint(
                ephemeris=ephem,
                target_ra=ra_deg,
                target_dec=dec_deg,
                time=dtutcfromtimestamp(utime),
                target_roll=roll_deg,
            ),
        )

    def _dot_exposure(
        self,
        sun_unit: npt.NDArray[np.float64] | None,
        earth_unit: npt.NDArray[np.float64] | None,
        in_eclipse: bool,
    ) -> tuple[float, float]:
        """Return (sun_exposure, earth_exposure) given pre-computed body-frame unit vectors."""
        normal = np.asarray(self.orientation.normal, dtype=np.float64)
        sun_exposure = (
            0.0
            if (sun_unit is None or in_eclipse)
            else max(0.0, float(np.dot(normal, sun_unit)))
        )
        earth_exposure = (
            0.0 if earth_unit is None else max(0.0, float(np.dot(normal, earth_unit)))
        )
        return sun_exposure, earth_exposure

    def exposure_factors(
        self,
        ra_deg: float,
        dec_deg: float,
        utime: float,
        ephem: rust_ephem.Ephemeris,
        roll_deg: float = 0.0,
    ) -> tuple[float, float]:
        """Return (sun_exposure, earth_exposure) as geometric fractions in [0, 1]."""
        dt = dtutcfromtimestamp(utime)
        idx = ephem.index(dt)
        ra_rad, dec_rad, roll_rad = (
            np.deg2rad(ra_deg),
            np.deg2rad(dec_deg),
            np.deg2rad(roll_deg),
        )

        sun_body = scbodyvector(
            ra_rad,
            dec_rad,
            roll_rad,
            radec2vec(
                np.deg2rad(ephem.sun_ra_deg[idx]), np.deg2rad(ephem.sun_dec_deg[idx])
            ),
        )
        earth_body = scbodyvector(
            ra_rad,
            dec_rad,
            roll_rad,
            radec2vec(
                np.deg2rad(ephem.earth_ra_deg[idx]),
                np.deg2rad(ephem.earth_dec_deg[idx]),
            ),
        )

        sun_norm = np.linalg.norm(sun_body)
        earth_norm = np.linalg.norm(earth_body)
        sun_unit: npt.NDArray[np.float64] | None = (
            sun_body / sun_norm if sun_norm > 0 else None
        )
        earth_unit: npt.NDArray[np.float64] | None = (
            earth_body / earth_norm if earth_norm > 0 else None
        )
        in_eclipse = bool(
            _ECLIPSE_CONSTRAINT.in_constraint(
                ephemeris=ephem, target_ra=0.0, target_dec=0.0, time=dt
            )
        )
        return self._dot_exposure(sun_unit, earth_unit, in_eclipse)

    def heat_dissipation_w(self, sun_exposure: float, earth_exposure: float) -> float:
        """Compute net radiator heat flow in Watts.

        Positive values indicate net heat rejection (dumping heat).
        Negative values indicate net absorption from external fluxes.
        """
        absorbed_flux_w_per_m2 = self.absorptivity * (
            self.solar_constant_w_per_m2 * self.sun_loading_factor * sun_exposure
            + self.earth_ir_flux_w_per_m2 * self.earth_loading_factor * earth_exposure
        )

        emitted_flux_w_per_m2 = (
            self.emissivity
            * STEFAN_BOLTZMANN_W_PER_M2_K4
            * (self.radiator_temperature_k**4 - self.sink_temperature_k**4)
        )

        # Keep legacy calibration lever by capping emitted flux.
        emitted_flux_w_per_m2 = min(
            emitted_flux_w_per_m2,
            self.dissipation_coefficient_w_per_m2,
        )

        net_flux_w_per_m2 = emitted_flux_w_per_m2 - absorbed_flux_w_per_m2
        return self.area_m2 * self.efficiency * net_flux_w_per_m2


class RadiatorConfiguration(ConfigModel):
    """Configuration for spacecraft radiator subsystem."""

    radiators: list[Radiator] = Field(default_factory=list)

    @cached_property
    def radiator_hard_constraint(self) -> ConstraintConfig | None:
        combined: ConstraintConfig | None = None

        for rad in self.radiators:
            offset_constraint = rad._hard_constraint_with_offset
            if offset_constraint is None:
                continue

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

        # Compute time- and pointing-dependent quantities once for all radiators.
        dt = dtutcfromtimestamp(utime)
        idx = ephem.index(dt)
        ra_rad, dec_rad, roll_rad = (
            np.deg2rad(ra_deg),
            np.deg2rad(dec_deg),
            np.deg2rad(roll_deg),
        )
        sun_body = scbodyvector(
            ra_rad,
            dec_rad,
            roll_rad,
            radec2vec(
                np.deg2rad(ephem.sun_ra_deg[idx]), np.deg2rad(ephem.sun_dec_deg[idx])
            ),
        )
        earth_body = scbodyvector(
            ra_rad,
            dec_rad,
            roll_rad,
            radec2vec(
                np.deg2rad(ephem.earth_ra_deg[idx]),
                np.deg2rad(ephem.earth_dec_deg[idx]),
            ),
        )
        sun_norm = np.linalg.norm(sun_body)
        earth_norm = np.linalg.norm(earth_body)
        sun_unit: npt.NDArray[np.float64] | None = (
            sun_body / sun_norm if sun_norm > 0 else None
        )
        earth_unit: npt.NDArray[np.float64] | None = (
            earth_body / earth_norm if earth_norm > 0 else None
        )
        in_eclipse = bool(
            _ECLIPSE_CONSTRAINT.in_constraint(
                ephemeris=ephem, target_ra=0.0, target_dec=0.0, time=dt
            )
        )

        area_total = sum(rad.area_m2 for rad in self.radiators)
        if area_total <= 0:
            area_total = 1.0

        sun_weighted = 0.0
        earth_weighted = 0.0
        heat_total = 0.0
        per_radiator: list[dict[str, float | str | bool]] = []

        for rad in self.radiators:
            sun_exp, earth_exp = rad._dot_exposure(sun_unit, earth_unit, in_eclipse)
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


def _default_radiators() -> list[Radiator]:
    return [
        Radiator(
            name="Default Radiator",
            orientation=RadiatorOrientation(normal=(0.0, -1.0, 0.0)),
        )
    ]


class DefaultRadiatorConfiguration(RadiatorConfiguration):
    """Default radiator configuration with a single radiator on the -Y face (opposite the default solar panel)."""

    radiators: list[Radiator] = Field(default_factory=_default_radiators)
