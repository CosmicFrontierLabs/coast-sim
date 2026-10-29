from typing import Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from ._base import ConfigModel


class ThrusterDumpConfig(ConfigModel):
    """A thruster momentum dump: a hold during which the wheels are unloaded."""

    duration_seconds: float = Field(
        default=120.0,
        gt=0.0,
        allow_inf_nan=False,
        description="How long a dump holds the spacecraft, in seconds",
    )
    residual_fraction: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        allow_inf_nan=False,
        description=(
            "The fraction of the stored momentum a dump leaves in the wheels; "
            "0 unloads them fully"
        ),
    )


class AtmosphereConfig(ConfigModel):
    """The atmosphere aerodynamic drag is computed in."""

    model: Literal["table", "msis"] = Field(
        default="table",
        description=(
            "The density model: 'table', the built-in piecewise exponential "
            "profile by height for moderate solar activity; or 'msis', "
            "NRLMSIS 2.1 through the optional pymsis package, at the solar "
            "flux and geomagnetic activity below"
        ),
    )
    f107: float = Field(
        default=150.0,
        ge=0.0,
        allow_inf_nan=False,
        description="The daily 10.7 cm solar flux the 'msis' model assumes, in sfu",
    )
    f107a: float = Field(
        default=150.0,
        ge=0.0,
        allow_inf_nan=False,
        description="The 81-day average 10.7 cm solar flux the 'msis' model assumes",
    )
    ap: float = Field(
        default=4.0,
        ge=0.0,
        allow_inf_nan=False,
        description="The geomagnetic activity index the 'msis' model assumes",
    )
    density_scale: float = Field(
        default=1.0,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "A factor on the density, for sensitivity studies: the density "
            "varies by a factor of twenty between solar minimum and maximum"
        ),
    )


class StoredMomentumConfig(ConfigModel):
    """Configuration for planning-level stored-momentum tracking."""

    @property
    def capacity_enabled(self) -> bool:
        """Whether the capacity is enforced as a hard limit on the executed
        motion (see ``enforce_capacity``)."""
        return self.capacity_n_m_s is not None and self.enforce_capacity

    enforce_capacity: bool = Field(
        default=False,
        description=(
            "Whether capacity_n_m_s is enforced as a hard limit on the executed "
            "motion: the momentum of every slew and hold, the body's own "
            "included, is certified against it before it is flown, an "
            "observation is admitted only at a roll and rate at which it fits "
            "with a return to the departure attitude (slew_rate_fractions, "
            "roll_step_deg), and a run that cannot continue within it fails. "
            "Off, the capacity is the red line thruster dumps keep the wheels "
            "under and fault management records; the two are exclusive, since "
            "enforcement integrates the momentum in the executor and dumps need "
            "the run's tracker"
        ),
    )
    slew_rate_fractions: tuple[float, ...] = Field(
        default=(1.0, 0.75, 0.5, 0.25, 0.125),
        description="Descending speed caps tried after a momentum violation; acceleration and settling are unchanged.",
    )
    roll_step_deg: int = Field(
        default=30,
        gt=0,
        le=360,
        description="Bounded alternative-roll search increment when momentum admission is enabled.",
    )

    @field_validator("slew_rate_fractions")
    @classmethod
    def _validate_speed_search(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if (
            not values
            or values[0] != 1.0
            or not all(np.isfinite(x) and 0 < x <= 1 for x in values)
            or any(a <= b for a, b in zip(values, values[1:]))
        ):
            raise ValueError(
                "slew_rate_fractions must start at 1 and strictly descend within (0, 1]"
            )
        return values

    @field_validator("roll_step_deg")
    @classmethod
    def _validate_roll_step(cls, value: int) -> int:
        if 360 % value:
            raise ValueError("roll_step_deg must divide 360")
        return value

    gravity_gradient_enabled: bool = Field(
        default=False,
        description=(
            "Accumulate stored momentum required to reject central-body "
            "gravity-gradient torque."
        ),
    )
    solar_radiation_pressure_enabled: bool = Field(
        default=False,
        description=(
            "Accumulate stored momentum required to reject the torque of "
            "sunlight on the spacecraft's surfaces (spacecraft_bus.surfaces, "
            "or a placeholder model built from the bus, panels and telescopes)"
        ),
    )
    aerodynamic_drag_enabled: bool = Field(
        default=False,
        description=(
            "Accumulate stored momentum required to reject the torque of the "
            "atmosphere on the spacecraft's surfaces (spacecraft_bus.surfaces, "
            "or a placeholder model built from the bus, panels and telescopes), "
            "at the density the atmosphere model gives"
        ),
    )
    atmosphere: AtmosphereConfig = Field(
        default_factory=AtmosphereConfig,
        description="The atmosphere drag is computed in",
    )
    geomagnetic_field: Literal["dipole", "igrf"] = Field(
        default="dipole",
        description=(
            "The field the magnetorquers unload against and the residual dipole "
            "is torqued by: 'dipole', the built-in tilted dipole; or 'igrf', the "
            "IGRF through the optional ppigrf package"
        ),
    )
    max_sample_interval_s: float = Field(
        default=10.0,
        gt=0.0,
        le=10.0,
        allow_inf_nan=False,
        description=(
            "Where the capacity is enforced, the longest interval the executor's "
            "budget integrates over, in seconds, further limited by "
            "max_attitude_step_deg at the fastest configured slew rate; the "
            "ephemeris must be at least as fine. The run's tracker samples at "
            "the ephemeris resolution regardless"
        ),
    )
    max_attitude_step_deg: float = Field(
        default=5.0,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "How far the attitude may move between momentum samples along a "
            "slew, in degrees. The simulations and the planners' forecast sample "
            "each slew at this step over the fastest configured slew rate, "
            "whatever the ephemeris resolution; a held attitude is sampled at "
            "the ephemeris resolution. Lower for convergence studies. This is "
            "not an integration-error guarantee."
        ),
    )
    initial_momentum_body_n_m_s: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description=(
            "Initial stored-momentum vector in spacecraft body coordinates, in N m s."
        ),
    )

    magnetorquer_max_dipole_a_m2: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "The magnetorquers' greatest dipole, in A m²; they unload the wheels "
            "against the geomagnetic field (a tilted dipole) continuously, taking "
            "out the stored momentum across the field. None: no magnetorquers"
        ),
    )
    magnetorquer_time_constant_s: float = Field(
        default=1800.0,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "How quickly the magnetorquers aim to unload the stored momentum, "
            "in seconds, within their dipole"
        ),
    )
    residual_dipole_body_a_m2: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "The spacecraft's residual magnetic dipole in body coordinates, in "
            "A m², torqued by the geomagnetic field; None: none"
        ),
    )

    capacity_n_m_s: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "The wheels' momentum capacity, in N m s, taken as the same about "
            "every axis; None sets no capacity"
        ),
    )
    planning_level: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
        description=(
            "The fraction of the capacity at which a thruster dump is called "
            "for, leaving the rest as margin"
        ),
    )
    thruster_dump: ThrusterDumpConfig | None = Field(
        default=None,
        description=(
            "Thruster dumps: dispatch begins one when the stored momentum reaches "
            "the planning level of the capacity, holding the attitude it has for "
            "the dump's length and starting no observation meanwhile; None: no "
            "thrusters, so the wheels fill until a run reports it"
        ),
    )

    slew_momentum_enabled: bool = Field(
        default=False,
        description=(
            "Whether the momentum the body carries while slewing, I·ω, counts "
            "against the capacity. On, capacity_n_m_s is the wheels' whole "
            "capacity: a slew is admitted against the room the stored momentum "
            "leaves, flown slower (slew_rate_fractions) where it does not fit, "
            "and the wheels are emptied first where no rate fits. Off, the "
            "capacity is the room left for stored momentum at the configured "
            "slew rates. Needs capacity_n_m_s"
        ),
    )

    @model_validator(mode="after")
    def _dumps_need_a_capacity(self) -> "StoredMomentumConfig":
        if self.thruster_dump is not None and self.capacity_n_m_s is None:
            raise ValueError("thruster dumps need capacity_n_m_s to be triggered at")
        if self.enforce_capacity and self.thruster_dump is not None:
            raise ValueError(
                "enforce_capacity and thruster_dump are exclusive: enforcement "
                "integrates the momentum in the executor, dumps need the run's tracker"
            )
        if self.slew_momentum_enabled and self.capacity_n_m_s is None:
            raise ValueError(
                "slew momentum needs capacity_n_m_s to be admitted against"
            )
        return self

    @property
    def dump_level_n_m_s(self) -> float | None:
        """The stored momentum at which a thruster dump is called for."""
        if self.capacity_n_m_s is None or self.thruster_dump is None:
            return None
        return self.planning_level * self.capacity_n_m_s

    @property
    def enabled(self) -> bool:
        """Whether momentum is tracked: any torque source, or the momentum of
        slews alone."""
        return (
            self.gravity_gradient_enabled
            or self.solar_radiation_pressure_enabled
            or self.aerodynamic_drag_enabled
            or self.magnetorquer_max_dipole_a_m2 is not None
            or self.residual_dipole_body_a_m2 is not None
            or self.slew_momentum_enabled
        )

    @field_validator("initial_momentum_body_n_m_s", "residual_dipole_body_a_m2")
    @classmethod
    def _validate_vector(
        cls, value: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        if value is None:
            return None
        if not all(np.isfinite(component) for component in value):
            raise ValueError("a body vector must contain finite values")
        return float(value[0]), float(value[1]), float(value[2])
