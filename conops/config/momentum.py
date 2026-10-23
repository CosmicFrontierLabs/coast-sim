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


class StoredMomentumConfig(ConfigModel):
    """Configuration for planning-level stored-momentum tracking."""

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

    @model_validator(mode="after")
    def _dumps_need_a_capacity(self) -> "StoredMomentumConfig":
        if self.thruster_dump is not None and self.capacity_n_m_s is None:
            raise ValueError("thruster dumps need capacity_n_m_s to be triggered at")
        return self

    @property
    def dump_level_n_m_s(self) -> float | None:
        """The stored momentum at which a thruster dump is called for."""
        if self.capacity_n_m_s is None or self.thruster_dump is None:
            return None
        return self.planning_level * self.capacity_n_m_s

    @property
    def enabled(self) -> bool:
        """Whether any torque source is tracked."""
        return (
            self.gravity_gradient_enabled
            or self.solar_radiation_pressure_enabled
            or self.magnetorquer_max_dipole_a_m2 is not None
            or self.residual_dipole_body_a_m2 is not None
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
