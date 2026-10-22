import numpy as np
from pydantic import Field, field_validator

from ._base import ConfigModel


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
    max_sample_interval_s: float = Field(
        default=10.0,
        gt=0.0,
        le=10.0,
        allow_inf_nan=False,
        description=(
            "Maximum executed-attitude and ephemeris sample interval in seconds. "
            "Further limited by max_attitude_step_deg at the fastest configured "
            "slew rate. Coarser runs fail instead of aliasing torque."
        ),
    )
    max_attitude_step_deg: float = Field(
        default=5.0,
        gt=0.0,
        le=5.0,
        allow_inf_nan=False,
        description=(
            "Maximum possible attitude motion between momentum samples in degrees. "
            "Default five-degree sampling guard; lower for convergence studies. "
            "This is not an integration-error guarantee."
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
