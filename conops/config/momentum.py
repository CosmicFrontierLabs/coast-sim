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

    @field_validator("initial_momentum_body_n_m_s")
    @classmethod
    def _validate_initial_momentum(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if not all(np.isfinite(component) for component in value):
            raise ValueError("initial stored momentum must contain finite values")
        return float(value[0]), float(value[1]), float(value[2])
