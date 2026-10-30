import numpy as np
from pydantic import Field, field_validator

from ._base import ConfigModel


class StoredMomentumConfig(ConfigModel):
    """Configuration for planning-level stored-momentum tracking."""

    @property
    def capacity_enabled(self) -> bool:
        return self.capacity_n_m_s is not None

    capacity_n_m_s: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description="Optional spherical actuator-momentum limit. None disables admission/execution enforcement.",
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
