"""Configuration for target selection and scheduling."""

from pydantic import Field

from ._base import ConfigModel


class TargetConfig(ConfigModel):
    """Configuration for target selection and scheduling behavior."""

    slew_distance_weight: float = Field(
        default=0.0,
        description="Weight to penalize long slews when selecting next target",
    )
    slew_time_weight: float = Field(
        default=0.0,
        description=(
            "Weight to penalize slew time during target selection, "
            "in merit points per minute"
        ),
    )
    collection_time_weight: float = Field(
        default=0.0,
        description=(
            "Weight to reward expected useful collection time during target "
            "selection, in merit points per minute"
        ),
    )
    radiator_sun_exposure_weight: float = Field(
        default=0.0,
        description="Weight to penalize radiator sun exposure during target selection",
    )
    radiator_earth_exposure_weight: float = Field(
        default=0.0,
        description="Weight to penalize radiator earth exposure during target selection",
    )
    urgency_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Merit points added at full urgency, for a target whose deadline "
            "(or last visibility window before it) closes within "
            "urgency_timescale_seconds"
        ),
    )
    urgency_timescale_seconds: float = Field(
        default=3600.0,
        gt=0.0,
        description=(
            "Time to close at which urgency reaches its maximum; urgency falls "
            "off as timescale / time-to-close beyond it"
        ),
    )
    cadence_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Merit points added once a target's category cadence interval has "
            "elapsed since its last visit, rising linearly before that"
        ),
    )
    completion_deficit_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Merit points per unit of program deficit (allocated time share "
            "minus delivered share of science time)"
        ),
    )
