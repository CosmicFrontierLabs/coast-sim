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
