"""Configuration for target selection and scheduling."""

from pydantic import Field

from ._base import ConfigModel


class TargetConfig(ConfigModel):
    """Configuration for target selection and scheduling behavior."""

    slew_distance_weight: float = Field(
        default=0.0,
        description="Weight to penalize long slews when selecting next target",
    )
