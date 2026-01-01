"""Configuration for target selection and scheduling."""

from pydantic import BaseModel, Field


class TargetConfig(BaseModel):
    """Configuration for target selection and scheduling behavior."""

    slew_distance_weight: float = Field(
        default=0.0,
        description="Weight to penalize long slews when selecting next target. "
        "Score = merit - slew_distance_weight * slew_distance. "
        "Set to 0.0 to disable (default).",
    )
