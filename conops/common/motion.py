"""Shared analytic rest-to-rest kinematics for planning and execution."""

from functools import cached_property
from math import isfinite, sqrt

from pydantic import BaseModel, ConfigDict, model_validator


class RestToRestMotion(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    angle: float
    acceleration: float
    rate: float

    @model_validator(mode="after")
    def _validate_state(self) -> "RestToRestMotion":
        if not isfinite(self.angle) or self.angle < 0:
            raise ValueError("Motion angle must be finite and nonnegative")
        if not all(
            isfinite(value) and value > 0 for value in (self.acceleration, self.rate)
        ):
            raise ValueError("Motion limits must be finite and positive")
        return self

    @cached_property
    def ramp(self) -> float:
        return min(self.rate / self.acceleration, sqrt(self.angle / self.acceleration))

    @cached_property
    def duration(self) -> float:
        return (
            self.angle / (self.acceleration * self.ramp) + self.ramp
            if self.angle
            else 0.0
        )

    def at(self, elapsed: float) -> tuple[float, float]:
        """Angular distance and speed (degrees and degrees/second)."""
        if elapsed <= 0:
            return 0.0, 0.0
        if elapsed >= self.duration:
            return self.angle, 0.0
        remaining = self.duration - elapsed
        peak = self.acceleration * self.ramp
        if elapsed < self.ramp:
            distance = 0.5 * self.acceleration * elapsed**2
        elif remaining < self.ramp:
            distance = self.angle - 0.5 * self.acceleration * remaining**2
        else:
            distance = peak * (elapsed - 0.5 * self.ramp)
        speed = min(self.acceleration * elapsed, peak, self.acceleration * remaining)
        return min(self.angle, max(0.0, distance)), max(0.0, speed)
