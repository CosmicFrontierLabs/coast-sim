"""Immutable snapshots of the prescribed slew law, not a dynamics model."""

from functools import cached_property

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..common.vector import quat_slerp, quat_to_attitude, quaternion_attitude_delta
from ..config.acs import AttitudeControlSystem

Quaternion = tuple[float, float, float, float]


def same_rotation(a: npt.ArrayLike, b: npt.ArrayLike) -> bool:
    """Compare unit quaternions, allowing either sign (about 1e-6 deg)."""
    qa, qb = np.asarray(a), np.asarray(b)
    return bool(min(np.linalg.norm(qa - qb), np.linalg.norm(qa + qb)) <= 1e-8)


def unit_quaternion(value: Quaternion) -> Quaternion:
    if not np.isfinite(value).all() or abs(np.linalg.norm(value) - 1.0) > 1e-8:
        raise ValueError("attitude must be a finite unit quaternion in wxyz order")
    return value


class SlewMotionSegment(BaseModel):
    """One rest-to-rest SLERP segment with resolved directional limits."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    start_quaternion: Quaternion
    end_quaternion: Quaternion
    distance_deg: float = Field(ge=0.0, le=180.0 + 1e-8)
    acceleration_deg_s2: float = Field(gt=0.0)
    max_rate_deg_s: float = Field(gt=0.0)

    _validate_quaternions = field_validator("start_quaternion", "end_quaternion")(
        unit_quaternion
    )

    @model_validator(mode="after")
    def _validate_distance(self) -> "SlewMotionSegment":
        distance, _ = quaternion_attitude_delta(
            *quat_to_attitude(np.asarray(self.start_quaternion)),
            *quat_to_attitude(np.asarray(self.end_quaternion)),
        )
        if abs(distance - self.distance_deg) > 1e-6:
            raise ValueError("segment distance disagrees with quaternion endpoints")
        return self

    @cached_property
    def _kinematics(self) -> AttitudeControlSystem:
        # Reuse the runtime law; never retain the mutable mission configuration.
        return AttitudeControlSystem(
            slew_acceleration=self.acceleration_deg_s2,
            max_slew_rate=self.max_rate_deg_s,
            settle_time=0.0,
        )

    @cached_property
    def duration_s(self) -> float:
        return self._kinematics.motion_time(self.distance_deg)

    def quaternion_at(self, elapsed_s: float) -> npt.NDArray[np.float64]:
        distance = self._kinematics.s_of_t(self.distance_deg, elapsed_s)
        fraction = distance / self.distance_deg if self.distance_deg else 0.0
        return quat_slerp(
            np.asarray(self.start_quaternion), np.asarray(self.end_quaternion), fraction
        )


class SlewMotionProfile(BaseModel):
    """Executed motion followed by a fixed hold, including settling/rounding.

    Quaternions inherit the containing attitude sidecar's GCRS-to-body,
    scalar-first Hamilton convention. Validity is supplied separately by the
    controller: the mathematical hold must not imply coverage after preemption.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    start_utime: float
    segments: tuple[SlewMotionSegment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_continuity(self) -> "SlewMotionProfile":
        for first, second in zip(self.segments, self.segments[1:]):
            if not same_rotation(first.end_quaternion, second.start_quaternion):
                raise ValueError("slew segments must have continuous attitudes")
        return self

    def quaternion_at(self, utime: float) -> npt.NDArray[np.float64]:
        if not np.isfinite(utime):
            raise ValueError("utime must be finite")
        elapsed = utime - self.start_utime
        for segment in self.segments:
            if elapsed <= segment.duration_s:
                return segment.quaternion_at(elapsed)
            elapsed -= segment.duration_s
        return np.asarray(self.segments[-1].end_quaternion)

    @property
    def breakpoints(self) -> tuple[float, ...]:
        """Motion phase boundaries for numerical quadrature."""
        result = [self.start_utime]
        for segment in self.segments:
            accel_time = min(
                segment.max_rate_deg_s / segment.acceleration_deg_s2,
                (segment.distance_deg / segment.acceleration_deg_s2) ** 0.5,
            )
            start = result[-1]
            result.extend(
                (
                    start + accel_time,
                    start + segment.duration_s - accel_time,
                    start + segment.duration_s,
                )
            )
        return tuple(result)


class ExecutedAttitudeInterval(BaseModel):
    """Closed interval where the controller followed this motion/hold exactly."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    start_utime: float
    end_utime: float
    motion: SlewMotionProfile

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ExecutedAttitudeInterval":
        if (
            self.end_utime < self.start_utime
            or self.start_utime < self.motion.start_utime
        ):
            raise ValueError("invalid executed attitude interval bounds")
        return self
