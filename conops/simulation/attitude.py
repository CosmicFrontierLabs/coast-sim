"""Continuous, rest-to-rest attitude execution independent of operating mode.

Only this executor advances physical attitude. Trajectories are immutable
snapshots; guidance changes cannot rewrite previously executed motion.
"""

from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
from math import inf, isfinite, nextafter
from typing import TYPE_CHECKING

import numpy as np

from ..common.motion import RestToRestMotion
from ..common.vector import (
    attitude_to_quat,
    quat_slerp,
    quat_to_attitude,
    quaternion_attitude_delta,
)
from ..config import AttitudeControlSystem

if TYPE_CHECKING:
    from .slew import Slew

Attitude = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
_ZERO: Attitude = (0.0, 0.0, 0.0)
_ANGLE_TOL = 1e-8


class AttitudeExecutionError(RuntimeError):
    """A requested physical trajectory cannot be executed as specified."""


def _quaternion(attitude: Attitude) -> Quaternion:
    if not all(isfinite(value) for value in attitude):
        raise AttitudeExecutionError("Attitude must be finite")
    q = attitude_to_quat(*attitude)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


@dataclass(frozen=True)
class AttitudeState:
    utime: float
    quaternion: Quaternion
    angular_velocity_body: Attitude = _ZERO  # degrees / second

    @cached_property
    def attitude(self) -> Attitude:
        ra, dec, roll = quat_to_attitude(np.asarray(self.quaternion))
        return float(ra) % 360.0, float(dec), float(roll) % 360.0


@dataclass(frozen=True)
class _MotionLeg:
    start: float
    end: float
    q0: Quaternion
    q1: Quaternion
    angle: float
    axis: Attitude
    motion: RestToRestMotion
    scale: float

    @classmethod
    def between(
        cls,
        start: float,
        first: Attitude,
        last: Attitude,
        limits: AttitudeControlSystem,
        end: float | None = None,
    ) -> "_MotionLeg":
        q0, q1 = _quaternion(first), _quaternion(last)
        angle, axis = quaternion_attitude_delta(*first, *last)
        acceleration = limits.effective_slew_acceleration(
            axis if angle else (1.0, 0.0, 0.0)
        )
        rate = limits.effective_max_slew_rate(axis if angle else (1.0, 0.0, 0.0))
        try:
            motion = RestToRestMotion(angle, acceleration, rate)
        except ValueError as exc:
            raise AttitudeExecutionError(str(exc)) from exc
        duration = motion.duration
        if end is None:
            end = start + duration
            # Unix timestamps can round a short turn down. Never compress motion
            # below its physical duration (even by one floating-point tick).
            if end - start < duration:
                end = nextafter(end, inf)
        if not (isfinite(start) and isfinite(end)) or end < start:
            raise AttitudeExecutionError("Motion times must be finite and ordered")
        interval = end - start
        if not isfinite(interval):
            raise AttitudeExecutionError("Motion interval must be finite")
        if duration > interval:
            raise AttitudeExecutionError(
                "Tracking interval exceeds rate or acceleration limits"
            )
        # Stretch a feasible rest-to-rest turn across a fixed tracking interval.
        scale = min(1.0, duration / interval) if interval > 0 else 1.0
        return cls(start, end, q0, q1, angle, axis, motion, scale)

    def state(self, utime: float) -> AttitudeState:
        if utime <= self.start:
            return AttitudeState(utime, self.q0)
        if utime >= self.end or self.angle == 0:
            return AttitudeState(utime, self.q1)
        distance, speed = self.motion.at((utime - self.start) * self.scale)
        speed *= self.scale
        q = quat_slerp(
            np.asarray(self.q0),
            np.asarray(self.q1),
            min(1.0, max(0.0, distance / self.angle)),
        )
        return AttitudeState(
            utime,
            (float(q[0]), float(q[1]), float(q[2]), float(q[3])),
            (speed * self.axis[0], speed * self.axis[1], speed * self.axis[2]),
        )


@dataclass(frozen=True)
class AttitudeTrajectory:
    """Piecewise bounded turns, at rest at each knot and outside the trajectory."""

    legs: tuple[_MotionLeg, ...]
    end: float

    @property
    def start(self) -> float:
        return self.legs[0].start

    @classmethod
    def from_slew(
        cls, slew: "Slew", limits: AttitudeControlSystem
    ) -> "AttitudeTrajectory":
        points = slew.attitude_waypoints()
        legs = []
        start = float(slew.slewstart)
        for first, last in zip(points, points[1:]):
            leg = _MotionLeg.between(start, first, last, limits)
            legs.append(leg)
            start = leg.end
        if not legs or not isfinite(slew.slewend) or slew.slewend < start - 1e-6:
            raise AttitudeExecutionError(
                "Slew duration cannot contain its physical trajectory"
            )
        if not isfinite(limits.settle_time) or limits.settle_time < 0:
            raise AttitudeExecutionError("Settling time must be finite and nonnegative")
        settle = limits.settle_time if any(leg.angle > 0 for leg in legs) else 0.0
        if slew.slewend < start + settle - 1e-6:
            raise AttitudeExecutionError("Slew duration omits configured settling time")
        return cls(tuple(legs), float(slew.slewend))

    @classmethod
    def turn(
        cls,
        start: float,
        first: Attitude,
        last: Attitude,
        limits: AttitudeControlSystem,
    ) -> "AttitudeTrajectory":
        """A bounded guidance correction, with no observation settling dwell."""
        leg = _MotionLeg.between(start, first, last, limits)
        return cls((leg,), leg.end)

    @classmethod
    def tracking(
        cls,
        samples: list[tuple[float, Attitude]],
        limits: AttitudeControlSystem,
    ) -> "AttitudeTrajectory":
        if len(samples) < 2:
            raise AttitudeExecutionError(
                "Tracking requires at least two timed attitudes"
            )
        legs = []
        for (start, first), (end, last) in zip(samples, samples[1:]):
            if end <= start:
                raise AttitudeExecutionError("Tracking times must increase strictly")
            legs.append(_MotionLeg.between(start, first, last, limits, end))
        return cls(tuple(legs), samples[-1][0])

    @cached_property
    def _starts(self) -> tuple[float, ...]:
        return tuple(leg.start for leg in self.legs)

    def _leg_at(self, utime: float) -> _MotionLeg:
        index = max(0, bisect_right(self._starts, utime) - 1)
        return self.legs[index]

    def state(self, utime: float) -> AttitudeState:
        return self._leg_at(utime).state(utime)

    def next_rest_time(self, utime: float) -> float:
        leg = self._leg_at(utime)
        if leg.start < utime < leg.end and leg.angle > 0:
            return leg.end
        return utime

    def stopped_at(self, utime: float) -> "AttitudeTrajectory":
        """Discard future legs at a rest boundary, preserving motion up to it."""
        if self.next_rest_time(utime) != utime:
            raise AttitudeExecutionError("Cannot truncate a trajectory during motion")
        return AttitudeTrajectory(
            tuple(leg for leg in self.legs if leg.start < utime), utime
        )


class AttitudeExecutor:
    """The sole owner of executed orientation and angular velocity."""

    def __init__(self, utime: float, attitude: Attitude) -> None:
        if not isfinite(utime):
            raise AttitudeExecutionError("Execution time must be finite")
        self._state = AttitudeState(utime, _quaternion(attitude))
        self._trajectory: AttitudeTrajectory | None = None

    @property
    def state(self) -> AttitudeState:
        return self._state

    def _check_time(self, utime: float) -> None:
        if not isfinite(utime) or utime < self._state.utime:
            raise AttitudeExecutionError("Execution time must be finite and monotonic")

    def predict(self, utime: float) -> AttitudeState:
        self._check_time(utime)
        if utime == self._state.utime:
            return self._state
        if self._trajectory is not None:
            return self._trajectory.state(utime)
        return AttitudeState(utime, self._state.quaternion)

    def advance(self, utime: float) -> None:
        self._state = self.predict(utime)

    def next_rest_time(self, utime: float) -> float:
        self._check_time(utime)
        return self._trajectory.next_rest_time(utime) if self._trajectory else utime

    def request_stop(self, utime: float) -> float:
        """Finish the current turn, then hold even if the next tick skips the knot."""
        ready = self.next_rest_time(utime)
        if ready == utime:
            self.hold(utime)
        elif self._trajectory is not None:
            self._trajectory = self._trajectory.stopped_at(ready)
        return ready

    def install(self, trajectory: AttitudeTrajectory) -> None:
        state = self.predict(trajectory.start)
        angle, _ = quaternion_attitude_delta(
            *state.attitude, *trajectory.state(trajectory.start).attitude
        )
        if angle > _ANGLE_TOL or any(
            abs(rate) > 1e-10 for rate in state.angular_velocity_body
        ):
            raise AttitudeExecutionError(
                "A new trajectory must join the current attitude at rest"
            )
        self._state = state
        self._trajectory = trajectory

    def hold(self, utime: float) -> None:
        state = self.predict(utime)
        if any(abs(rate) > 1e-10 for rate in state.angular_velocity_body):
            raise AttitudeExecutionError("Cannot stop moving attitude instantaneously")
        self._state = state
        self._trajectory = None
