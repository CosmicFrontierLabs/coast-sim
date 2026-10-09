"""Continuous attitude execution independent of operating mode.

Only this executor advances physical attitude. Trajectories are immutable
snapshots; guidance changes cannot rewrite previously executed motion.
"""

from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
from math import inf, isclose, isfinite, nextafter
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ..common.motion import RestToRestMotion
from ..common.quaternion_curve import QuaternionHermite
from ..common.vector import (
    _quat_mul,
    _quaternion_delta,
    attitude_to_quat,
    quat_slerp,
    quat_to_attitude,
)
from ..config import AttitudeControlSystem

if TYPE_CHECKING:
    from .slew import Slew

Attitude = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
_ZERO: Attitude = (0.0, 0.0, 0.0)
# Numerical handoff tolerances, not allowances for physical steps. Compare in
# quaternion space so coordinate singularities cannot consume this budget.
_HANDOFF_ANGLE_TOL_DEG = 1e-8
_HANDOFF_RATE_TOL_DEG_S = 1e-9


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


def _vector(values: np.ndarray) -> Attitude:
    return float(values[0]), float(values[1]), float(values[2])


def _body_rotation(quaternion: Quaternion, angle_body: np.ndarray) -> Quaternion:
    angle = float(np.linalg.norm(angle_body))
    if angle == 0:
        return quaternion
    half_angle = np.deg2rad(angle) / 2
    delta = np.r_[np.cos(half_angle), -np.sin(half_angle) * angle_body / angle]
    q = _quat_mul(delta, np.asarray(quaternion))
    q /= np.linalg.norm(q)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


class _Leg(Protocol):
    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...

    def state(self, utime: float) -> AttitudeState: ...


@dataclass(frozen=True)
class _SpinLeg:
    """Constant-axis coasting or linear braking of body angular velocity."""

    start: float
    end: float
    quaternion: Quaternion
    first_rate: Attitude
    last_rate: Attitude

    def state(self, utime: float) -> AttitudeState:
        duration = self.end - self.start
        elapsed = min(duration, max(0.0, utime - self.start))
        first, last = np.asarray(self.first_rate), np.asarray(self.last_rate)
        acceleration = (last - first) / duration
        q = _body_rotation(
            self.quaternion, elapsed * first + 0.5 * elapsed**2 * acceleration
        )
        rate = (
            self.last_rate
            if elapsed == duration
            else _vector(first + elapsed * acceleration)
        )
        return AttitudeState(utime, q, rate)


@dataclass(frozen=True)
class _TrackingLeg:
    start: float
    end: float
    curve: QuaternionHermite

    def state(self, utime: float) -> AttitudeState:
        fraction = min(1.0, max(0.0, (utime - self.start) / (self.end - self.start)))
        q, rate, _ = self.curve.evaluate(fraction)
        return AttitudeState(
            utime, (float(q[0]), float(q[1]), float(q[2]), float(q[3])), _vector(rate)
        )


def _braking_leg(
    state: AttitudeState, limits: AttitudeControlSystem
) -> _SpinLeg | None:
    rate = np.asarray(state.angular_velocity_body)
    speed = float(np.linalg.norm(rate))
    if speed < 1e-12:
        return None
    axis = _vector(rate / speed)
    acceleration = limits.effective_slew_acceleration(axis)
    maximum = limits.effective_max_slew_rate(axis)
    if not all(
        isfinite(value) and value > 0 for value in (speed, acceleration, maximum)
    ):
        raise AttitudeExecutionError(
            "Motion limits and rates must be finite and positive"
        )
    if speed > maximum * (1 + 1e-10):
        raise AttitudeExecutionError("Initial angular velocity exceeds motion limits")
    duration = speed / acceleration
    end = state.utime + duration
    if end - state.utime < duration:
        end = nextafter(end, inf)
    if not isfinite(end) or end <= state.utime:
        raise AttitudeExecutionError("Braking duration must be finite and positive")
    return _SpinLeg(
        state.utime, end, state.quaternion, state.angular_velocity_body, _ZERO
    )


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
        q0: Quaternion,
        q1: Quaternion,
        limits: AttitudeControlSystem,
        end: float | None = None,
    ) -> "_MotionLeg":
        angle, axis = _quaternion_delta(q0, q1)
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
        # Never speed up the profile when rounding its physical time interval.
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
    """Bounded, rate-continuous motion with a physical stop at its end."""

    legs: tuple[_Leg, ...]
    end: float

    @property
    def start(self) -> float:
        return self.legs[0].start

    @classmethod
    def from_slew(
        cls,
        slew: "Slew",
        limits: AttitudeControlSystem,
        initial_state: AttitudeState | None = None,
    ) -> "AttitudeTrajectory":
        try:
            points = slew.quaternion_waypoints()
        except ValueError as exc:
            raise AttitudeExecutionError(str(exc)) from exc
        legs = []
        start = float(slew.slewstart)
        if initial_state is not None:
            if initial_state.utime != start:
                raise AttitudeExecutionError("Initial state must be at slew start")
            points[0] = initial_state.quaternion
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
        leg = _MotionLeg.between(start, _quaternion(first), _quaternion(last), limits)
        return cls((leg,), leg.end)

    @classmethod
    def turn_from_state(
        cls, state: AttitudeState, target: Attitude, limits: AttitudeControlSystem
    ) -> "AttitudeTrajectory":
        """Start a rest-to-rest guidance turn at the exact executed quaternion."""
        leg = _MotionLeg.between(
            state.utime, state.quaternion, _quaternion(target), limits
        )
        return cls((leg,), leg.end)

    @classmethod
    def tracking(
        cls,
        samples: list[tuple[float, Attitude]],
        limits: AttitudeControlSystem,
        initial_rate: Attitude | None = None,
        *,
        initial_state: AttitudeState | None = None,
    ) -> "AttitudeTrajectory":
        if len(samples) < 2:
            raise AttitudeExecutionError(
                "Tracking requires at least two timed attitudes"
            )
        quaternions = [_quaternion(attitude) for _, attitude in samples]
        if initial_state is not None:
            if initial_state.utime != samples[0][0] or initial_rate is not None:
                raise AttitudeExecutionError(
                    "Initial state must be at tracking start and supplies its rate"
                )
            quaternions[0] = initial_state.quaternion
            initial_rate = initial_state.angular_velocity_body
        intervals, rates = [], []
        for index, ((start, _), (end, _)) in enumerate(zip(samples, samples[1:])):
            if (
                not all(isfinite(time) for time in (start, end, end - start))
                or end <= start
            ):
                raise AttitudeExecutionError(
                    "Tracking times must be finite and increase strictly"
                )
            distance, axis = _quaternion_delta(
                quaternions[index], quaternions[index + 1]
            )
            intervals.append(end - start)
            rates.append(np.asarray(axis) * distance / (end - start))
        node_rates = [rates[0]]
        for index in range(1, len(samples) - 1):
            previous, following = intervals[index - 1], intervals[index]
            node_rates.append(
                (following * rates[index - 1] + previous * rates[index])
                / (previous + following)
            )
        node_rates.append(rates[-1])
        if initial_rate is not None:
            if not all(isfinite(value) for value in initial_rate):
                raise AttitudeExecutionError("Initial angular velocity must be finite")
            node_rates[0] = np.asarray(initial_rate)
        rate_axes = limits.max_slew_rate_body or (limits.max_slew_rate,) * 3
        accel_axes = limits.slew_acceleration_body or (limits.slew_acceleration,) * 3
        if not all(
            isfinite(value) and value > 0 for value in (*rate_axes, *accel_axes)
        ):
            raise AttitudeExecutionError("Motion limits must be finite and positive")
        legs: list[_Leg] = []
        for index, ((start, _), (end, _)) in enumerate(zip(samples, samples[1:])):
            q0, q1 = quaternions[index], quaternions[index + 1]
            w0, w1 = _vector(node_rates[index]), _vector(node_rates[index + 1])
            spin = _vector(rates[index])
            if all(
                isclose(first, speed, rel_tol=0, abs_tol=1e-11)
                and isclose(last, speed, rel_tol=0, abs_tol=1e-11)
                for first, last, speed in zip(w0, w1, spin)
            ):
                if np.linalg.norm(rates[index] / rate_axes) > 1 + 1e-10:
                    raise AttitudeExecutionError("Tracking rate exceeds motion limits")
                # Snap roundoff-equivalent knot rates to the exact secant so
                # this analytic leg has zero acceleration and one fixed axis.
                legs.append(_SpinLeg(start, end, q0, spin, spin))
            else:
                curve = QuaternionHermite(q0, q1, w0, w1, end - start)
                if not curve.within_limits(rate_axes, accel_axes):
                    raise AttitudeExecutionError(
                        "Tracking interval cannot be certified within rate and acceleration limits"
                    )
                legs.append(_TrackingLeg(start, end, curve))
        # Profile exhaustion cannot reset a nonzero terminal tracking rate.
        stop = _braking_leg(legs[-1].state(samples[-1][0]), limits)
        if stop is not None:
            legs.append(stop)
        return cls(tuple(legs), legs[-1].end)

    @classmethod
    def braking(
        cls, state: AttitudeState, limits: AttitudeControlSystem
    ) -> "AttitudeTrajectory | None":
        leg = _braking_leg(state, limits)
        return cls((leg,), leg.end) if leg is not None else None

    @cached_property
    def _starts(self) -> tuple[float, ...]:
        return tuple(leg.start for leg in self.legs)

    def _leg_at(self, utime: float) -> _Leg:
        index = max(0, bisect_right(self._starts, utime) - 1)
        return self.legs[index]

    def state(self, utime: float) -> AttitudeState:
        if utime < self.start:
            raise AttitudeExecutionError("Cannot evaluate motion before its start")
        return self._leg_at(utime).state(utime)

    def next_rest_time(self, utime: float) -> float:
        if np.linalg.norm(self.state(utime).angular_velocity_body) < 1e-10:
            return utime
        for leg in self.legs:
            if (
                leg.end >= utime
                and np.linalg.norm(leg.state(leg.end).angular_velocity_body) < 1e-10
            ):
                return leg.end
        raise AttitudeExecutionError("Trajectory has no physical stopping boundary")


class AttitudeExecutor:
    """The sole owner of executed orientation and angular velocity."""

    def __init__(
        self, utime: float, attitude: Attitude, angular_velocity_body: Attitude = _ZERO
    ) -> None:
        if not isfinite(utime):
            raise AttitudeExecutionError("Execution time must be finite")
        if not all(isfinite(rate) for rate in angular_velocity_body):
            raise AttitudeExecutionError("Initial angular velocity must be finite")
        self._state = AttitudeState(utime, _quaternion(attitude), angular_velocity_body)
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
        elapsed = utime - self._state.utime
        return AttitudeState(
            utime,
            _body_rotation(
                self._state.quaternion,
                elapsed * np.asarray(self._state.angular_velocity_body),
            ),
            self._state.angular_velocity_body,
        )

    def advance(self, utime: float) -> None:
        self._state = self.predict(utime)

    def next_rest_time(self, utime: float) -> float:
        self._check_time(utime)
        if self._trajectory is not None:
            return self._trajectory.next_rest_time(utime)
        return (
            inf
            if any(abs(rate) > 1e-10 for rate in self._state.angular_velocity_body)
            else utime
        )

    def request_stop(self, utime: float, limits: AttitudeControlSystem) -> float:
        """Brake from the current rate, then hold even if a coarse tick skips it."""
        trajectory = AttitudeTrajectory.braking(self.predict(utime), limits)
        if trajectory is None:
            self.hold(utime)
            return utime
        self.install(trajectory)
        return trajectory.end

    def install(self, trajectory: AttitudeTrajectory) -> None:
        state = self.predict(trajectory.start)
        incoming = trajectory.state(trajectory.start)
        angle, _ = _quaternion_delta(state.quaternion, incoming.quaternion)
        rate_difference = (
            np.asarray(state.angular_velocity_body) - incoming.angular_velocity_body
        )
        if (
            not isfinite(angle)
            or not np.all(np.isfinite(rate_difference))
            or angle > _HANDOFF_ANGLE_TOL_DEG
            or np.linalg.norm(rate_difference) > _HANDOFF_RATE_TOL_DEG_S
        ):
            raise AttitudeExecutionError(
                "A new trajectory must join the current attitude and angular velocity"
            )
        self._state = state
        self._trajectory = trajectory

    def hold(self, utime: float) -> None:
        state = self.predict(utime)
        if any(abs(rate) > 1e-10 for rate in state.angular_velocity_body):
            raise AttitudeExecutionError("Cannot stop moving attitude instantaneously")
        self._state = state
        self._trajectory = None
