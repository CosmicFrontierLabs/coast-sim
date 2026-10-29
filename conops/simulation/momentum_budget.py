"""Shared sampled momentum prediction and execution guard.

Tracks inertial total angular momentum; actuator demand is R H - I omega.
Candidate forecasts are independent copies, never updates to the live state.
Trajectory intervals bound the interior actuator demand using a shared sampled
torque model. This is not CMG steering or certification of the true continuous external
torque. No unloading or unknown disturbance torques are invented here.
"""

from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import floor

import numpy as np
import rust_ephem

from ..common.ephemeris import position_vectors
from ..common.vector import quaternion_to_rotation_matrix
from ..config import MissionConfig
from .attitude import AttitudeExecutionError, AttitudeState, AttitudeTrajectory
from .momentum import MomentumSample, StoredMomentumTracker


class MomentumLimitError(AttitudeExecutionError):
    """No physically admitted continuation: the run must not report success."""


@dataclass(frozen=True)
class MomentumForecast:
    tracker: StoredMomentumTracker
    sample: MomentumSample
    utime: float
    peak_n_m_s: float
    first_violation: float | None = None
    uncertified_interval: tuple[float, float] | None = None
    path: "_MomentumPath | None" = field(default=None, compare=False, repr=False)


@dataclass
class _MomentumPath:
    """Integration knots owned by an immutable physical trajectory.

    Partial scheduler ticks read the same torque polynomial, without inserting
    a new integration knot or changing the forecast for subsequent ticks.
    """

    trajectory: AttitudeTrajectory | None
    state_at: Callable[[float], AttitudeState]
    signature: tuple[float, float, bytes]
    nodes: list[MomentumForecast]
    times: list[float]


class MomentumBudget:
    def __init__(
        self, config: MissionConfig, ephem: rust_ephem.Ephemeris, initial: AttitudeState
    ):
        acs = config.spacecraft_bus.attitude_control
        settings = acs.stored_momentum
        assert settings.capacity_n_m_s is not None
        inertia = config.spacecraft_bus.inertia_tensor_body_kg_m2
        if inertia is None:
            raise ValueError("Momentum capacity enforcement requires an inertia tensor")
        self.inertia = np.asarray(inertia, dtype=float).copy()
        self.capacity = settings.capacity_n_m_s
        if not np.isfinite(self.capacity) or self.capacity <= 0:
            raise ValueError("Momentum capacity must be finite and positive")
        fastest = max(acs.max_slew_rate_body or (acs.max_slew_rate,))
        if not np.isfinite(fastest) or fastest <= 0:
            raise ValueError("Momentum enforcement requires finite positive slew rates")
        self.step = min(
            settings.max_sample_interval_s, settings.max_attitude_step_deg / fastest
        )
        self.times = np.array([t.timestamp() for t in ephem.timestamp])
        self.positions = np.asarray(position_vectors(ephem, "gcrs"), dtype=float)
        if (
            len(self.times) < 2
            or self.positions.shape != (len(self.times), 3)
            or not np.isfinite(self.times).all()
            or not np.isfinite(self.positions).all()
            or np.any(np.diff(self.times) <= 0)
        ):
            raise ValueError(
                "Momentum enforcement requires finite, ordered orbit samples"
            )
        # Do not pretend sub-sampling motion repairs an under-sampled orbit.
        if (
            settings.gravity_gradient_enabled
            and np.diff(self.times).max() > settings.max_sample_interval_s + 1e-6
        ):
            raise ValueError(
                "Gravity-gradient enforcement requires finer ephemeris samples"
            )
        tracker = StoredMomentumTracker(
            self.inertia,
            settings.initial_momentum_body_n_m_s,
            max_sample_interval_s=self.step,
            gravity_gradient_enabled=settings.gravity_gradient_enabled,
        )
        sample = self._sample(tracker, initial)
        assert sample.actuator_momentum_norm_n_m_s is not None
        peak = sample.actuator_momentum_norm_n_m_s
        self.current = MomentumForecast(
            tracker,
            sample,
            initial.utime,
            peak,
            initial.utime if peak > self.capacity else None,
        )
        self.require(self.current)

    def _sample(
        self,
        tracker: StoredMomentumTracker,
        state: AttitudeState,
        torque: np.ndarray | None = None,
    ) -> MomentumSample:
        if not self.times[0] <= state.utime <= self.times[-1]:
            raise ValueError("Momentum prediction is outside ephemeris coverage")
        position = np.array(
            [np.interp(state.utime, self.times, self.positions[:, i]) for i in range(3)]
        )
        return tracker.update(
            utime=state.utime,
            position_eci_km=position,
            attitude_quaternion_eci_to_body=state.quaternion,
            angular_velocity_body_rad_s=np.deg2rad(state.angular_velocity_body),
            torque_eci_n_m=torque,
        )

    @staticmethod
    def _torque(sample: MomentumSample, state: AttitudeState) -> np.ndarray:
        return np.asarray(
            quaternion_to_rotation_matrix(state.quaternion).T
            @ np.asarray(sample.gravity_gradient_torque_body_n_m)
        )

    def _fixed_axis_within_limit(
        self,
        state_at: Callable[[float], AttitudeState],
        start: float,
        end: float,
        momentum: np.ndarray,
    ) -> bool:
        """Bound slew/braking/coast demand between acceleration boundaries.

        Body rate is linear on these intervals. Inertial momentum is quadratic
        under the sampled, linearly interpolated torque model. Bound changes
        from each midpoint using rotation angle, torque, and angular acceleration;
        subdivide when the bound is too loose. Failure to certify is rejection.
        """
        duration = end - start
        first_rate = np.deg2rad(state_at(start).angular_velocity_body)
        last_rate = np.deg2rad(state_at(end).angular_velocity_body)
        acceleration = (last_rate - first_rate) / duration
        inertia_accel = float(np.linalg.norm(self.inertia @ acceleration))

        def certify(left: float, right: float, depth: int = 0) -> bool:
            middle = (left + right) / 2
            state = state_at(start + middle * duration)
            total = momentum[0] + middle * momentum[1] + middle**2 * momentum[2]
            rotation = quaternion_to_rotation_matrix(state.quaternion)
            demand = rotation @ total - self.inertia @ np.deg2rad(
                state.angular_velocity_body
            )
            norm = float(np.linalg.norm(demand))
            if norm > self.capacity * (1 + 1e-10):
                return False
            rates = [first_rate + f * duration * acceleration for f in (left, right)]
            torques = [
                (momentum[1] + 2 * f * momentum[2]) / duration for f in (left, right)
            ]
            derivative = (
                np.linalg.norm(total) * max(np.linalg.norm(rate) for rate in rates)
                + max(np.linalg.norm(torque) for torque in torques)
                + inertia_accel
            )
            if norm + derivative * (right - left) * duration / 2 <= self.capacity * (
                1 + 1e-10
            ):
                return True
            return (
                depth < 18
                and certify(left, middle, depth + 1)
                and certify(middle, right, depth + 1)
            )

        return certify(0.0, 1.0)

    def _interval(
        self,
        path: _MomentumPath,
        base: MomentumForecast,
        end: float,
        *,
        torque: np.ndarray | None = None,
    ) -> MomentumForecast:
        previous, state = path.state_at(base.utime), path.state_at(end)
        tracker = base.tracker.copy()
        sample = self._sample(tracker, state, torque)
        assert sample.actuator_momentum_norm_n_m_s is not None
        peak = max(base.peak_n_m_s, sample.actuator_momentum_norm_n_m_s)
        violation, uncertified = base.first_violation, base.uncertified_interval
        if sample.actuator_momentum_norm_n_m_s > self.capacity and violation is None:
            violation = end
        elapsed = end - base.utime
        if elapsed > 0 and violation is None:
            before_rotation = quaternion_to_rotation_matrix(previous.quaternion).T
            first_torque = self._torque(base.sample, previous)
            last_torque = self._torque(sample, state)
            momentum = np.asarray(
                [
                    before_rotation
                    @ np.asarray(base.sample.stored_momentum_body_n_m_s),
                    elapsed * first_torque,
                    0.5 * elapsed * (last_torque - first_torque),
                ]
            )
            interval = (
                path.trajectory.tracking_curve_interval(base.utime, end)
                if path.trajectory is not None
                else None
            )
            if interval is not None:
                curve, first, last = interval
                safe = curve.within_momentum_limit(
                    first, last, self.inertia, momentum, self.capacity
                )
            else:
                safe = self._fixed_axis_within_limit(
                    path.state_at, base.utime, end, momentum
                )
            if not safe:
                violation, uncertified = base.utime, (base.utime, end)
        return MomentumForecast(tracker, sample, end, peak, violation, uncertified)

    def forecast(
        self,
        state_at: Callable[[float], AttitudeState],
        end: float,
        *,
        trajectory: AttitudeTrajectory | None,
        initial: MomentumForecast | None = None,
    ) -> MomentumForecast:
        """Share integration knots across admission, holds and execution ticks."""
        base = self.current if initial is None else initial
        if not np.isfinite(end) or end < base.utime:
            raise ValueError("Momentum forecast time must be finite and monotonic")
        if not self.times[0] <= end <= self.times[-1]:
            raise ValueError("Momentum prediction is outside ephemeris coverage")
        signature = (self.capacity, self.step, self.inertia.tobytes())
        path = base.path
        if (
            path is None
            or path.trajectory is not trajectory
            or path.signature != signature
        ):
            state = state_at(base.utime)
            frozen_state_at = (
                trajectory.state if trajectory is not None else state.coast
            )
            tracker = base.tracker.copy()
            sample = self._sample(tracker, state)
            assert sample.actuator_momentum_norm_n_m_s is not None
            peak = max(
                base.peak_n_m_s if initial is not None else 0.0,
                sample.actuator_momentum_norm_n_m_s,
            )
            violation = base.first_violation if initial is not None else None
            if (
                sample.actuator_momentum_norm_n_m_s > self.capacity
                and violation is None
            ):
                violation = base.utime
            origin = MomentumForecast(
                tracker,
                sample,
                base.utime,
                peak,
                violation,
                base.uncertified_interval if initial is not None else None,
            )
            path = _MomentumPath(
                trajectory, frozen_state_at, signature, [origin], [base.utime]
            )
        boundaries = trajectory.boundaries if trajectory is not None else ()
        origin_time = path.times[0]
        while path.times[-1] < end:
            last = path.times[-1]
            next_time = (
                origin_time + (floor((last - origin_time) / self.step) + 1) * self.step
            )
            if next_time <= last:
                next_time += self.step
            index = bisect_right(boundaries, last)
            if index < len(boundaries):
                next_time = min(next_time, boundaries[index])
            next_time = min(next_time, float(self.times[-1]))
            path.nodes.append(self._interval(path, path.nodes[-1], next_time))
            path.times.append(next_time)
        index = bisect_left(path.times, end)
        if path.times[index] == end:
            result = path.nodes[index]
        else:
            before, after = path.nodes[index - 1], path.nodes[index]
            fraction = (end - before.utime) / (after.utime - before.utime)
            first = self._torque(before.sample, path.state_at(before.utime))
            last_torque = self._torque(after.sample, path.state_at(after.utime))
            result = self._interval(
                path, before, end, torque=first + fraction * (last_torque - first)
            )
        return replace(result, path=path)

    def require(self, forecast: MomentumForecast) -> None:
        if forecast.uncertified_interval is not None:
            start, end = forecast.uncertified_interval
            raise MomentumLimitError(
                f"Actuator momentum could not be certified within {self.capacity:g} N m s "
                f"on [{start:.6f}, {end:.6f}]; sampled peak {forecast.peak_n_m_s:.6f} N m s"
            )
        if forecast.first_violation is not None:
            raise MomentumLimitError(
                f"Actuator momentum exceeds {self.capacity:g} N m s at "
                f"{forecast.first_violation:.6f}; predicted peak {forecast.peak_n_m_s:.6f} N m s"
            )

    def commit(self, forecast: MomentumForecast) -> None:
        self.require(forecast)
        self.current = forecast
