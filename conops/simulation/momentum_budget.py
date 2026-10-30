"""Shared sampled momentum prediction and execution guard.

Tracks inertial total angular momentum; actuator demand is R H - I omega.
Candidate forecasts are independent copies, never updates to the live state;
a committed forecast is written back to the live tracker, which the
simulation shares for its dumps and telemetry. Trajectory intervals bound the
interior actuator demand using a shared sampled torque model. This is not CMG
steering or certification of the true continuous external torque. A thruster
dump that empties the live tracker starts the forecast afresh from what it
holds; no other unloading or unknown disturbance torque is invented here.
"""

from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import floor

import numpy as np
import rust_ephem

from ..common.vector import quaternion_to_rotation_matrix
from ..config import MissionConfig
from .attitude import AttitudeExecutionError, AttitudeState, AttitudeTrajectory
from .momentum import (
    MomentumSample,
    OrbitTables,
    StoredMomentumTracker,
    build_torque_sources,
)
from .slew import Slew


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
    """The capacity enforced on the executed motion.

    ``tracker`` is the simulation's live tracker where it has one: the
    budget integrates into it, so the dumps, the fault checks and the
    telemetry read the momentum enforcement certified. Without one the
    budget builds a tracker from the configuration's torque sources.
    """

    def __init__(
        self,
        config: MissionConfig,
        ephem: rust_ephem.Ephemeris,
        initial: AttitudeState,
        tracker: StoredMomentumTracker | None = None,
    ):
        self.config = config
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
        self.orbit = OrbitTables.from_ephem(ephem)
        self.times = self.orbit.times
        self.positions = self.orbit.positions_eci_km
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
        if tracker is None:
            tracker = StoredMomentumTracker(
                self.inertia,
                settings.initial_momentum_body_n_m_s,
                sources=build_torque_sources(config),
            )
        # Do not pretend sub-sampling motion repairs an under-sampled orbit.
        if (
            tracker.sources
            and np.diff(self.times).max() > settings.max_sample_interval_s + 1e-6
        ):
            raise ValueError(
                "Enforcement with environmental torques requires finer ephemeris samples"
            )
        tracker.max_sample_interval_s = self.step
        self.live = tracker
        """The tracker the simulation reads: every committed forecast is
        written back to it."""
        for source in tracker.sources:
            if not callable(getattr(source, "torque_bound_n_m", None)):
                raise ValueError(
                    f"torque source {source.name} has no bound for capacity enforcement"
                )
        self.max_external_torque = float(
            sum(source.torque_bound_n_m(self.orbit) for source in tracker.sources)
        )
        sample = self._sample(self.live, initial)
        assert sample.actuator_momentum_norm_n_m_s is not None
        peak = sample.actuator_momentum_norm_n_m_s
        self.current = MomentumForecast(
            self.live,
            sample,
            initial.utime,
            peak,
            initial.utime if peak > self.capacity else None,
        )
        self.require(self.current)
        slowest_fraction = settings.slew_rate_fractions[-1]
        slow_limits = acs.model_copy(
            update={
                "max_slew_rate": acs.max_slew_rate * slowest_fraction,
                "max_slew_rate_body": tuple(
                    rate * slowest_fraction for rate in acs.max_slew_rate_body
                )
                if acs.max_slew_rate_body is not None
                else None,
            }
        )
        self.idle_escape_duration = Slew.duration_upper_bound(slow_limits)
        self.idle_escape_momentum = float(np.linalg.norm(self.inertia, 2)) * np.deg2rad(
            fastest * slowest_fraction
        )

    def adopt(self, tracker: StoredMomentumTracker, state: AttitudeState) -> None:
        """Integrate into ``tracker`` from now on, as the simulation's live
        tracker, starting the forecast from what it holds at ``state``."""
        tracker.max_sample_interval_s = self.step
        self.live = tracker
        self.unloaded(state)

    def unloaded(self, state: AttitudeState) -> MomentumForecast:
        """The live tracker's momentum has changed outside the forecast, as
        a thruster dump changes it: the forecast starts again from what it
        holds at ``state``, and any certified path is dropped."""
        tracker = self.live.copy()
        sample = self._sample(tracker, state)
        assert sample.actuator_momentum_norm_n_m_s is not None
        norm = sample.actuator_momentum_norm_n_m_s
        self.current = MomentumForecast(
            tracker,
            sample,
            state.utime,
            norm,
            state.utime if norm > self.capacity else None,
        )
        self.live.assign(tracker)
        return self.current

    def _interpolate(self, table: np.ndarray, utime: float) -> np.ndarray:
        return np.array(
            [np.interp(utime, self.times, table[:, i]) for i in range(3)], dtype=float
        )

    def _sunlit(self, utime: float) -> bool:
        """Whether the Sun shines on the spacecraft at the nearest ephemeris
        sample; it does where the configuration cannot say."""
        constraint = self.config.constraint
        if getattr(constraint, "ephem", None) is None or not callable(
            getattr(constraint, "in_eclipse", None)
        ):
            return True
        index = int(np.clip(np.searchsorted(self.times, utime), 0, len(self.times) - 1))
        if index > 0 and abs(self.times[index - 1] - utime) < abs(
            self.times[index] - utime
        ):
            index -= 1
        return not bool(constraint.in_eclipse(0.0, 0.0, float(self.times[index])))

    def _sample(
        self,
        tracker: StoredMomentumTracker,
        state: AttitudeState,
        torque: np.ndarray | None = None,
    ) -> MomentumSample:
        if not self.times[0] <= state.utime <= self.times[-1]:
            raise ValueError("Momentum prediction is outside ephemeris coverage")
        position = self._interpolate(self.positions, state.utime)
        orbit = self.orbit
        return tracker.update(
            utime=state.utime,
            position_eci_km=position,
            attitude_quaternion_eci_to_body=state.quaternion,
            sun_eci_km=(
                self._interpolate(orbit.sun_positions_eci_km, state.utime) - position
                if orbit.sun_positions_eci_km is not None
                else None
            ),
            sunlit=self._sunlit(state.utime),
            velocity_eci_km_s=(
                self._interpolate(orbit.velocities_eci_km_s, state.utime)
                if orbit.velocities_eci_km_s is not None
                else None
            ),
            position_ecef_km=(
                self._interpolate(orbit.positions_ecef_km, state.utime)
                if orbit.positions_ecef_km is not None
                else None
            ),
            angular_velocity_body_rad_s=np.deg2rad(state.angular_velocity_body),
            torque_eci_n_m=torque,
        )

    @staticmethod
    def _torque(sample: MomentumSample, state: AttitudeState) -> np.ndarray:
        return np.asarray(
            quaternion_to_rotation_matrix(state.quaternion).T
            @ np.asarray(sample.torque_body_n_m)
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
            # This bounds a finite difference, not d(R H)/dt. Split it as
            # (R(t)-R(mid)) H(mid) + R(t) (H(t)-H(mid)). Rotations have unit
            # norm, so midpoint |H| plus the torque bound covers growing H
            # too; no product of changing rotation and changing H is omitted.
            variation_rate_bound = (
                np.linalg.norm(total) * max(np.linalg.norm(rate) for rate in rates)
                + max(np.linalg.norm(torque) for torque in torques)
                + inertia_accel
            )
            if norm + variation_rate_bound * (
                right - left
            ) * duration / 2 <= self.capacity * (1 + 1e-10):
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
        if forecast.tracker is not self.live:
            self.live.assign(forecast.tracker)
