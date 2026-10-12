"""Shared sampled momentum prediction and execution guard.

Tracks inertial total angular momentum; actuator demand is R H - I omega.
Candidate forecasts are independent copies, never updates to the live state.
The spherical bound is a planning model, not CMG steering or a continuous-time
certification. No unloading or unknown disturbance torques are invented here.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import ceil

import numpy as np
import rust_ephem

from ..common.ephemeris import position_vectors
from ..config import MissionConfig
from .attitude import AttitudeState
from .momentum import MomentumSample, StoredMomentumTracker


class MomentumLimitError(RuntimeError):
    """No physically admitted continuation: the run must not report success."""


@dataclass(frozen=True)
class MomentumForecast:
    tracker: StoredMomentumTracker
    sample: MomentumSample
    utime: float
    peak_n_m_s: float
    first_violation: float | None = None


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
            inertia,
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
        self, tracker: StoredMomentumTracker, state: AttitudeState
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
        )

    def forecast(
        self,
        state_at: Callable[[float], AttitudeState],
        end: float,
        *,
        initial: MomentumForecast | None = None,
        boundaries: Iterable[float] = (),
    ) -> MomentumForecast:
        base = self.current if initial is None else initial
        start = base.utime
        if not np.isfinite(end) or end < start:
            raise ValueError("Momentum forecast time must be finite and monotonic")
        tracker = base.tracker.copy()
        # Include exact motion/acceleration boundaries, including a complete
        # short slew hidden between scheduler ticks. Check start and end too.
        times = np.unique(
            np.r_[
                np.linspace(start, end, max(1, ceil((end - start) / self.step)) + 1),
                [t for t in boundaries if start <= t <= end],
            ]
        )
        peak, violation = (
            (base.peak_n_m_s, base.first_violation)
            if initial is not None
            else (0.0, None)
        )
        sample = base.sample
        for t in times:
            sample = self._sample(tracker, state_at(float(t)))
            assert sample.actuator_momentum_norm_n_m_s is not None
            norm = sample.actuator_momentum_norm_n_m_s
            peak = max(peak, norm)
            if norm > self.capacity and violation is None:
                violation = float(t)
        return MomentumForecast(tracker, sample, end, peak, violation)

    def require(self, forecast: MomentumForecast) -> None:
        if forecast.first_violation is not None:
            raise MomentumLimitError(
                f"Actuator momentum exceeds {self.capacity:g} N m s at "
                f"{forecast.first_violation:.6f}; predicted peak {forecast.peak_n_m_s:.6f} N m s"
            )

    def commit(self, forecast: MomentumForecast) -> None:
        self.require(forecast)
        self.current = forecast
