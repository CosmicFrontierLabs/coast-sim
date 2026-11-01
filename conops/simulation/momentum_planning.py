"""Optional bounded observation admission using the executor's momentum model.

The queue retains its usual target ranking. For each selected target, try rolls
and momentum-limited speeds, then shorter legal snapshots. A finite return to
the departure attitude is reserved, not executed if another target is admitted.
This is deliberately not a global schedule/CMG-steering optimizer.
"""

from dataclasses import dataclass
from math import ceil, floor
from typing import TYPE_CHECKING

import numpy as np

from ..common import ACSMode, ObsType
from ..common.vector import quaternion_attitude_delta
from .attitude import Attitude, AttitudeState, AttitudeTrajectory
from .momentum_budget import MomentumForecast
from .roll import optimum_body_roll
from .slew import Slew

if TYPE_CHECKING:
    from ..ditl.queue_ditl import QueueDITL, _ObservationRequest
    from ..targets import Pointing


@dataclass(frozen=True)
class MomentumObservation:
    slew: Slew
    end: float
    recovery: Slew


@dataclass(frozen=True)
class IdleMomentumCertificate:
    """A certified hold followed by an escape at one fixed absolute time.

    The departure is never slid forward on a cache hit. Execution must still
    match the original attitude, integration history and planning inputs.
    """

    initial: AttitudeState
    origin: MomentumForecast
    departure: float
    escape: Slew | None
    context: tuple[object, ...]


class MomentumObservationPlanner:
    # Bound speculative work and how long an unchanged hold can reuse a proof.
    # This is a planning horizon, not a relaxed physical sampling interval.
    _IDLE_CERTIFICATE_SECONDS = 60.0

    def __init__(self, simulation: "QueueDITL"):
        self.sim = simulation
        self.acs = simulation.acs
        executor = self.acs._executor
        assert executor is not None and executor.momentum is not None
        self.executor = executor
        self.budget = executor.momentum
        self.settings = (
            simulation.config.spacecraft_bus.attitude_control.stored_momentum
        )
        self.end = float(
            min(simulation._simulation_end_deadline(), self.budget.times[-1])
        )

    def _first_violation(
        self,
        trajectory: AttitudeTrajectory,
        end: float,
        mode: ACSMode,
        *,
        start: float | None = None,
        target: "Pointing | None" = None,
    ) -> float | None:
        if end > self.end:
            return self.end
        start = trajectory.start if start is None else start
        if mode == ACSMode.SLEWING:
            times = np.unique(
                np.r_[
                    np.linspace(
                        start, end, max(1, ceil((end - start) / self.budget.step)) + 1
                    ),
                    [t for t in trajectory.boundaries if start <= t <= end],
                ]
            )
        else:
            violation = self.sim._check_attitude_hold_window(
                start, end, trajectory.state(start).attitude, mode, target=target
            )
            return violation[0] if violation is not None else None
        for t in times:
            if (
                self.sim._attitude_constraint_name_for_attitude(
                    *trajectory.state(max(start, float(t))).attitude,
                    float(t),
                    mode,
                    target=target,
                )
                is not None
            ):
                return float(t)
        return None

    def _safe(
        self,
        trajectory: AttitudeTrajectory,
        end: float,
        mode: ACSMode,
        *,
        start: float | None = None,
        target: "Pointing | None" = None,
    ) -> bool:
        return (
            self._first_violation(trajectory, end, mode, start=start, target=target)
            is None
        )

    def _slew(
        self,
        initial: AttitudeState,
        attitude: Attitude,
        fraction: float,
    ) -> tuple[Slew, AttitudeTrajectory]:
        slew = Slew(
            config=self.sim.config,
            slewrequest=initial.utime,
            slewstart=initial.utime,
            startra=initial.attitude[0],
            startdec=initial.attitude[1],
            startroll=initial.attitude[2],
            endra=attitude[0],
            enddec=attitude[1],
            endroll=attitude[2],
            rate_fraction=fraction,
            obstype=ObsType.IDLE,
            obsid=0,
        )
        slew.calc_slewtime()
        trajectory = AttitudeTrajectory.from_slew(slew, slew.motion_limits, initial)
        slew._momentum_trajectory = trajectory
        return slew, trajectory

    def recovery(
        self, initial: AttitudeState, momentum: MomentumForecast, destination: Attitude
    ) -> Slew | None:
        for fraction in self.settings.slew_rate_fractions:
            slew, trajectory = self._slew(initial, destination, fraction)
            if slew.slewend > self.end:
                break
            reserve = self.sim._idle_safety.reserve if self.sim._idle_safety else 0.0
            hold_end = min(
                self.end,
                slew.slewend
                + reserve
                + self.sim.config.spacecraft_bus.attitude_control.idle_min_hold_s
                + self.sim.step_size,
            )
            forecast = self.budget.forecast(
                trajectory.state,
                hold_end,
                initial=momentum,
                trajectory=trajectory,
                stop_on_violation=True,
            )
            if forecast.first_violation is not None:
                continue
            if not self.sim._validate_idle_slew(slew, hold_end):
                return None
            slew._momentum_hold_end = hold_end
            return slew
        return None

    def observation(
        self,
        target: "Pointing",
        now: float,
        *,
        ready_time: float | None = None,
        request: "_ObservationRequest | None" = None,
    ) -> MomentumObservation | None:
        if request is None:
            room = self.sim._recorder_room_gb(now, *self.executor.state.attitude)
            request = self.sim._observation_request(
                target, now, room, ready_time=ready_time
            )
        if request is None or request.start >= self.end:
            return None
        start = request.start
        initial = self.executor.predict(start)
        before = self.executor.forecast_momentum(start, stop_on_violation=True)
        if before.first_violation is not None:
            return None
        preferred = self.sim._ppt_optimum_roll(target, start)
        rolls = dict.fromkeys([preferred, *range(0, 360, self.settings.roll_step_deg)])
        timing = self.sim.config.payload.observation_timing
        best: MomentumObservation | None = None
        best_collection = -1.0
        for roll in rolls:
            attitude = self.sim._target_body_attitude(target, roll)
            trial = target.model_copy()
            trial.roll, trial.spacecraft_attitude = roll, attitude
            for fraction in self.settings.slew_rate_fractions:
                slew = self.sim._new_ppt_slew(trial, now)
                slew.rate_fraction = fraction
                self.sim._complete_ppt_slew(
                    slew, trial, now, start, instrument_roll=float(roll)
                )
                trajectory = AttitudeTrajectory.from_slew(
                    slew, slew.motion_limits, initial
                )
                slew._momentum_trajectory = trajectory
                if slew.slewend > self.end:
                    break
                arrival = self.budget.forecast(
                    trajectory.state,
                    slew.slewend,
                    initial=before,
                    trajectory=trajectory,
                    stop_on_violation=True,
                )
                if arrival.first_violation is not None:
                    continue  # Only a momentum violation permits a slower cap.
                window = self.sim._validate_ppt_slew(trial, slew, now, request)
                if window is None:
                    break
                requested, collection_begin = window.requested, window.collection_begin
                teardown = timing.post_collection_seconds
                end = (
                    now
                    + min(
                        floor(
                            (min(self.end, window.deadline) - now) / self.sim.step_size
                        ),
                        ceil((trial.end - now) / self.sim.step_size),
                    )
                    * self.sim.step_size
                )
                minimum = collection_begin + target.ss_min + teardown
                while end >= minimum:
                    held = self.budget.forecast(
                        trajectory.state,
                        end,
                        trajectory=trajectory,
                        initial=arrival,
                        stop_on_violation=True,
                    )
                    if held.first_violation is not None:
                        end = min(
                            end - self.sim.step_size,
                            now
                            + floor((held.first_violation - now) / self.sim.step_size)
                            * self.sim.step_size
                            - self.sim.step_size,
                        )
                        continue
                    recovery = self.recovery(
                        trajectory.state(end), held, initial.attitude
                    )
                    if recovery is not None:
                        slew.at = target
                        slew._momentum_hold_end = end
                        collection = min(requested, end - collection_begin - teardown)
                        slew._momentum_collection_begin = collection_begin
                        slew._momentum_collection_end = collection_begin + collection
                        if collection > best_collection:
                            best = MomentumObservation(slew, end, recovery)
                            best_collection = collection
                        # A full snapshot needs no further roll search.
                        # Otherwise try other rolls before accepting
                        # the longest available shortened snapshot.
                        if collection >= requested:
                            return best
                        break
                    end -= self.sim.step_size
                break  # This slew fits; try a different roll, not a blanket slowdown.
        return best

    def _idle_escape(
        self, initial: AttitudeState, momentum: MomentumForecast
    ) -> Slew | None:
        for ra, dec in self.acs._idle_safe_attitude_candidates(initial.utime):
            preferred = optimum_body_roll(
                ra,
                dec,
                initial.utime,
                self.sim.ephem,
                self.sim.config.solar_panel,
                self.sim.constraint,
                drive_state=self.acs.solar_array_drive_state,
            )
            for roll in self.acs._idle_safe_roll_candidates(preferred):
                # Re-certifying a short zero-motion hold is not proof that a
                # momentum-limited escape remains possible at the next tick.
                if (
                    quaternion_attitude_delta(*initial.attitude, ra, dec, roll)[0]
                    <= 1e-8
                ):
                    continue
                result = self.recovery(initial, momentum, (ra, dec, roll))
                if result is not None:
                    return result
        return None

    def _idle_context(self) -> tuple[object, ...]:
        """Physical inputs are immutable during a run; replacement invalidates.

        Configuration is mutable, so include its values, not just identities.
        Pass conflicts are checked separately against the current pass list.
        """
        return (
            id(self.executor),
            id(self.executor._trajectory),
            id(self.budget),
            id(self.budget.times),
            id(self.budget.positions),
            id(self.sim.ephem),
            id(self.acs.ephem),
            id(self.sim.constraint),
            id(self.sim.constraint.ephem),
            id(self.sim._idle_safety),
            self.budget.capacity,
            self.budget.step,
            self.budget.inertia.tobytes(),
            self.end,
            self.sim.step_size,
            len(self.acs.executed_commands),
            self.sim.config.model_dump_json(
                include={
                    "spacecraft_bus",
                    "payload",
                    "solar_panel",
                    "constraint",
                    "attitude_constraint_scopes",
                    "ground_stations",
                }
            ),
        )

    def _idle_certificate(self, now: float) -> IdleMomentumCertificate | None:
        certificate = self.sim._momentum_idle_certificate
        if certificate is None:
            return None
        # Drop before checking, so an exception or a failed match cannot leave
        # a stale proof available to another planner instance.
        self.sim._momentum_idle_certificate = None
        state = self.executor.state
        if (
            not certificate.initial.utime <= now <= certificate.departure
            or state.utime != now
            or self.acs.command_queue
            or certificate.context != self._idle_context()
            or state != certificate.initial.coast(now)
            or self.budget.current.path is not certificate.origin.path
            or self.budget.current.first_violation is not None
        ):
            return None
        expected = self.budget.forecast(
            self.executor.predict,
            now,
            trajectory=self.executor._trajectory,
            initial=certificate.origin,
            stop_on_violation=True,
        )
        if (
            expected.first_violation is not None
            or self.budget.current.sample != expected.sample
            or not self.budget.current.tracker.same_state(expected.tracker)
            or (
                certificate.escape is not None
                and not self.sim._idle_slew_fits_pass(certificate.escape)
            )
        ):
            return None
        self.sim._momentum_idle_certificate = certificate
        return certificate

    def _remember_idle(self, forecast: MomentumForecast, escape: Slew | None) -> None:
        initial = self.executor.state
        if (
            any(initial.angular_velocity_body)
            or self.executor.next_rest_time(initial.utime) != initial.utime
            or (
                self.executor._trajectory is not None
                and self.executor._trajectory.end > initial.utime
            )
            or self.acs.command_queue
            or self.budget.current.path is not forecast.path
        ):
            return
        self.sim._momentum_idle_certificate = IdleMomentumCertificate(
            initial=initial,
            origin=self.budget.current,
            departure=forecast.utime,
            escape=escape,
            context=self._idle_context(),
        )

    def _extend_idle_certificate(
        self, now: float, hold: AttitudeTrajectory, escape: Slew
    ) -> None:
        """Try the known destination later, without another candidate search.

        Only replace the one-tick proof after certifying both the entire
        longer hold and an escape from its predicted momentum at departure.
        A failed extension leaves the original one-tick proof intact.
        """
        ticks = max(1, floor(self._IDLE_CERTIFICATE_SECONDS / self.sim.step_size))
        departure = min(self.end, now + ticks * self.sim.step_size)
        if departure <= escape.slewstart or not self._safe(
            hold, departure, ACSMode.IDLE
        ):
            return
        forecast = self.executor.forecast_momentum(departure, stop_on_violation=True)
        if forecast.first_violation is not None:
            return
        if departure == self.end:
            self._remember_idle(forecast, None)
            return
        later = self.recovery(
            self.executor.predict(departure),
            forecast,
            (escape.endra, escape.enddec, escape.endroll),
        )
        if later is not None:
            self._remember_idle(forecast, later)

    def idle_recovery(self, now: float) -> Slew | None:
        """Wait only along a certified hold with a feasible future escape."""
        initial = self.executor.state
        certificate = self._idle_certificate(now)
        next_tick = min(self.end, now + self.sim.step_size)
        if certificate is not None and next_tick <= certificate.departure:
            return None
        # At expiry, re-certify this destination before searching the grid.
        # If waiting has become impossible, the original escape still starts
        # exactly now and can be dispatched without changing its trajectory.
        self.sim._momentum_idle_certificate = None
        planner = self.sim._idle_safety
        # With no external torque, a rest hold preserves the momentum norm.
        # A geometry certificate to run-end then proves no escape is needed.
        if (
            not self.settings.gravity_gradient_enabled
            and planner is not None
            and planner.first_violation(initial.attitude, now) - planner.step > self.end
        ):
            return None
        if planner is not None:
            # Reserve the slowest permitted escape, not the nominal full-speed
            # reserve. The envelope includes possible torque growth during both
            # the hold and that maneuver, so this shortcut cannot spend the
            # momentum needed to leave. Near capacity we retain the full search.
            escape_duration = self.budget.idle_escape_duration
            violation = planner.first_violation(initial.attitude, now)
            deadline = min(self.end, violation - escape_duration - 2 * planner.step)
            demand_bound = (
                self.budget.current.sample.stored_momentum_norm_n_m_s
                + self.budget.max_external_torque
                * max(0.0, deadline - now + escape_duration)
                + self.budget.idle_escape_momentum
            )
            if (
                now + self.sim.step_size < deadline
                and demand_bound <= self.budget.capacity
            ):
                return None
        forecast = self.executor.forecast_momentum(next_tick, stop_on_violation=True)
        hold = AttitudeTrajectory.turn_from_state(
            initial, initial.attitude, self.sim.config.spacecraft_bus.attitude_control
        )
        can_wait = forecast.first_violation is None and self._safe(
            hold, next_tick, ACSMode.IDLE
        )
        if can_wait:
            if next_tick == self.end:
                self._remember_idle(forecast, None)
                return None
            escape = None
            if certificate is not None and certificate.escape is not None:
                prior = certificate.escape
                escape = self.recovery(
                    self.executor.predict(next_tick),
                    forecast,
                    (prior.endra, prior.enddec, prior.endroll),
                )
            if escape is None:
                escape = self._idle_escape(self.executor.predict(next_tick), forecast)
            if escape is not None:
                self._remember_idle(forecast, escape)
                self._extend_idle_certificate(now, hold, escape)
                return None
            # No escape is required if the current hold itself remains safe
            # through run-end (including its accumulated external torque).
            if self._safe(hold, self.end, ACSMode.IDLE):
                remaining = self.executor.forecast_momentum(
                    self.end, stop_on_violation=True
                )
                if remaining.first_violation is None:
                    self._remember_idle(remaining, None)
                    return None
        # The full-speed time reserve is not sufficient once momentum caps a
        # recovery's speed. Leave now if waiting would lose a feasible escape.
        result = (
            certificate.escape
            if certificate is not None and certificate.departure == now
            else None
        )
        if result is None:
            result = self._idle_escape(initial, self.budget.current)
        if result is not None:
            return result
        # Geometry/pass conflicts are ordinary idle planning failures. Waiting
        # is still legal before the conservative departure deadline, provided
        # momentum is safe through that deadline (a pass can move us sooner).
        if can_wait and planner is not None:
            deadline = min(self.end, planner.departure_deadline(initial.attitude, now))
            if (
                next_tick < deadline
                and self.executor.forecast_momentum(
                    deadline, stop_on_violation=True
                ).first_violation
                is None
            ):
                return None
        self.acs.report_idle_safety_fault(
            now, "No path- and momentum-validated idle continuation found"
        )
        return None
