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


class MomentumObservationPlanner:
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
        before = self.executor.forecast_momentum(start)
        if before.first_violation is not None:
            return None
        preferred = self.sim._ppt_optimum_roll(target, start)
        rolls = dict.fromkeys([preferred, *range(0, 360, self.settings.roll_step_deg)])
        timing = self.sim._ppt_timing(target)
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
                minimum = collection_begin + target.snapshot_min_seconds + teardown
                while end >= minimum:
                    held = self.budget.forecast(
                        trajectory.state, end, trajectory=trajectory, initial=arrival
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

    def idle_recovery(self, now: float) -> Slew | None:
        """Wait one tick only with a feasible escape from that future state."""
        initial = self.executor.state
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
            escape = self.budget.idle_escape_duration
            violation = planner.first_violation(initial.attitude, now)
            deadline = min(self.end, violation - escape - 2 * planner.step)
            demand_bound = (
                self.budget.current.sample.stored_momentum_norm_n_m_s
                + self.budget.max_external_torque * max(0.0, deadline - now + escape)
                + self.budget.idle_escape_momentum
            )
            if (
                now + self.sim.step_size < deadline
                and demand_bound <= self.budget.capacity
            ):
                return None
        next_tick = min(self.end, now + self.sim.step_size)
        forecast = self.executor.forecast_momentum(next_tick)
        hold = AttitudeTrajectory.turn_from_state(
            initial, initial.attitude, self.sim.config.spacecraft_bus.attitude_control
        )
        can_wait = forecast.first_violation is None and self._safe(
            hold, next_tick, ACSMode.IDLE
        )
        if can_wait:
            if (
                next_tick == self.end
                or self._idle_escape(self.executor.predict(next_tick), forecast)
                is not None
            ):
                return None
            # No escape is required if the current hold itself remains safe
            # through run-end (including its accumulated external torque).
            if self._safe(hold, self.end, ACSMode.IDLE) and (
                self.executor.forecast_momentum(self.end).first_violation is None
            ):
                return None
        # The full-speed time reserve is not sufficient once momentum caps a
        # recovery's speed. Leave now if waiting would lose a feasible escape.
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
                and self.executor.forecast_momentum(deadline).first_violation is None
            ):
                return None
        self.acs.report_idle_safety_fault(
            now, "No path- and momentum-validated idle continuation found"
        )
        return None
