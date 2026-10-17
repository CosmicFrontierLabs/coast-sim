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
from .momentum_budget import MomentumForecast, MomentumLimitError
from .slew import Slew

if TYPE_CHECKING:
    from ..ditl.queue_ditl import QueueDITL
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
        self.end = min(simulation._simulation_end_deadline(), self.budget.times[-1])

    def _safe(
        self,
        trajectory: AttitudeTrajectory,
        end: float,
        mode: ACSMode,
        *,
        start: float | None = None,
        target: "Pointing | None" = None,
    ) -> bool:
        if end > self.end:
            return False
        start = trajectory.start if start is None else start
        times = np.unique(
            np.r_[
                np.linspace(
                    start,
                    end,
                    max(1, ceil((end - start) / self.budget.step)) + 1,
                ),
                [t for t in trajectory.boundaries if start <= t <= end],
            ]
        )
        return all(
            self.sim._attitude_constraint_name_for_attitude(
                *trajectory.state(float(t)).attitude, float(t), mode, target=target
            )
            is None
            for t in times
        )

    def _slew(
        self,
        initial: AttitudeState,
        attitude: Attitude,
        fraction: float,
        *,
        obsid: int = 0,
        science: bool = False,
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
            obstype=ObsType.PPT if science else ObsType.IDLE,
            obsid=obsid,
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
            if not self._safe(
                trajectory, slew.slewend, ACSMode.SLEWING
            ) or not self._safe(trajectory, hold_end, ACSMode.IDLE, start=slew.slewend):
                return None
            # Recovery must not invalidate an already reserved ground pass.
            next_pass = self.acs.passrequests.next_pass(initial.utime)
            if next_pass is not None and any(
                slew.slewend >= deadline
                for _, deadline in next_pass.tracking_profile_slew_deadlines(
                    initial.utime, *destination, for_admission=True
                )
            ):
                return None
            slew._momentum_hold_end = hold_end
            return slew
        return None

    def observation(self, target: "Pointing", now: float) -> MomentumObservation | None:
        visible = target.next_vis(now)
        if visible is False:
            return None
        start = max(self.sim._ppt_slew_execution_time(now), visible)
        start = now + ceil((start - now) / self.sim.step_size) * self.sim.step_size
        if start >= self.end or self.acs.passrequests.current_pass(start) is not None:
            return None
        initial = self.executor.predict(start)
        before = self.executor.forecast_momentum(start)
        if before.first_violation is not None:
            return None
        # Pending motion/wait remains subject to hardware constraints.
        for t in np.linspace(
            now, start, max(1, ceil((start - now) / self.budget.step)) + 1
        ):
            if (
                self.sim._attitude_constraint_name_for_attitude(
                    *self.executor.predict(float(t)).attitude, float(t), ACSMode.IDLE
                )
                is not None
            ):
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
                slew, trajectory = self._slew(
                    initial, attitude, fraction, obsid=target.obsid, science=True
                )
                if slew.slewend > self.end:
                    break
                if any(
                    p.begin <= slew.slewend and p.end >= start
                    for p in self.acs.passrequests.passes
                ):
                    break
                arrival = self.budget.forecast(
                    trajectory.state,
                    slew.slewend,
                    initial=before,
                    trajectory=trajectory,
                )
                if arrival.first_violation is not None:
                    continue  # Only a momentum violation permits a slower cap.
                if not self._safe(trajectory, slew.slewend, ACSMode.SLEWING):
                    break
                deadline, _ = self.sim._next_science_deadline(
                    slew.slewend,
                    current_time=now,
                    target=trial,
                    target_roll=attitude[2],
                )
                requested = min(
                    target.ss_max,
                    target.exptime if target.exptime is not None else target.ss_max,
                )
                # Collection can finish between scheduler ticks. Certify the
                # physical hold through the next tick, without collecting extra
                # data or rounding a visibility/recovery deadline upward.
                requested_end = slew.slewend + timing.total_seconds + requested
                end = (
                    now
                    + min(
                        floor((min(self.end, deadline) - now) / self.sim.step_size),
                        ceil((requested_end - now) / self.sim.step_size),
                    )
                    * self.sim.step_size
                )
                minimum = slew.slewend + timing.total_seconds + target.ss_min
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
                    if not self._safe(
                        trajectory,
                        end,
                        ACSMode.SCIENCE,
                        start=slew.slewend,
                        target=trial,
                    ):
                        end -= self.sim.step_size
                        continue
                    recovery = self.recovery(
                        trajectory.state(end), held, initial.attitude
                    )
                    if recovery is not None:
                        telescope = target.science_telescope()
                        slew.instrument_name = (
                            telescope.name if telescope is not None else None
                        )
                        slew.instrument_roll = float(roll)
                        slew._momentum_hold_end = end
                        collection = min(
                            requested, end - slew.slewend - timing.total_seconds
                        )
                        slew._momentum_collection_end = (
                            slew.slewend + timing.setup_seconds + collection
                        )
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
            for roll in self.acs._idle_safe_roll_candidates(initial.attitude[2]):
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
        next_tick = min(self.end, now + self.sim.step_size)
        forecast = self.executor.forecast_momentum(next_tick)
        hold = AttitudeTrajectory.turn_from_state(
            initial, initial.attitude, self.sim.config.spacecraft_bus.attitude_control
        )
        if forecast.first_violation is None and self._safe(
            hold, next_tick, ACSMode.IDLE
        ):
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
        raise MomentumLimitError(
            "No path- and momentum-validated idle continuation found"
        )
