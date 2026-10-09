"""Predictive inertial-hold safety, shared by science admission and idle recovery.

This is a conservative scheduling policy, not a flight attitude controller.
It reserves a worst-case direct rest-to-rest turn before a hold becomes unsafe;
the actual recovery must still pass sampled path and destination-hold checks.
"""

from functools import lru_cache
from math import inf

import numpy as np
import numpy.typing as npt

from ..common import ACSMode
from ..config import AttitudeConstraintScope, AttitudeControlSystem, MissionConfig
from ..config.constraint import in_attitude_constraint_scopes


class IdleSafetyPlanner:
    def __init__(self, config: MissionConfig, end: float) -> None:
        self.config = config
        ephem = config.constraint.ephem
        assert ephem is not None
        self.ephem = ephem
        self.times = [time.timestamp() for time in self.ephem.timestamp]
        self.end = end
        self.step = float(self.ephem.step_size)
        self.scopes = config.attitude_constraint_scopes_for_mode(ACSMode.IDLE)
        acs = config.spacecraft_bus.attitude_control
        limits = (
            *(acs.max_slew_rate_body or (acs.max_slew_rate,)),
            *(acs.slew_acceleration_body or (acs.slew_acceleration,)),
        )
        if not all(np.isfinite(value) and value > 0 for value in limits):
            raise ValueError(
                "Predictive idle safety requires finite positive slew limits"
            )
        # The smallest semiaxis bounds every direction of an ellipsoidal limit,
        # even when the acceleration and rate minima occur on different axes.
        slowest = AttitudeControlSystem(
            max_slew_rate=min(acs.max_slew_rate_body or (acs.max_slew_rate,)),
            slew_acceleration=min(
                acs.slew_acceleration_body or (acs.slew_acceleration,)
            ),
            settle_time=acs.settle_time,
        )
        self.reserve = float(np.ceil(slowest.slew_time(180.0))) + 2 * self.step

    @lru_cache(maxsize=256)
    def _violations(
        self, attitude: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Cache one immutable hold forecast; rebuild the planner for each run."""
        constraint = self.config.constraint
        if not self.scopes:
            return np.array([], dtype=float)
        if self.scopes == [AttitudeConstraintScope.HARDWARE_SAFETY]:
            tree = constraint.hardware_safety_constraint_config
            if tree is None:
                return np.array([], dtype=float)
            result = tree.evaluate(
                self.ephem, attitude[0], attitude[1], target_roll=attitude[2]
            )
            mask = np.asarray(result.constraint_array, dtype=bool)
        else:
            # Preserve mode-gated semantics for non-default IDLE scopes.
            mask = np.array(
                [
                    in_attitude_constraint_scopes(
                        constraint,
                        self.scopes,
                        attitude[0],
                        attitude[1],
                        time,
                        target_roll=attitude[2],
                        acs_mode=ACSMode.IDLE,
                    )
                    for time in self.times
                ]
            )
        return np.asarray(np.asarray(self.times)[mask], dtype=np.float64)

    def first_violation(
        self, attitude: tuple[float, float, float], start: float
    ) -> float:
        """Return the first violating sample, not an exact crossing time."""
        violations = self._violations(attitude)
        index = int(np.searchsorted(violations, start))
        return float(violations[index]) if index < len(violations) else inf

    def departure_deadline(
        self, attitude: tuple[float, float, float], start: float
    ) -> float:
        """Latest conservative departure, or infinity if safe to simulation end."""
        violation = self.first_violation(attitude, start)
        # A crossing can precede its first violating sample by up to one tick,
        # including when that sample is at or just beyond the simulation end.
        # The existing two-tick reserve already covers this sampling uncertainty
        # and the scheduler's reaction time; don't subtract another tick from it.
        return violation - self.reserve if violation - self.step < self.end else inf

    def hold_is_safe(self, attitude: tuple[float, float, float], start: float) -> bool:
        """Require a useful dwell after arrival, not just a safe endpoint."""
        # Require the whole hold to precede the potentially unsafe interval,
        # not merely the first violating sample at its right-hand edge.
        return self.first_violation(attitude, start) - self.step >= min(
            self.end,
            start
            + self.reserve
            + self.config.spacecraft_bus.attitude_control.idle_min_hold_s,
        )
