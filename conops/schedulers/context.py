"""The spacecraft model a planner schedules against.

:class:`SchedulingContext` answers the questions a planner asks while it builds
a timeline (how long a slew takes, whether an attitude is allowed in a given
ACS mode, which ground passes are available) using the same models
:class:`~conops.ditl.DITL` executes with. A plan built against it should
therefore execute as planned, step for step.
"""

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import rust_ephem

from ..common import ACSMode, ObsType
from ..common.enums import SlewAlgorithm
from ..config import MissionConfig, bind_ephemeris
from ..config.constraint import attitude_constraint_name_for_scopes
from ..simulation.acs import ACS
from ..simulation.passes import Pass, PassTimes
from ..simulation.roll import optimum_roll
from ..simulation.slew import Slew
from ..targets import PlanEntry

Attitude = tuple[float, float, float]
"""Spacecraft body attitude: RA, Dec and roll in degrees."""


class SchedulingContext:
    """Spacecraft, sky and timing model for offline planning.

    Times are Unix seconds. Plans execute on a fixed step grid starting at
    ``begin``; activities start on grid steps and constraints are evaluated at
    grid steps, exactly as :class:`~conops.ditl.DITL` samples them.

    Args:
        config: Mission configuration, with an ephemeris on its constraint.
        begin: Start of the planning horizon.
        end: End of the planning horizon.
        step_size: Simulation step in seconds. Defaults to the ephemeris step.
            Execute the plan with a DITL using the same step.
    """

    def __init__(
        self,
        config: MissionConfig,
        begin: datetime,
        end: datetime,
        step_size: int | None = None,
    ) -> None:
        ephem = config.constraint.ephem
        if ephem is None:
            raise ValueError("config.constraint.ephem must be set to plan")
        self.config = config
        self.ephem: rust_ephem.Ephemeris = ephem
        bind_ephemeris(config, ephem)
        self.begin = begin
        self.end = end
        self.ustart = begin.timestamp()
        self.uend = end.timestamp()
        self.step_size = int(step_size if step_size is not None else ephem.step_size)
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        timing = config.payload.observation_timing
        self.setup_seconds = float(timing.setup_seconds)
        self.post_collection_seconds = float(timing.post_collection_seconds)
        self._initial_acs: ACS | None = None
        self._initial_attitudes: list[Attitude] = []
        # Quaternion slews do not depend on when they start, so one slew per
        # attitude pair is computed and re-timed for each start.
        self._slew_cache: dict[tuple[Attitude, Attitude, ObsType, int], Slew] = {}
        self._time_independent_slews = (
            config.spacecraft_bus.attitude_control.slew_algorithm
            == SlewAlgorithm.QUATERNION
        )

    # ── Step grid ─────────────────────────────────────────────────────────

    def ceil_step(self, utime: float) -> float:
        """Return the first grid step at or after ``utime``."""
        steps = np.ceil((utime - self.ustart) / self.step_size - 1e-9)
        return self.ustart + max(0.0, float(steps)) * self.step_size

    def floor_step(self, utime: float) -> float:
        """Return the last grid step at or before ``utime``."""
        steps = np.floor((utime - self.ustart) / self.step_size + 1e-9)
        return self.ustart + float(steps) * self.step_size

    def steps(self, begin: float, end: float) -> np.ndarray:
        """Return the grid steps in ``[begin, end)``."""
        first = self.ceil_step(begin)
        if first >= end:
            return np.empty(0)
        return np.arange(first, end, self.step_size, dtype=float)

    # ── Attitudes and slews ───────────────────────────────────────────────

    def initial_attitude(self, utime: float) -> Attitude:
        """Return the attitude a freshly started simulation holds at ``utime``.

        Before its first command the spacecraft idles at an Earth-opposite
        attitude whose roll ACS keeps optimizing, so the attitude depends on
        time. It is replayed step by step with a real ACS.
        """
        if self._initial_acs is None:
            self._initial_acs = ACS(config=self.config)
        index = int(round((self.floor_step(utime) - self.ustart) / self.step_size))
        while len(self._initial_attitudes) <= index:
            step = self.ustart + len(self._initial_attitudes) * self.step_size
            ra, dec, roll, _ = self._initial_acs.pointing(step)
            self._initial_attitudes.append((float(ra), float(dec), float(roll)))
        return self._initial_attitudes[index]

    def instrument_roll(self, target: PlanEntry, utime: float) -> float:
        """Return the power-optimal instrument roll for a target at ``utime``."""
        return optimum_roll(
            target.ra,
            target.dec,
            utime,
            self.ephem,
            self.config.solar_panel,
            self.config.constraint,
            telescope=target.science_telescope(),
        )

    def slew(
        self,
        start: Attitude,
        end: Attitude,
        slewstart: float,
        *,
        obstype: ObsType = ObsType.PPT,
        obsid: int = 0,
    ) -> Slew:
        """Return the slew ACS would fly between two attitudes from ``slewstart``."""
        key = (start, end, obstype, obsid)
        cached = self._slew_cache.get(key) if self._time_independent_slews else None
        if cached is not None:
            return cached.model_copy(
                update={
                    "slewrequest": slewstart,
                    "slewstart": slewstart,
                    "slewend": slewstart + cached.slewtime,
                }
            )
        slew = Slew(config=self.config)
        slew.slewrequest = slewstart
        slew.slewstart = slewstart
        slew.startra, slew.startdec, slew.startroll = start
        slew.endra, slew.enddec, slew.endroll = end
        slew.obstype = obstype
        slew.obsid = obsid
        slew.calc_slewtime()
        if self._time_independent_slews:
            self._slew_cache[key] = slew
            return slew.model_copy()
        return slew

    # ── Constraints ───────────────────────────────────────────────────────

    def attitude_violation(
        self, attitude: Attitude, utime: float, mode: ACSMode
    ) -> str | None:
        """Return the constraint a body attitude violates in ``mode``, if any."""
        ra, dec, roll = attitude
        return attitude_constraint_name_for_scopes(
            self.config.constraint,
            self.config.attitude_constraint_scopes_for_mode(mode),
            ra,
            dec,
            utime,
            target_roll=roll,
            acs_mode=mode,
        )

    def science_violation(
        self, entry: PlanEntry, attitude: Attitude, utime: float
    ) -> str | None:
        """Return the constraint violated while observing ``entry``, if any.

        Mounted instruments are checked with science and body constraints in
        their own frames, as DITL records them.
        """
        if entry.uses_mounted_attitude():
            names = entry.attitude_constraint_names(
                self.config.attitude_constraint_scopes_for_mode(ACSMode.SCIENCE),
                attitude,
                utime,
                ACSMode.SCIENCE,
            )
            return names[0] if names else None
        return self.attitude_violation(attitude, utime, ACSMode.SCIENCE)

    def first_hold_violation(
        self, attitude: Attitude, begin: float, end: float, mode: ACSMode
    ) -> float | None:
        """Return the first step in ``[begin, end)`` where a held attitude violates ``mode``."""
        for utime in self.steps(begin, end):
            if self.attitude_violation(attitude, float(utime), mode) is not None:
                return float(utime)
        return None

    def first_science_violation(
        self, entry: PlanEntry, attitude: Attitude, begin: float, end: float
    ) -> float | None:
        """Return the first step in ``[begin, end)`` where observing ``entry`` is not allowed."""
        for utime in self.steps(begin, end):
            if self.science_violation(entry, attitude, float(utime)) is not None:
                return float(utime)
        return None

    def first_slew_violation(self, slew: Slew, mode: ACSMode) -> float | None:
        """Return the first step during a slew where its path violates ``mode``."""
        for utime in self.steps(slew.slewstart, slew.slewend):
            sample = slew.attitude(float(utime))
            attitude = (float(sample[0]), float(sample[1]), float(sample[2]))
            if self.attitude_violation(attitude, float(utime), mode) is not None:
                return float(utime)
        return None

    def visibility_windows(self, target: PlanEntry) -> list[tuple[float, float]]:
        """Return when a target is clear of its roll-independent constraints.

        Every roll is excluded outside these windows, so a planner can skip
        them before checking a specific roll.
        """
        constraint = self.config.constraint.roll_independent_constraint
        if constraint is None:
            return [(self.ustart, self.uend)]
        result = constraint.evaluate(
            ephemeris=self.ephem,
            target_ra=target.ra,
            target_dec=target.dec,
            target_roll=None,
        )
        return [
            (window.start_time.timestamp(), window.end_time.timestamp())
            for window in result.visibility
        ]

    # ── Ground passes ─────────────────────────────────────────────────────

    def predict_passes(self) -> Sequence[Pass]:
        """Predict the ground passes in the horizon, as DITL will when executing.

        Uses the configuration's random seed, so the same passes are predicted
        when the plan is executed.
        """
        passtimes = PassTimes(config=self.config)
        length = max(1, int(np.ceil((self.uend - self.ustart) / 86400)))
        passtimes.get(self.begin.year, self.begin.timetuple().tm_yday, length)
        return [
            gspass
            for gspass in passtimes.passes
            if self.ustart <= gspass.begin and gspass.end <= self.uend
        ]
