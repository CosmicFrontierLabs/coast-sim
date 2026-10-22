"""The spacecraft model a scheduler schedules against.

:class:`SchedulingContext` answers the questions a scheduler asks while it builds
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
from ..config import AttitudeControlSystem, MissionConfig
from ..config.constraint import attitude_constraint_name_for_scopes
from ..config.observation_timing import ObservationTiming, timing_for
from ..ditl.ditl_log import DITLLog
from ..simulation.acs import ACS
from ..simulation.attitude import AttitudeExecutionError, AttitudeTrajectory
from ..simulation.coverage import StationCoverage, cut_windows
from ..simulation.passes import Pass, PassTimes
from ..simulation.roll import optimum_roll
from ..simulation.slew import Slew
from ..targets import PlanEntry

Attitude = tuple[float, float, float]
"""Spacecraft body attitude: RA, Dec and roll in degrees."""


class SchedulingContext:
    """Spacecraft, sky and timing model for offline planning.

    Times are Unix seconds. Plans execute on a fixed step grid starting at
    the simulation start; activities start on grid steps and constraints are
    evaluated at grid steps, exactly as :class:`~conops.ditl.DITL` samples them.

    Args:
        config: Mission configuration, with an ephemeris on its constraint.
        begin: Start of the planning horizon.
        end: End of the planning horizon.
        step_size: Simulation step in seconds. Defaults to the ephemeris step.
            Execute the plan with a DITL using the same step.
        simulation_start: Start of the simulation that will execute the plan,
            which anchors the step grid and the initial attitude. Defaults to
            ``begin``; set it when replanning part-way through a simulation.
        simulation_end: End of that simulation. Ground passes are predicted
            over the whole simulation, as DITL predicts them, so that a plan for
            part of it reserves the same passes. Defaults to ``end``.
        passes: The run's predicted ground passes, where the caller has them:
            :meth:`predict_passes` then filters them to the horizon instead of
            predicting, which takes several seconds a day of simulation.
    """

    def __init__(
        self,
        config: MissionConfig,
        begin: datetime,
        end: datetime,
        step_size: int | None = None,
        simulation_start: datetime | None = None,
        simulation_end: datetime | None = None,
        passes: Sequence[Pass] | None = None,
        coverage: StationCoverage | None = None,
    ) -> None:
        self._passes: list[Pass] | None = list(passes) if passes is not None else None
        """The run's predicted ground passes, when the caller has them."""
        self._coverage = coverage
        ephem = config.constraint.ephem
        if ephem is None:
            raise ValueError("config.constraint.ephem must be set to plan")
        # The star tracker and radiator constraints need the ephemeris too. A
        # DITL attaches it when it is created, but a plan can be built before
        # any DITL exists.
        config.spacecraft_bus.star_trackers.set_ephem(ephem)
        config.spacecraft_bus.radiators.set_ephem(ephem)
        self.config = config
        self.ephem: rust_ephem.Ephemeris = ephem
        self.begin = begin
        self.end = end
        self.ustart = begin.timestamp()
        self.uend = end.timestamp()
        self.simulation_start = simulation_start or begin
        self.simulation_end = simulation_end or end
        self.origin = self.simulation_start.timestamp()
        if self.origin > self.ustart:
            raise ValueError("simulation_start must not be after begin")
        self.step_size = int(step_size if step_size is not None else ephem.step_size)
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        bound = ObservationTiming.bound(config.observation_timings())
        self.setup_seconds = float(bound.setup_seconds)
        """The longest setup any observation can have: a bound where the
        observation is not known. :meth:`setup_for` gives an observation's own."""
        self.post_collection_seconds = float(bound.post_collection_seconds)
        """The longest cleanup and handoff any observation can have."""
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
        steps = np.ceil((utime - self.origin) / self.step_size - 1e-9)
        return self.origin + max(0.0, float(steps)) * self.step_size

    def floor_step(self, utime: float) -> float:
        """Return the last grid step at or before ``utime``."""
        steps = np.floor((utime - self.origin) / self.step_size + 1e-9)
        return self.origin + float(steps) * self.step_size

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
        time. It is replayed step by step with a real ACS. The replay's events
        go to a log of their own, which is discarded: the simulation records
        its own, and an ACS without a log prints them.
        """
        if self._initial_acs is None:
            self._initial_acs = ACS(config=self.config, log=DITLLog())
        index = int(round((self.floor_step(utime) - self.origin) / self.step_size))
        while len(self._initial_attitudes) <= index:
            step = self.origin + len(self._initial_attitudes) * self.step_size
            ra, dec, roll, _ = self._initial_acs.pointing(step)
            self._initial_attitudes.append((float(ra), float(dec), float(roll)))
        return self._initial_attitudes[index]

    def optimum_roll(self, target: PlanEntry, utime: float) -> float:
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

    def instrument_roll(self, target: PlanEntry, utime: float) -> float:
        """The instrument roll a target is observed at, at ``utime``: the
        power-optimal roll, or the roll the target asks for (see
        :meth:`Pointing.requested_roll`). An absolute roll beyond the ACS
        limit from the optimum gives the optimum here; :meth:`roll_allowed`
        says whether the target can be observed at that time at all."""
        optimum = self.optimum_roll(target, utime)
        requested = getattr(target, "requested_roll", None)
        if not callable(requested):
            return optimum
        roll = requested(
            optimum,
            self.config.spacecraft_bus.attitude_control.max_roll_from_optimum_deg,
        )
        return optimum if roll is None else float(roll)

    def roll_allowed(self, target: PlanEntry, utime: float) -> bool:
        """Whether the roll ``target`` asks for is within the ACS limit of the
        power-optimal roll at ``utime``; True without a request."""
        requested = getattr(target, "requested_roll", None)
        if not callable(requested):
            return True
        limit = self.config.spacecraft_bus.attitude_control.max_roll_from_optimum_deg
        return requested(self.optimum_roll(target, utime), limit) is not None

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

        Every roll is excluded outside these windows, so a scheduler can skip
        them before checking a specific roll. A mounted instrument's windows
        follow its science line of sight, not the body attitude.
        """
        constraint = (
            target.roll_independent_visibility_constraint()
            if target.constraint is not None
            else self.config.constraint.roll_independent_constraint
        )
        if constraint is None:
            windows = [(self.ustart, self.uend)]
        else:
            result = constraint.evaluate(
                ephemeris=self.ephem,
                target_ra=target.ra,
                target_dec=target.dec,
                target_roll=None,
            )
            windows = [
                (window.start_time.timestamp(), window.end_time.timestamp())
                for window in result.visibility
            ]
        return self.without_repoints(target, windows)

    def without_repoints(
        self, target: PlanEntry, windows: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """``windows`` less the antenna repoints, for a target whose category
        forbids a repoint during an observation; unchanged otherwise."""
        coverage = self.coverage
        if coverage is None or not coverage.repoints:
            return windows
        if self.config.observation_categories.antenna_repoints(int(target.obsid)) != (
            "forbidden"
        ):
            return windows
        return cut_windows(windows, coverage.repoint_cuts(self.ustart, self.uend))

    # ── Ground passes ─────────────────────────────────────────────────────

    @property
    def coverage(self) -> StationCoverage | None:
        """Link coverage for passes without spacecraft tracking, or None
        when passes require tracking activities.

        Built from the configuration unless the caller gave it, which takes
        a while for a long run (see
        :class:`~conops.simulation.coverage.StationCoverage`).
        """
        if self._coverage is None:
            self._coverage = StationCoverage.for_config(
                self.config,
                scheduled_passes=(
                    [(p.begin, p.end, p.station) for p in self._passes]
                    if self._passes is not None
                    else None
                ),
            )
        return self._coverage

    @coverage.setter
    def coverage(self, coverage: StationCoverage | None) -> None:
        self._coverage = coverage

    # ── Observation timing ────────────────────────────────────────────────

    def timing_for(self, obsid: int) -> ObservationTiming:
        """The setup, cleanup and handoff budgets of observation ``obsid``."""
        return timing_for(self.config, obsid)

    def setup_for(self, obsid: int) -> float:
        return float(self.timing_for(obsid).setup_seconds)

    def post_collection_for(self, obsid: int) -> float:
        return float(self.timing_for(obsid).post_collection_seconds)

    def predict_passes(self) -> Sequence[Pass]:
        """Predict the ground passes in the horizon, as DITL will when executing.

        Uses the configuration's random seed, so the same passes are predicted
        when the plan is executed. Predicting covers the whole simulation and
        takes several seconds a day of it, so a caller that has the run's
        passes gives them as ``passes`` and they are filtered to the horizon
        instead.
        """
        if self._passes is not None:
            return [
                gspass
                for gspass in self._passes
                if self.ustart <= gspass.begin and gspass.end <= self.uend
            ]
        coverage = self.coverage
        if coverage is not None:
            return [
                p
                for p in coverage.pass_times.passes
                if self.ustart <= p.begin and p.end <= self.uend
            ]
        passtimes = PassTimes(config=self.config)
        # PassTimes.get counts days from midnight UTC of the start date.
        day_start = self.origin - self.origin % 86400
        span = self.simulation_end.timestamp() - day_start
        length = max(1, int(np.ceil(span / 86400)))
        start = self.simulation_start
        passtimes.get(start.year, start.timetuple().tm_yday, length)
        return [
            gspass
            for gspass in passtimes.passes
            if self.ustart <= gspass.begin and gspass.end <= self.uend
        ]

    def tracking_rest(
        self, gspass: Pass, profile: Sequence[Attitude], acquire: float
    ) -> tuple[float, Attitude] | None:
        """Return when a tracked contact comes to rest, and the attitude it rests at.

        See :func:`tracking_rest`.
        """
        return tracking_rest(
            gspass, profile, acquire, self.config.spacecraft_bus.attitude_control
        )


def tracking_samples(
    gspass: Pass, profile: Sequence[Attitude], acquire: float
) -> list[tuple[float, Attitude]]:
    """Return the timed attitudes ACS tracks after acquiring at ``acquire``.

    Acquisition holds the attitude the profile calls for at that moment, and
    the profile is picked up from the first sample after it, exactly as
    :meth:`~conops.simulation.acs.ACS._start_pass` builds its trajectory.
    """
    held = gspass.attitude_for_profile_at(list(profile), acquire)
    if held is None:
        return []
    samples: list[tuple[float, Attitude]] = [(float(acquire), held)]
    samples.extend(
        (float(utime), (float(ra), float(dec), float(roll)))
        for utime, (ra, dec, roll) in zip(gspass.utime, profile, strict=True)
        if float(utime) > acquire
    )
    return samples


def tracking_rest(
    gspass: Pass,
    profile: Sequence[Attitude],
    acquire: float,
    attitude_control: AttitudeControlSystem,
) -> tuple[float, Attitude] | None:
    """Return when a tracked contact comes to rest, and the attitude it rests at.

    ACS cannot stop a tracking rate instantly: it brakes past the profile's
    last sample, so a contact neither ends at that sample's attitude nor at the
    contact's end time. Tracking starts from rest at ``acquire``, so a late
    acquisition has to turn harder to catch the profile up. Returns None when
    the profile cannot be flown within the motion limits.
    """
    samples = tracking_samples(gspass, profile, acquire)
    if not samples:
        return None
    if len(samples) < 2:
        # A single sample is held, not tracked, so there is nothing to brake.
        return float(gspass.end), samples[-1][1]
    try:
        trajectory = AttitudeTrajectory.tracking(
            samples, attitude_control, initial_rate=(0.0, 0.0, 0.0)
        )
    except AttitudeExecutionError:
        return None
    return float(trajectory.end), trajectory.state(trajectory.end).attitude
