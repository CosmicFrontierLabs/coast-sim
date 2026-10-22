"""The battery as a consumable: energy drawn from it builds up until the panels put it back.

The spacecraft draws power all the time and its panels generate some whenever
they are lit, so the battery's charge moves continuously. For planning, the
consumable is the *deficit*: the energy below full, in watt-hours. It rises
while the draw exceeds the generation and falls while the panels generate
more than is drawn, and it can never fall below zero: a full battery wastes
what its panels make.

:class:`BatteryLedger` maps the deficit onto
:class:`~conops.schedulers.consumable.ConsumableLedger`. Unlike data on the
recorder, which only comes off during a contact, energy comes back during
science too, so the ledger's additions are ordered in time: each activity is
an :class:`Addition` spanning its slew and its hold, with the net energy it
draws (its mode's draw less what its panels generate at its attitude), and a
segment is run by walking its additions in order. The panels generate
whenever they are lit, whatever the spacecraft is doing, so every span is
counted at the attitude held through it: a slew along its path, an
observation or a charge at its attitude, a contact along its tracking
profile, and idle time at the attitude the spacecraft is left in. Charging
is an addition too: the schedulers put a charging entry, pointing the panels
at the Sun, into every idle sunlit stretch a charge fits as they build the
plan, and the walk counts what those entries put back (see
:meth:`~conops.schedulers.PriorityScheduler.schedule`). The ledger knows the
horizon's sunlit runs, the charging attitude in each and what it generates
step by step, which is what the schedulers place charges from. The segments
start at each sunrise. The limit is the battery's planning level.

The walk is the simulation's own arithmetic, mode by mode and step by step,
at the attitudes the simulation flies. A panel on a finite drive is carried
along as the simulation carries it, tracking the Sun in the modes its
control lists and holding its angle in the rest, so each addition is counted
at the drive angles the plan reaches it with.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import numpy as np

from ..common.enums import ACSMode, ObsType
from ..config import MissionConfig
from ..config.constraint import attitude_constraint_name_for_scopes
from ..config.solar_panel import SolarArrayDriveState
from ..ditl.ditl_log import DITLLog
from ..ditl.plan_validator import PLAN_SCIENCE_OBSTYPES, entry_obstype
from ..simulation.emergency_charging import EmergencyCharging
from ..simulation.slew import Slew
from ..targets import PlanEntry
from .consumable import ConsumableLedger, Removal
from .context import Attitude

MIN_CHARGE_SECONDS = 300.0
"""The shortest charge a plan reserves: a charge costs two slews."""


class Addition(NamedTuple):
    """What one planned activity adds to the deficit, over its span."""

    begin: float
    """When its slew starts."""
    end: float
    """When it ends."""
    amount: float
    """Net energy it draws, in Wh; negative when its panels put energy back."""
    deltas: tuple[float, ...] = ()
    """Its energy step by step, in Wh, one for each grid step that starts
    within it: the deficit cannot fall below zero within it, so a surplus
    before a draw is lost, as it is in the simulation. With a solar-array
    drive these are at the drive angles the horizon starts with; the walk
    counts the addition at the angles it is reached with."""
    profile: int = -1
    """The ledger's record of the mode and attitude held at each of its
    steps, from which the walk counts it at any drive angles; -1 if none."""


Additions = tuple[Addition, ...]
"""A segment's additions, in time order."""

Angles = tuple[float | None, ...]
"""Solar-array drive angles, one per panel; None for an undriven panel."""

Modes = Mapping[str, str] | None
"""The mode each instrument is in during an observation, by name; None
outside observations."""
_Profile = tuple[tuple[ACSMode, Attitude | None, Modes], ...]
"""The mode and attitude held at each grid step of an addition."""


class _Stretch(NamedTuple):
    """A sunlit run's charging terms, by step of the horizon."""

    begin: float
    end: float
    attitude: Attitude | None
    """The charging attitude; None if no valid one was found."""
    charge: np.ndarray
    """Net charge rate, in watts, at each grid step of the run while the
    attitude is allowed; 0 from the first step it is not."""


class BatteryLedger(ConsumableLedger[Additions, Addition]):
    """Battery deficit over a horizon, by the segments its sunrises cut it into.

    Args:
        config: Mission configuration: the battery and its planning level, the
            bus's and payload's draw by mode, the solar panels, and the
            ephemeris on its constraint.
        begin: Start of the horizon (Unix seconds).
        end: End of the horizon.
        start_deficit_wh: The energy below full at ``begin``.
        step_size: Simulation step in seconds: energy is summed step by step,
            as the simulation drains and charges the battery.
        origin: A time the step grid passes through; ``begin`` if None.
        drive_state: The solar-array drive's state at ``begin``; the
            configured initial state if None.

    Energy is counted the way the simulation counts it, in watt-hours per step,
    with the draw of each mode and the eclipse state of each step. Where a
    panel is on a finite drive, the walk carries the drive along as the
    simulation does: it tracks the Sun in the modes its control lists and
    holds its angle in the rest, so what an activity generates depends on
    the angles it is reached with, and the walk counts it at those.
    """

    def __init__(
        self,
        config: MissionConfig,
        begin: float,
        end: float,
        *,
        start_deficit_wh: float,
        step_size: float,
        origin: float | None = None,
        drive_state: SolarArrayDriveState | None = None,
    ) -> None:
        self.config = config
        ephem = config.constraint.ephem
        if ephem is None:
            raise ValueError("config.constraint.ephem must be set to plan")
        self.ephem = ephem
        self.step_size = float(step_size)
        self.origin = float(begin if origin is None else origin)
        self.driven = any(
            panel.single_axis_drive is not None for panel in config.solar_panel.panels
        )
        """Whether a panel is on a finite drive, so that generation depends
        on where the drive is."""
        self.drive_start: Angles = tuple(
            (
                config.solar_panel.initial_drive_state()
                if drive_state is None
                else drive_state
            ).angles_deg
        )
        """The drive angles at the start of the horizon."""
        self._profiles: list[_Profile] = []
        """What each addition holds at each of its steps, by profile index."""
        self._driven_cache: dict[
            tuple[int, tuple[float | None, ...]], tuple[tuple[float, ...], Angles]
        ] = {}
        """An addition's deltas and the drive angles it leaves, by profile and
        the angles it is reached with."""
        self._generation_cache: dict[
            tuple[float, float, float, float, float], float
        ] = {}
        self._addition_cache: dict[tuple[object, ...], Addition | None] = {}
        """Additions by entry identity and the fields they depend on: a
        scheduler re-slews entries as it places others before them, and reads
        a timeline's additions again each time."""
        self._charger = EmergencyCharging(config, log=DITLLog())
        self.recharge_floor = float(config.battery.planning_recharge_deficit_wh)
        """The deficit generation is not counted below: a plan never assumes
        the battery above ``planning_recharge_level``."""
        self.slew_allowance = float(
            Slew.duration_upper_bound(config.spacecraft_bus.attitude_control)
        )
        """Seconds kept clear at each end of idle sunlit time for the slews a
        charge needs."""
        self.grid = self.steps(begin, end)
        """The grid steps of the horizon."""
        self.eclipse = np.array([self.in_eclipse(float(t)) for t in self.grid])
        self.idle_rate = np.array(
            [self._watts(ACSMode.IDLE, bool(e)) for e in self.eclipse]
        )
        self.slew_rate = np.array(
            [self._watts(ACSMode.SLEWING, bool(e)) for e in self.eclipse]
        )
        self.stretches = [
            self._stretch(run_begin, run_end)
            for run_begin, run_end in self.sunlit_runs(begin, end)
        ]
        """The horizon's sunlit runs with their charging terms."""
        super().__init__(
            begin,
            end,
            [Removal(s.begin, s.end, 0.0) for s in self.stretches],
            limit=config.battery.planning_deficit_wh,
            start=start_deficit_wh,
            accrual=lambda span_begin, span_end: 0.0,
        )

    # ── The step grid ─────────────────────────────────────────────────────

    def _first_step(self, begin: float) -> float:
        """The first grid step at or after ``begin``."""
        return float(
            self.origin
            + self.step_size * np.ceil((begin - self.origin) / self.step_size - 1e-9)
        )

    def steps(self, begin: float, end: float) -> np.ndarray:
        """The grid steps in ``[begin, end)``."""
        first = self._first_step(begin)
        if first >= end:
            return np.empty(0)
        return np.arange(first, end, self.step_size, dtype=float)

    def index(self, utime: float) -> int:
        """The grid step at or before ``utime``, as an index into the horizon."""
        if len(self.grid) == 0:
            return 0
        k = int(np.floor((utime - self.grid[0]) / self.step_size + 1e-9))
        return min(max(k, 0), len(self.grid) - 1)

    def in_eclipse(self, utime: float) -> bool:
        return self.config.constraint.in_eclipse(ra=0.0, dec=0.0, time=utime)

    def sunlit_runs(self, begin: float, end: float) -> list[tuple[float, float]]:
        """Maximal runs of sunlit steps in ``[begin, end)``, as time spans."""
        runs: list[tuple[float, float]] = []
        for utime in self.steps(begin, end):
            if self.in_eclipse(float(utime)):
                continue
            if runs and runs[-1][1] == utime:
                runs[-1] = (runs[-1][0], float(utime) + self.step_size)
            else:
                runs.append((float(utime), float(utime) + self.step_size))
        return [(b, min(e, end)) for b, e in runs]

    # ── Power ─────────────────────────────────────────────────────────────

    def _watts(self, mode: ACSMode, in_eclipse: bool, modes: Modes = None) -> float:
        """The bus's and payload's draw in ``mode``, in watts, the
        instruments in ``modes`` during an observation."""
        return self.config.spacecraft_bus.power(
            mode.value, in_eclipse=in_eclipse
        ) + self.config.payload.power(
            mode.value, in_eclipse=in_eclipse, instrument_modes=modes
        )

    def _integrate(
        self, begin: float, end: float, watts_at: Callable[[float], float]
    ) -> float:
        """Watt-hours of a step-wise rate over ``[begin, end)``.

        The rate is sampled at the grid steps in the span, or at ``begin`` if
        the span holds none, and scaled to the span's length.
        """
        if end <= begin:
            return 0.0
        samples = self.steps(begin, end)
        if len(samples) == 0:
            samples = np.array([begin])
        rates = [watts_at(float(utime)) for utime in samples]
        return float(np.mean(rates)) * (end - begin) / 3600.0

    def draw(self, mode: ACSMode, begin: float, end: float) -> float:
        """Energy the bus and payload draw in ``mode`` over a span, in Wh."""
        return self._integrate(
            begin, end, lambda utime: self._watts(mode, self.in_eclipse(utime))
        )

    def generation_watts(
        self, attitude: Attitude, utime: float, angles: Angles | None = None
    ) -> float:
        """What the panels generate holding ``attitude`` at ``utime``, in
        watts, with the drive at ``angles``, or where the horizon starts."""
        ra, dec, roll = attitude
        return float(
            self.config.solar_panel.power(
                utime,
                ra,
                dec,
                self.ephem,
                roll=roll,
                drive_state=self._drive_state(
                    self.drive_start if angles is None else angles
                ),
            )
        )

    def _drive_state(
        self, angles: Angles, updated_at: float | None = None
    ) -> SolarArrayDriveState | None:
        """A drive state at ``angles``, or None where nothing is driven."""
        if not self.driven:
            return None
        return SolarArrayDriveState(angles_deg=angles, updated_at_s=updated_at)

    def generation(self, attitude: Attitude, begin: float, end: float) -> float:
        """Energy the panels generate holding ``attitude`` over a span, in Wh,
        with the drive where the horizon starts."""
        ra, dec, roll = attitude
        key = (round(ra, 6), round(dec, 6), round(roll, 6), begin, end)
        cached = self._generation_cache.get(key)
        if cached is not None:
            return cached
        energy = self._integrate(
            begin, end, lambda utime: self.generation_watts(attitude, utime)
        )
        self._generation_cache[key] = energy
        return energy

    def _step_generation(
        self,
        utime: float,
        mode: ACSMode,
        attitude: Attitude | None,
        angles: Angles,
    ) -> tuple[float, Angles]:
        """What the panels generate at one step, in watts, and where the
        drive is after it: advanced over the step as the simulation advances
        it, tracking the Sun in ``mode`` if its control says so and holding
        otherwise; it holds through a step with no attitude known."""
        if attitude is None or not self.driven:
            watts = 0.0 if attitude is None else self.generation_watts(attitude, utime)
            return watts, angles
        ra, dec, roll = attitude
        _, watts, state = self.config.solar_panel.evaluate_executed_attitude(
            utime,
            ra,
            dec,
            self.ephem,
            SolarArrayDriveState(
                angles_deg=angles, updated_at_s=utime - self.step_size
            ),
            roll=roll,
            acs_mode=mode,
        )
        return float(watts), tuple(state.angles_deg)

    # ── Charging ──────────────────────────────────────────────────────────

    def charging_attitude(
        self, utime: float, angles: Angles | None = None
    ) -> Attitude | None:
        """A constraint-valid attitude that points the panels at the Sun at
        ``utime``, with the drive at ``angles``, or where the horizon starts:
        the roll ACS's charging guidance settles on depends on where the
        drive is."""
        pointing = self._charger.create_charging_pointing(
            utime,
            self.ephem,
            drive_state=self._drive_state(
                self.drive_start if angles is None else angles
            ),
        )
        if pointing is None:
            return None
        return (float(pointing.ra), float(pointing.dec), float(pointing.roll))

    def charging_allowed(self, attitude: Attitude, utime: float) -> bool:
        """Whether ``attitude`` clears the charging mode's constraints at ``utime``."""
        ra, dec, roll = attitude
        return (
            attitude_constraint_name_for_scopes(
                self.config.constraint,
                self.config.attitude_constraint_scopes_for_mode(ACSMode.CHARGING),
                ra,
                dec,
                utime,
                target_roll=roll,
                acs_mode=ACSMode.CHARGING,
            )
            is None
        )

    def _stretch(self, begin: float, end: float) -> _Stretch:
        """A sunlit run's charging attitude and net charge rate by step."""
        steps = self.steps(begin, end)
        charge = np.zeros(len(steps))
        attitude = self.charging_attitude(begin)
        if attitude is not None:
            # With a drive, as a charge from the start of the run would find
            # it: tracking the Sun in charging mode if its control says so.
            angles = self.drive_start
            for k, utime in enumerate(steps):
                if not self.charging_allowed(attitude, float(utime)):
                    break
                watts, angles = self._step_generation(
                    float(utime), ACSMode.CHARGING, attitude, angles
                )
                charge[k] = max(0.0, watts - self._watts(ACSMode.CHARGING, False))
        if not np.any(charge > 0.0):
            # Nothing to gain by pointing at the Sun: no panels, or none that
            # beat the draw. The run is idle time like any other.
            attitude = None
        return _Stretch(begin, end, attitude, charge)

    def stretch_at(self, utime: float) -> _Stretch | None:
        """The sunlit run ``utime`` falls in, if any."""
        for stretch in self.stretches:
            if stretch.begin <= utime < stretch.end:
                return stretch
        return None

    def net_charge(self, attitude: Attitude, begin: float, end: float) -> float:
        """The net energy charging at ``attitude`` over a span puts back, in Wh."""
        if end <= begin:
            return 0.0
        return max(
            0.0,
            self.generation(attitude, begin, end)
            - self.draw(ACSMode.CHARGING, begin, end),
        )

    # ── What a plan adds ──────────────────────────────────────────────────

    def addition(
        self,
        entry: PlanEntry,
        *,
        after: float | None = None,
        rest: float | None = None,
        path: Callable[[float], Attitude] | None = None,
        path_start: Attitude | None = None,
        track: Callable[[float], tuple[float | None, float | None, float]]
        | None = None,
    ) -> Addition | None:
        """What a plan entry adds to the deficit over its span, or None if nothing.

        Every step draws its mode's power less what the panels generate at the
        attitude held then: along the slew's ``path`` from ``path_start``
        (none counted without one) at ``SLEWING`` power, or ``PASS`` power
        into a contact, as ACS reports the slew; then the entry's own
        attitude in ``SCIENCE`` or ``CHARGING`` mode, or the contact's
        ``track`` in ``PASS`` mode, and ``SLEWING`` power at the contact's
        last attitude while it brakes to ``rest`` after its end. Only the
        part from ``after`` counts, if given.
        """
        obstype = entry_obstype(entry)
        key = (
            id(entry),
            float(entry.begin),
            float(entry.slewtime or 0),
            float(entry.end),
            float(entry.ra),
            float(entry.dec),
            float(entry.roll),
            after,
            rest,
            path_start,
            track is not None,
            tuple(sorted(entry.effective_instrument_modes().items())),
        )
        if key in self._addition_cache:
            return self._addition_cache[key]
        begin = float(entry.begin)
        arrival = begin + float(entry.slewtime or 0)
        end = float(entry.end)
        braking = max(end, rest) if rest is not None else end
        if after is not None:
            begin = max(begin, after)
        if braking <= begin:
            self._addition_cache[key] = None
            return None
        if obstype in PLAN_SCIENCE_OBSTYPES:
            mode = ACSMode.SCIENCE
        elif obstype == ObsType.GSP:
            mode = ACSMode.PASS
        elif obstype == ObsType.CHARGE:
            mode = ACSMode.CHARGING
        elif obstype == ObsType.ACTIVITY:
            mode = ACSMode.IDLE
        else:
            self._addition_cache[key] = None
            return None
        own: Attitude = (
            entry.target_body_attitude()
            if mode is ACSMode.SCIENCE
            else (float(entry.ra), float(entry.dec), float(entry.roll))
        )

        def attitude_at(utime: float) -> Attitude | None:
            if utime < arrival:
                if path is None:
                    return None
                ra, dec, roll = path(utime)
                return (float(ra), float(dec), float(roll))
            if mode is ACSMode.PASS and track is not None:
                track_ra, track_dec, track_roll = track(utime)
                if track_ra is not None and track_dec is not None:
                    return (float(track_ra), float(track_dec), float(track_roll))
            return own

        # The instruments an observation, or an activity, uses while it holds.
        modes = (
            entry.effective_instrument_modes()
            if mode is ACSMode.SCIENCE or obstype == ObsType.ACTIVITY
            else None
        )

        def held_at(utime: float) -> tuple[ACSMode, Attitude | None, Modes]:
            if utime < arrival:
                # ACS reports the slew into a contact as PASS, and the slew
                # into anything else as SLEWING.
                step_mode = ACSMode.PASS if mode is ACSMode.PASS else ACSMode.SLEWING
            elif utime >= end:
                step_mode = ACSMode.SLEWING
            else:
                step_mode = mode
            return (
                step_mode,
                attitude_at(utime),
                modes if arrival <= utime < end else None,
            )

        addition = self._addition_over(begin, braking, held_at)
        self._addition_cache[key] = addition
        return addition

    def visit_addition(
        self,
        steps: Sequence[PlanEntry],
        slews: Sequence[Slew | None],
        path_start: Attitude | None,
    ) -> Addition | None:
        """What a sequence visit adds over its span: each step's slew at
        ``SLEWING`` power along its path, from ``path_start`` for the first,
        then its hold at ``SCIENCE`` power at its own attitude."""
        if not steps:
            return None
        begin = float(steps[0].begin)
        end = float(steps[-1].end)
        if end <= begin:
            return None
        key = (
            "visit",
            tuple(
                (
                    float(step.begin),
                    float(step.slewtime or 0),
                    float(step.end),
                    float(step.ra),
                    float(step.dec),
                    float(step.roll),
                    tuple(sorted(step.effective_instrument_modes().items())),
                )
                for step in steps
            ),
            path_start,
        )
        if key in self._addition_cache:
            return self._addition_cache[key]
        spans = []
        for step, slew in zip(steps, slews):
            arrival = float(step.begin) + float(step.slewtime or 0)
            spans.append(
                (
                    float(step.begin),
                    arrival,
                    float(step.end),
                    slew.attitude if slew is not None else None,
                    step.target_body_attitude(),
                    step.effective_instrument_modes(),
                )
            )

        def held_at(utime: float) -> tuple[ACSMode, Attitude | None, Modes]:
            for span_begin, arrival, span_end, path, own, modes in spans:
                if span_begin <= utime < arrival:
                    if path is None:
                        return ACSMode.SLEWING, None, None
                    ra, dec, roll = path(utime)
                    return ACSMode.SLEWING, (float(ra), float(dec), float(roll)), None
                if arrival <= utime < span_end:
                    return ACSMode.SCIENCE, own, modes
            return ACSMode.SLEWING, None, None

        addition = self._addition_over(begin, end, held_at)
        self._addition_cache[key] = addition
        return addition

    def hold_addition(
        self, attitude: Attitude, mode: ACSMode, begin: float, end: float
    ) -> Addition | None:
        """What holding ``attitude`` in ``mode`` over a span adds to the deficit:
        the mode's draw less what the panels generate there, step by step."""
        if end <= begin:
            return None
        ra, dec, roll = attitude
        key = ("hold", round(ra, 6), round(dec, 6), round(roll, 6), mode, begin, end)
        if key in self._addition_cache:
            return self._addition_cache[key]
        held = (float(ra), float(dec), float(roll))
        addition = self._addition_over(begin, end, lambda utime: (mode, held, None))
        self._addition_cache[key] = addition
        return addition

    def _addition_over(
        self,
        begin: float,
        end: float,
        held_at: Callable[[float], tuple[ACSMode, Attitude | None, Modes]],
    ) -> Addition:
        """An addition over ``[begin, end)`` holding what ``held_at`` says at
        each grid step: the mode's draw less what the panels generate at the
        attitude, with the drive where the horizon starts.

        As the simulation does, each grid step that starts in the span counts
        for the whole step at the rate then, and a span no step starts in
        counts for nothing.
        """
        profile = tuple(held_at(float(step)) for step in self.steps(begin, end))
        self._profiles.append(profile)
        deltas = self._deltas(begin, profile, self.drive_start)[0]
        return Addition(begin, end, sum(deltas), deltas, len(self._profiles) - 1)

    def _deltas(
        self, begin: float, profile: _Profile, angles: Angles
    ) -> tuple[tuple[float, ...], Angles]:
        """An addition's energy step by step from ``angles``, in Wh, and the
        drive angles it leaves."""
        deltas = []
        utime = self._first_step(begin)
        for mode, attitude, modes in profile:
            watts, angles = self._step_generation(utime, mode, attitude, angles)
            draw = self._watts(mode, self.in_eclipse(utime), modes) - watts
            deltas.append(draw * self.step_size / 3600.0)
            utime += self.step_size
        return tuple(deltas), angles

    def driven_deltas(
        self, addition: Addition, angles: Angles
    ) -> tuple[tuple[float, ...], Angles]:
        """An addition's deltas as the walk counts them, reached with the drive
        at ``angles``, and the angles it leaves: its own deltas where nothing
        is driven or it has no profile."""
        if not self.driven or addition.profile < 0:
            return addition.deltas, angles
        key = (
            addition.profile,
            tuple(None if a is None else round(a, 3) for a in angles),
        )
        cached = self._driven_cache.get(key)
        if cached is None:
            cached = self._deltas(
                addition.begin, self._profiles[addition.profile], angles
            )
            self._driven_cache[key] = cached
        return cached

    def additions(self, entries: Sequence[PlanEntry]) -> tuple[Additions, ...]:
        """The plan's additions by segment, in time order."""
        planned: list[list[Addition]] = [[] for _ in range(self.segments)]
        for entry in entries:
            addition = self.addition(entry)
            if addition is not None:
                planned[self.segment(addition.begin)].append(addition)
        return tuple(tuple(sorted(group)) for group in planned)

    def add(
        self, planned: Sequence[Additions], segment: int, extra: Addition
    ) -> tuple[Additions, ...]:
        """The plan's additions with ``extra`` in ``segment``, in time order."""
        group = list(planned[segment])
        group.insert(bisect_left(group, extra), extra)
        added = list(planned)
        added[segment] = tuple(group)
        return tuple(added)

    # ── Running the segments ──────────────────────────────────────────────

    def _peaks_and_levels(
        self, planned: Sequence[Additions]
    ) -> tuple[list[float], list[float]]:
        level = self.start
        angles = self.drive_start
        cursor = self.bounds[0]
        peaks = []
        levels = []
        for k in range(self.segments):
            peak = level
            for addition in planned[k]:
                if addition.begin > cursor:
                    idle_peak, level = self._idle(level, cursor, addition.begin)
                    peak = max(peak, idle_peak)
                deltas, angles = self.driven_deltas(addition, angles)
                for delta in deltas:
                    level = self._after(level, delta)
                    peak = max(peak, level)
                cursor = max(cursor, addition.end)
            if self.bounds[k + 1] > cursor:
                idle_peak, level = self._idle(level, cursor, self.bounds[k + 1])
                peak = max(peak, idle_peak)
                cursor = self.bounds[k + 1]
            peaks.append(peak)
            levels.append(level)
        return peaks, levels

    def first_over_limit(self, planned: Sequence[Additions]) -> float | None:
        """The first time the deficit exceeds the limit, or None if it never does."""
        if self.limit is None:
            return None
        level = self.start
        angles = self.drive_start
        cursor = self.bounds[0]
        for group in planned:
            for addition in group:
                if addition.begin > cursor:
                    for step in self.steps(cursor, addition.begin):
                        stop = min(float(step) + self.step_size, addition.begin)
                        _, level = self._idle(level, float(step), stop)
                        if level > self.limit + 1e-9:
                            return float(step)
                deltas, angles = self.driven_deltas(addition, angles)
                for step, delta in zip(
                    self.steps(addition.begin, addition.end), deltas
                ):
                    level = self._after(level, delta)
                    if level > self.limit + 1e-9:
                        return float(step)
                cursor = max(cursor, addition.end)
        for step in self.steps(cursor, self.bounds[-1]):
            stop = min(float(step) + self.step_size, self.bounds[-1])
            _, level = self._idle(level, float(step), stop)
            if level > self.limit + 1e-9:
                return float(step)
        return None

    def walk_deltas(
        self, planned: Sequence[Additions]
    ) -> dict[Addition, tuple[float, ...]]:
        """Each addition's deltas as the walk counts them, reached with the
        drive where the plan's additions before it leave it."""
        angles = self.drive_start
        counted: dict[Addition, tuple[float, ...]] = {}
        for group in planned:
            for addition in group:
                counted[addition], angles = self.driven_deltas(addition, angles)
        return counted

    def _after(self, level: float, delta: float) -> float:
        """The deficit after a step adds ``delta``: a draw adds in full, and
        generation takes the deficit no lower than the recharge floor, or
        than where it already is below the floor, and never below zero."""
        if delta < 0.0:
            return max(level + delta, min(level, self.recharge_floor), 0.0)
        return level + delta

    def _idle(self, level: float, begin: float, end: float) -> tuple[float, float]:
        """Walk idle time from ``level``: the peak reached and the level at its end.

        Idle time with no attitude known draws the idle power and generates
        nothing. The schedulers give every idle span the attitude the
        spacecraft holds through it, as a hold addition
        (:meth:`hold_addition`), so this is the fallback for a plan read on
        its own.
        """
        peak = level
        for step in self.steps(begin, end):
            utime = float(step)
            seconds = min(utime + self.step_size, end) - utime
            level = level + float(self.idle_rate[self.index(utime)]) * seconds / 3600.0
            peak = max(peak, level)
        return peak, level

    def charges_in(
        self, begin: float, end: float
    ) -> list[tuple[float, float, _Stretch]]:
        """The sunlit spans of idle time ``[begin, end)`` long enough to charge in."""
        spans = []
        for stretch in self.stretches:
            if stretch.attitude is None:
                continue
            span_begin = max(begin, stretch.begin)
            span_end = min(end, stretch.end)
            if span_end - span_begin >= MIN_CHARGE_SECONDS + 2 * self.slew_allowance:
                spans.append((span_begin, span_end, stretch))
        return spans

    def time_to_fit(
        self,
        planned: Sequence[Additions],
        addition: Addition,
        until: float,
        excess: float,
    ) -> float | None:
        """When an addition refused at its time could start instead, charging first.

        ``excess`` is how far over the limit the plan went with the addition
        where it is. The idle time before the addition, from the end of the
        activity before it, is in sunlight for some of its length; starting
        the addition later leaves more of that for a charge. Returns the
        earliest start at which a charge in that time, at the sunlit run's net
        charge rate after a slew allowance, would have put back the excess, or
        None if no charge before ``until`` would, or the deficit at the start
        is too small for charging to take off the excess. The start can be at
        or after ``until`` when the time gains less than the excess, which
        the walk may overstate: a start late in the gap may still fit, and is
        worth trying. It is a hint: whether a charge then fits is for the
        scheduler to find out.
        """
        if self.limit is None or excess <= 1e-9:
            return None
        at_start = self.level_at(planned, addition.begin)
        if at_start - excess < self.recharge_floor:
            # Charging cannot take the deficit low enough for the addition.
            return None
        # The idle time before the addition: a hold spanning its start, or
        # the time since the last activity before it.
        gap_begin = self.bounds[0]
        for group in planned:
            for other in group:
                if other.begin >= addition.begin:
                    break
                gap_begin = other.begin if other.end > addition.begin else other.end
        for stretch in self.stretches:
            span_begin = max(gap_begin, stretch.begin)
            if stretch.end <= addition.begin or span_begin >= until:
                continue
            offset = self.index(stretch.begin)
            gained = 0.0
            first = span_begin + self.slew_allowance
            for step in self.steps(first, min(stretch.end, until)):
                t = float(step)
                net = float(stretch.charge[self.index(t) - offset])
                if net <= 0.0:
                    break
                gained += net * self.step_size / 3600.0
                if gained >= excess:
                    start = t + self.step_size + self.slew_allowance
                    return max(
                        start, span_begin + MIN_CHARGE_SECONDS + 2 * self.slew_allowance
                    )
        return None

    def level_at(self, planned: Sequence[Additions], utime: float) -> float:
        """The deficit at ``utime``, walking the plan's additions up to it."""
        return self._at(planned, utime)[0]

    def drive_state_at(
        self, planned: Sequence[Additions], utime: float
    ) -> SolarArrayDriveState | None:
        """Where the drive is at ``utime``, walking the plan's additions up to
        it; None where nothing is driven."""
        return self._drive_state(self._at(planned, utime)[1])

    def _at(self, planned: Sequence[Additions], utime: float) -> tuple[float, Angles]:
        """The deficit and the drive angles at ``utime``."""
        level = self.start
        angles = self.drive_start
        cursor = self.bounds[0]
        for group in planned:
            for addition in group:
                if addition.begin >= utime:
                    break
                if addition.begin > cursor:
                    _, level = self._idle(level, cursor, addition.begin)
                deltas, left = self.driven_deltas(addition, angles)
                if addition.end <= utime:
                    angles = left
                else:
                    # Only the steps before ``utime``: the drive is where
                    # they leave it, step by step.
                    profile = (
                        self._profiles[addition.profile]
                        if addition.profile >= 0
                        else ()
                    )
                    step = self._first_step(addition.begin)
                    for k, (mode, attitude, _) in enumerate(profile):
                        if step >= utime:
                            break
                        _, angles = self._step_generation(step, mode, attitude, angles)
                        step += self.step_size
                for step, delta in zip(
                    self.steps(addition.begin, addition.end), deltas
                ):
                    if float(step) >= utime:
                        break
                    level = self._after(level, delta)
                cursor = max(cursor, min(addition.end, utime))
        if utime > cursor:
            _, level = self._idle(level, cursor, utime)
        return level, angles

    # ── In the battery's terms ────────────────────────────────────────────

    @property
    def limit_wh(self) -> float | None:
        """The planning limit as a deficit in Wh, or None."""
        return self.limit

    @property
    def watthour(self) -> float:
        return float(self.config.battery.watthour)

    def charge_levels(self, planned: Sequence[Additions]) -> list[float]:
        """The lowest battery level, as a fraction of full, in each segment."""
        return [1.0 - peak / self.watthour for peak in self.peaks(planned)]
