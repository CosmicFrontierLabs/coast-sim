"""Time-sequential first-fit planner.

:class:`SequentialPlanner` walks forward through the horizon and, at each
point, observes the first target in list order whose attitude stays clear of
constraints long enough. It is simple and quick, and list order acts as
priority. It is myopic: it never looks ahead or back, observes each target at
most once, and stops at the first time nothing fits. It does not reserve
ground passes or model slews and observation timing the way
:class:`~conops.ditl.DITL` executes them.
"""

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import rust_ephem

from ..common import dtutcfromtimestamp
from ..config import AttitudeConstraintScope, MissionConfig
from ..ditl.ditl_log import DITLLog
from ..simulation.roll import optimum_body_roll, optimum_instrument_roll
from ..simulation.saa import SAA
from ..targets import Plan, PlanEntry

Attitude = tuple[float, float, float]

FIRST_SLEW_SECONDS = 180
"""Slew time assumed for the first observation, which has no prior attitude."""


def _is_plan_entry(target: object) -> bool:
    # Duck-typed targets without instrument geometry are pointed by body roll.
    return issubclass(type(target), PlanEntry)


@dataclass
class _Selection:
    """A target chosen at one time, with the geometry it was chosen for."""

    target: PlanEntry
    roll: float
    attitude: Attitude
    """Spacecraft body attitude held for the observation."""
    slewtime: float
    clear_seconds: float
    """Contiguous constraint-free time from the slew start, slew included."""


class SequentialPlanner:
    """Build a plan by observing, at each time, the first target that fits.

    Steps:

    1. Start at ``begin``. Candidates are targets with exposure left, in list
       order, that have not been observed yet.
    2. For each candidate, choose the roll at the current time, estimate the
       slew from the previous observation, and measure how long the attitude
       stays clear of constraints. Take the first candidate whose clear window
       covers its slew plus ``min_window``.
    3. Observe it until its constraint-free window closes, capping the
       exposure at the target's remaining ``exptime``, then continue from
       there.
    4. Stop at ``end``, or at the first time no candidate fits.

    Targets are updated in place: each observed target's ``roll``,
    ``spacecraft_attitude`` and ``slewtime`` are set and its ``exptime`` is
    reduced by the exposure planned.

    Args:
        config: Mission configuration, with an ephemeris on its constraint.
        targets: Targets to plan, in priority order.
        begin: Start of the planning horizon.
        end: End of the planning horizon.
        min_window: Shortest constraint-free time, after the slew, worth
            observing, in seconds.
        log: Event log for planning events.
    """

    ephem: rust_ephem.Ephemeris

    def __init__(
        self,
        config: MissionConfig,
        targets: Sequence[PlanEntry],
        begin: datetime,
        end: datetime,
        *,
        min_window: float = 300,
        log: DITLLog | None = None,
    ) -> None:
        self.ephem = self._validated_ephem(config)
        self.config = config
        self.constraint = config.constraint
        self.targets = list(targets)
        self.begin = begin
        self.end = end
        self.min_window = min_window
        self.log = log if log is not None else DITLLog()
        self.step_size = self.ephem.step_size
        self.saa: SAA | None = None
        """South Atlantic Anomaly model; computed on first use if not set."""
        self.scheduled: list[int] = []
        """Obsids observed, in plan order."""
        self.unplaced: list[PlanEntry] = []
        """Targets left with exposure unplanned."""
        self.plan = Plan()

    # ── Public API ────────────────────────────────────────────────────────

    def schedule(self) -> Plan:
        """Build and return the plan."""
        self._check_supported()
        self._init_saa()
        self._start_run()

        ustart, uend = self._horizon()
        ephem_utime = [dt.timestamp() for dt in self.ephem.timestamp]
        latest_end = self._latest_end(uend, ephem_utime[-1])
        # First ephemeris step at or after the horizon start, so no entry
        # begins before ``begin``.
        i = bisect.bisect_left(ephem_utime, ustart)
        while i < len(ephem_utime) and ephem_utime[i] < uend:
            utime = ephem_utime[i]
            selection = self._select(utime, latest_end)
            if selection is None:
                self._log(
                    utime,
                    f"WARNING: No target found at time index {i} (utime={utime}); "
                    "stopping scheduling",
                )
                break
            entry = self._entry(selection, utime, latest_end)
            self.scheduled.append(selection.target.obsid)
            self.plan.extend([entry])
            # First step at or after the entry ends, so entries never overlap.
            i = bisect.bisect_left(ephem_utime, entry.end)

        # Observed targets are not revisited, so any left with exposure,
        # including partially observed ones, stay unplaced.
        self.unplaced = self._candidates()
        self._log(ephem_utime[0], f"Scheduled {len(self.plan)} targets")
        return self.plan

    # ── Setup ─────────────────────────────────────────────────────────────

    @staticmethod
    def _validated_ephem(config: MissionConfig) -> rust_ephem.Ephemeris:
        """Return the configuration's ephemeris, rejecting what this planner can't plan."""
        if config.payload.observation_timing.total_seconds > 0:
            raise ValueError(
                "Observation timing budgets require QueueDITL, not "
                "SequentialPlanner or DumbScheduler"
            )
        ephem = config.constraint.ephem
        if ephem is None:
            raise ValueError("Constraint.ephem must be set")
        return ephem

    def _check_supported(self) -> None:
        if any(
            panel.single_axis_drive is not None
            and bool(panel.drive_control.sun_tracking_modes)
            for panel in self.config.solar_panel.panels
        ):
            raise NotImplementedError(
                f"{type(self).__name__} cannot propagate evolving single-axis "
                "solar-array drive state; use the queue simulation for dynamic "
                "finite drives"
            )

    def _init_saa(self) -> None:
        if self.saa is None:
            self.saa = SAA()
            self.saa.ephem = self.ephem
            self.saa.calc()

    def _start_run(self) -> None:
        """Clear the results of any previous run."""
        self.scheduled = []
        self.plan = Plan(metadata={"planner": "sequential"})

    def _horizon(self) -> tuple[float, float]:
        """Return the planning horizon in Unix seconds."""
        return self.begin.timestamp(), self.end.timestamp()

    def _latest_end(self, uend: float, ephem_end: float) -> float:
        """Return the latest time an observation may end."""
        return min(uend, ephem_end)

    # ── Selection ─────────────────────────────────────────────────────────

    def _candidates(self) -> list[PlanEntry]:
        """Return the targets with exposure left, in priority order."""
        return [t for t in self.targets if t.exptime is not None and t.exptime > 0]

    def _select(self, utime: float, latest_end: float) -> _Selection | None:
        """Return the first candidate that fits at ``utime``, or None.

        Constraint-free time past ``latest_end`` doesn't count, since the
        observation is cut off there.
        """
        min_window = self.min_window
        for target in self._candidates():
            if target.obsid in self.scheduled:
                continue
            roll, attitude = self._point(target, utime)
            slewtime = self._slewtime(target)
            assert target.exptime is not None
            clear_seconds = min(
                self._clear_seconds(
                    target, attitude, utime, utime + target.exptime + slewtime
                ),
                latest_end - utime,
            )
            if clear_seconds >= min_window + slewtime:
                return _Selection(target, roll, attitude, slewtime, clear_seconds)
        return None

    def _point(self, target: PlanEntry, utime: float) -> tuple[float, Attitude]:
        """Set the target's attitude for an observation starting at ``utime``.

        The complete attitude is assigned before timing the slew so
        direction-dependent limits have both endpoints.

        Returns:
            The roll and the spacecraft body attitude.
        """
        is_entry = _is_plan_entry(target)
        telescope = target.science_telescope() if is_entry else None
        if telescope is not None:
            roll = optimum_instrument_roll(
                target.ra,
                target.dec,
                utime,
                self.ephem,
                telescope,
                self.config.solar_panel,
                self.constraint,
            )
        else:
            roll = optimum_body_roll(
                target.ra,
                target.dec,
                utime,
                self.ephem,
                self.config.solar_panel,
                self.constraint,
            )
        target.roll = roll
        attitude = (
            target.target_body_attitude(roll)
            if is_entry
            else (target.ra, target.dec, roll)
        )
        mounted = is_entry and target.uses_mounted_attitude()
        target.spacecraft_attitude = attitude if mounted else None
        return roll, attitude

    def _slewtime(self, target: PlanEntry) -> float:
        """Return the slew time into ``target`` from the previous observation."""
        if len(self.plan) == 0:
            return FIRST_SLEW_SECONDS
        last = self.plan[-1]
        last_attitude = last.spacecraft_attitude or (last.ra, last.dec, last.roll)
        estimate = target.calc_slewtime(*last_attitude)
        if estimate is not None:
            target.slewtime = estimate
        return target.slewtime

    def _clear_seconds(
        self, target: PlanEntry, attitude: Attitude, begin: float, end: float
    ) -> float:
        """Return how long ``attitude`` stays clear of constraints from ``begin``."""
        begin_idx = self.ephem.index(dtutcfromtimestamp(begin))
        end_idx = self.ephem.index(dtutcfromtimestamp(end)) + 1
        times = self.ephem.timestamp[begin_idx:end_idx]
        if _is_plan_entry(target) and target.uses_mounted_attitude():
            blocked = [
                bool(
                    target.attitude_constraint_names(
                        list(AttitudeConstraintScope), attitude, t.timestamp()
                    )
                )
                for t in times
            ]
        else:
            ra, dec, roll = attitude
            blocked = [
                self.constraint.in_constraint(
                    ra=ra, dec=dec, utime=t.timestamp(), target_roll=roll
                )
                for t in times
            ]
        clear = np.bitwise_not(blocked).astype(int).tolist()
        steps = next((k for k, ok in enumerate(clear) if ok != 1), len(clear))
        return float(steps * self.step_size)

    # ── Plan entries ──────────────────────────────────────────────────────

    def _entry(
        self, selection: _Selection, begin: float, latest_end: float
    ) -> PlanEntry:
        """Build the plan entry for a selection and charge its exposure to the target."""
        target = selection.target
        assert target.exptime is not None
        telescope = target.science_telescope() if _is_plan_entry(target) else None
        entry = PlanEntry(config=self.config)
        entry.ephem = self.ephem
        entry.ra = target.ra
        entry.dec = target.dec
        entry.roll = selection.roll
        entry.instrument_name = telescope.name if telescope is not None else None
        entry.spacecraft_attitude = target.spacecraft_attitude
        entry.begin = begin
        entry.slewtime = int(selection.slewtime)
        # Exposure is the time after the slew, capped at what the target still needs.
        entry.end = min(
            begin + selection.clear_seconds,
            latest_end,
            begin + entry.slewtime + target.exptime,
        )
        target.exptime -= entry.exposure
        entry.obsid = target.obsid
        assert self.saa is not None
        entry.saa = self.saa
        entry.merit = target.merit
        entry.name = target.name
        return entry

    def _log(self, utime: float, description: str) -> None:
        self.log.log_event(
            utime=utime,
            event_type="SCHEDULER",
            description=description,
            obsid=None,
            acs_mode=None,
        )
