"""Priority-first construction scheduler.

:class:`PriorityScheduler` reserves ground contacts first, then takes requests
in priority order and places each snapshot in the earliest slot that still fits,
without moving anything already placed. It is fast and deterministic, and its
plans are traceable to the priority order. It is myopic: an early placement can
block a better packing.
"""

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from enum import Enum, auto
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from ..common import ACSMode, ObsType, unixtime2date
from ..config import MissionConfig
from ..config.observation_categories import ObservationCategory
from ..config.observation_timing import ObservationTiming
from ..config.solar_panel import SolarArrayDriveState
from ..ditl.ditl_log import DITLLog
from ..ditl.plan_validator import entry_obstype
from ..simulation.coverage import StationCoverage
from ..simulation.passes import Pass
from ..simulation.slew import Slew
from ..targets import (
    DESATURATION_OBSID,
    Plan,
    PlanEntry,
    Pointing,
    ScheduledActivity,
    assign_obsids,
)
from ..targets.merit import MeritBreakdown, MeritModel
from ..targets.plan_entry import snapshot_min_seconds
from .allocator import AllocatedTime
from .battery import MIN_CHARGE_SECONDS, Addition, Additions, Angles, BatteryLedger
from .context import Attitude, SchedulingContext
from .momentum import MomentumForecast, SpanCache
from .recorder import RecorderLedger

CHARGE_OBSID = 999000
"""Obsid of the first planned charge; the rest follow."""

TrackingProfile = list[tuple[float, float, float]]


_MAX_DUMPS = 50
"""How many thruster dumps a plan reserves at most."""


class _Verdict(Enum):
    """Outcome of checking one slew into a timeline block."""

    OK = auto()
    SLEW_NOT_ALLOWED = auto()
    """The slew itself cannot start then; another start time may work."""
    WAIT_NOT_ALLOWED = auto()
    """The wait at the block's attitude fails; earlier starts only lengthen it."""


class _Block(BaseModel):
    """One commanded activity on the timeline, with the slew that reaches it."""

    entry: PlanEntry
    attitude_in: Attitude
    attitude_out: Attitude
    ready: float
    """When the activity needs its attitude: collection setup, or contact start."""
    end: float
    slew: Slew | None = None
    gspass: Pass | None = None
    steps: list[PlanEntry] = Field(default_factory=list)
    """For a sequence visit, the entries of its steps in order, the first
    reached by the block's slew; empty for anything else."""
    step_slews: list[Slew | None] = Field(default_factory=list)
    """The slew into each step: the block's own for the first, the slew
    between pointings for a later step, None where it stays put."""
    follows: bool = False
    """Whether the block holds the attitude of whatever precedes it, as a
    scheduled activity does: it is reached by no slew, its attitude is set
    again whenever the block before it changes, and the block after it is
    then re-slewed from it."""

    @property
    def is_visit(self) -> bool:
        return bool(self.steps)

    @property
    def is_pass(self) -> bool:
        return self.gspass is not None

    @property
    def is_charge(self) -> bool:
        return entry_obstype(self.entry) == ObsType.CHARGE

    @property
    def is_activity(self) -> bool:
        return entry_obstype(self.entry) == ObsType.ACTIVITY

    @property
    def is_fixed(self) -> bool:
        """A pass, a charge or an activity: not science, never moved."""
        return self.is_pass or self.is_charge or self.is_activity

    @property
    def slew_start(self) -> float:
        assert self.slew is not None
        return float(self.slew.slewstart)

    @property
    def arrival(self) -> float:
        assert self.slew is not None
        return float(self.slew.slewend)


class StartState(BaseModel):
    """Where the spacecraft is when a plan takes over part-way through a run.

    Attributes:
        time: First time (Unix seconds) a new activity may start; the scheduler
            rounds it up to a simulation step.
        attitude: Body attitude the spacecraft holds from ``time`` until the
            first planned slew.
    """

    model_config = ConfigDict(frozen=True)

    time: float
    attitude: Attitude


class _Request(BaseModel):
    """A target being planned, with its remaining exposure."""

    target: Pointing
    merit: MeritBreakdown
    """Merit at the start of the horizon."""
    remaining: float
    windows: list[tuple[float, float]] = Field(default_factory=list)
    category: ObservationCategory
    steady_value: float = 0.0
    """Value of a snapshot apart from the completion deficit, which depends
    on what the plan has collected before it. For allocated time it includes
    the allocation's factor."""
    value_factor: float = 1.0
    """Factor on the completion deficit's value: the allocation's, for
    allocated time, as on the rest of its value."""
    span: tuple[float, float] | None = None
    """The span of allocated time the request is for, if it is for one."""
    whole: bool = False
    """Whether the exposure is placed as one snapshot or not at all: allocated
    time that took the rest of the target's exposure, so that none of it is
    left too short to plan."""
    cadence: float | None = None
    """Seconds a visit waits after the target's last one, when cadence counts."""
    visit_candidates: list[int] = Field(default_factory=list)
    """For a target with visit windows, the windows a visit may still be
    placed in, in order: not visited, not committed, and not yet closed."""
    skipped: set[int] = Field(default_factory=set)
    """Visit windows no placement could meet."""
    visit: int | None = None
    """The visit window the snapshot being placed is for."""

    @property
    def pointing(self) -> Pointing:
        """The pointing the snapshot being placed observes: the target's,
        or the visit's own where its window gives it one."""
        return self.target.visit_pointing(self.visit)

    def select_visit(self, visit: int | None, ctx: SchedulingContext) -> None:
        """Place the next snapshot in visit window ``visit``: a visit with a
        pointing of its own is visible when that pointing is."""
        if visit == self.visit:
            return
        self.visit = visit
        self.windows = ctx.visibility_windows(self.pointing)

    @property
    def wants_more(self) -> bool:
        """Whether enough exposure is left for another snapshot."""
        return self.remaining > 0 and self.remaining >= float(
            snapshot_min_seconds(self.target)
        )

    @property
    def timed(self) -> bool:
        """Whether the target's visits have windows of their own."""
        return bool(self.target.visit_windows)

    @property
    def earliest_start(self) -> float | None:
        """When the snapshot being placed may begin collecting."""
        if self.timed and self.visit is not None:
            return self.target.visit_windows[self.visit].start
        return self.target.earliest_start

    @property
    def deadline(self) -> float | None:
        """By when the snapshot being placed must begin collecting."""
        if self.timed and self.visit is not None:
            return self.target.visit_windows[self.visit].deadline
        return self.target.deadline

    def open_visits(self, last_end: float | None, not_before: float) -> list[int]:
        """The windows a visit may still go in, in order: the candidates not
        skipped, after the window of the visit that ended at ``last_end``,
        and still open at ``not_before``."""
        target = self.target
        last = (
            target.visit_index_at(last_end - target.per_visit_seconds)
            if last_end is not None
            else None
        )
        return [
            k
            for k in self.visit_candidates
            if k not in self.skipped
            and (last is None or k > last)
            and target.visit_windows[k].deadline >= not_before
        ]

    def next_visit(self, last_end: float | None, not_before: float) -> int | None:
        """The window the next visit goes in, or None when none is left."""
        return next(iter(self.open_visits(last_end, not_before)), None)


def program_shares(seconds: Mapping[str, float]) -> dict[str, float]:
    """Return each program's fraction of the science seconds given."""
    total = sum(seconds.values())
    if total <= 0.0:
        return {}
    return {program: value / total for program, value in seconds.items()}


_MAX_DELAYED = 4
"""Most snapshots moved later to fit one before them."""

_BATTERY_RETRIES = 5
"""Later starts tried in a gap for a snapshot the battery refuses."""


class _Visit(NamedTuple):
    """A sequence visit built for one start: its block entry, its steps and
    their slews, where it leaves the spacecraft and when it ends."""

    entry: PlanEntry
    steps: list[PlanEntry]
    slews: list[Slew | None]
    attitude_out: Attitude
    end: float


class _Move(NamedTuple):
    """A snapshot placed again later, to make room before it."""

    old_block: _Block
    request: _Request
    new_block: _Block
    slew: Slew
    lost: float
    """Collection seconds the move cut short."""


def _collection_seconds(block: _Block) -> float:
    """Return the science a block collects: a visit's science steps' collection."""
    if block.steps:
        return sum(
            step.collection_seconds_between(step.begin, step.end)
            for step in block.steps
            if not step.calibration
        )
    return block.entry.collection_seconds_between(block.entry.begin, block.entry.end)


def _usable(request: _Request, seconds: float) -> bool:
    """Whether ``seconds`` of the request's exposure are none or a snapshot's worth."""
    return seconds <= 0.0 or seconds >= float(snapshot_min_seconds(request.target))


def _long_enough(collection_seconds: float, ss_min: float) -> bool:
    """Whether a snapshot collects something, and at least ``ss_min``."""
    return collection_seconds > 0 and collection_seconds >= ss_min


class PriorityScheduler:
    """Build a plan by placing requests in priority order at their earliest fit.

    Steps:

    1. Locked entries keep their collection windows and are placed first.
    2. Ground passes the spacecraft can reach are reserved.
    3. Requests are sorted by merit value at the start of the
       horizon (see :class:`~conops.targets.merit.MeritModel`). Each request is
       split into snapshots of up to ``ss_max`` seconds, never shorter than
       ``ss_min``, until its ``exptime`` is used or it no longer fits.

       Merit follows the plan as it would follow execution. With a
       completion-deficit weight, programs with a time share are re-ranked
       after every snapshot by the shares the plan delivers so far, counting
       science already collected. With a cadence weight, a target in a
       category with a cadence is visited no sooner than its cadence after
       its last collected visit, and kept a cadence clear of its other
       planned visits, before or after; each visit then has the full
       cadence value.
    4. Each snapshot goes in the earliest slot where every check
       :class:`~conops.ditl.DITL` will apply passes: the slew starts on a step,
       its path and the held attitude clear their mode's constraints, the
       target is visible when the slew starts (or, if ACS may slew before a
       target is visible, when it arrives), collection starts no earlier than
       the target's earliest start and by its deadline, and the following
       activity can still be reached.

    5. The plan keeps the battery above its planning level: every snapshot
       is checked against the energy the plan draws and generates, with a
       charge in every idle sunlit stretch one fits (see
       :class:`~conops.schedulers.BatteryLedger`), and the finished plan
       carries those charges as ``CHARGE`` entries.

    The plan is built for execution by :class:`~conops.ditl.DITL` with the same
    configuration, ephemeris, horizon and step size.

    Args:
        config: Mission configuration, with an ephemeris on its constraint.
        targets: Requests to plan. They are not modified.
        begin: Start of the planning horizon.
        end: End of the planning horizon.
        step_size: Simulation step in seconds; defaults to the ephemeris step.
        include_passes: Whether to reserve spacecraft tracking activities for passes.
        locked: Science entries to keep exactly where their collection windows
            are; the slews into them are recomputed.
        log: Event log for placements and rejections.
        simulation_start: Start of the simulation that will execute the plan,
            when the plan covers only part of it. Defaults to ``begin``.
        simulation_end: End of that simulation. Defaults to ``end``.
        passes: The run's predicted ground passes, where the caller has them;
            the plan then reserves those in its horizon instead of predicting
            the whole simulation's, which takes several seconds a day.
        start_state: Time and attitude the plan starts from, when it takes
            over from activities already committed. Without it the plan
            starts at ``begin`` from a freshly started spacecraft.
        reserved_seconds: Exposure, by obsid, already scheduled outside this
            plan and not yet collected; it is subtracted from each target's
            remaining exposure and counts towards its program's share.
        reserved_visits: When observations scheduled outside this plan and not
            yet collected will finish collecting, by obsid; a cadence target's
            next visit waits for its cadence after them.
        held: Obsids of targets this plan does not place, though their
            science, collected and reserved, counts towards their programs'
            shares as every target's does.
        successor_retries: When a snapshot fits a gap but the activity after
            it cannot then be reached, how many later starts to try in that gap
            before giving up on it (default 3). This plans much faster when many
            requests cannot be placed, but can miss a fit late in a gap. None
            tries every start, which finds every fit.
        start_volume_gb: What is on the recorder when the horizon starts; the
            configured recorder's volume if None.
        start_deficit_wh: How far below full the battery is, in Wh, when the
            horizon starts; the configured battery's charge if None.
        start_drive_state: Where the solar-array drive is when the horizon
            starts; its configured initial state if None.
        allocated: Time allocated to requests, by obsid, such as from
            :meth:`~conops.schedulers.Allocation.allocated`. A request's
            allocated seconds in each span are planned in that span, worth
            ``allocation_bonus`` more; collection starting in the span counts.
            The rest of its exposure, and any allocated seconds that do not fit
            their span, are planned like an unallocated request, which may work
            ahead. None (the default) treats all requests alike.
        allocation_bonus: Fraction of a request's value its allocated time
            gains, so plans trade keeping to the allocation against merit.
            Zero ignores ``allocated`` altogether.
        start_momentum_eci_n_m_s: The stored momentum in inertial coordinates
            when the horizon starts, for the momentum forecast; the configured
            initial momentum if None.
    """

    scheduler_name = "priority"
    """Recorded in the plan's metadata."""

    def __init__(
        self,
        config: MissionConfig,
        targets: Sequence[Pointing],
        begin: datetime,
        end: datetime,
        *,
        step_size: int | None = None,
        include_passes: bool = True,
        locked: Sequence[PlanEntry] = (),
        log: DITLLog | None = None,
        simulation_start: datetime | None = None,
        simulation_end: datetime | None = None,
        start_state: StartState | None = None,
        reserved_seconds: Mapping[int, float] | None = None,
        reserved_visits: Mapping[int, float] | None = None,
        held: Iterable[int] = (),
        successor_retries: int | None = 3,
        start_volume_gb: float | None = None,
        start_deficit_wh: float | None = None,
        start_drive_state: SolarArrayDriveState | None = None,
        passes: Sequence[Pass] | None = None,
        allocated: Mapping[int, Sequence[AllocatedTime]] | None = None,
        allocation_bonus: float = 0.5,
        coverage: StationCoverage | None = None,
        activities: Sequence[ScheduledActivity] = (),
        start_momentum_eci_n_m_s: Sequence[float] | None = None,
    ) -> None:
        self.config = config
        self.targets = list(targets)
        self.activities = assign_obsids(activities)
        """Fixed-time activities to reserve, in time order."""
        self._slidable: dict[int, ScheduledActivity] = {}
        """The activities reserved with a slide window, by their block."""
        self.dumps: list[ScheduledActivity] = []
        """The thruster dumps this scheduler has reserved where its momentum
        forecast reached the level, in time order."""
        self._momentum_rejected: set[int] = set()
        """Targets whose visit stores more than the wheels' capacity from
        empty: not planned."""
        self.start_momentum_eci_n_m_s = (
            tuple(float(v) for v in start_momentum_eci_n_m_s)
            if start_momentum_eci_n_m_s is not None
            else None
        )
        """The stored momentum in inertial coordinates when the horizon
        starts; the configured initial momentum if None."""
        self.held = frozenset(int(obsid) for obsid in held)
        self.start_drive_state = start_drive_state
        """Where the solar-array drive is when the horizon starts; its
        configured initial state if None."""
        self.begin = begin
        self.end = end
        self.step_size = step_size
        self.include_passes = include_passes
        self.locked = list(locked)
        self.log = log if log is not None else DITLLog()
        self.simulation_start = simulation_start
        self.simulation_end = simulation_end
        self.passes = passes
        """The run's predicted ground passes, where the caller has them."""
        self.coverage = coverage
        """Link coverage for passes without spacecraft tracking, where the
        caller has it; built from the configuration otherwise."""
        self.start_state = start_state
        self.reserved_seconds = dict(reserved_seconds or {})
        self.reserved_visits = dict(reserved_visits or {})
        if successor_retries is not None and successor_retries < 0:
            raise ValueError("successor_retries must not be negative")
        self.successor_retries = successor_retries
        self.start_volume_gb = start_volume_gb
        """What is on the recorder when the horizon starts; the configured
        recorder's volume if None."""
        self.start_deficit_wh = start_deficit_wh
        """How far below full the battery is when the horizon starts; the
        configured battery's charge if None."""
        if allocation_bonus < 0.0:
            raise ValueError("allocation_bonus must not be negative")
        # A bonus of zero ignores when a request was allocated: its seconds
        # are then not held to their spans either.
        self.allocated = (
            None
            if allocated is None or allocation_bonus == 0.0
            else {int(obsid): list(spans) for obsid, spans in allocated.items()}
        )
        self.allocation_bonus = allocation_bonus
        self.unplaced: list[Pointing] = []
        """Targets left with at least ``ss_min`` of exposure unplanned."""
        self.plan = Plan()
        self._rest_of: dict[int, _Request] = {}
        """For each request for allocated time, by id, the request for the
        rest of its target's exposure."""
        self._planned_visits: dict[int, list[_Block]] = defaultdict(list)
        """Cadence visits placed so far, by obsid. Allocated time can place
        a later visit first, so a visit is spaced from these on both sides."""
        self._refitting: _Block | None = None
        """A visit being placed again later, not spaced from itself."""

    # ── Public API ────────────────────────────────────────────────────────

    def schedule(self) -> Plan:
        """Build and return the plan, within the wheels' momentum where the
        configuration tracks it (see :meth:`_plan_within_momentum`)."""
        self._momentum_cache: SpanCache = {}
        """Spans of the timeline integrated for the momentum forecast, shared
        by the passes of this plan: each pass changes the timeline only from
        the dump it reserves onwards."""
        return self._plan_within_momentum(self._plan_once)

    def _plan_once(self) -> Plan:
        """Build the plan with the activities and dumps reserved so far."""
        self._reserve_fixed()
        self._place_requests(self._with_allocated_time(self._requests()))
        return self._finish(self.timeline)

    def _plan_within_momentum(self, build: Callable[[], Plan]) -> Plan:
        """Build the plan, and where the forecast of the stored momentum
        along it reaches the level a thruster dump is called for at, reserve
        a dump there and build again, until no dump is called for.

        Each pass plans from the start with the dumps so far as fixed
        activities, so the plan before a dump is the plan that fills the
        wheels to it; a dump found earlier than one already reserved
        replaces that one and those after it, which the next pass finds
        again if they are still needed. Without thrusters the forecast is
        logged where it passes the capacity, and the plan stands.

        A dump goes between observations (see :meth:`_dump_placement`): after
        the one the forecast reaches the level inside, where the capacity is
        not reached inside it too; before it, where it is but the observation
        fits from empty wheels; a visit that does not fit from empty is not
        planned, and a plain observation that does not is cut at the level.
        """
        plan = build()
        forecast = self._momentum
        if forecast is None:
            return plan
        level = forecast.level_n_m_s
        capacity = forecast.capacity_n_m_s
        if level is None:
            if capacity is not None:
                crossing = forecast.first_crossing(self.timeline, capacity)
                if crossing is not None:
                    self._log(
                        crossing,
                        f"The stored momentum reaches the {capacity} N m s "
                        f"capacity at {unixtime2date(crossing)}; no thrusters to dump it",
                    )
            return plan
        dump = self.config.spacecraft_bus.attitude_control.stored_momentum.thruster_dump
        assert dump is not None
        for _ in range(_MAX_DUMPS):
            crossing = self._next_momentum_crossing(level, waiting=True)
            if crossing is None:
                self._log_forecast_past_capacity()
                return plan
            placement = self._dump_placement(crossing, capacity)
            if placement is None:
                plan = build()
                continue
            begin, where = placement
            if begin + dump.duration_seconds > self.ctx.uend:
                return plan
            if any(abs(d.begin - begin) < 1e-6 for d in self.dumps):
                # The same dump again: the crossing it answers stands with
                # the wheels emptied there, so another pass would not change it.
                self._log(
                    crossing,
                    f"Stored momentum reaches the planning level {level:.3f} N m s "
                    f"at {unixtime2date(crossing)} with a dump already reserved "
                    f"{where}; the plan stands",
                )
                self._log_forecast_past_capacity()
                return plan
            kept = [d for d in self.dumps if d.end <= begin]
            reserved = ScheduledActivity(
                name="Desaturation",
                begin=begin,
                duration_seconds=dump.duration_seconds,
                desaturation=True,
                obsid=DESATURATION_OBSID + len(kept),
            )
            self.dumps = [*kept, reserved]
            self._log(
                begin,
                f"Stored momentum reaches the planning level {level:.3f} N m s at "
                f"{unixtime2date(crossing)}: thruster dump reserved {where} at "
                f"{unixtime2date(begin)}",
                reserved.obsid,
            )
            plan = build()
        self._log(
            self.ctx.uend,
            f"{_MAX_DUMPS} thruster dumps reserved; the plan may still fill the wheels",
        )
        return plan

    def _log_forecast_past_capacity(self) -> None:
        """Say where the plan's forecast reaches the capacity, if it does:
        the dumps reserved could not keep it within."""
        forecast = self._momentum
        if forecast is None or forecast.capacity_n_m_s is None:
            return
        capacity = forecast.capacity_n_m_s
        crossing = forecast.first_crossing(self.timeline, capacity)
        if crossing is not None:
            self._log(
                crossing,
                f"The plan's forecast reaches the {capacity} N m s capacity at "
                f"{unixtime2date(crossing)} with the dumps reserved",
            )

    def _dump_placement(
        self, crossing: float, capacity: float | None
    ) -> tuple[float, str] | None:
        """Where the dump called for by the forecast reaching the level at
        ``crossing`` goes: when it begins and, for the log, where. Between
        observations: at the crossing where no observation is under way
        there; after the observation under way where the capacity is not
        reached inside it; before it where the capacity is but the
        observation fits from empty wheels; at the crossing where a plain
        observation does not fit from empty, which cuts it there. None where
        a visit does not fit from empty: it is left out of the plan and
        nothing is reserved this pass."""
        assert self._momentum is not None
        dump = self.config.spacecraft_bus.attitude_control.stored_momentum.thruster_dump
        assert dump is not None
        block = next(
            (
                b
                for b in self.timeline
                if float(b.entry.begin) <= crossing < float(b.end)
                and entry_obstype(b.entry) not in (ObsType.ACTIVITY, ObsType.GSP)
            ),
            None,
        )
        if block is None:
            return self._dump_around_fixed(crossing, capacity, dump.duration_seconds)
        obsid = int(block.entry.obsid)
        at_capacity = (
            self._next_momentum_crossing(capacity) if capacity is not None else None
        )
        # The wheels go on filling through the dump: it must end within the capacity.
        after = self.ctx.ceil_step(float(block.end))
        if at_capacity is None or at_capacity > after + dump.duration_seconds:
            return after, f"after observation {obsid}"
        assert capacity is not None
        loading = self._momentum.loading(block)
        if loading <= capacity:
            return (
                self.ctx.floor_step(float(block.entry.begin)),
                f"before observation {obsid}, which would reach the capacity",
            )
        if block.steps:
            self._momentum_rejected.add(obsid)
            self._log(
                float(block.entry.begin),
                f"Visit of {obsid} not planned: its loading of {loading:.3f} N m s "
                f"from empty wheels exceeds the {capacity} N m s capacity",
                obsid,
            )
            return None
        return (
            self.ctx.ceil_step(crossing),
            f"inside observation {obsid}, which stores more than the capacity",
        )

    def _dump_around_fixed(
        self, crossing: float, capacity: float | None, duration: float
    ) -> tuple[float, str]:
        """Where a dump goes for a crossing that falls in no observation: at
        the crossing, unless the crossing falls inside a scheduled activity
        or a contact, which a dump cannot interrupt. Then after that block
        where the capacity is not reached through the dump's own length,
        else before it, so the crossing is addressed rather than a dump held
        after the block for a crossing it leaves standing."""
        inside = next(
            (
                b
                for b in self.timeline
                if float(b.entry.begin) <= crossing < float(b.end)
                and entry_obstype(b.entry) in (ObsType.ACTIVITY, ObsType.GSP)
                and not b.entry.desaturation
            ),
            None,
        )
        if inside is None:
            return self.ctx.ceil_step(crossing), "between observations"
        name = inside.entry.name or str(inside.entry.obsid)
        what = "contact" if inside.is_pass else "activity"
        after = self.ctx.ceil_step(float(inside.end))
        at_capacity = (
            self._next_momentum_crossing(capacity) if capacity is not None else None
        )
        if at_capacity is None or at_capacity > after + duration:
            return after, f"after {what} {name}"
        before = self.ctx.floor_step(float(inside.ready) - duration)
        if before < self.ctx.ustart:
            return after, f"after {what} {name}, which leaves no room before it"
        return before, f"before {what} {name}, which would reach the capacity inside"

    def _next_momentum_crossing(
        self, level: float, *, waiting: bool = False
    ) -> float | None:
        """When the forecast first reaches ``level`` that the dumps already
        reserved do not answer. A crossing inside a reserved dump is
        answered: the wheels are empty when it ends. With ``waiting``, so is
        a crossing a reserved dump follows, as long as the capacity is not
        reached before that dump begins: the dump was reserved to follow the
        observation the crossing falls in."""
        assert self._momentum is not None
        capacity = self._momentum.capacity_n_m_s
        after: float | None = None
        while True:
            crossing = self._momentum.first_crossing(self.timeline, level, after=after)
            if crossing is None:
                return None
            answered = next(
                (
                    d
                    for d in self.dumps
                    if d.begin - self.ctx.step_size <= crossing <= d.end
                ),
                None,
            )
            if answered is None and waiting:
                following = next((d for d in self.dumps if d.begin > crossing), None)
                if following is not None:
                    at_capacity = (
                        self._momentum.first_crossing(
                            self.timeline, capacity, after=crossing
                        )
                        if capacity is not None
                        else None
                    )
                    if at_capacity is None or at_capacity >= following.begin:
                        answered = following
            if answered is None:
                return crossing
            after = answered.end

    def _reserve_fixed(self) -> None:
        """Start a timeline holding the locked entries and reachable passes."""
        self.ctx = SchedulingContext(
            self.config,
            self.begin,
            self.end,
            step_size=self.step_size,
            simulation_start=self.simulation_start,
            simulation_end=self.simulation_end,
            passes=self.passes,
            coverage=self.coverage,
        )
        for target in self.targets:
            own = getattr(target, "own_timing", None)
            if isinstance(own, ObservationTiming):
                self.ctx.register_timing(int(target.obsid), own)
        self._battery = self._battery_ledger()
        self._momentum: MomentumForecast | None = None
        if self.config.spacecraft_bus.attitude_control.stored_momentum.enabled:
            self._momentum = MomentumForecast(
                self.config,
                self.ctx,
                start_momentum_eci_n_m_s=self.start_momentum_eci_n_m_s,
                start_attitude=(
                    self.start_state.attitude if self.start_state is not None else None
                ),
                cache=getattr(self, "_momentum_cache", None),
            )
        self._gap_charges_cache: dict[
            tuple[object, ...], tuple[list[_Block], Slew | None]
        ] = {}
        """Charges by gap (see :meth:`_gap_charges`)."""
        self._next_charge_obsid = CHARGE_OBSID
        self._energy: tuple[Additions, ...] = tuple(
            () for _ in range(self._battery.segments)
        )
        """What the plan draws from the battery, by segment and in time order."""
        self.timeline: list[_Block] = []
        self._movable: dict[int, _Request] = {}
        """Science blocks placed by this scheduler that may move later, by id,
        with their requests; locked entries and passes never move."""
        self._successor_blocked = False
        """Whether the last gap searched held a fit that only the activity
        after it prevented."""
        self.unplaced = []
        self._origin: _Block | None = None
        if self.start_state is not None:
            time = self.ctx.ceil_step(self.start_state.time)
            attitude = self.start_state.attitude
            self._origin = _Block(
                entry=PlanEntry(),
                attitude_in=attitude,
                attitude_out=attitude,
                ready=time,
                end=time,
            )

        for entry in self.locked:
            self._place_locked(entry)
        self._locked_visits: dict[int, list[tuple[float, float]]] = defaultdict(list)
        """Locked science visits by obsid: when each slew starts and its
        collection ends. Cadence visits are kept clear of them."""
        for block in self.timeline:
            entry = block.entry
            if entry.collection_end is not None:
                self._locked_visits[int(entry.obsid)].append(
                    (float(entry.begin), float(entry.collection_end))
                )
        if self.include_passes:
            for gspass in self.ctx.predict_passes():
                if gspass.requires_tracking:
                    self._place_pass(gspass)
        for activity in sorted([*self.activities, *self.dumps], key=lambda a: a.begin):
            if self.ctx.ustart <= activity.begin < self.ctx.uend:
                self._place_activity(activity)
        self._ledger = self._recorder_ledger()
        self._volumes = self._ledger.initial_volumes(
            [block.entry for block in self.timeline]
        )
        """Science volume planned in each of the ledger's segments."""
        limit = self._ledger.limit
        if limit is not None:
            over = max(self._ledger.peaks(self._volumes))
            if over + 1e-9 >= limit:
                # What is fixed already, with the housekeeping, reaches the
                # limit: nothing placed here changes that, so it is said.
                self._log(
                    self.ctx.ustart,
                    f"The recorder reaches {over:.2f} Gb, its planning limit "
                    f"being {limit:.2f} Gb, from what is on it and the "
                    "locked entries and housekeeping alone; no science is "
                    "placed before it comes down",
                )
        self._energy = self._charged_energy(self.timeline)

    def _recorder_ledger(self) -> RecorderLedger:
        """The recorder over the horizon, downlinking during the reserved contacts."""
        start_volume = (
            self.start_volume_gb
            if self.start_volume_gb is not None
            else self.config.recorder.current_volume_gb
        )
        return RecorderLedger(
            self.config,
            self.ctx.ustart,
            self.ctx.uend,
            [block.entry for block in self.timeline if block.is_pass],
            start_volume_gb=start_volume,
            step_size=self.ctx.step_size,
            origin=self.ctx.origin,
            coverage=self.ctx.coverage,
        )

    def _ingress_deltas(
        self, block: _Block | None, slew: Slew | None
    ) -> list[tuple[int, float]]:
        """How what the slew into ``block`` adds to the recorder changes, by
        segment, were the slew flown as ``slew`` instead of as the block has
        it: nothing unless the block is a contact."""
        if block is None or slew is None or not block.is_pass:
            return []
        ledger = self._ledger
        entry = block.entry
        return [
            (
                ledger.ingress_segment(entry, begin=float(slew.slewstart)),
                ledger.ingress_volume(
                    entry, begin=float(slew.slewstart), slewtime=float(slew.slewtime)
                ),
            ),
            (ledger.ingress_segment(entry), -ledger.ingress_volume(entry)),
        ]

    def _battery_ledger(self) -> BatteryLedger:
        """The battery over the horizon, charging in its idle sunlit time."""
        battery = self.config.battery
        start_deficit = (
            self.start_deficit_wh
            if self.start_deficit_wh is not None
            else max(0.0, float(battery.watthour) - float(battery.charge_level))
        )
        return BatteryLedger(
            self.config,
            self.ctx.ustart,
            self.ctx.uend,
            start_deficit_wh=start_deficit,
            step_size=self.ctx.step_size,
            origin=self.ctx.origin,
            drive_state=self.start_drive_state,
        )

    def _timeline_energy(self, timeline: Sequence[_Block]) -> tuple[Additions, ...]:
        """What a timeline draws from the battery, by segment, every span at the
        attitude the spacecraft holds through it: each block over its slew and
        hold, and the idle time between blocks at the attitude the block before
        leaves it in (the start state's, or a freshly started spacecraft's,
        before the first)."""
        ledger = self._battery
        planned: list[list[Addition]] = [[] for _ in range(ledger.segments)]
        cursor = self.ctx.ustart
        held = (
            self._origin.attitude_out
            if self._origin is not None
            else self.ctx.initial_attitude(self.ctx.ustart)
        )

        def add(addition: Addition | None) -> None:
            if addition is not None:
                planned[ledger.segment(addition.begin)].append(addition)

        for block in sorted(timeline, key=lambda b: float(b.entry.begin)):
            begin = float(block.entry.begin)
            if begin > cursor:
                add(ledger.hold_addition(held, ACSMode.IDLE, cursor, begin))
            addition = self._block_addition(block)
            add(addition)
            cursor = max(cursor, addition.end)
            held = block.attitude_out
        if self.ctx.uend > cursor:
            add(ledger.hold_addition(held, ACSMode.IDLE, cursor, self.ctx.uend))
        return tuple(tuple(sorted(group)) for group in planned)

    def _block_addition(self, block: _Block) -> Addition:
        """What a block draws from the battery over its slew and hold, with the
        panels' generation along the slew, at the attitude held, or along a
        contact's tracking profile. A contact goes on drawing until the
        spacecraft is at rest after it."""
        slew = block.slew
        if block.steps:
            addition = self._battery.visit_addition(
                block.steps,
                block.step_slews,
                (
                    (float(slew.startra), float(slew.startdec), float(slew.startroll))
                    if slew is not None
                    else None
                ),
            )
            assert addition is not None
            return addition
        addition = self._battery.addition(
            block.entry,
            rest=block.end if block.is_pass else None,
            path=slew.attitude if slew is not None else None,
            path_start=(
                (float(slew.startra), float(slew.startdec), float(slew.startroll))
                if slew is not None
                else None
            ),
            track=block.gspass.attitude_at if block.gspass is not None else None,
        )
        assert addition is not None
        return addition

    def _charged_energy(self, timeline: Sequence[_Block]) -> tuple[Additions, ...]:
        """A timeline's energy with the charges its idle sunlit time holds."""
        return self._timeline_energy(self._charged_timeline(list(timeline)))

    def _block_volume(self, block: _Block) -> float:
        """What a science block puts on the recorder: each science step's, for a visit."""
        if block.steps:
            return sum(
                self._ledger.science_volume(
                    step, step.collection_seconds_between(step.begin, step.end)
                )
                for step in block.steps
                if not step.calibration
            )
        return self._ledger.science_volume(block.entry, _collection_seconds(block))

    def _block_segment(self, block: _Block) -> int:
        return self._ledger.segment(float(block.entry.collection_begin or block.ready))

    def _finish(self, timeline: Sequence[_Block]) -> Plan:
        """Record and return the plan for a finished timeline, with its charging."""
        timeline = self._plan_charging(list(timeline))
        self._settle_sliding_activities(timeline)
        self.plan = Plan(
            entries=[
                step for block in timeline for step in (block.steps or [block.entry])
            ],
            ground_contacts=(
                [
                    (span.begin, span.end, span.station)
                    for span in self.ctx.coverage.scheduled_passes
                ]
                if self.ctx.coverage is not None
                else None
            ),
            metadata={
                "scheduler": self.scheduler_name,
                "step_size": self.ctx.step_size,
            },
        )
        return self.plan

    def _place_requests(self, requests: Sequence[_Request]) -> None:
        """Place snapshots of the highest-ranked request at their earliest fit.

        Re-evaluate values after each snapshot: program shares can change,
        and merging leftover allocated seconds can dilute a request's bonus.
        """
        # The request for the rest of a target's exposure can start with none,
        # all of it allocated, and take allocated seconds that do not fit.
        active = [request for request in requests if request.wants_more]
        order = {id(request): k for k, request in enumerate(requests)}
        placed = dict.fromkeys(order, 0)
        self._program_seconds = dict(self._delivered)
        """Science by program: collected, committed and planned so far."""
        self._cut_short: list[_Request] = []
        """Requests a move has cut short since the last placement."""
        while active:
            shares = program_shares(self._program_seconds)
            # The original order breaks ties only; a merge can change value
            # even when completion-deficit balancing is disabled.
            request = max(
                active,
                key=lambda r: (
                    self._value(r, shares),
                    -order[id(r)],
                ),
            )
            if request.timed:
                # The next window to visit; none left ends the request.
                request.select_visit(
                    request.next_visit(
                        self._last_visit.get(int(request.target.obsid)),
                        self.ctx.ustart,
                    ),
                    self.ctx,
                )
                if request.visit is None:
                    request.remaining = 0.0
            block = (
                self._place_snapshot(request, self._cadence_release(request))
                if request.wants_more
                else None
            )
            self._reactivate_cut_short(active, order)
            if block is None and request.timed and request.visit is not None:
                # This window cannot be met; the later ones may be.
                request.skipped.add(request.visit)
                request.remaining -= request.target.per_visit_seconds
                if request.wants_more:
                    continue
            if block is not None:
                # Freeze the value the snapshot was selected at onto its entry,
                # as dispatch does: a Target of Opportunity must beat it to
                # interrupt the observation.
                block.entry.merit = self._value(request, shares)
                collected = _collection_seconds(block)
                request.remaining -= collected
                placed[id(request)] += 1
                self._record_visit(request, block)
                if request.wants_more:
                    continue
            # By identity: a request for allocated time can equal the request
            # for the rest of its target's exposure field for field.
            active[:] = [r for r in active if r is not request]
            self._return_allocated_time(request, active, order)

        # Checked once all are placed: moving a snapshot later to make room
        # can cut short a request that was already finished.
        for request in requests:
            self._return_allocated_time(request, None, order)
        # Subclasses place again from their own state, in time order.
        self._planned_visits = defaultdict(list)
        for request in requests:
            if id(request) not in self._rest_of and request.wants_more:
                self.unplaced.append(request.target)
                self._log(
                    self.ctx.ustart,
                    f"Target {request.target.obsid} left {request.remaining:.0f}s "
                    f"unplanned after {placed[id(request)]} snapshot(s)",
                    request.target.obsid,
                )

    def _reactivate_cut_short(
        self, active: list[_Request], order: Mapping[int, int]
    ) -> None:
        """Put requests a move cut short back among those still to place.

        A finished request cut short by at least ``ss_min`` can be placed
        again. ``active`` keeps the original tie order; selection uses each
        request's current value. This always ends: a cut is
        only made when it raises the plan's value-weighted science.
        """
        waiting = {id(request) for request in active}
        for request in self._cut_short:
            if id(request) not in waiting and request.wants_more:
                active.append(request)
                waiting.add(id(request))
        self._cut_short = []
        active.sort(key=lambda request: order[id(request)])

    def _return_allocated_time(
        self,
        request: _Request,
        active: list[_Request] | None,
        order: Mapping[int, int],
    ) -> None:
        """Give allocated seconds that did not fit their span to the rest of
        the target's exposure, and put that back among ``active`` if it can
        now take another snapshot."""
        rest = self._rest_of.get(id(request))
        if rest is None or request.remaining <= 0.0:
            return
        sibling = None
        if active is not None and not _usable(rest, rest.remaining + request.remaining):
            # Too few for the rest to plan: a snapshot of the target's
            # allocated time still to place takes them instead.
            sibling = next(
                (
                    r
                    for r in active
                    if r is not request and self._rest_of.get(id(r)) is rest
                ),
                None,
            )
        if sibling is not None:
            self._merge_allocated_time(sibling, request)
            return
        rest.remaining += request.remaining
        request.remaining = 0.0
        if (
            active is not None
            and all(r is not rest for r in active)
            and rest.wants_more
        ):
            active.append(rest)
            active.sort(key=lambda r: order[id(r)])

    def _merge_allocated_time(self, into: _Request, request: _Request) -> None:
        """Move ``request``'s remaining seconds to ``into``, a request for
        allocated time of the same target, with the bonus each part of them
        has in ``into``'s span: the factor becomes their seconds-weighted
        mean. Seconds from another span have none of their bonus there.
        ``into`` is placed whole while one snapshot holds it."""
        seconds = into.remaining + request.remaining
        moved = request.value_factor if request.span == into.span else 1.0
        factor = (into.value_factor * into.remaining + moved * request.remaining) / (
            seconds
        )
        into.steady_value = into.steady_value / into.value_factor * factor
        unallocated = into.merit.value - into.merit.allocation
        into.merit = into.merit.model_copy(
            update={"allocation": (factor - 1.0) * unallocated}
        )
        into.value_factor = factor
        into.remaining = seconds
        into.whole = seconds <= float(into.target.ss_max)
        request.remaining = 0.0

    def _value(self, request: _Request, shares: Mapping[str, float]) -> float:
        """Value of a snapshot of ``request`` when the plan delivers ``shares``."""
        if not self._deficit_weight:
            return request.steady_value
        return request.steady_value + request.value_factor * self._deficit_weight * (
            MeritModel.completion_deficit(request.category, shares)
        )

    def _cadence_release(self, request: _Request) -> float | None:
        """Earliest slew start of the request's next visit, if cadence spaces them."""
        if request.cadence is None:
            return None
        # Planned visits are spaced in _next_locked_visit, before or after.
        last = self._visited_before.get(int(request.target.obsid))
        return None if last is None else last + request.cadence

    def _next_locked_visit(
        self, request: _Request, start: float
    ) -> tuple[float, float] | None:
        """Return the first locked or planned visit a cadence visit starting
        at ``start`` must end before.

        Visits are spaced by the cadence: a visit must start at least a
        cadence after another visit's collection ends, or end at least a
        cadence before its slew starts. Returns that visit's slew start and
        collection end, or None if the request has no cadence or ``start`` is
        clear of every other visit.
        """
        if request.cadence is None:
            return None
        obsid = int(request.target.obsid)
        visits = sorted(
            [
                *self._locked_visits.get(obsid, ()),
                *(
                    (float(block.entry.begin), float(block.entry.collection_end))
                    for block in self._planned_visits.get(obsid, ())
                    if block is not self._refitting
                    and block.entry.collection_end is not None
                ),
            ]
        )
        return next(
            (
                (locked_start, locked_end)
                for locked_start, locked_end in visits
                if start < locked_end + request.cadence
            ),
            None,
        )

    def _record_visit(self, request: _Request, block: _Block) -> None:
        """Count a placed snapshot towards its program and the target's last visit."""
        entry = block.entry
        seconds = entry.collection_seconds_between(entry.begin, entry.end)
        program = request.category.program_name
        self._program_seconds[program] = (
            self._program_seconds.get(program, 0.0) + seconds
        )
        if (
            request.cadence is not None or request.timed
        ) and entry.collection_end is not None:
            obsid = int(request.target.obsid)
            # Allocated time can place a target's later visit first.
            self._last_visit[obsid] = max(
                self._last_visit.get(obsid, float("-inf")), float(entry.collection_end)
            )
            self._planned_visits[obsid].append(block)

    # ── Requests ──────────────────────────────────────────────────────────

    def _requests(self) -> list[_Request]:
        """Return the requests in planning order: value, then a stable tie-break."""
        merit_model = MeritModel(self.config)
        # Science committed outside this plan or locked into it will be
        # collected before what this plan places.
        reserved = self._reserved_collection()
        for obsid, seconds in self.reserved_seconds.items():
            reserved[obsid] += seconds
        seed = self.config.random_seed if self.config.random_seed is not None else 0
        self._delivered: dict[str, float] = {}
        """Science already collected, by program."""
        self._last_visit: dict[int, float] = {}
        """When each cadence target was last observed, collected or planned."""
        for target in self.targets:
            obsid = int(target.obsid)
            seconds = target.collected_seconds + reserved.get(obsid, 0.0)
            if seconds > 0.0:
                program = merit_model.category(target).program_name
                self._delivered[program] = self._delivered.get(program, 0.0) + seconds
            last_collection = target.last_collection_time
            if last_collection is not None and target.visit_windows:
                # A visit cut short is taken again in its window: it is
                # the last visit only once a whole visit has been collected.
                index = target.visit_index_at(last_collection)
                if index is not None and not target.visit_complete(index):
                    last_collection = None
            visits = [
                t
                for t in (last_collection, self.reserved_visits.get(obsid))
                if t is not None
            ]
            if visits:
                self._last_visit[obsid] = max(visits)
        self._visited_before = dict(self._last_visit)
        """When each cadence target was last observed before this plan."""
        self._planned_visits = defaultdict(list)
        self._refitting = None
        shares = program_shares(self._delivered)
        cadence_weight = merit_model.cadence_weight
        requests = []
        for target in self.targets:
            check = getattr(target, "check_instrument_modes", None)
            if callable(check) and not target.done:
                # A request naming no modes uses every instrument with modes.
                check()
            exptime = target.exptime if target.exptime is not None else target.ss_max
            remaining = float(exptime) - reserved.get(int(target.obsid), 0.0)
            candidates: list[int] = []
            if target.visit_windows:
                # One visit per window still to come: not visited, not after
                # a committed visit's window, and not closed before the plan.
                last = self._last_visit.get(int(target.obsid))
                visited = set(target.visits_done)
                if last is not None:
                    index = target.visit_index_at(last - target.per_visit_seconds)
                    if index is not None:
                        visited.update(range(index + 1))
                candidates = [
                    k
                    for k, window in enumerate(target.visit_windows)
                    if k not in visited and window.deadline >= self.ctx.ustart
                ]
                remaining = target.per_visit_seconds * len(candidates)
            if (
                target.done
                or int(target.obsid) in self.held
                or int(target.obsid) in self._momentum_rejected
                or remaining < snapshot_min_seconds(target)
            ):
                continue
            category = merit_model.category(target)
            merit = merit_model.value_terms(
                target, self.ctx.ustart, delivered_shares=shares
            )
            cadence = category.cadence_seconds if cadence_weight > 0.0 else None
            requests.append(
                _Request(
                    target=target,
                    merit=merit,
                    remaining=remaining,
                    windows=self.ctx.visibility_windows(target),
                    category=category,
                    # Visits wait for their cadence, so each has full pressure.
                    steady_value=merit.base
                    + merit.urgency
                    + (cadence_weight if cadence is not None else 0.0),
                    cadence=cadence,
                    visit_candidates=candidates,
                )
            )
        self._deficit_weight = merit_model.completion_deficit_weight
        self._seed = seed
        self._shares = shares
        # Allocated time is worth its request's value plus the bonus.
        self._allocated_factor = 1.0 + self.allocation_bonus
        requests.sort(key=self._rank, reverse=True)
        return requests

    def _rank(self, request: _Request) -> tuple[float, int]:
        """Planning order: value, then a stable tie-break.

        Ranked by the value of a visit when it is placed, as selection uses:
        a cadence target's visits wait until they are due, so each has the
        full cadence value, not the part it has at the start of the horizon.
        """
        payload = f"{self._seed}:{request.target.obsid}".encode()
        tie_break = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "big"
        )
        return self._value(request, self._shares), tie_break

    def _with_allocated_time(self, requests: Sequence[_Request]) -> list[_Request]:
        """Requests to place: allocated time first, in its span, then the rest.

        Each span of a request's allocated time becomes a request for those
        seconds, limited to the span, its value raised by
        ``_allocated_factor``. Seconds too few for a snapshot stay with the
        rest of the target's exposure. Where the rest would otherwise be too
        few for a snapshot, a span takes them too, in one more snapshot if
        need be; that snapshot is placed whole or not at all, with the bonus
        for its allocated seconds spread over it. Allocated
        seconds that do not fit their span go back to the rest (see
        :meth:`_place_requests`).
        """
        self._rest_of = {}
        if self.allocated is None:
            return list(requests)
        placing = []
        for request in requests:
            rest = request.model_copy()
            left = request.remaining
            ss_min = float(snapshot_min_seconds(request.target))
            ss_max = float(request.target.ss_max)
            for span in self.allocated.get(int(request.target.obsid), []):
                allocated = min(span.seconds, left)
                if allocated <= 0.0 or allocated < ss_min:
                    continue
                # Pieces of the span's time: seconds, the allocated ones among
                # them, and whether they are placed whole.
                pieces = [(allocated, allocated, False)]
                if 0.0 < left - allocated < ss_min:
                    # What is left would be too short for a snapshot of its
                    # own. Take it all, as one snapshot so that cutting it
                    # short strands none, or, if that is longer than a
                    # snapshot, split off a snapshot's worth to take it with.
                    if left - ss_min >= ss_min:
                        first = left - ss_min
                        pieces = [
                            (first, first, False),
                            (ss_min, allocated - first, ss_min <= ss_max),
                        ]
                    else:
                        pieces = [(left, allocated, left <= ss_max)]
                for seconds, bonused, whole in pieces:
                    # The bonus is for the allocated seconds, spread over all.
                    factor = 1.0 + self.allocation_bonus * bonused / seconds
                    part = request.model_copy(
                        update={
                            "remaining": seconds,
                            "windows": [
                                (max(w0, span.begin), min(w1, span.end))
                                for w0, w1 in request.windows
                                if w0 < span.end and span.begin < w1
                            ],
                            "merit": request.merit.model_copy(
                                update={
                                    "allocation": (factor - 1.0) * request.merit.value,
                                }
                            ),
                            "steady_value": request.steady_value * factor,
                            "value_factor": factor,
                            "span": (span.begin, span.end),
                            "whole": whole,
                        }
                    )
                    self._rest_of[id(part)] = rest
                    placing.append(part)
                    left -= seconds
            rest.remaining = left
            placing.append(rest)
        # Stable, so a request's spans keep their time order.
        placing.sort(key=self._rank, reverse=True)
        return placing

    def _reserved_collection(self) -> dict[int, float]:
        """Return the collection seconds accepted locks already hold per request."""
        reserved: dict[int, float] = defaultdict(float)
        for block in self.timeline:
            if block.is_fixed:
                continue
            entry = block.entry
            reserved[int(entry.obsid)] += entry.collection_seconds_between(
                entry.begin, entry.end
            )
        return reserved

    # ── Timeline neighbours and connections ──────────────────────────────

    def _pred_at(self, index: int) -> _Block | None:
        """Return the block before timeline position ``index``.

        The first position follows the start state, if the plan has one.
        """
        return self.timeline[index - 1] if index > 0 else self._origin

    def _start_attitude(self, pred: _Block | None, utime: float) -> Attitude:
        """Return the attitude a slew starting at ``utime`` leaves from."""
        if pred is None:
            return self.ctx.initial_attitude(utime)
        return pred.attitude_out

    def _earliest_start(self, pred: _Block | None) -> float:
        """Return the first step a slew after ``pred`` can start."""
        if pred is None:
            return self.ctx.ustart
        return self.ctx.ceil_step(pred.end)

    def _idle_limit(self, pred: _Block | None, earliest: float, until: float) -> float:
        """Return the latest step a slew can start while ``pred``'s held attitude stays safe."""
        if pred is None:
            # Before its first command ACS keeps the initial attitude safe itself.
            return until
        violation = self.ctx.first_hold_violation(
            pred.attitude_out, earliest, until, ACSMode.IDLE
        )
        return until if violation is None else violation

    def _connect(
        self,
        pred: _Block | None,
        block: _Block,
        earliest: float,
        attitude_in: Attitude | None = None,
    ) -> Slew | None:
        """Find the slew into ``block`` after ``pred``, starting as late as allowed.

        Starting late keeps any wait at ``pred``'s attitude rather than at the
        block's. Returns None if no start from ``earliest`` reaches the block by
        its ready time with every check passing. A block that follows is
        reached by a slew of no length to the attitude it is at, which it
        must then hold through its length; where that attitude cannot be
        held, through the block or on through the idle time to the block
        after it, the block slews to the safe idle attitude of its start
        instead, the one the simulation holds before its first command.
        """
        slew = self._connect_to(pred, block, earliest, attitude_in)
        if slew is None and block.follows and attitude_in is None:
            slew = self._connect_to(
                pred, block, earliest, self.ctx.initial_attitude(block.ready)
            )
        return slew

    def _connect_to(
        self,
        pred: _Block | None,
        block: _Block,
        earliest: float,
        attitude_in: Attitude | None,
    ) -> Slew | None:
        """Find the slew into ``block`` after ``pred`` to ``attitude_in``, or to
        the block's own attitude, or, for a block that follows, the attitude
        it is at."""
        if block.is_pass:
            obstype = ObsType.GSP
        elif block.is_charge:
            obstype = ObsType.CHARGE
        elif block.is_activity:
            obstype = ObsType.IDLE
        else:
            obstype = ObsType.PPT
        idle_limit = self._idle_limit(pred, earliest, block.ready)

        start = self.ctx.floor_step(block.ready)
        while start >= earliest:
            if start > idle_limit:
                start -= self.ctx.step_size
                continue
            if attitude_in is not None:
                target = attitude_in
            elif block.follows:
                # The block takes the attitude it is at: no slew.
                target = self._start_attitude(pred, start)
            else:
                target = block.attitude_in
            slew = self.ctx.slew(
                self._start_attitude(pred, start),
                target,
                start,
                obstype=obstype,
                obsid=block.entry.obsid,
            )
            if slew.slewend > block.ready:
                start = min(
                    start - self.ctx.step_size,
                    self.ctx.floor_step(block.ready - slew.slewtime),
                )
                continue
            verdict = self._connection_verdict(block, slew, target)
            if verdict is _Verdict.OK:
                return slew
            if verdict is _Verdict.WAIT_NOT_ALLOWED:
                # Starting earlier only lengthens the wait that failed.
                return None
            start -= self.ctx.step_size
        return None

    def _connection_verdict(
        self, block: _Block, slew: Slew, target: Attitude
    ) -> "_Verdict":
        """Check a slew into ``block`` and the wait at its attitude before it starts."""
        mode = ACSMode.PASS if block.is_pass else ACSMode.SLEWING
        if self.ctx.first_slew_violation(slew, mode) is not None:
            return _Verdict.SLEW_NOT_ALLOWED
        if block.is_pass:
            # After the ingress slew, ACS idles on the track until contact.
            wait_violation = self.ctx.first_hold_violation(
                target,
                slew.slewend,
                self.ctx.ceil_step(block.ready),
                ACSMode.IDLE,
            )
        elif block.is_charge or block.is_activity:
            # A following block holds the attitude it is given through its
            # length too; a reserved one was checked when it was placed.
            until = block.end if block.follows else block.ready
            wait_violation = self.ctx.first_hold_violation(
                target, slew.slewend, self.ctx.ceil_step(until), ACSMode.IDLE
            )
        else:
            # ACS only flies a science slew the target is visible for.
            needed = self._visibility_needed_at(slew)
            if not block.entry.visible(needed, needed):
                return _Verdict.SLEW_NOT_ALLOWED
            wait_violation = self.ctx.first_science_violation(
                block.entry, target, slew.slewend, block.ready
            )
        return _Verdict.OK if wait_violation is None else _Verdict.WAIT_NOT_ALLOWED

    def _insertion_index(self, ready: float) -> int:
        """Return where a block ready at ``ready`` goes in the timeline."""
        for index, block in enumerate(self.timeline):
            if block.ready > ready:
                return index
        return len(self.timeline)

    def _insert_fixed(self, block: _Block) -> bool:
        """Insert a block whose timing is fixed, reconnecting its successor."""
        index = self._insertion_index(block.ready)
        pred = self._pred_at(index)
        succ = self.timeline[index] if index < len(self.timeline) else None
        if pred is not None and self.ctx.ceil_step(pred.end) > block.ready:
            return False
        slew = self._connect(pred, block, self._earliest_start(pred))
        if slew is None:
            return False
        succ_slew = None
        if succ is not None:
            reached = self._followed(block, slew) if block.follows else block
            chain = self._connect_chain(reached, self.timeline[index:])
            if chain is None:
                return False
            succ_slew = chain[0][1]
        self._commit(index, block, slew, succ, succ_slew)
        return True

    def _commit(
        self,
        index: int,
        block: _Block,
        slew: Slew,
        succ: _Block | None,
        succ_slew: Slew | None,
    ) -> None:
        """Insert ``block`` and record its slew and its successor's new slew.

        A successor that is a contact adds what its slew generates to the
        recorder; once the ledger exists, that is kept current here.
        """
        self._apply_slew(block, slew)
        if succ is not None and succ_slew is not None:
            if getattr(self, "_ledger", None) is not None:
                self._volumes = self._ledger.add_all(
                    self._volumes, self._ingress_deltas(succ, succ_slew)
                )
            self._apply_slew(succ, succ_slew)
        self.timeline.insert(index, block)
        if block.follows:
            self._reconnect_following(index)
        elif succ is not None and succ.follows:
            self._reconnect_following(index + 1)
        self._energy = self._charged_energy(self.timeline)

    @staticmethod
    def _apply_slew(block: _Block, slew: Slew) -> None:
        """Record a block's incoming slew on the block and its plan entry; a
        following block takes the attitude the slew ends at."""
        block.slew = slew
        entries = [block.entry, *block.steps[:1]]
        for entry in entries:
            entry.begin = float(slew.slewstart)
            entry.slewtime = int(slew.slewtime)
            entry.slewdist = float(slew.slewdist)
        if block.step_slews:
            block.step_slews[0] = slew
        if block.follows:
            attitude = (float(slew.endra), float(slew.enddec), float(slew.endroll))
            block.attitude_in = block.attitude_out = attitude
            block.entry.ra, block.entry.dec, block.entry.roll = attitude

    @staticmethod
    def _followed(block: _Block, slew: Slew) -> _Block:
        """A stand-in for a following ``block`` reached by ``slew``, at the
        attitude the slew ends at, for connecting the block after it."""
        attitude = (float(slew.endra), float(slew.enddec), float(slew.endroll))
        return block.model_copy(
            update={"attitude_in": attitude, "attitude_out": attitude, "slew": slew}
        )

    def _connect_chain(
        self, pred: _Block | None, after: Sequence[_Block]
    ) -> list[tuple[_Block, Slew]] | None:
        """The slews into the first of ``after`` from ``pred`` and, while the
        block reached follows, into the block after it, which the following
        block's new attitude must reach; None if any cannot be found."""
        chain: list[tuple[_Block, Slew]] = []
        for block in after:
            slew = self._connect(pred, block, self._earliest_start(pred))
            if slew is None:
                return None
            chain.append((block, slew))
            if not block.follows:
                break
            pred = self._followed(block, slew)
        return chain

    @staticmethod
    def _chain_led(beyond: Sequence[_Block]) -> list[_Block]:
        """The blocks a following block's attitude reaches: those after it
        up to and including the first that does not follow."""
        led: list[_Block] = []
        for block in beyond:
            led.append(block)
            if not block.follows:
                break
        return led

    def _reconnected(
        self, pred: _Block | None, after: Sequence[_Block], first: Slew | None
    ) -> list[_Block]:
        """The blocks ``after`` ``pred`` with the slews a following chain
        needs, on copies: the first takes ``first``, each following block
        the attitude it is given, and the block after a following one a
        slew from it; the rest are as they are."""
        if not after:
            return []
        result: list[_Block] = []
        index = 0
        for index, block in enumerate(after):
            if index == 0:
                slew = first
                if slew is None and block.follows:
                    slew = self._connect(pred, block, self._earliest_start(pred))
            else:
                slew = self._connect(pred, block, self._earliest_start(pred))
            if slew is None and index > 0:
                raise RuntimeError(
                    f"The block after {pred.entry.obsid if pred else 'the start'} "
                    "cannot be reached from the attitude it follows"
                )
            copy = block.model_copy(update={"entry": block.entry.model_copy()})
            if slew is not None:
                self._apply_slew(copy, slew)
            result.append(copy)
            if not copy.follows:
                break
            pred = copy
        result.extend(after[index + 1 :])
        return result

    def _reconnect_following(self, index: int) -> None:
        """Re-derive, in place, the slews into the blocks after timeline
        position ``index`` while the block before each follows."""
        while index + 1 < len(self.timeline) and self.timeline[index].follows:
            pred, block = self.timeline[index], self.timeline[index + 1]
            slew = self._connect(pred, block, self._earliest_start(pred))
            if slew is None:
                raise RuntimeError(
                    f"Block {block.entry.obsid} cannot be reached from the "
                    f"attitude activity {pred.entry.obsid} follows"
                )
            self._apply_slew(block, slew)
            index += 1

    # ── Locked entries and passes ────────────────────────────────────────

    def _place_locked(self, entry: PlanEntry) -> None:
        """Keep a locked science entry's collection window, recomputing its slew."""
        if entry.collection_begin is None or entry.collection_end is None:
            raise ValueError(
                f"Locked entry {entry.obsid} needs a collection window to keep"
            )
        if (
            float(entry.collection_begin) < self.ctx.ustart
            or float(entry.end) > self.ctx.uend
        ):
            self._log(
                float(entry.begin),
                f"Locked entry {entry.obsid} is outside the planning horizon; "
                "leaving it out",
                entry.obsid,
            )
            return
        if float(entry.collection_end) + self.ctx.post_collection_for(
            int(entry.obsid)
        ) > float(entry.end):
            self._log(
                float(entry.begin),
                f"Locked entry {entry.obsid} does not reserve its post-collection "
                "time; leaving it out",
                entry.obsid,
            )
            return
        Plan(entries=[entry]).bind_runtime(self.config, self.ctx.ephem)
        attitude = entry.target_body_attitude()
        block = _Block(
            entry=entry,
            attitude_in=attitude,
            attitude_out=attitude,
            ready=float(entry.collection_begin) - self.ctx.setup_for(int(entry.obsid)),
            end=float(entry.end),
        )
        entry.visibility()
        # The slew in only reaches the attitude; the lock must also hold it for
        # its whole setup, collection and cleanup budget.
        violation = self.ctx.first_science_violation(
            entry, attitude, block.ready, self.ctx.ceil_step(block.end)
        )
        if violation is not None:
            self._log(
                violation,
                f"Locked entry {entry.obsid} cannot hold its attitude at "
                f"{unixtime2date(violation)}; leaving it out",
                entry.obsid,
            )
            return
        if not self._insert_fixed(block):
            self._log(
                float(entry.begin),
                f"Locked entry {entry.obsid} cannot be reached; leaving it out",
                entry.obsid,
            )

    def _settle_sliding_activities(self, timeline: Sequence[_Block]) -> None:
        """Hold each activity that may begin late, within a slide window or
        after another activity, from when what precedes it on the timeline
        has ended: at its start at the earliest and where it was reserved at
        the latest, for its length."""
        for index, block in enumerate(timeline):
            activity = self._slidable.get(id(block))
            if activity is None:
                continue
            begin = activity.begin
            if index > 0:
                previous = timeline[index - 1]
                # A settled activity before it ends where its entry does.
                ended = (
                    float(previous.entry.end)
                    if id(previous) in self._slidable
                    else float(previous.end)
                )
                begin = max(begin, self.ctx.ceil_step(ended))
            begin = min(begin, block.ready)
            block.entry.begin = begin
            block.entry.end = begin + activity.duration_seconds

    def _activity_ready(self, activity: ScheduledActivity) -> float:
        """When an activity is reserved from: its latest start, or after
        each activity already on the timeline that is under way then, as
        one due while another is under way begins when that one ends."""
        ready = activity.latest_start
        for block in self.timeline:
            if (
                block.is_activity
                and block.ready < ready + activity.duration_seconds
                and block.end > ready
            ):
                ready = max(ready, self.ctx.ceil_step(float(block.end)))
        return ready

    def _place_activity(self, activity: ScheduledActivity) -> None:
        """Reserve an activity as a fixed block in IDLE mode.

        It holds the attitude of whatever precedes it: as reserved, the
        block before it or the start attitude, and later the science placed
        before it (see :attr:`_Block.follows`). An attitude IDLE cannot hold
        for the activity's length is not used; the plan then leaves the
        activity out and says so.
        """
        ready = self._activity_ready(activity)
        end = ready + activity.duration_seconds
        if ready > activity.latest_start:
            self._log(
                activity.begin,
                f"Activity {activity.name} is due during another activity and "
                f"begins when it ends, by {unixtime2date(ready)}",
                activity.obsid,
            )
        before = [b for b in self.timeline if b.end <= ready]
        attitude = (
            before[-1].attitude_out
            if before
            else (
                self._origin.attitude_out
                if self._origin is not None
                else self.ctx.initial_attitude(self.ctx.ustart)
            )
        )
        mode = ACSMode.DESAT if activity.desaturation else ACSMode.IDLE
        violation = self.ctx.first_hold_violation(
            attitude, activity.begin, self.ctx.ceil_step(end), mode
        )
        if violation is not None:
            self._log(
                activity.begin,
                f"Activity {activity.name} cannot hold its attitude in {mode.name} at "
                f"{unixtime2date(violation)}; leaving it out",
                activity.obsid,
            )
            return
        entry = activity.plan_entry(self.config, attitude)
        # An activity that may slide, or that waits for another, is reserved
        # from its latest start to its latest end, so science before it may
        # run into its window and nothing is placed where it may still be;
        # its entry begins after what precedes it, at its start at the
        # earliest, when the plan is finished.
        late = activity.slides or ready > activity.latest_start
        if late:
            entry.end = end
        block = _Block(
            entry=entry,
            attitude_in=attitude,
            attitude_out=attitude,
            ready=ready,
            end=end,
            follows=True,
        )
        if late:
            self._slidable[id(block)] = activity
        if not self._insert_fixed(block):
            self._log(
                activity.begin,
                f"Activity {activity.name} cannot be reached; leaving it out",
                activity.obsid,
            )

    def _place_pass(self, gspass: Pass) -> None:
        """Reserve a ground pass at its earliest reachable acquisition step.

        Each step inside the pass is tried in turn, on every tracking profile,
        so a pass whose first step is blocked can still be acquired later.
        """
        acquisitions = self._acquisition_steps(gspass)
        if not acquisitions:
            self._log(
                gspass.begin,
                f"Pass at {gspass.station} has no simulation step left to acquire "
                "it on; not reserved",
                gspass.obsid,
            )
            return
        profiles = gspass.available_tracking_profiles()
        for acquire in acquisitions:
            for profile in profiles:
                block = self._pass_block(gspass, profile, acquire)
                if block is not None and self._insert_fixed(block):
                    gspass.select_tracking_profile(profile)
                    return
        self._log(
            gspass.begin,
            f"Pass at {gspass.station} cannot be reached at any acquisition step "
            "on any safe tracking profile; not reserved",
            gspass.obsid,
        )

    def _acquisition_steps(self, gspass: Pass) -> list[float]:
        """Return the steps a contact could be acquired on, earliest first.

        Contact can only start on a step, where the spacecraft must already
        hold the tracking attitude of that moment. A pass beginning between
        steps is therefore acquired late, and the time before that is lost.
        """
        last_sample = float(gspass.utime[-1]) if gspass.utime else float(gspass.end)
        end = min(float(gspass.end), last_sample)
        return [float(step) for step in self.ctx.steps(gspass.begin, end)]

    def _pass_block(
        self, gspass: Pass, profile: TrackingProfile, acquire: float
    ) -> _Block | None:
        """Build the plan entry and timeline block for a ground contact.

        Returns None if the profile cannot be tracked within the motion limits.
        """
        rest = self.ctx.tracking_rest(gspass, profile, acquire)
        if rest is None:
            return None
        rest_time, rest_attitude = rest
        start = gspass.attitude_for_profile_at(profile, acquire)
        if start is None:
            return None
        finish = profile[-1]
        entry = PlanEntry(config=self.config)
        entry.name = f"{gspass.station}_PASS"
        entry.ra, entry.dec, entry.roll = start
        entry.begin = acquire
        entry.end = gspass.end
        entry.obsid = gspass.obsid
        entry.obstype = ObsType.GSP
        entry.ss_min = 0
        duration = max(0, int(round(gspass.end - acquire)))
        entry.ss_max = duration
        entry.exptime = duration
        entry.station = gspass.station
        try:
            station = self.config.ground_stations.get(gspass.station)
            entry.station_lat_deg = float(station.latitude_deg)
            entry.station_lon_deg = float(station.longitude_deg)
            entry.station_alt_m = float(station.elevation_m)
        except KeyError:
            pass
        # A contact only earns the time from acquisition, but it is still the
        # whole predicted pass: its end identifies which one.
        entry.contact_begin = acquire
        entry.contact_end = gspass.end
        entry.track_start_ra, entry.track_start_dec, entry.track_start_roll = start
        entry.track_end_ra, entry.track_end_dec, entry.track_end_roll = finish
        return _Block(
            entry=entry,
            attitude_in=start,
            # Tracking brakes past the last sample, so the contact leaves the
            # spacecraft at rest later, and elsewhere, than the profile ends.
            attitude_out=rest_attitude,
            ready=acquire,
            end=max(float(gspass.end), rest_time),
            gspass=gspass,
        )

    # ── Science snapshots ────────────────────────────────────────────────

    def _place_snapshot(
        self, request: _Request, not_before: float | None = None
    ) -> _Block | None:
        """Place one snapshot of a request in the earliest slot that fits.

        Its slew starts no earlier than ``not_before``, if given. A snapshot
        goes only where the recorder has room for what it generates, before
        the next contact and after; it is never shortened to fit.
        """
        recorder_full = battery_low = False
        for index in range(len(self.timeline) + 1):
            pred = self._pred_at(index)
            succ = self.timeline[index] if index < len(self.timeline) else None

            # First try the gap as it is. Where the recorder has no room
            # before the next contact, nothing earlier in the gap fits
            # either; a pass that requires no spacecraft tracking
            # can fall within the gap, so the gap is tried again after it.
            after = not_before
            beyond = self.timeline[index + 1 :]
            while True:
                candidate = self._fit_in_gap(request, pred, succ, after, beyond=beyond)
                if candidate is None:
                    full = False
                    break
                block, slew, succ_slew = candidate
                volume = self._block_volume(block)
                segment = self._block_segment(block)
                # Include any change to the next tracking pass's ingress.
                additions = [
                    (segment, volume),
                    *self._ingress_deltas(succ, succ_slew),
                ]
                if self._ledger.fits_all(self._volumes, additions):
                    full = False
                    break
                recorder_full = full = True
                if segment + 1 >= self._ledger.segments:
                    break
                after = self._ledger.bounds[segment + 1]
                if succ is not None and after >= succ.ready:
                    break
            if full:
                # Nothing before the gap's last contact fits: try later gaps.
                continue
            if candidate is not None:
                charged = self._fit_within_battery(
                    request, pred, succ, after, candidate, self.timeline, index
                )
                if charged is None:
                    # A later gap is after more sunlit idle time, which charges.
                    battery_low = True
                    continue
                block, slew, succ_slew = charged
                volume = self._block_volume(block)
                segment = self._block_segment(block)
                additions = [
                    (segment, volume),
                    *self._ingress_deltas(succ, succ_slew),
                ]
                if not self._ledger.fits_all(self._volumes, additions):
                    recorder_full = True
                    continue
                # _commit adds the contact's change itself.
                self._commit(index, block, slew, succ, succ_slew)
                self._volumes = self._ledger.add(self._volumes, segment, volume)
            elif self._can_delay(succ):
                # The snapshot fitted, but only the one after it was then
                # unreachable: try again with that one, and any packed behind
                # it, moved later.
                delayed = self._fit_delaying_successors(request, index, after)
                if delayed is None:
                    continue
                block = delayed
            else:
                continue

            self._movable[id(block)] = request
            self._log(
                block.entry.begin,
                f"Placed {request.target.obsid} at "
                f"{unixtime2date(float(block.entry.collection_begin or 0))} "
                f"for {block.entry.exposure}s: {request.merit.describe()}",
                request.target.obsid,
            )
            return block
        if recorder_full:
            snapshot = min(float(request.target.ss_max), request.remaining)
            self._log(
                self.ctx.ustart,
                f"No room on the recorder for {request.target.obsid}: a snapshot "
                f"generates {self._ledger.science_volume(request.target, snapshot):.2f} Gb",
                request.target.obsid,
            )
        if battery_low:
            self._log(
                self.ctx.ustart,
                f"Not enough charge for {request.target.obsid}: a snapshot would "
                f"take the battery below {self.config.battery.planning_level:.0%}",
                request.target.obsid,
            )
        return None

    def _fit_within_battery(
        self,
        request: _Request,
        pred: _Block | None,
        succ: _Block | None,
        not_before: float | None,
        candidate: tuple[_Block, Slew, Slew | None],
        timeline: Sequence[_Block],
        index: int,
    ) -> tuple[_Block, Slew, Slew | None] | None:
        """Keep a fitted snapshot where the battery stays above its level.

        The snapshot is checked in the timeline it would join at ``index``,
        with the activity after it re-slewed from it. If it would take the
        battery below the level there, it is tried again later in the same
        gap, after the charging time its idle sunlit start then leaves, as
        long as that could cover the shortfall. Returns the fit, or None.
        """
        if not self._battery.active:
            return candidate
        until = succ.ready if succ is not None else self.ctx.uend
        target = request.target
        snapshot = min(float(target.ss_max), request.remaining)
        # The latest a slew can start and still fit the whole snapshot before
        # ``until``, with the longest slew; a later gap holds it otherwise.
        latest = self.ctx.floor_step(
            until
            - snapshot
            - self.ctx.timing_for(int(target.obsid)).total_seconds
            - self._battery.slew_allowance
        )
        for _ in range(_BATTERY_RETRIES + 1):
            block, _, succ_slew = candidate
            trial = self._charged_energy(
                self._trial_timeline(timeline, index, block, succ, succ_slew)
            )
            if self._within_level(trial):
                return candidate
            addition = self._block_addition(block)
            segment = self._battery.segment(addition.begin)
            limit = self._battery.limit
            assert limit is not None
            excess = max(self._battery.peaks(trial)[segment:]) - limit
            later = self._battery.time_to_fit(
                self._charged_energy(timeline), addition, until, excess
            )
            if later is None:
                return None
            later = min(later, latest)
            if later <= addition.begin:
                return None
            retry = self._fit_in_gap(
                request, pred, succ, later, beyond=timeline[index + 1 :]
            )
            if retry is None:
                return None
            candidate = retry
        return None

    def _trial_timeline(
        self,
        timeline: Sequence[_Block],
        index: int,
        block: _Block,
        succ: _Block | None,
        succ_slew: Slew | None,
    ) -> list[_Block]:
        """The timeline with ``block`` at ``index`` and its successor re-slewed,
        with the chain a following successor leads, on copies, so nothing is
        committed."""
        trial = [*timeline[:index], block]
        if succ is not None:
            trial.extend(self._reconnected(block, timeline[index:], succ_slew))
        return trial

    def _within_level(self, energy: Sequence[Additions]) -> bool:
        """Whether a timeline's energy keeps the battery above its level throughout."""
        limit = self._battery.limit
        if limit is None:
            return True
        return all(peak <= limit + 1e-9 for peak in self._battery.peaks(energy))

    def _can_delay(self, succ: _Block | None) -> bool:
        """Whether the last gap search failed only because of ``succ``, and it may move."""
        return succ is not None and self._successor_blocked and self._may_move(succ)

    def _may_move(self, block: _Block) -> bool:
        """Whether a block may be placed again later to make room before it.

        Only science snapshots this scheduler placed may move. A cadence
        target's visit may move only if it is the target's latest, as its
        next visit is spaced from it; visits are placed in time order, so it
        usually is. Locked entries and passes never move.
        """
        request = self._movable.get(id(block))
        if request is None:
            return False
        if request.cadence is None and not request.timed:
            return True
        end = block.entry.collection_end
        return end is not None and self._last_visit.get(
            int(request.target.obsid)
        ) == float(end)

    def _fit_delaying_successors(
        self, request: _Request, index: int, not_before: float | None
    ) -> _Block | None:
        """Fit a snapshot before timeline block ``index`` by moving that block later.

        This handles a gap the snapshot fits, but only if the science snapshot
        after it could start later. Typically that snapshot starts the moment
        its target rises, and the spacecraft could only be there on time by
        staying on the target while it was hidden: anything else placed in
        that time would get there late. Without this, the whole time the
        target was hidden is lost.

        The later snapshot is moved by as little as it can be. When it is
        packed against the snapshots after it, those move too: first the one
        snapshot is tried, then it and the next, and so on, up to
        ``_MAX_DELAYED`` snapshots or the first that may not move. Returns the
        new snapshot's block, or None with the timeline unchanged.
        """
        for count in range(1, _MAX_DELAYED + 1):
            run = self.timeline[index : index + count]
            if len(run) < count or not self._may_move(run[-1]):
                return None
            block = self._fit_delaying_run(request, index, count, not_before)
            if block is not None:
                return block
        return None

    def _fit_delaying_run(
        self, request: _Request, index: int, count: int, not_before: float | None
    ) -> _Block | None:
        """Fit a snapshot at timeline position ``index``, moving the ``count``
        snapshots there later.

        Moving a snapshot may cut it short, where its target's visibility ends
        sooner after the move. That is accepted only if the trade is worth it:
        the new snapshot's value-weighted seconds exceed those lost. Returns the new
        snapshot's block, or None with the timeline unchanged.
        """
        pred = self._pred_at(index)
        run = self.timeline[index : index + count]
        end = index + count
        after = self.timeline[end] if end < len(self.timeline) else None
        # Everything placed here must be done before the block after the run.
        until = after.ready if after is not None else None

        # 1. Fit the new snapshot as if the run were not there. The run follows
        #    it, so it need not reach the block after the run itself.
        new_fit = self._fit_in_gap(request, pred, None, not_before, until)
        if new_fit is None:
            return None
        new_block, new_slew, _ = new_fit
        gained = _collection_seconds(new_block)

        # 2. Fit each snapshot of the run again, in order, after the new one,
        #    for up to the collection time it had. Only the last must reach the
        #    block after the run. Nothing changes until all fit.
        moves: list[_Move] = []
        after_slew = None
        previous = new_block
        for position, old_block in enumerate(run):
            old_request = self._movable[id(old_block)]
            seconds = _collection_seconds(old_block)
            same_length = old_request.model_copy(
                update={"remaining": seconds, "visit": old_block.entry.visit}
            )
            last = position == len(run) - 1
            self._refitting = old_block
            try:
                moved_fit = self._fit_in_gap(
                    same_length,
                    previous,
                    after if last else None,
                    # A cadence visit starts no earlier than it did, which
                    # kept it a cadence after the target's previous visit.
                    not_before=float(old_block.entry.begin)
                    if old_request.cadence is not None
                    else None,
                    until=None if last else until,
                    beyond=self.timeline[end + 1 :] if last else (),
                )
            finally:
                self._refitting = None
            if moved_fit is None:
                return None
            moved_block, moved_slew, after_slew = moved_fit
            lost = max(0.0, seconds - _collection_seconds(moved_block))
            moves.append(_Move(old_block, old_request, moved_block, moved_slew, lost))
            previous = moved_block

        # 3. Check the trade, if moving cut anything short, and that the
        #    recorder has room for what the new snapshot adds, less what the
        #    moves cut. The run and the new snapshot are all before the same
        #    contact, so they share a segment.
        if not self._worth_cutting(request, gained, moves):
            return None
        segment = self._block_segment(new_block)
        extra = self._block_volume(new_block) - sum(
            self._block_volume(move.old_block) - self._block_volume(move.new_block)
            for move in moves
        )
        # With what the slew into a contact after the run comes to.
        additions = [(segment, extra), *self._ingress_deltas(after, after_slew)]
        if not self._ledger.fits_all(self._volumes, additions):
            return None
        trial = [*self.timeline[:index], new_block, *(m.new_block for m in moves)]
        if after is not None:
            trial.extend(self._reconnected(previous, self.timeline[end:], after_slew))
        if not self._within_level(self._charged_energy(trial)):
            return None
        self._volumes = self._ledger.add_all(self._volumes, additions)

        # 4. Record the new slews and put the blocks in place.
        self._apply_slew(new_block, new_slew)
        for move in moves:
            self._apply_slew(move.new_block, move.slew)
        if after is not None and after_slew is not None:
            self._apply_slew(after, after_slew)

        self.timeline[index:end] = [new_block] + [move.new_block for move in moves]
        if after is not None and after.follows:
            self._reconnect_following(index + 1 + len(moves))
        self._energy = self._charged_energy(self.timeline)

        for move in moves:
            # A moved snapshot is a new block, so it takes the old one's place
            # among the blocks that may move again.
            del self._movable[id(move.old_block)]
            self._movable[id(move.new_block)] = move.request
            # It keeps the value it was selected at.
            move.new_block.entry.merit = move.old_block.entry.merit
            moved_end = move.new_block.entry.collection_end
            if move.request.cadence is not None and moved_end is not None:
                # Only a cadence target's latest visit moves (see
                # _may_move): its next visit is spaced from the new end.
                obsid = int(move.request.target.obsid)
                self._last_visit[obsid] = float(moved_end)
                visits = self._planned_visits[obsid]
                visits[:] = [
                    move.new_block if block is move.old_block else block
                    for block in visits
                ]
            # Exposure cut short is left for the request to plan elsewhere,
            # and no longer counts towards its program's share.
            move.request.remaining += move.lost
            program = move.request.category.program_name
            self._program_seconds[program] -= move.lost
            if move.lost:
                self._cut_short.append(move.request)
            moved = move.new_block.entry
            self._log(
                float(moved.begin),
                f"Moved {move.request.target.obsid} to "
                f"{unixtime2date(float(moved.collection_begin or 0))}"
                + (f", {move.lost:.0f}s shorter," if move.lost else "")
                + f" to fit {request.target.obsid} before it",
                move.request.target.obsid,
            )
        return new_block

    @staticmethod
    def _worth_cutting(
        request: _Request, gained: float, moves: Sequence["_Move"]
    ) -> bool:
        """Whether a new snapshot is worth the science its moves cut short."""
        cut = [move for move in moves if move.lost > 1e-6]
        if not cut:
            return True
        lost_value = sum(move.lost * move.request.steady_value for move in cut)
        return gained * request.steady_value > lost_value

    def _successor_slew(
        self, fitted: _Block, succ: _Block, beyond: Sequence[_Block]
    ) -> Slew | None:
        """The slew from a fitted snapshot into its successor, found with
        the chain the successor leads if it follows; None if unreachable."""
        chain = self._connect_chain(fitted, [succ, *beyond])
        return chain[0][1] if chain is not None else None

    def _fit_in_gap(
        self,
        request: _Request,
        pred: _Block | None,
        succ: _Block | None,
        not_before: float | None = None,
        until: float | None = None,
        beyond: Sequence[_Block] = (),
    ) -> tuple[_Block, Slew, Slew | None] | None:
        """Find the earliest snapshot of ``request`` between ``pred`` and ``succ``.

        The slew starts no earlier than ``not_before``, if given. Without a
        ``succ``, the snapshot ends by ``until``, if given, rather than the end
        of the horizon. ``beyond`` holds the blocks after ``succ``: a
        successor that follows takes the snapshot's attitude, so the block
        after it must be reachable from there too.
        """
        self._successor_blocked = False
        target = request.pointing
        timing = self.ctx.timing_for(int(target.obsid))
        ss_min = (
            request.remaining if request.whole else float(snapshot_min_seconds(target))
        )
        earliest = self._earliest_start(pred)
        horizon_end = self.ctx.uend if until is None else min(until, self.ctx.uend)
        gap_end = succ.ready if succ is not None else horizon_end
        start = earliest
        if not_before is not None:
            start = max(start, self.ctx.ceil_step(not_before))
        if gap_end - start < ss_min:
            return None
        idle_limit = self._idle_limit(pred, earliest, gap_end)
        snapshot = min(float(target.ss_max), request.remaining)
        # Starts at which the snapshot fitted but its successor was unreachable.
        unreachable = 0
        search_end = min(idle_limit, gap_end)

        # A visit arrives at its first step, which may point elsewhere than
        # the target and ask for its own roll.
        arrival_pointing = (
            target.step_pointing(0) if target.sequence is not None else target
        )
        while start <= search_end:
            if not self.ctx.roll_allowed(arrival_pointing, start):
                # The roll the target asks for is beyond the ACS limit from
                # the power-optimal roll now; it may be within it later.
                start += self.ctx.step_size
                continue
            instrument_roll = self.ctx.instrument_roll(arrival_pointing, start)
            attitude = arrival_pointing.target_body_attitude(instrument_roll)
            slew = self.ctx.slew(
                self._start_attitude(pred, start),
                attitude,
                start,
                obsid=target.obsid,
            )

            # The target must be visible when ACS needs it to be. If it is
            # not, start again when it next becomes visible, less the slew
            # time if the slew may start before then.
            needed = self._visibility_needed_at(slew)
            window = next((w for w in request.windows if w[0] <= needed < w[1]), None)
            if window is None:
                upcoming = [w[0] for w in request.windows if w[0] > needed]
                if not upcoming:
                    return None
                lead = needed - float(slew.slewstart)
                start = max(
                    start + self.ctx.step_size,
                    self.ctx.ceil_step(min(upcoming) - lead),
                )
                continue

            collection_begin = float(slew.slewend) + timing.setup_seconds
            earliest_start = request.earliest_start
            if earliest_start is not None and collection_begin < earliest_start:
                # Start later by the shortfall, so collection begins at the
                # earliest start if the slew takes as long. It need not: its
                # duration changes with the instrument roll, so a longer slew
                # there can leave this step too late for a deadline that an
                # earlier step meets. So step back while the step before
                # still begins collection by the earliest start. No earlier
                # start can then do so: a slew's duration changes by far less
                # than a step from one step to the next, so collection begins
                # later the later the slew starts.
                jump = max(
                    start + self.ctx.step_size,
                    self.ctx.ceil_step(start + earliest_start - collection_begin),
                )
                jump = min(jump, self.ctx.floor_step(search_end) + self.ctx.step_size)
                while (
                    jump - self.ctx.step_size > start
                    and self._collection_begin_at(
                        target, pred, jump - self.ctx.step_size
                    )
                    >= earliest_start
                ):
                    jump -= self.ctx.step_size
                start = jump
                continue
            deadline = request.deadline
            if deadline is not None and collection_begin > deadline:
                return None

            visit: _Visit | None = None
            departure = attitude
            if target.sequence is not None:
                visit = self._visit(
                    request, slew, attitude, instrument_roll, collection_begin
                )
                if visit is None:
                    # A step cannot be observed from this start; try later.
                    start += self.ctx.step_size
                    continue
                departure = visit.attitude_out

            latest_end = min(window[1], horizon_end)
            succ_slewtime = 0.0
            if succ is not None:
                if not succ.follows:
                    # A following successor is reached by no slew.
                    succ_slewtime = self.ctx.slew(
                        departure, succ.attitude_in, collection_begin
                    ).slewtime
                # The successor's slew must start on a step after this ends.
                latest_end = min(
                    latest_end, self.ctx.floor_step(succ.ready - succ_slewtime)
                )
            if visit is not None:
                # A visit runs whole or not at all.
                if visit.end > latest_end:
                    if succ is not None and latest_end < window[1]:
                        return None
                    start += self.ctx.step_size
                    continue
                block = _Block(
                    entry=visit.entry,
                    attitude_in=attitude,
                    attitude_out=visit.attitude_out,
                    ready=collection_begin - timing.setup_seconds,
                    end=visit.end,
                    steps=visit.steps,
                    step_slews=visit.slews,
                )
                fitted = self._check_snapshot(block, slew, ss_min)
                if fitted is not None:
                    succ_slew = None
                    if succ is not None:
                        succ_slew = self._successor_slew(fitted, succ, beyond)
                    if succ is None or succ_slew is not None:
                        return fitted, slew, succ_slew
                    unreachable += 1
                    self._successor_blocked = True
                    if (
                        self.successor_retries is not None
                        and unreachable > self.successor_retries
                    ):
                        return None
                start += self.ctx.step_size
                continue
            collection_end = min(
                collection_begin + snapshot,
                latest_end - timing.post_collection_seconds,
            )

            # A cadence visit before a locked visit of its target must end a
            # cadence before that visit starts, so try a shorter snapshot that
            # does. If even the shortest is too long, try again a cadence after
            # the locked visit.
            locked = self._next_locked_visit(request, start)
            if locked is not None:
                assert request.cadence is not None
                locked_start, locked_end = locked
                cadence_end = locked_start - request.cadence
                if not _long_enough(cadence_end - collection_begin, ss_min):
                    start = max(
                        start + self.ctx.step_size,
                        self.ctx.ceil_step(locked_end + request.cadence),
                    )
                    continue
                collection_end = min(collection_end, cadence_end)

            if not _long_enough(collection_end - collection_begin, ss_min):
                if succ is not None and latest_end < window[1]:
                    # Starting later only leaves less room before the successor.
                    return None
                start += self.ctx.step_size
                continue

            entry = self._science_entry(
                request,
                instrument_roll,
                attitude,
                slew,
                collection_begin,
                collection_end,
            )
            block = _Block(
                entry=entry,
                attitude_in=attitude,
                attitude_out=attitude,
                ready=collection_begin - timing.setup_seconds,
                end=float(entry.end),
            )
            fitted = self._check_snapshot(block, slew, ss_min)
            if fitted is not None:
                succ_slew = None
                if succ is not None:
                    succ_slew = self._successor_slew(fitted, succ, beyond)
                if succ is None or succ_slew is not None:
                    return fitted, slew, succ_slew
                unreachable += 1
                self._successor_blocked = True
                if (
                    self.successor_retries is not None
                    and unreachable > self.successor_retries
                ):
                    return None
            start += self.ctx.step_size
        return None

    def _collection_begin_at(
        self, target: Pointing, pred: _Block | None, start: float
    ) -> float:
        """Return when collection of ``target`` begins if the slew to it after
        ``pred`` starts at ``start``."""
        attitude = target.target_body_attitude(self.ctx.instrument_roll(target, start))
        slew = self.ctx.slew(
            self._start_attitude(pred, start), attitude, start, obsid=target.obsid
        )
        return float(slew.slewend) + self.ctx.setup_for(int(target.obsid))

    def _visibility_needed_at(self, slew: Slew) -> float:
        """When a science target must be visible for ACS to fly ``slew`` to it.

        ACS starts the slew only once the target is visible, unless it may
        slew before then (``slew_before_visible``), when the target need only
        be visible on arrival.
        """
        if self.config.spacecraft_bus.attitude_control.slew_before_visible:
            return float(slew.slewend)
        return float(slew.slewstart)

    def _check_snapshot(
        self, block: _Block, slew: Slew, ss_min: float
    ) -> _Block | None:
        """Check a snapshot's slew and observation, shortening it to fit if needed."""
        entry = block.entry
        if self.ctx.first_slew_violation(slew, ACSMode.SLEWING) is not None:
            return None
        entry.visibility()
        needed = self._visibility_needed_at(slew)
        if not entry.visible(needed, needed):
            return None
        if block.steps:
            # Each step's hold was checked when the visit was built.
            return block
        violation = self.ctx.first_science_violation(
            entry, block.attitude_in, slew.slewend, self.ctx.ceil_step(block.end)
        )
        if violation is None:
            return block
        # End the observation at the step the attitude stops being allowed.
        assert entry.collection_begin is not None
        collection_end = violation - self.ctx.post_collection_for(int(entry.obsid))
        if not _long_enough(collection_end - entry.collection_begin, ss_min):
            return None
        entry.collection_end = collection_end
        entry.end = violation
        block.end = violation
        return block

    def _visit(
        self,
        request: _Request,
        slew: Slew,
        attitude: Attitude,
        instrument_roll: float,
        collection_begin: float,
    ) -> "_Visit | None":
        """Build a visit of a sequence target whose first step's collection
        begins at ``collection_begin``, after ``slew``.

        Each later step follows the one before: a slew where it points
        elsewhere, the setup budget where the step asks, then its duration;
        cleanup and handoff after the last. Each step must be visible and
        allowed throughout, and each slew between steps allowed, else None.
        A visit that runs past the end of the horizon is None too, checked
        before anything is sampled past it: the horizon can end with the
        ephemeris, which has nothing beyond.
        """
        target = request.pointing
        assert target.sequence is not None
        sequence = target.sequence
        # The visit's own budgets, as dispatch reserves them: a step that
        # asks for setup pays the target's, and the last step its cleanup.
        timing = self.ctx.timing_for(int(target.obsid))
        steps: list[PlanEntry] = []
        slews: list[Slew | None] = []
        cursor = collection_begin
        previous = attitude
        previous_roll = instrument_roll
        limit = self.config.spacecraft_bus.attitude_control.max_roll_from_optimum_deg
        first_pointing = target.step_pointing(0)
        previous_optimum = self.ctx.optimum_roll(first_pointing, float(slew.slewstart))
        step_slew: Slew | None = slew
        for index, step in enumerate(sequence.steps):
            pointing = target.step_pointing(index)
            if index > 0:
                # A step at the pointing of the one before keeps the
                # power-optimal roll found there, so its own roll offset is
                # a roll-only slew and no offset leaves it where it is,
                # though ACS still settles as for any commanded slew; one
                # elsewhere slews to its own attitude.
                same_pointing = sequence.pointing_of(
                    index, target.ra, target.dec
                ) == sequence.pointing_of(index - 1, target.ra, target.dec)
                if not same_pointing:
                    previous_optimum = self.ctx.optimum_roll(pointing, cursor)
                requested = pointing.requested_roll(previous_optimum, limit)
                if requested is None:
                    # The step's roll is beyond the ACS limit now.
                    return None
                step_roll = float(requested)
                step_attitude = (
                    previous
                    if same_pointing and abs(step_roll - previous_roll) < 1e-9
                    else pointing.target_body_attitude(step_roll)
                )
                # Commands execute on simulation steps: the slew starts on the
                # first step after the collection before it ends.
                slew_begin = self.ctx.ceil_step(cursor)
                step_slew = self.ctx.slew(
                    previous, step_attitude, slew_begin, obsid=target.obsid
                )
                if float(step_slew.slewend) > self.ctx.uend:
                    return None
                if self.ctx.first_slew_violation(step_slew, ACSMode.SLEWING):
                    return None
                cursor = float(step_slew.slewend)
                if step.setup:
                    cursor += timing.setup_seconds
                begin, slewtime = slew_begin, float(step_slew.slewtime)
            else:
                step_attitude = attitude
                step_roll = instrument_roll
                begin, slewtime = float(slew.slewstart), float(slew.slewtime)
            step_begin = cursor
            step_end = cursor + step.duration_seconds
            last = index == len(sequence.steps) - 1
            end = step_end + (timing.post_collection_seconds if last else 0.0)
            if end > self.ctx.uend or not pointing.visible(step_begin, step_end):
                return None
            if (
                self.ctx.first_science_violation(
                    pointing, step_attitude, step_begin, self.ctx.ceil_step(end)
                )
                is not None
            ):
                return None
            entry = PlanEntry(
                config=self.config,
                name=pointing.name,
                instrument_name=pointing.instrument_name,
                instrument_modes=dict(pointing.instrument_modes),
                science_data=pointing.science_data,
                own_timing=target.own_timing,
                ra=pointing.ra,
                dec=pointing.dec,
                roll=step_roll,
                spacecraft_attitude=(
                    step_attitude if pointing.uses_mounted_attitude() else None
                ),
                begin=begin,
                slewtime=int(slewtime),
                slewdist=float(step_slew.slewdist) if step_slew is not None else 0.0,
                end=end,
                collection_begin=step_begin,
                collection_end=step_end,
                obsid=target.obsid,
                obstype=target.obstype,
                merit=request.merit.value,
                ss_min=0,
                ss_max=step.duration_seconds,
                step=step.name,
                calibration=not step.science,
            )
            entry.exptime = step.duration_seconds
            steps.append(entry)
            slews.append(step_slew)
            previous = step_attitude
            previous_roll = step_roll
            cursor = step_end
        first = steps[0]
        visit_entry = PlanEntry(
            config=self.config,
            name=target.name,
            instrument_name=first.instrument_name,
            instrument_modes=dict(target.instrument_modes),
            science_data=target.science_data,
            own_timing=target.own_timing,
            ra=first.ra,
            dec=first.dec,
            roll=first.roll,
            spacecraft_attitude=first.spacecraft_attitude,
            begin=float(slew.slewstart),
            slewtime=int(slew.slewtime),
            slewdist=float(slew.slewdist),
            end=steps[-1].end,
            collection_begin=collection_begin,
            collection_end=steps[-1].collection_end,
            obsid=target.obsid,
            obstype=target.obstype,
            merit=request.merit.value,
            ss_min=snapshot_min_seconds(target),
            ss_max=target.ss_max,
        )
        visit_entry.exptime = sequence.science_seconds
        return _Visit(visit_entry, steps, slews, previous, float(steps[-1].end))

    def _science_entry(
        self,
        request: _Request,
        instrument_roll: float,
        attitude: Attitude,
        slew: Slew,
        collection_begin: float,
        collection_end: float,
    ) -> PlanEntry:
        """Build the plan entry for one science snapshot."""
        target = request.pointing
        mounted = target.uses_mounted_attitude()
        telescope = target.science_telescope()
        end = collection_end + self.ctx.post_collection_for(int(target.obsid))
        entry = PlanEntry(
            config=self.config,
            name=target.name,
            instrument_name=telescope.name if telescope is not None else None,
            instrument_modes=dict(target.instrument_modes),
            own_timing=target.own_timing,
            science_data=target.science_data,
            ra=target.ra,
            dec=target.dec,
            roll=instrument_roll,
            spacecraft_attitude=attitude if mounted else None,
            begin=float(slew.slewstart),
            slewtime=int(slew.slewtime),
            slewdist=float(slew.slewdist),
            end=end,
            collection_begin=collection_begin,
            collection_end=collection_end,
            obsid=target.obsid,
            obstype=target.obstype,
            merit=request.merit.value,
            ss_min=snapshot_min_seconds(target),
            ss_max=target.ss_max,
            visit=request.visit if request.timed else None,
        )
        entry.exptime = collection_end - collection_begin
        return entry

    # ── Charging ─────────────────────────────────────────────────────────

    def _plan_charging(self, timeline: list[_Block]) -> list[_Block]:
        """Put the charges a finished plan relies on into it.

        Every placement was checked with the charges its idle sunlit time
        holds (see :meth:`_charged_timeline`); this commits them: each charge
        goes into the plan, and the activity after it is re-slewed from the
        charging attitude.
        """
        if not self._battery.active:
            return timeline
        charged = self._charged_timeline(timeline, commit=True)
        for block in charged:
            if block.is_charge:
                self._log(
                    float(block.entry.begin),
                    f"Charging at {unixtime2date(block.ready)} for "
                    f"{block.end - block.ready:.0f}s, putting back "
                    f"{self._battery.net_charge(block.attitude_in, block.ready, block.end):.0f} Wh",
                    block.entry.obsid,
                )
        violation = self._battery.first_over_limit(self._timeline_energy(charged))
        if violation is not None:
            self._log(
                violation,
                "The battery goes below its planning level at "
                f"{unixtime2date(violation)}",
            )
        return charged

    def _charged_timeline(
        self, timeline: Sequence[_Block], commit: bool = False
    ) -> list[_Block]:
        """The timeline with a charge in every idle sunlit stretch one fits.

        Between each pair of blocks, charges are chained through the sunlit
        stretches of the idle time (see :meth:`_gap_charges`), and the block
        after them is re-slewed from the last charging attitude. Unless
        ``commit`` is set, that re-slew goes on a copy, so the timeline given
        is not changed.
        """
        if not self._battery.active:
            return list(timeline)
        charged: list[_Block] = []
        # Where the solar-array drive is as the timeline goes: a charge's
        # attitude depends on it. Idle time holds the drive, so only the
        # blocks move it.
        angles = self._battery.drive_start
        stale = False
        """Whether the block before the gap follows and was given a new
        attitude here, so the block after it must be re-slewed from it."""
        for index in range(len(timeline) + 1):
            pred = timeline[index - 1] if index > 0 else self._origin
            if stale:
                # The block as it is here, on its copy where nothing is committed.
                pred = charged[-1]
            succ = timeline[index] if index < len(timeline) else None
            charges, succ_slew = self._gap_charges(
                pred, succ, angles, timeline[index + 1 :]
            )
            charged.extend(charges)
            for charge in charges:
                angles = self._angles_after(charge, angles)
            if succ is None:
                continue
            if not charges and stale:
                assert pred is not None
                succ_slew = self._connect(pred, succ, self._earliest_start(pred))
                if succ_slew is None:
                    raise RuntimeError(
                        f"Block {succ.entry.obsid} cannot be reached from the "
                        f"attitude activity {pred.entry.obsid} follows"
                    )
            if (charges or stale) and succ_slew is not None:
                if commit:
                    self._apply_slew(succ, succ_slew)
                else:
                    succ = succ.model_copy(update={"entry": succ.entry.model_copy()})
                    self._apply_slew(succ, succ_slew)
            stale = bool(charges or stale) and succ.follows
            charged.append(succ)
            angles = self._angles_after(succ, angles)
        return charged

    def _angles_after(self, block: _Block, angles: Angles) -> Angles:
        """Where the solar-array drive is after ``block``, reached at ``angles``."""
        if not self._battery.driven:
            return angles
        return self._battery.driven_deltas(self._block_addition(block), angles)[1]

    def _gap_charges(
        self,
        pred: _Block | None,
        succ: _Block | None,
        angles: Angles,
        beyond: Sequence[_Block] = (),
    ) -> tuple[list[_Block], Slew | None]:
        """The charges that fit between ``pred`` and ``succ``, and the slew that
        then takes ``succ`` from the last of them.

        Charges are chained through the gap's sunlit stretches, each slewing
        from the block before it, at the charging attitude for where the
        solar-array drive is then (``angles`` as the gap starts, and where
        each charge leaves it after). The last must leave ``succ`` reachable
        from its charging attitude; where it does not, it is shortened, and
        failing that dropped and the one before it tried, until the chain
        connects. A ``succ`` that follows takes the charging attitude, so the
        first block it leads to in ``beyond`` must be reachable from there as
        well. Keyed on what the gap depends on, since the same gap is seen
        again and again as placements are tried around it.
        """
        ledger = self._battery
        led: tuple[tuple[int, float, Attitude], ...] = ()
        if succ is not None and succ.follows:
            led = tuple(
                (id(block), block.ready, block.attitude_in)
                for block in self._chain_led(beyond)
            )
        key = (
            id(pred) if pred is not None else None,
            pred.end if pred is not None else None,
            pred.attitude_out if pred is not None else None,
            id(succ) if succ is not None else None,
            succ.ready if succ is not None else None,
            succ.attitude_in if succ is not None else None,
            led,
            tuple(None if a is None else round(a, 3) for a in angles)
            if ledger.driven
            else (),
        )
        cached = self._gap_charges_cache.get(key)
        if cached is not None:
            return cached
        idle_begin = self._earliest_start(pred)
        idle_end = succ.ready if succ is not None else self.ctx.uend
        chain: list[tuple[_Block, float, float]] = []
        last = pred
        for span_begin, span_end, stretch in ledger.charges_in(idle_begin, idle_end):
            if stretch.attitude is None or last is not None and span_end <= last.end:
                continue
            start = max(span_begin, self._earliest_start(last))
            attitude = (
                ledger.charging_attitude(start, angles)
                if ledger.driven
                else stretch.attitude
            )
            if attitude is None:
                continue
            placed = self._charge_block(last, None, attitude, start, span_end)
            if placed is None:
                continue
            chain.append((placed[0], span_begin, span_end))
            last = placed[0]
            angles = self._angles_after(last, angles)
        succ_slew: Slew | None = None
        while chain and succ is not None:
            block, span_begin, span_end = chain[-1]
            succ_slew = self._successor_slew(block, succ, beyond)
            if succ_slew is not None:
                break
            chain.pop()
            before = chain[-1][0] if chain else pred
            shorter = self._charge_block(
                before,
                succ,
                block.attitude_in,
                max(span_begin, self._earliest_start(before)),
                span_end,
                beyond,
            )
            if shorter is not None:
                chain.append((shorter[0], span_begin, span_end))
                succ_slew = shorter[1]
                break
        result = ([block for block, _, _ in chain], succ_slew)
        self._gap_charges_cache[key] = result
        return result

    def _charge_block(
        self,
        pred: _Block | None,
        succ: _Block | None,
        attitude: Attitude,
        span_begin: float,
        span_end: float,
        beyond: Sequence[_Block] = (),
    ) -> tuple[_Block, Slew | None] | None:
        """Build a charge in a sunlit idle span, or None if none fits.

        The charge slews from ``pred`` as soon as it can, holds the charging
        attitude while it is allowed until the span ends, and ends early
        enough for ``succ`` to be reached from the charging attitude.
        """
        ledger = self._battery
        start = self.ctx.ceil_step(span_begin)
        slew = self.ctx.slew(
            self._start_attitude(pred, start),
            attitude,
            start,
            obstype=ObsType.CHARGE,
            obsid=self._next_charge_obsid,
        )
        if self.ctx.first_slew_violation(slew, ACSMode.SLEWING) is not None:
            return None
        hold_begin = self.ctx.ceil_step(float(slew.slewend))
        latest = self.ctx.floor_step(span_end - ledger.slew_allowance)
        violation = self.ctx.first_hold_violation(
            attitude, hold_begin, latest, ACSMode.CHARGING
        )
        if violation is not None:
            latest = min(latest, violation)
        hold_end = latest
        while hold_end - hold_begin >= MIN_CHARGE_SECONDS:
            block = self._charge_entry(attitude, slew, hold_begin, hold_end)
            if succ is None:
                return block, None
            succ_slew = self._successor_slew(block, succ, beyond)
            if succ_slew is not None:
                return block, succ_slew
            hold_end -= self.ctx.step_size
        return None

    def _charge_entry(
        self, attitude: Attitude, slew: Slew, hold_begin: float, hold_end: float
    ) -> _Block:
        """The plan entry and block of a charge holding ``attitude``."""
        ra, dec, roll = attitude
        duration = max(0, int(round(hold_end - hold_begin)))
        entry = PlanEntry(
            config=self.config,
            name="CHARGE",
            ra=ra,
            dec=dec,
            roll=roll,
            begin=float(slew.slewstart),
            slewtime=int(slew.slewtime),
            slewdist=float(slew.slewdist),
            end=hold_end,
            obsid=self._next_charge_obsid,
            obstype=ObsType.CHARGE,
            ss_min=0,
            ss_max=duration,
        )
        entry.exptime = duration
        self._next_charge_obsid += 1
        return _Block(
            entry=entry,
            attitude_in=attitude,
            attitude_out=attitude,
            ready=hold_begin,
            end=hold_end,
            slew=slew,
        )

    def _log(self, utime: float, description: str, obsid: int | None = None) -> None:
        self.log.log_event(
            utime=utime,
            event_type="SCHEDULER",
            description=description,
            obsid=obsid,
            acs_mode=None,
        )
