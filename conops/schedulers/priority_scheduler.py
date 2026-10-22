"""Priority-first construction scheduler.

:class:`PriorityScheduler` reserves ground contacts first, then takes requests
in priority order and places each snapshot in the earliest slot that still fits,
without moving anything already placed. It is fast and deterministic, and its
plans are traceable to the priority order. It is myopic: an early placement can
block a better packing.
"""

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from enum import Enum, auto
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from ..common import ACSMode, ObsType, unixtime2date
from ..config import MissionConfig
from ..config.observation_categories import ObservationCategory
from ..config.solar_panel import SolarArrayDriveState
from ..ditl.ditl_log import DITLLog
from ..ditl.plan_validator import entry_obstype
from ..simulation.coverage import StationCoverage
from ..simulation.passes import Pass
from ..simulation.slew import Slew
from ..targets import Plan, PlanEntry, Pointing
from ..targets.merit import MeritBreakdown, MeritModel
from ..targets.plan_entry import snapshot_min_seconds
from .allocator import AllocatedTime
from .battery import MIN_CHARGE_SECONDS, Addition, Additions, Angles, BatteryLedger
from .context import Attitude, SchedulingContext
from .recorder import RecorderLedger

CHARGE_OBSID = 999000
"""Obsid of the first planned charge; the rest follow."""

TrackingProfile = list[tuple[float, float, float]]


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
    on what the plan has collected before it."""
    cadence: float | None = None
    """Seconds a visit waits after the target's last one, when cadence counts."""

    @property
    def wants_more(self) -> bool:
        """Whether enough exposure is left for another snapshot."""
        return self.remaining > 0 and self.remaining >= float(
            snapshot_min_seconds(self.target)
        )


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
       its last visit, planned or collected; each visit then has the full
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
        include_passes: Whether to reserve ground-station passes.
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
        allocated: Time allocated to requests, by obsid, such as from
            :meth:`~conops.schedulers.Allocation.allocated`. A request's
            allocated seconds in each span are planned in that span, worth
            ``allocation_bonus`` more; collection starting in the span counts.
            The rest of its exposure, and any allocated seconds that do not fit
            their span, are planned like an unallocated request, which may work
            ahead. None (the default) treats all requests alike.
        allocation_bonus: Fraction of a request's value its allocated time
            gains, so plans trade keeping to the allocation against merit.
        start_volume_gb: What is on the recorder when the horizon starts; the
            configured recorder's volume if None.
        start_deficit_wh: How far below full the battery is, in Wh, when the
            horizon starts; the configured battery's charge if None.
        start_drive_state: Where the solar-array drive is when the horizon
            starts; its configured initial state if None.
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
        allocated: Mapping[int, Sequence[AllocatedTime]] | None = None,
        allocation_bonus: float = 0.5,
        start_volume_gb: float | None = None,
        start_deficit_wh: float | None = None,
        start_drive_state: SolarArrayDriveState | None = None,
        passes: Sequence[Pass] | None = None,
        coverage: StationCoverage | None = None,
    ) -> None:
        self.config = config
        self.targets = list(targets)
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
        """Station coverage for an antenna that needs no attitude, where the
        caller has it; built from the configuration otherwise."""
        self.start_state = start_state
        self.reserved_seconds = dict(reserved_seconds or {})
        self.reserved_visits = dict(reserved_visits or {})
        if successor_retries is not None and successor_retries < 0:
            raise ValueError("successor_retries must not be negative")
        self.successor_retries = successor_retries
        if allocation_bonus < 0.0:
            raise ValueError("allocation_bonus must not be negative")
        self.allocated = (
            None
            if allocated is None
            else {int(obsid): list(spans) for obsid, spans in allocated.items()}
        )
        self.allocation_bonus = allocation_bonus
        self.start_volume_gb = start_volume_gb
        """What is on the recorder when the horizon starts; the configured
        recorder's volume if None."""
        self.start_deficit_wh = start_deficit_wh
        """How far below full the battery is when the horizon starts; the
        configured battery's charge if None."""
        self.unplaced: list[Pointing] = []
        """Targets left with at least ``ss_min`` of exposure unplanned."""
        self.plan = Plan()
        self._rest_of: dict[int, _Request] = {}
        """For each request for allocated time, by id, the request for the
        rest of its target's exposure."""

    # ── Public API ────────────────────────────────────────────────────────

    def schedule(self) -> Plan:
        """Build and return the plan."""
        self._reserve_fixed()
        self._place_requests(self._with_allocated_time(self._requests()))
        return self._finish(self.timeline)

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
        self._battery = self._battery_ledger()
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
                self._place_pass(gspass)
        self._ledger = self._recorder_ledger()
        self._volumes = self._ledger.initial_volumes(
            [block.entry for block in self.timeline]
        )
        """Science volume planned in each of the ledger's segments."""
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
            coverage=self.ctx.coverage,
        )

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
        self.plan = Plan(
            entries=[
                step for block in timeline for step in (block.steps or [block.entry])
            ],
            metadata={
                "scheduler": self.scheduler_name,
                "step_size": self.ctx.step_size,
            },
        )
        return self.plan

    def _place_requests(self, requests: Sequence[_Request]) -> None:
        """Place snapshots of the highest-ranked request at their earliest fit.

        Without a completion deficit the ranking never changes, so each
        request is placed in full before the next.
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
            request = active[0]
            if self._balancing:
                request = max(
                    active,
                    key=lambda r: (
                        self._value(r, shares),
                        -order[id(r)],
                    ),
                )
            block = self._place_snapshot(request, self._cadence_release(request))
            self._reactivate_cut_short(active, order)
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
        again. ``active`` stays in planning order. This always ends: a cut is
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
        rest.remaining += request.remaining
        request.remaining = 0.0
        if (
            active is not None
            and all(r is not rest for r in active)
            and rest.wants_more
        ):
            active.append(rest)
            active.sort(key=lambda r: order[id(r)])

    def _value(self, request: _Request, shares: Mapping[str, float]) -> float:
        """Value of a snapshot of ``request`` when the plan delivers ``shares``."""
        if not self._deficit_weight:
            return request.steady_value
        return request.steady_value + self._deficit_weight * (
            MeritModel.completion_deficit(request.category, shares)
        )

    def _cadence_release(self, request: _Request) -> float | None:
        """Earliest slew start of the request's next visit, if cadence spaces them."""
        if request.cadence is None:
            return None
        last = self._last_visit.get(int(request.target.obsid))
        return None if last is None else last + request.cadence

    def _next_locked_visit(
        self, request: _Request, start: float
    ) -> tuple[float, float] | None:
        """Return the first locked visit a cadence visit starting at ``start``
        must end before.

        Visits are spaced by the cadence: a visit must start at least a
        cadence after a locked visit's collection ends, or end at least a
        cadence before its slew starts. Returns that locked visit's slew start
        and collection end, or None if the request has no cadence or
        ``start`` is clear of every locked visit.
        """
        if request.cadence is None:
            return None
        visits = sorted(self._locked_visits.get(int(request.target.obsid), ()))
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
        if request.cadence is not None and entry.collection_end is not None:
            self._last_visit[int(request.target.obsid)] = float(entry.collection_end)

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
            visits = [
                t
                for t in (target.last_collection_time, self.reserved_visits.get(obsid))
                if t is not None
            ]
            if visits:
                self._last_visit[obsid] = max(visits)
        shares = program_shares(self._delivered)
        cadence_weight = merit_model.cadence_weight
        requests = []
        for target in self.targets:
            exptime = target.exptime if target.exptime is not None else target.ss_max
            remaining = float(exptime) - reserved.get(int(target.obsid), 0.0)
            if (
                target.done
                or int(target.obsid) in self.held
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
                )
            )
        self._deficit_weight = merit_model.completion_deficit_weight
        self._balancing = self._deficit_weight > 0.0 and any(
            r.category.time_share is not None for r in requests
        )

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
        rest of the target's exposure. Allocated
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
            for span in self.allocated.get(int(request.target.obsid), []):
                seconds = min(span.seconds, left)
                if seconds <= 0.0 or seconds < float(
                    snapshot_min_seconds(request.target)
                ):
                    continue
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
                                "allocation": (self._allocated_factor - 1.0)
                                * request.merit.value,
                            }
                        ),
                        "steady_value": request.steady_value * self._allocated_factor,
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
            if block.is_pass or block.is_charge:
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
        its ready time with every check passing.
        """
        target = attitude_in if attitude_in is not None else block.attitude_in
        if block.is_pass:
            obstype = ObsType.GSP
        elif block.is_charge:
            obstype = ObsType.CHARGE
        else:
            obstype = ObsType.PPT
        idle_limit = self._idle_limit(pred, earliest, block.ready)

        start = self.ctx.floor_step(block.ready)
        while start >= earliest:
            if start > idle_limit:
                start -= self.ctx.step_size
                continue
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
        elif block.is_charge:
            wait_violation = self.ctx.first_hold_violation(
                target, slew.slewend, self.ctx.ceil_step(block.ready), ACSMode.IDLE
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
            succ_slew = self._connect(block, succ, self.ctx.ceil_step(block.end))
            if succ_slew is None:
                return False
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
        """Insert ``block`` and record its slew and its successor's new slew."""
        self._apply_slew(block, slew)
        if succ is not None and succ_slew is not None:
            self._apply_slew(succ, succ_slew)
        self.timeline.insert(index, block)
        self._energy = self._charged_energy(self.timeline)

    @staticmethod
    def _apply_slew(block: _Block, slew: Slew) -> None:
        """Record a block's incoming slew on the block and its plan entry."""
        block.slew = slew
        entries = [block.entry, *block.steps[:1]]
        for entry in entries:
            entry.begin = float(slew.slewstart)
            entry.slewtime = int(slew.slewtime)
            entry.slewdist = float(slew.slewdist)
        if block.step_slews:
            block.step_slews[0] = slew

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
        if float(entry.collection_end) + self.ctx.post_collection_seconds > float(
            entry.end
        ):
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
            ready=float(entry.collection_begin) - self.ctx.setup_seconds,
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
            # either; a contact that is no block on the timeline (an antenna
            # that needs no attitude downlinks in whatever mode ACS is in)
            # can fall within the gap, so the gap is tried again after it.
            after = not_before
            while True:
                candidate = self._fit_in_gap(request, pred, succ, after)
                if candidate is None:
                    full = False
                    break
                block, slew, succ_slew = candidate
                volume = self._block_volume(block)
                segment = self._block_segment(block)
                if self._ledger.fits(self._volumes, segment, volume):
                    full = False
                    break
                recorder_full = full = True
                if segment + 1 >= self._ledger.segments:
                    break
                after = self._ledger.bounds[segment + 1]
                if succ is not None and after >= succ.ready:
                    break
            if full:
                # Nothing before the gap's last contact fits: try the gaps after it.
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
                if not self._ledger.fits(self._volumes, segment, volume):
                    recorder_full = True
                    continue
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
            - self.ctx.setup_seconds
            - self.ctx.post_collection_seconds
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
            retry = self._fit_in_gap(request, pred, succ, later)
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
        on copies, so nothing is committed."""
        trial = [*timeline[:index], block]
        if succ is not None:
            moved = succ.model_copy(update={"entry": succ.entry.model_copy()})
            if succ_slew is not None:
                self._apply_slew(moved, succ_slew)
            trial.append(moved)
            trial.extend(timeline[index + 1 :])
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
        if request.cadence is None:
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
            same_length = old_request.model_copy(update={"remaining": seconds})
            last = position == len(run) - 1
            moved_fit = self._fit_in_gap(
                same_length,
                previous,
                after if last else None,
                # A cadence visit starts no earlier than it did, which kept
                # it a cadence after the target's previous visit.
                not_before=float(old_block.entry.begin)
                if old_request.cadence is not None
                else None,
                until=None if last else until,
            )
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
        if not self._ledger.fits(self._volumes, segment, extra):
            return None
        trial = [*self.timeline[:index], new_block, *(m.new_block for m in moves)]
        if after is not None:
            moved_after = after.model_copy(update={"entry": after.entry.model_copy()})
            if after_slew is not None:
                self._apply_slew(moved_after, after_slew)
            trial.append(moved_after)
            trial.extend(self.timeline[end + 1 :])
        if not self._within_level(self._charged_energy(trial)):
            return None
        self._volumes = self._ledger.add(self._volumes, segment, extra)

        # 4. Record the new slews and put the blocks in place.
        self._apply_slew(new_block, new_slew)
        for move in moves:
            self._apply_slew(move.new_block, move.slew)
        if after is not None and after_slew is not None:
            self._apply_slew(after, after_slew)

        self.timeline[index:end] = [new_block] + [move.new_block for move in moves]
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
                self._last_visit[int(move.request.target.obsid)] = float(moved_end)
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

    def _fit_in_gap(
        self,
        request: _Request,
        pred: _Block | None,
        succ: _Block | None,
        not_before: float | None = None,
        until: float | None = None,
    ) -> tuple[_Block, Slew, Slew | None] | None:
        """Find the earliest snapshot of ``request`` between ``pred`` and ``succ``.

        The slew starts no earlier than ``not_before``, if given. Without a
        ``succ``, the snapshot ends by ``until``, if given, rather than the end
        of the horizon.
        """
        self._successor_blocked = False
        target = request.target
        ss_min = float(snapshot_min_seconds(target))
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

        while start <= search_end:
            instrument_roll = self.ctx.instrument_roll(target, start)
            attitude = target.target_body_attitude(instrument_roll)
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

            collection_begin = float(slew.slewend) + self.ctx.setup_seconds
            if (
                target.earliest_start is not None
                and collection_begin < target.earliest_start
            ):
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
                    self.ctx.ceil_step(
                        start + target.earliest_start - collection_begin
                    ),
                )
                jump = min(jump, self.ctx.floor_step(search_end) + self.ctx.step_size)
                while (
                    jump - self.ctx.step_size > start
                    and self._collection_begin_at(
                        target, pred, jump - self.ctx.step_size
                    )
                    >= target.earliest_start
                ):
                    jump -= self.ctx.step_size
                start = jump
                continue
            if target.deadline is not None and collection_begin > target.deadline:
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
                    ready=collection_begin - self.ctx.setup_seconds,
                    end=visit.end,
                    steps=visit.steps,
                    step_slews=visit.slews,
                )
                fitted = self._check_snapshot(block, slew, ss_min)
                if fitted is not None:
                    succ_slew = None
                    if succ is not None:
                        succ_slew = self._connect(
                            fitted, succ, self.ctx.ceil_step(fitted.end)
                        )
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
                latest_end - self.ctx.post_collection_seconds,
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
                ready=collection_begin - self.ctx.setup_seconds,
                end=float(entry.end),
            )
            fitted = self._check_snapshot(block, slew, ss_min)
            if fitted is not None:
                succ_slew = None
                if succ is not None:
                    succ_slew = self._connect(
                        fitted, succ, self.ctx.ceil_step(fitted.end)
                    )
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
        return float(slew.slewend) + self.ctx.setup_seconds

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
        collection_end = violation - self.ctx.post_collection_seconds
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
        """
        target = request.target
        assert target.sequence is not None
        sequence = target.sequence
        steps: list[PlanEntry] = []
        slews: list[Slew | None] = []
        cursor = collection_begin
        previous = attitude
        previous_roll = instrument_roll
        step_slew: Slew | None = slew
        for index, step in enumerate(sequence.steps):
            pointing = target.step_pointing(index)
            if index > 0:
                # A step at the pointing of the one before stays put, at the
                # attitude already held, though ACS still settles as it does
                # for any commanded slew; one elsewhere slews to its own.
                same_pointing = sequence.pointing_of(
                    index, target.ra, target.dec
                ) == sequence.pointing_of(index - 1, target.ra, target.dec)
                step_roll = (
                    previous_roll
                    if same_pointing
                    else self.ctx.instrument_roll(pointing, cursor)
                )
                step_attitude = (
                    previous
                    if same_pointing
                    else pointing.target_body_attitude(step_roll)
                )
                # Commands execute on simulation steps: the slew starts on the
                # first step after the collection before it ends.
                slew_begin = self.ctx.ceil_step(cursor)
                step_slew = self.ctx.slew(
                    previous, step_attitude, slew_begin, obsid=target.obsid
                )
                if self.ctx.first_slew_violation(step_slew, ACSMode.SLEWING):
                    return None
                cursor = float(step_slew.slewend)
                if step.setup:
                    cursor += self.ctx.setup_seconds
                begin, slewtime = slew_begin, float(step_slew.slewtime)
            else:
                step_attitude = attitude
                step_roll = instrument_roll
                begin, slewtime = float(slew.slewstart), float(slew.slewtime)
            step_begin = cursor
            step_end = cursor + step.duration_seconds
            last = index == len(sequence.steps) - 1
            end = step_end + (self.ctx.post_collection_seconds if last else 0.0)
            if not pointing.visible(step_begin, step_end):
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
                science_data=pointing.science_data,
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
            science_data=target.science_data,
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
        target = request.target
        mounted = target.uses_mounted_attitude()
        telescope = target.science_telescope()
        end = collection_end + self.ctx.post_collection_seconds
        entry = PlanEntry(
            config=self.config,
            name=target.name,
            instrument_name=telescope.name if telescope is not None else None,
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
        for index in range(len(timeline) + 1):
            pred = timeline[index - 1] if index > 0 else self._origin
            succ = timeline[index] if index < len(timeline) else None
            charges, succ_slew = self._gap_charges(pred, succ, angles)
            charged.extend(charges)
            for charge in charges:
                angles = self._angles_after(charge, angles)
            if succ is None:
                continue
            if charges and succ_slew is not None:
                if commit:
                    self._apply_slew(succ, succ_slew)
                else:
                    succ = succ.model_copy(update={"entry": succ.entry.model_copy()})
                    self._apply_slew(succ, succ_slew)
            charged.append(succ)
            angles = self._angles_after(succ, angles)
        return charged

    def _angles_after(self, block: _Block, angles: Angles) -> Angles:
        """Where the solar-array drive is after ``block``, reached at ``angles``."""
        if not self._battery.driven:
            return angles
        return self._battery.driven_deltas(self._block_addition(block), angles)[1]

    def _gap_charges(
        self, pred: _Block | None, succ: _Block | None, angles: Angles
    ) -> tuple[list[_Block], Slew | None]:
        """The charges that fit between ``pred`` and ``succ``, and the slew that
        then takes ``succ`` from the last of them.

        Charges are chained through the gap's sunlit stretches, each slewing
        from the block before it, at the charging attitude for where the
        solar-array drive is then (``angles`` as the gap starts, and where
        each charge leaves it after). The last must leave ``succ`` reachable
        from its charging attitude; where it does not, it is shortened, and
        failing that dropped and the one before it tried, until the chain
        connects. Keyed on what the gap depends on, since the same gap is
        seen again and again as placements are tried around it.
        """
        ledger = self._battery
        key = (
            id(pred) if pred is not None else None,
            pred.end if pred is not None else None,
            pred.attitude_out if pred is not None else None,
            id(succ) if succ is not None else None,
            succ.ready if succ is not None else None,
            succ.attitude_in if succ is not None else None,
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
            succ_slew = self._connect(block, succ, self._earliest_start(block))
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
            succ_slew = self._connect(block, succ, self.ctx.ceil_step(hold_end))
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
