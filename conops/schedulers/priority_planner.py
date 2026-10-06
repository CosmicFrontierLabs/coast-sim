"""Priority-first construction planner.

:class:`PriorityPlanner` reserves ground contacts first, then takes requests
in priority order and places each snapshot in the earliest slot that still fits,
without moving anything already placed. It is fast and deterministic, and its
plans are traceable to the priority order. It is myopic: an early placement can
block a better packing.
"""

import hashlib
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from enum import Enum, auto

from pydantic import BaseModel, ConfigDict, Field

from ..common import ACSMode, ObsType, unixtime2date
from ..config import AllocationStrictness, MissionConfig
from ..config.observation_categories import ObservationCategory
from ..ditl.ditl_log import DITLLog
from ..simulation.passes import Pass
from ..simulation.slew import Slew
from ..targets import Plan, PlanEntry, Pointing
from ..targets.merit import MeritBreakdown, MeritModel
from .allocator import AllocatedTime
from .context import Attitude, SchedulingContext

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

    @property
    def is_pass(self) -> bool:
        return self.gspass is not None

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
        time: First time (Unix seconds) a new activity may start; the planner
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
    """Merit at the start of the horizon; its tier and ranking."""
    remaining: float
    windows: list[tuple[float, float]] = Field(default_factory=list)
    category: ObservationCategory
    steady_value: float = 0.0
    """Value of a snapshot apart from the completion deficit, which depends
    on what the plan has collected before it."""
    cadence: float | None = None
    """Seconds a visit waits after the target's last one, when cadence counts."""


def program_shares(seconds: Mapping[str, float]) -> dict[str, float]:
    """Return each program's fraction of the science seconds given."""
    total = sum(seconds.values())
    if total <= 0.0:
        return {}
    return {program: value / total for program, value in seconds.items()}


class PriorityPlanner:
    """Build a plan by placing requests in priority order at their earliest fit.

    Steps:

    1. Locked entries keep their collection windows and are placed first.
    2. Ground passes the spacecraft can reach are reserved.
    3. Requests are sorted by tier, then by merit value at the start of the
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
       target is visible when the slew starts, collection starts no earlier
       than the target's earliest start and by its deadline, and the following
       activity can still be reached.

    The plan is built for execution by :class:`~conops.ditl.DITL` with the same
    configuration, ephemeris, horizon and step size. The planner does not model
    battery state, so it does not schedule charging.

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
        start_state: Time and attitude the plan starts from, when it takes
            over from activities already committed. Without it the plan
            starts at ``begin`` from a freshly started spacecraft.
        reserved_seconds: Exposure, by obsid, already scheduled outside this
            plan and not yet collected; it is subtracted from each target's
            remaining exposure and counts towards its program's share.
        reserved_visits: When observations scheduled outside this plan and not
            yet collected will finish collecting, by obsid; a cadence target's
            next visit waits for its cadence after them.
        successor_retries: When a snapshot fits a gap but the activity after
            it cannot then be reached, how many later starts to try in that gap
            before giving up on it (default 3). This plans much faster when many
            requests cannot be placed, but can miss a fit late in a gap. None
            tries every start, which finds every fit.
        preferred: Obsids to plan ahead of the other requests in their tier,
            anywhere in the horizon; the others fill the time left. Tiers still
            come first. None (the default) treats all requests alike.
        allocated: Time allocated to requests, by obsid, such as from
            :meth:`~conops.schedulers.Allocation.allocated`. A request's
            allocated seconds in each span are planned in that span, ranked by
            ``allocation_strictness``; collection starting in the span counts.
            The rest of its exposure, and any allocated seconds that do not fit
            their span, are planned like an unallocated request, which may work
            ahead. Not with ``preferred``.
        allocation_strictness: How allocated time ranks against unallocated
            work: ``strict`` (the default) ahead of all of it, whatever its
            tier; ``tier`` ahead of the unallocated work in its own tier;
            ``weighted`` in its own tier, with ``allocation_bonus`` of its
            value added (see :data:`~conops.config.AllocationStrictness`).
        allocation_bonus: Fraction of a request's value its allocated time
            gains with ``weighted`` strictness.
    """

    planner_name = "priority"
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
        successor_retries: int | None = 3,
        preferred: Collection[int] | None = None,
        allocated: Mapping[int, Sequence[AllocatedTime]] | None = None,
        allocation_strictness: AllocationStrictness = "strict",
        allocation_bonus: float = 0.5,
    ) -> None:
        self.config = config
        self.targets = list(targets)
        self.begin = begin
        self.end = end
        self.step_size = step_size
        self.include_passes = include_passes
        self.locked = list(locked)
        self.log = log if log is not None else DITLLog()
        self.simulation_start = simulation_start
        self.simulation_end = simulation_end
        self.start_state = start_state
        self.reserved_seconds = dict(reserved_seconds or {})
        self.reserved_visits = dict(reserved_visits or {})
        if successor_retries is not None and successor_retries < 0:
            raise ValueError("successor_retries must not be negative")
        self.successor_retries = successor_retries
        if preferred is not None and allocated is not None:
            raise ValueError("give preferred or allocated, not both")
        if allocation_strictness not in ("strict", "tier", "weighted"):
            raise ValueError(
                'allocation_strictness must be "strict", "tier" or "weighted"'
            )
        if allocation_bonus < 0.0:
            raise ValueError("allocation_bonus must not be negative")
        self.allocation_strictness = allocation_strictness
        self.allocation_bonus = allocation_bonus
        self.preferred = None if preferred is None else frozenset(preferred)
        self.allocated = (
            None
            if allocated is None
            else {int(obsid): list(spans) for obsid, spans in allocated.items()}
        )
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
        )
        self.timeline: list[_Block] = []
        self._movable: dict[int, _Request] = {}
        """Science blocks placed by this planner that may move later, by id,
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
        if self.include_passes:
            for gspass in self.ctx.predict_passes():
                self._place_pass(gspass)

    def _place_requests(self, requests: Sequence[_Request]) -> None:
        """Place snapshots of the highest-ranked request at their earliest fit.

        Without a completion deficit the ranking never changes, so each
        request is placed in full before the next.
        """
        active = list(requests)
        order = {id(request): k for k, request in enumerate(active)}
        placed = dict.fromkeys(order, 0)
        program_seconds = dict(self._delivered)
        while active:
            request = active[0]
            if self._balancing:
                shares = program_shares(program_seconds)
                request = max(
                    active,
                    key=lambda r: (
                        r.merit.tier,
                        self._value(r, shares),
                        -order[id(r)],
                    ),
                )
            block = (
                # Such as allocated time shorter than a snapshot.
                None
                if request.remaining < float(request.target.ss_min)
                else self._place_snapshot(request, self._cadence_release(request))
            )
            if block is not None:
                collected = block.entry.collection_seconds_between(
                    block.entry.begin, block.entry.end
                )
                request.remaining -= collected
                placed[id(request)] += 1
                self._record_visit(request, block, program_seconds)
                if request.remaining >= float(request.target.ss_min):
                    continue
            active.remove(request)
            rest = self._rest_of.get(id(request))
            if rest is not None:
                # Allocated seconds that did not fit their span are planned
                # with the rest of the request's exposure.
                rest.remaining += max(request.remaining, 0.0)
                continue
            if request.remaining >= float(request.target.ss_min):
                self.unplaced.append(request.target)
                self._log(
                    self.ctx.ustart,
                    f"Target {request.target.obsid} left {request.remaining:.0f}s "
                    f"unplanned after {placed[id(request)]} snapshot(s)",
                    request.target.obsid,
                )

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

    def _record_visit(
        self, request: _Request, block: _Block, program_seconds: dict[str, float]
    ) -> None:
        """Count a placed snapshot towards its program and the target's last visit."""
        entry = block.entry
        seconds = entry.collection_seconds_between(entry.begin, entry.end)
        program = request.category.program_name
        program_seconds[program] = program_seconds.get(program, 0.0) + seconds
        if request.cadence is not None and entry.collection_end is not None:
            self._last_visit[int(request.target.obsid)] = float(entry.collection_end)

    def _finish(self, timeline: Sequence[_Block]) -> Plan:
        """Record and return the plan for a finished timeline."""
        self.plan = Plan(
            entries=[block.entry for block in timeline],
            metadata={"planner": self.planner_name, "step_size": self.ctx.step_size},
        )
        return self.plan

    # ── Requests ──────────────────────────────────────────────────────────

    def _requests(self) -> list[_Request]:
        """Return the requests in planning order: tier, then value, then a stable tie-break."""
        merit_model = MeritModel(self.config)
        seed = self.config.random_seed if self.config.random_seed is not None else 0
        self._delivered: dict[str, float] = {}
        """Science already collected, by program."""
        self._last_visit: dict[int, float] = {}
        """When each cadence target was last observed, collected or planned."""
        for target in self.targets:
            obsid = int(target.obsid)
            # Committed science will be collected before this plan's.
            seconds = target.collected_seconds + self.reserved_seconds.get(obsid, 0.0)
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
            exptime -= self.reserved_seconds.get(int(target.obsid), 0.0)
            if target.done or exptime < target.ss_min:
                continue
            category = merit_model.category(target)
            merit = merit_model.value_terms(
                target, self.ctx.ustart, base=float(target.fom), delivered_shares=shares
            )
            if self.preferred is not None or (
                self.allocated is not None and self.allocation_strictness == "tier"
            ):
                # Preferred requests, or with tier strictness allocated time,
                # rank above the others in their tier, and below every request
                # in the tiers above: every ranking and tier-by-tier objective
                # in the planners follows this tier.
                preferred = (
                    self.preferred is not None and int(target.obsid) in self.preferred
                )
                merit = merit.model_copy(
                    update={"tier": 2 * merit.tier + int(preferred)}
                )
            cadence = category.cadence_seconds if cadence_weight > 0.0 else None
            requests.append(
                _Request(
                    target=target,
                    merit=merit,
                    remaining=float(exptime),
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
        tiers = [r.merit.tier for r in requests]
        # The tier allocated time is raised by, and the factor on its value.
        # Strict: above every unallocated request, in tier order among itself,
        # as the allocation has already weighed the tiers. Tier: just above its
        # own tier's unallocated requests (tiers are doubled above). Weighted:
        # in its own tier, worth more.
        self._allocated_offset = {
            "strict": max(tiers) - min(tiers) + 1 if tiers else 1,
            "tier": 1,
            "weighted": 0,
        }[self.allocation_strictness]
        self._allocated_factor = (
            1.0 + self.allocation_bonus
            if self.allocation_strictness == "weighted"
            else 1.0
        )
        requests.sort(key=self._rank, reverse=True)
        return requests

    def _rank(self, request: _Request) -> tuple[tuple[int, float], int]:
        """Planning order: tier, then value, then a stable tie-break."""
        payload = f"{self._seed}:{request.target.obsid}".encode()
        tie_break = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "big"
        )
        return request.merit.value_rank, tie_break

    def _with_allocated_time(self, requests: Sequence[_Request]) -> list[_Request]:
        """Requests to place: allocated time first, in its span, then the rest.

        Each span of a request's allocated time becomes a request for those
        seconds, limited to the span, ranked by ``allocation_strictness``: its
        tier raised by ``_allocated_offset`` and its value by
        ``_allocated_factor``. The rest of its exposure stays a request in its
        own tier. Allocated seconds
        that do not fit their span go back to the rest (see _place_requests).
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
                if seconds <= 0.0:
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
                                "tier": request.merit.tier + self._allocated_offset,
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
        obstype = ObsType.GSP if block.is_pass else ObsType.PPT
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
        else:
            # ACS only starts a science slew while the target is visible.
            if not block.entry.visible(slew.slewstart, slew.slewstart):
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

    @staticmethod
    def _apply_slew(block: _Block, slew: Slew) -> None:
        """Record a block's incoming slew on the block and its plan entry."""
        block.slew = slew
        entry = block.entry
        entry.begin = float(slew.slewstart)
        entry.slewtime = int(slew.slewtime)
        entry.slewdist = float(slew.slewdist)

    # ── Locked entries and passes ────────────────────────────────────────

    def _place_locked(self, entry: PlanEntry) -> None:
        """Keep a locked science entry's collection window, recomputing its slew."""
        if entry.collection_begin is None or entry.collection_end is None:
            raise ValueError(
                f"Locked entry {entry.obsid} needs a collection window to keep"
            )
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
        if not self._insert_fixed(block):
            self._log(
                float(entry.begin),
                f"Locked entry {entry.obsid} cannot be reached; leaving it out",
                entry.obsid,
            )

    def _place_pass(self, gspass: Pass) -> None:
        """Reserve a ground pass on the first tracking profile that can be reached."""
        for profile in gspass.available_tracking_profiles():
            block = self._pass_block(gspass, profile)
            if self._insert_fixed(block):
                gspass.select_tracking_profile(profile)
                return
        self._log(
            gspass.begin,
            f"Pass at {gspass.station} cannot be reached on any safe tracking "
            "profile; not reserved",
            gspass.obsid,
        )

    def _pass_block(self, gspass: Pass, profile: TrackingProfile) -> _Block:
        """Build the plan entry and timeline block for a ground contact."""
        start, finish = profile[0], profile[-1]
        entry = PlanEntry(config=self.config)
        entry.name = f"{gspass.station}_PASS"
        entry.ra, entry.dec, entry.roll = start
        entry.begin = gspass.begin
        entry.end = gspass.end
        entry.obsid = gspass.obsid
        entry.obstype = ObsType.GSP
        entry.ss_min = 0
        duration = max(0, int(round(gspass.end - gspass.begin)))
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
        entry.contact_begin = gspass.begin
        entry.contact_end = gspass.end
        entry.track_start_ra, entry.track_start_dec, entry.track_start_roll = start
        entry.track_end_ra, entry.track_end_dec, entry.track_end_roll = finish
        return _Block(
            entry=entry,
            attitude_in=start,
            attitude_out=finish,
            ready=gspass.begin,
            end=gspass.end,
            gspass=gspass,
        )

    # ── Science snapshots ────────────────────────────────────────────────

    def _place_snapshot(
        self, request: _Request, not_before: float | None = None
    ) -> _Block | None:
        """Place one snapshot of a request in the earliest slot that fits.

        Its slew starts no earlier than ``not_before``, if given.
        """
        for index in range(len(self.timeline) + 1):
            pred = self._pred_at(index)
            succ = self.timeline[index] if index < len(self.timeline) else None
            candidate = self._fit_in_gap(request, pred, succ, not_before)
            if candidate is not None:
                block, slew, succ_slew = candidate
                self._commit(index, block, slew, succ, succ_slew)
            elif (
                succ is not None
                and self._successor_blocked
                and id(succ) in self._movable
            ):
                delayed = self._fit_delaying_successor(request, index, not_before)
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
        return None

    def _fit_delaying_successor(
        self, request: _Request, index: int, not_before: float | None
    ) -> _Block | None:
        """Fit a snapshot before timeline block ``index`` by moving that block later.

        Used when a snapshot fits the gap but the science snapshot after it
        cannot then be reached in time, such as one placed at the start of its
        target's next visibility window. That snapshot is placed again after the
        new one, collecting as long as before, and must still reach the block
        after it; otherwise nothing changes. Snapshots of requests whose visits
        follow a cadence stay where they are, as later visits are spaced from
        them.
        """
        succ = self.timeline[index]
        succ_request = self._movable[id(succ)]
        if succ_request.cadence is not None:
            return None
        pred = self._pred_at(index)
        after = self.timeline[index + 1] if index + 1 < len(self.timeline) else None
        candidate = self._fit_in_gap(request, pred, after, not_before)
        if candidate is None:
            return None
        block, slew, _ = candidate
        seconds = succ.entry.collection_seconds_between(
            succ.entry.begin, succ.entry.end
        )
        again = succ_request.model_copy(update={"remaining": seconds})
        moved = self._fit_in_gap(again, block, after)
        if moved is None:
            return None
        moved_block, moved_slew, after_slew = moved
        entry = moved_block.entry
        if entry.collection_seconds_between(entry.begin, entry.end) < seconds - 1e-6:
            return None

        self._apply_slew(block, slew)
        self._apply_slew(moved_block, moved_slew)
        if after is not None and after_slew is not None:
            self._apply_slew(after, after_slew)
        self.timeline[index] = moved_block
        self.timeline.insert(index, block)
        del self._movable[id(succ)]
        self._movable[id(moved_block)] = succ_request
        self._log(
            float(moved_block.entry.begin),
            f"Moved {succ_request.target.obsid} to "
            f"{unixtime2date(float(entry.collection_begin or 0))} to fit "
            f"{request.target.obsid} before it",
            succ_request.target.obsid,
        )
        return block

    def _fit_in_gap(
        self,
        request: _Request,
        pred: _Block | None,
        succ: _Block | None,
        not_before: float | None = None,
    ) -> tuple[_Block, Slew, Slew | None] | None:
        """Find the earliest snapshot of ``request`` between ``pred`` and ``succ``.

        The slew starts no earlier than ``not_before``, if given.
        """
        self._successor_blocked = False
        target = request.target
        ss_min = float(target.ss_min)
        earliest = self._earliest_start(pred)
        gap_end = succ.ready if succ is not None else self.ctx.uend
        start = earliest
        if not_before is not None:
            start = max(start, self.ctx.ceil_step(not_before))
        if gap_end - start < ss_min:
            return None
        idle_limit = self._idle_limit(pred, earliest, gap_end)
        snapshot = min(float(target.ss_max), request.remaining)
        # Starts at which the snapshot fitted but its successor was unreachable.
        unreachable = 0

        while start <= min(idle_limit, gap_end):
            window = next((w for w in request.windows if w[0] <= start < w[1]), None)
            if window is None:
                upcoming = [w[0] for w in request.windows if w[0] > start]
                if not upcoming:
                    return None
                start = self.ctx.ceil_step(min(upcoming))
                continue

            instrument_roll = self.ctx.instrument_roll(target, start)
            attitude = target.target_body_attitude(instrument_roll)
            slew = self.ctx.slew(
                self._start_attitude(pred, start),
                attitude,
                start,
                obsid=target.obsid,
            )
            collection_begin = float(slew.slewend) + self.ctx.setup_seconds
            if (
                target.earliest_start is not None
                and collection_begin < target.earliest_start
            ):
                # Start later by the shortfall, so collection begins at the
                # earliest start if the slew takes as long.
                start = max(
                    start + self.ctx.step_size,
                    self.ctx.ceil_step(
                        start + target.earliest_start - collection_begin
                    ),
                )
                continue
            if target.deadline is not None and collection_begin > target.deadline:
                return None

            latest_end = min(window[1], self.ctx.uend)
            succ_slewtime = 0.0
            if succ is not None:
                succ_slewtime = self.ctx.slew(
                    attitude, succ.attitude_in, collection_begin
                ).slewtime
                # The successor's slew must start on a step after this ends.
                latest_end = min(
                    latest_end, self.ctx.floor_step(succ.ready - succ_slewtime)
                )
            collection_end = min(
                collection_begin + snapshot,
                latest_end - self.ctx.post_collection_seconds,
            )
            if collection_end - collection_begin < ss_min:
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

    def _check_snapshot(
        self, block: _Block, slew: Slew, ss_min: float
    ) -> _Block | None:
        """Check a snapshot's slew and observation, shortening it to fit if needed."""
        entry = block.entry
        if self.ctx.first_slew_violation(slew, ACSMode.SLEWING) is not None:
            return None
        entry.visibility()
        if not entry.visible(slew.slewstart, slew.slewstart):
            return None
        violation = self.ctx.first_science_violation(
            entry, block.attitude_in, slew.slewend, self.ctx.ceil_step(block.end)
        )
        if violation is None:
            return block
        # End the observation at the step the attitude stops being allowed.
        assert entry.collection_begin is not None
        collection_end = violation - self.ctx.post_collection_seconds
        if collection_end - entry.collection_begin < ss_min:
            return None
        entry.collection_end = collection_end
        entry.end = violation
        block.end = violation
        return block

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
            ss_min=target.ss_min,
            ss_max=target.ss_max,
        )
        entry.exptime = collection_end - collection_begin
        return entry

    def _log(self, utime: float, description: str, obsid: int | None = None) -> None:
        self.log.log_event(
            utime=utime,
            event_type="SCHEDULER",
            description=description,
            obsid=obsid,
            acs_mode=None,
        )
