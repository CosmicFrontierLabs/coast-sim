"""Priority-first construction planner.

:class:`PriorityPlanner` reserves ground contacts first, then takes requests
in priority order and places each snapshot in the earliest slot that still fits,
without moving anything already placed. It is fast and deterministic, and its
plans are traceable to the priority order. It is myopic: an early placement can
block a better packing.
"""

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum, auto

from pydantic import BaseModel, ConfigDict, Field

from ..common import ACSMode, ObsType, unixtime2date
from ..config import MissionConfig
from ..ditl.ditl_log import DITLLog
from ..simulation.passes import Pass
from ..simulation.slew import Slew
from ..targets import Plan, PlanEntry, Pointing
from ..targets.merit import MeritBreakdown, MeritModel
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
    remaining: float
    windows: list[tuple[float, float]] = Field(default_factory=list)


class PriorityPlanner:
    """Build a plan by placing requests in priority order at their earliest fit.

    Steps:

    1. Locked entries keep their collection windows and are placed first.
    2. Ground passes the spacecraft can reach are reserved.
    3. Requests are sorted by tier, then by merit value at the start of the
       horizon (see :class:`~conops.targets.merit.MeritModel`). Each request is
       split into snapshots of up to ``ss_max`` seconds, never shorter than
       ``ss_min``, until its ``exptime`` is used or it no longer fits.
    4. Each snapshot goes in the earliest slot where every check
       :class:`~conops.ditl.DITL` will apply passes: the slew starts on a step,
       its path and the held attitude clear their mode's constraints, the
       target is visible when the slew starts, collection starts by the
       target's deadline, and the following activity can still be reached.

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
            remaining exposure.
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
        self.unplaced: list[Pointing] = []
        """Targets left with at least ``ss_min`` of exposure unplanned."""
        self.plan = Plan()

    # ── Public API ────────────────────────────────────────────────────────

    def schedule(self) -> Plan:
        """Build and return the plan."""
        self._reserve_fixed()
        self._place_requests(self._requests())
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
        """Place each request's snapshots in order at their earliest fit."""
        for request in requests:
            placed = 0
            while request.remaining >= float(request.target.ss_min):
                block = self._place_snapshot(request)
                if block is None:
                    break
                collected = block.entry.collection_seconds_between(
                    block.entry.begin, block.entry.end
                )
                request.remaining -= collected
                placed += 1
            if request.remaining >= float(request.target.ss_min):
                self.unplaced.append(request.target)
                self._log(
                    self.ctx.ustart,
                    f"Target {request.target.obsid} left {request.remaining:.0f}s "
                    f"unplanned after {placed} snapshot(s)",
                    request.target.obsid,
                )

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
        requests = []
        for target in self.targets:
            exptime = target.exptime if target.exptime is not None else target.ss_max
            exptime -= self.reserved_seconds.get(int(target.obsid), 0.0)
            if target.done or exptime < target.ss_min:
                continue
            requests.append(
                _Request(
                    target=target,
                    merit=merit_model.value_terms(
                        target, self.ctx.ustart, base=float(target.fom)
                    ),
                    remaining=float(exptime),
                    windows=self.ctx.visibility_windows(target),
                )
            )

        def tie_break(request: _Request) -> int:
            payload = f"{seed}:{request.target.obsid}".encode()
            return int.from_bytes(
                hashlib.blake2b(payload, digest_size=8).digest(), "big"
            )

        requests.sort(
            key=lambda r: (r.merit.value_rank, tie_break(r)),
            reverse=True,
        )
        return requests

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

    def _place_snapshot(self, request: _Request) -> _Block | None:
        """Place one snapshot of a request in the earliest slot that fits."""
        for index in range(len(self.timeline) + 1):
            pred = self._pred_at(index)
            succ = self.timeline[index] if index < len(self.timeline) else None
            candidate = self._fit_in_gap(request, pred, succ)
            if candidate is None:
                continue
            block, slew, succ_slew = candidate
            self._commit(index, block, slew, succ, succ_slew)
            self._log(
                block.entry.begin,
                f"Placed {request.target.obsid} at "
                f"{unixtime2date(float(block.entry.collection_begin or 0))} "
                f"for {block.entry.exposure}s: {request.merit.describe()}",
                request.target.obsid,
            )
            return block
        return None

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
        target = request.target
        ss_min = float(target.ss_min)
        earliest = self._earliest_start(pred)
        gap_end = succ.ready if succ is not None else self.ctx.uend
        if gap_end - earliest < ss_min:
            return None
        idle_limit = self._idle_limit(pred, earliest, gap_end)
        snapshot = min(float(target.ss_max), request.remaining)

        start = earliest
        if not_before is not None:
            start = max(start, self.ctx.ceil_step(not_before))
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
