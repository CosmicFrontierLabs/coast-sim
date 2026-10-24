"""Improve a priority-first plan by local search over its observation sequence.

A plan's science can be described by the order of its snapshots alone. A
decoder turns an order into a timeline by placing each snapshot at the earliest
time it fits after the one before, with the checks
:class:`~conops.schedulers.PriorityScheduler` applies, so every decoded timeline
executes as planned. Decoding in time order also closes the gaps that placing
requests in priority order leaves behind, and lets each snapshot's merit follow
the plan before it: its completion deficit counts the program shares delivered
so far, and a cadence target's visits wait for their cadence, as in
:class:`~conops.schedulers.PriorityScheduler`.

:class:`LocalSearchScheduler` starts from the priority-first plan and searches
for a better order by inserting, replacing, removing, swapping and moving snapshots,
keeping the best plan it finds within a time or iteration budget.
"""

import random
import time
from collections.abc import Mapping, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from ..common import ObsType
from ..config import MissionConfig
from ..ditl.plan_validator import PLAN_SCIENCE_OBSTYPES, entry_obstype
from ..targets import Plan, Pointing
from ..targets.plan_entry import snapshot_min_seconds
from .battery import Additions
from .priority_scheduler import (
    PriorityScheduler,
    _Block,
    _collection_seconds,
    _Request,
    program_shares,
)

_VOLUME_TOLERANCE_GB = 1e-9
"""Within which two ledgers' volumes are the same, allowing for rounding."""


class _Snapshot(BaseModel):
    """One snapshot in the sequence: its request, length and release time."""

    model_config = ConfigDict(frozen=True)

    obsid: int
    seconds: float
    """Longest collection to place; shorter if exposure or the gap runs out."""
    release: float | None = None
    """Earliest slew start, which keeps the snapshot where an earlier plan had
    it until a move releases it."""


class _State(BaseModel):
    """The timeline after decoding a prefix of the sequence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timeline: list[_Block]
    cursor: int
    """Timeline position after the last placed snapshot; the next goes after it."""
    remaining: dict[int, float]
    score: float
    """Merit-weighted science seconds."""
    programs: dict[str, float]
    """Science collected and planned so far, by program."""
    last_visit: dict[int, float]
    """When each cadence target was last observed, collected or planned."""
    allocated_left: dict[tuple[int, int], float]
    """Allocated seconds not yet used, by obsid and span (see ``allocated``)."""
    volumes: tuple[float, ...]
    """Science volume planned in each segment of the recorder ledger."""
    energy: tuple[Additions, ...]
    """What the plan draws from the battery, by segment of the battery ledger."""


class _Decoded(BaseModel):
    """A sequence and the timeline states it decodes to."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sequence: list[_Snapshot]
    """Snapshots in time order."""
    states: list[_State]
    """``states[k]`` is the state before snapshot ``k``; the last is final."""

    @property
    def final(self) -> _State:
        return self.states[-1]


def _wants_more(request: _Request, remaining: float, exposure: float) -> bool:
    """Whether ``remaining`` seconds are enough for another snapshot of
    ``request``, of ``exposure`` to place at the start: a remainder of any
    length once some of the exposure has been placed."""
    if remaining <= 0:
        return False
    if remaining < exposure:
        return True
    return remaining >= float(snapshot_min_seconds(request.target))


class LocalSearchScheduler(PriorityScheduler):
    """Build a priority-first plan, then improve it by local search.

    The search changes the order of the plan's science snapshots and decodes
    each order into a timeline (see the module description). It tries:

    * inserting a snapshot of a request with exposure left to place,
    * replacing a snapshot with one of another request with exposure left,
    * releasing a snapshot from the time and length the priority-first plan
      gave it,
    * removing a snapshot,
    * swapping two nearby snapshots, and
    * moving a snapshot to a nearby position.

    A change is kept by late-acceptance hill climbing: when it is no worse than
    the current plan, or than the plan of ``history_length`` steps ago, so the
    search can cross plateaus. The objective is merit-weighted science time.
    A snapshot of a request with a
    deadline is worth less the later it starts: starting at the deadline
    rather than the start of the horizon, or its earliest start if later,
    costs ``earliness_weight`` of its value. With ``allocated`` time, a
    snapshot's seconds that use what is left of the span its collection starts
    in are worth ``allocation_bonus`` more. The best plan
    found is returned, and never one worse than the priority-first plan.

    Each plan entry's merit is the value its snapshot has where the decoder
    places it, with the program shares the plan delivers before it; a Target
    of Opportunity in a rolling run must beat it to interrupt.

    Ground passes and locked entries stay where the priority-first plan put
    them. Takes the same arguments as
    :class:`~conops.schedulers.PriorityScheduler`, plus:

    Args:
        time_limit: Seconds of search after the priority-first plan is built.
        max_iterations: Changes to try at most. Set this rather than
            ``time_limit`` alone for reproducible plans.
        seed: Random seed; defaults to the configuration's.
        neighborhood: How many positions apart a swap or move can be.
        history_length: Length of the late-acceptance history.
        earliness_weight: Fraction of a deadline request's value lost by
            starting at its deadline, in [0, 1]. Values below 1 keep a late
            snapshot worth more than none.
    """

    scheduler_name = "local_search"

    def __init__(
        self,
        config: MissionConfig,
        targets: Sequence[Pointing],
        begin: datetime,
        end: datetime,
        *,
        time_limit: float = 10.0,
        max_iterations: int | None = None,
        seed: int | None = None,
        neighborhood: int = 3,
        history_length: int = 50,
        earliness_weight: float = 0.5,
        start_plan: Plan | None = None,
        **options: object,
    ) -> None:
        super().__init__(config, targets, begin, end, **options)  # type: ignore[arg-type]
        self.start_plan = start_plan
        """A plan to start from where it scores higher than the priority-first
        plan: its science entries decoded in time order at their own times,
        such as the plan a dispatch run of the same requests executed. The
        search, or the solver, then improves on the better of the two, so the
        plan returned scores at least as well as either."""
        self.seed_score: float | None = None
        """Objective of ``start_plan`` decoded, where one was given."""
        if time_limit < 0:
            raise ValueError("time_limit must not be negative")
        if max_iterations is not None and max_iterations < 0:
            raise ValueError("max_iterations must not be negative")
        if neighborhood < 1 or history_length < 1:
            raise ValueError("neighborhood and history_length must be positive")
        if not 0.0 <= earliness_weight <= 1.0:
            raise ValueError("earliness_weight must be between 0 and 1")
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.seed = seed if seed is not None else (config.random_seed or 0)
        self.neighborhood = neighborhood
        self.history_length = history_length
        self.earliness_weight = earliness_weight
        self.iterations = 0
        """Changes tried in the last :meth:`schedule`."""
        self.accepted = 0
        """Changes the search kept."""
        self.initial_score = 0.0
        """Objective of the priority-first plan."""
        self.start_score = 0.0
        """Objective of the decoded starting sequence; equals ``initial_score``."""
        self.score = 0.0
        """Objective of the returned plan."""

    def schedule(self) -> Plan:
        """Build the priority-first plan, then search for a better one, within
        the wheels' momentum where the configuration tracks it."""
        return self._plan_within_momentum(self._plan_once)

    def _plan_once(self) -> Plan:
        start = self._start_from_priority_plan()
        best = self._best = self._search(start)
        return self._finish_best(best)

    def _start_from_priority_plan(self) -> _Decoded:
        """Build the priority-first plan and return it as a decoded sequence."""
        self._reserve_fixed()
        # Copies, because placing science reconnects the fixed blocks' slews.
        fixed = [self._copy_block(block) for block in self.timeline]
        fixed_entries = {id(block.entry) for block in self.timeline}
        requests = self._requests()
        self._by_obsid = {int(r.target.obsid): r for r in requests}
        self._initial_remaining = {
            obsid: r.remaining for obsid, r in self._by_obsid.items()
        }
        allocated = self.allocated or {}
        self._initial_allocated = {
            (obsid, n): span.seconds
            for obsid, spans in allocated.items()
            for n, span in enumerate(spans)
        }
        self._initial_last_visit = dict(self._last_visit)
        self._initial_volumes = tuple(self._volumes)
        self._initial_energy = tuple(self._energy)

        self._place_requests(self._with_allocated_time(requests))
        self._greedy_timeline = list(self.timeline)
        science = [b for b in self._greedy_timeline if id(b.entry) not in fixed_entries]
        self.initial_score = self._timeline_score(science)

        self._fixed = fixed
        if self.start_plan is not None:
            self._settle_from(self.start_plan)
        # Released at their priority-first times, the snapshots decode to
        # the priority-first plan; moves release them one at a time.
        start = self._decode(
            [
                _Snapshot(
                    obsid=int(b.entry.obsid),
                    seconds=_collection_seconds(b),
                    release=float(b.entry.begin),
                )
                for b in science
            ]
        )
        self.start_score = self._objective(start.final)
        if self.start_plan is not None:
            seeded = self._decode_plan(self.start_plan)
            self.seed_score = self._objective(seeded.final)
            if self.seed_score > self.start_score:
                start, self.start_score = seeded, self.seed_score
        return start

    def _settle_from(self, plan: Plan) -> None:
        """Hold each fixed block that may begin late where ``plan`` held its
        activity, within the block's reservation, and reserve it there: the
        plan's science around the activity then decodes at its own times,
        before and after the block as the plan had it."""
        held = {
            int(entry.obsid): float(entry.begin)
            for entry in plan
            if entry_obstype(entry) == ObsType.ACTIVITY
        }
        for block in self._fixed:
            activity = block.activity
            if activity is None or activity.obsid is None:
                continue
            begin = held.get(int(activity.obsid))
            if begin is None:
                continue
            self._settle_block(block, begin)
            block.latest_ready = block.ready

    def _decode_plan(self, plan: Plan) -> _Decoded:
        """Decode a plan's science as a start sequence: one snapshot per
        observation, a visit's steps as one, each released at its entry's
        start, for the requests being planned; entries of other targets,
        such as the ToOs a dispatch run observed, are left out."""
        snapshots: list[_Snapshot] = []
        visit_of: int | None = None
        """The target whose visit the last snapshot is, while its steps follow."""
        entries = sorted(
            (e for e in plan if entry_obstype(e) in PLAN_SCIENCE_OBSTYPES),
            key=lambda e: float(e.begin),
        )
        for entry in entries:
            obsid = int(entry.obsid)
            request = self._by_obsid.get(obsid)
            if request is None:
                visit_of = None
                continue
            seconds = (
                0.0
                if entry.calibration
                else entry.collection_seconds_between(
                    float(entry.begin), float(entry.end)
                )
            )
            sequence = request.target.sequence
            if entry.step is not None and sequence is not None:
                if visit_of == obsid and entry.step != sequence.steps[0].name:
                    # Another step of the visit under way.
                    last = snapshots[-1]
                    snapshots[-1] = _Snapshot(
                        obsid=obsid,
                        seconds=last.seconds + seconds,
                        release=last.release,
                    )
                    continue
                visit_of = obsid
                snapshots.append(
                    _Snapshot(obsid=obsid, seconds=seconds, release=float(entry.begin))
                )
                continue
            visit_of = None
            if seconds > 0.0:
                snapshots.append(
                    _Snapshot(obsid=obsid, seconds=seconds, release=float(entry.begin))
                )
        # Each snapshot where the plan had it, before the plan's next one
        # begins: one the timeline cannot take there is left out, where
        # decoding it at its next fit would push everything after it later;
        # so is one it cannot take at all, such as a remainder shorter than
        # the request's minimum.
        kept: list[_Snapshot] = []
        states = [self._initial_state()]
        for snapshot, following in zip(snapshots, [*snapshots[1:], None], strict=True):
            state = self._step(states[-1], snapshot)
            if state.cursor == states[-1].cursor:
                continue
            block = state.timeline[state.cursor - 1]
            assert snapshot.release is not None
            before = (
                following.release
                if following is not None and following.release is not None
                else self.ctx.uend
            )
            if float(block.entry.begin) >= max(
                before, snapshot.release + self.ctx.step_size
            ):
                continue
            kept.append(snapshot)
            states.append(state)
        return _Decoded.model_construct(sequence=kept, states=states)

    def _finish_best(self, best: _Decoded) -> Plan:
        """Return the best plan found, or the priority-first plan if better."""
        best_score = self._objective(best.final)
        if best_score >= self.initial_score:
            self.score = best_score
            self._set_unplaced(best.final.remaining)
            return self._finish(best.final.timeline)
        self.score = self.initial_score
        return self._finish(self._greedy_timeline)

    # ── Objective ────────────────────────────────────────────────────────

    @staticmethod
    def _objective(state: _State) -> float:
        return state.score

    def _timeline_score(self, science: Sequence[_Block]) -> float:
        """Score a priority-first timeline's science blocks, in time order."""
        total = 0.0
        programs = dict(self._delivered)
        left = dict(self._initial_allocated)
        for block in sorted(science, key=lambda b: float(b.entry.begin)):
            request = self._by_obsid.get(int(block.entry.obsid))
            if request is None:
                continue
            value = self._value(request, program_shares(programs))
            total += self._scored(left, request, block, value)
            self._add_program_seconds(programs, request, block)
        return total

    @staticmethod
    def _add_program_seconds(
        programs: dict[str, float], request: _Request, block: _Block
    ) -> None:
        program = request.category.program_name
        programs[program] = programs.get(program, 0.0) + _collection_seconds(block)

    def _scored(
        self,
        left: dict[tuple[int, int], float],
        request: _Request,
        block: _Block,
        value: float,
    ) -> float:
        """Return a snapshot's contribution to the score.

        Its seconds, up to what is left of the allocated time of the span its
        collection starts in, are worth ``_allocated_factor`` more, and use up
        that time in ``left``.
        """
        worth = self._contribution(request, block, value)
        entry = block.entry
        seconds = _collection_seconds(block)
        span = self._span_of(int(entry.obsid), entry.collection_begin)
        counted = min(seconds, left.get(span, 0.0)) if span is not None else 0.0
        if counted > 0.0 and span is not None:
            left[span] -= counted
            worth *= 1.0 + (self._allocated_factor - 1.0) * counted / seconds
        return worth

    def _span_of(
        self, obsid: int, collection_begin: float | None
    ) -> tuple[int, int] | None:
        """The span of allocated time a collection starting then falls in."""
        if self.allocated is None or collection_begin is None:
            return None
        for n, span in enumerate(self.allocated.get(obsid, [])):
            if span.begin <= collection_begin < span.end:
                return obsid, n
        return None

    def _contribution(self, request: _Request, block: _Block, value: float) -> float:
        """Return a snapshot's merit-weighted science, discounted for lateness.

        ``value`` is the value of a snapshot of the request where it is
        placed (see :meth:`_value`).
        """
        entry = block.entry
        worth = value * _collection_seconds(block)
        deadline = request.deadline
        if deadline is None or self.earliness_weight == 0.0:
            return worth
        # Lateness counts from when the request could first start.
        opens = max(self.ctx.ustart, request.earliest_start or self.ctx.ustart)
        available = deadline - opens
        if available <= 0.0 or entry.collection_begin is None:
            return worth
        delay = max(0.0, float(entry.collection_begin) - opens)
        return worth * (1.0 - self.earliness_weight * min(1.0, delay / available))

    def _set_unplaced(self, remaining: Mapping[int, float]) -> None:
        self.unplaced = [
            request.target
            for obsid, request in self._by_obsid.items()
            if _wants_more(request, remaining[obsid], self._initial_remaining[obsid])
        ]

    # ── Decoding ─────────────────────────────────────────────────────────

    @staticmethod
    def _copy_block(block: _Block) -> _Block:
        """Copy a block and its plan entry, so the copy can be changed alone."""
        return block.model_copy(update={"entry": block.entry.model_copy()})

    def _initial_state(self) -> _State:
        return _State.model_construct(
            timeline=list(self._fixed),
            cursor=0,
            remaining=dict(self._initial_remaining),
            score=0.0,
            programs=dict(self._delivered),
            last_visit=dict(self._initial_last_visit),
            allocated_left=dict(self._initial_allocated),
            volumes=self._initial_volumes,
            energy=self._initial_energy,
        )

    def _decode(self, sequence: list[_Snapshot]) -> _Decoded:
        """Decode a whole sequence from the start."""
        states = [self._initial_state()]
        for snapshot in sequence:
            states.append(self._step(states[-1], snapshot))
        return _Decoded.model_construct(sequence=sequence, states=states)

    def _step(self, state: _State, snapshot: _Snapshot) -> _State:
        """Place the next snapshot at its earliest fit after the cursor.

        Returns ``state`` unchanged if the snapshot does not fit.
        """
        obsid = snapshot.obsid
        request = self._by_obsid[obsid]
        remaining = state.remaining[obsid]
        exposure = self._initial_remaining[obsid]
        if not _wants_more(request, remaining, exposure):
            return state

        # _fit_in_gap sizes the snapshot from the request's remaining exposure.
        request.remaining = min(remaining, snapshot.seconds)
        request.placed_before = remaining < exposure
        not_before = snapshot.release
        last = state.last_visit.get(obsid)
        if request.cadence is not None and last is not None:
            due = last + request.cadence
            not_before = due if not_before is None else max(not_before, due)
        if request.timed:
            # The next window still open where the decoding has got to.
            cursor_time = (
                state.timeline[state.cursor - 1].end
                if state.cursor > 0
                else self.ctx.ustart
            )
            request.select_visit(
                request.next_visit(last, max(cursor_time, not_before or cursor_time)),
                self.ctx,
            )
            if request.visit is None:
                return state

        timeline = state.timeline
        for index in range(state.cursor, len(timeline) + 1):
            pred = timeline[index - 1] if index > 0 else self._origin
            succ = timeline[index] if index < len(timeline) else None
            fit = self._fit_in_gap(
                request, pred, succ, not_before, beyond=timeline[index + 1 :]
            )
            if fit is None:
                continue
            # Only where the battery stays above its planning level, charging
            # first in the gap if that is what it takes.
            charged = self._fit_within_battery(
                request, pred, succ, not_before, fit, timeline, index
            )
            if charged is None:
                continue
            block, slew, succ_slew = charged
            # And only where the recorder has room for it; later gaps may be
            # after a contact that makes room.
            volume = self._block_volume(block)
            segment = self._block_segment(block)
            # With what the slew into a contact after it comes to.
            additions = [(segment, volume), *self._ingress_deltas(succ, succ_slew)]
            if not self._ledger.fits_all(state.volumes, additions):
                continue

            # Put the snapshot in place, reconnecting the fixed block after
            # it, with the chain it leads where it follows.
            self._apply_slew(block, slew)
            placed = [*timeline[:index], block]
            if succ is not None:
                assert succ_slew is not None
                placed.extend(self._reconnected(block, timeline[index:], succ_slew))

            # Freeze the value the snapshot has here onto its entry, as the
            # priority-first scheduler does: a Target of Opportunity must beat
            # it to interrupt the observation.
            value = self._value(request, program_shares(state.programs))
            block.entry.merit = value
            allocated_left = dict(state.allocated_left)
            score = state.score + self._scored(allocated_left, request, block, value)

            left = dict(state.remaining)
            left[obsid] = remaining - _collection_seconds(block)
            programs = dict(state.programs)
            self._add_program_seconds(programs, request, block)
            last_visit = state.last_visit
            if (
                request.cadence is not None or request.timed
            ) and block.entry.collection_end is not None:
                last_visit = {**last_visit, obsid: float(block.entry.collection_end)}
            return _State.model_construct(
                timeline=placed,
                cursor=index + 1,
                remaining=left,
                score=score,
                programs=programs,
                last_visit=last_visit,
                allocated_left=allocated_left,
                volumes=self._ledger.add_all(state.volumes, additions),
                energy=self._charged_energy(placed),
            )
        return state

    def _same_state(self, a: _State, b: _State) -> bool:
        """Return whether decoding continues identically from two states.

        Equal remaining exposure means equal program totals, so the program
        shares, and with them every later snapshot's value, agree too. When
        the recorder limits the plan, what is on it must agree as well.
        """
        if (
            a.remaining != b.remaining
            or a.last_visit != b.last_visit
            or a.allocated_left != b.allocated_left
            or len(a.timeline) - a.cursor != len(b.timeline) - b.cursor
        ):
            return False
        if self._ledger.active and any(
            abs(x - y) > _VOLUME_TOLERANCE_GB
            for x, y in zip(a.volumes, b.volumes, strict=True)
        ):
            return False
        if self._battery.active and a.energy != b.energy:
            return False
        if a.cursor == 0 or b.cursor == 0:
            return a.cursor == b.cursor
        last_a, last_b = a.timeline[a.cursor - 1], b.timeline[b.cursor - 1]
        return last_a is last_b or (
            last_a.entry.obsid == last_b.entry.obsid
            and last_a.entry.begin == last_b.entry.begin
            and last_a.end == last_b.end
            and last_a.attitude_out == last_b.attitude_out
        )

    def _redecode(
        self,
        current: _Decoded,
        sequence: list[_Snapshot],
        first: int,
        last: int,
        shift: int,
    ) -> _Decoded:
        """Decode a changed sequence, reusing the current decoding where it agrees.

        Positions before ``first`` are unchanged. After ``last``, position ``k``
        holds what the current sequence holds at ``k - shift``; once the states
        there agree, the rest of the current decoding is reused.
        """
        states = current.states[: first + 1]
        for k in range(first, len(sequence)):
            if k > last:
                old = k - shift
                previous = current.states[old]
                if self._same_state(states[-1], previous):
                    return self._splice(current, sequence, states, old)
            states.append(self._step(states[-1], sequence[k]))
        return _Decoded.model_construct(sequence=sequence, states=states)

    def _splice(
        self,
        current: _Decoded,
        sequence: list[_Snapshot],
        states: list[_State],
        old: int,
    ) -> _Decoded:
        """Finish a decoding with the current one's states from position ``old``."""
        here = states[-1]
        there = current.states[old]
        prefix = here.timeline[: here.cursor]
        offset = here.cursor - there.cursor
        for later in current.states[old + 1 :]:
            # What the reused snapshots add to the score is the same from here.
            score = here.score + later.score - there.score
            states.append(
                _State.model_construct(
                    timeline=[*prefix, *later.timeline[there.cursor :]],
                    cursor=later.cursor + offset,
                    remaining=later.remaining,
                    score=score,
                    programs=later.programs,
                    last_visit=later.last_visit,
                    allocated_left=later.allocated_left,
                    volumes=tuple(
                        a + b - c
                        for a, b, c in zip(
                            here.volumes, later.volumes, there.volumes, strict=True
                        )
                    ),
                    energy=self._charged_energy(
                        [*prefix, *later.timeline[there.cursor :]]
                    ),
                )
            )
        return _Decoded.model_construct(sequence=sequence, states=states)

    # ── Search ───────────────────────────────────────────────────────────

    def _search(self, start: _Decoded) -> _Decoded:
        """Late-acceptance hill climbing over the snapshot sequence."""
        rng = random.Random(self.seed)
        current = best = start
        current_score = best_score = self._objective(start.final)
        history = [current_score] * self.history_length
        deadline = time.perf_counter() + self.time_limit
        self.iterations = self.accepted = 0

        while time.perf_counter() < deadline and (
            self.max_iterations is None or self.iterations < self.max_iterations
        ):
            move = self._random_move(current, rng)
            slot = self.iterations % self.history_length
            self.iterations += 1
            if move is None:
                continue
            candidate = self._redecode(current, *move)
            score = self._objective(candidate.final)
            if score >= current_score or score >= history[slot]:
                current, current_score = candidate, score
                self.accepted += 1
                if score > best_score:
                    best, best_score = candidate, score
            history[slot] = current_score
        return best

    def _random_move(
        self, current: _Decoded, rng: random.Random
    ) -> tuple[list[_Snapshot], int, int, int] | None:
        """Return a changed sequence and the span it changed, or None.

        The span is as :meth:`_redecode` takes it: the first and last changed
        positions, and how far the positions after it shifted.
        """
        sequence = current.sequence
        size = len(sequence)
        kind = rng.random()

        if kind < 0.25:
            # Insert a full-length snapshot of a request with exposure left.
            open_requests = self._open_requests(current)
            if not open_requests:
                return None
            obsid = rng.choice(open_requests)
            at = rng.randint(0, size)
            added = self._released(obsid)
            return [*sequence[:at], added, *sequence[at:]], at, at, 1
        if size == 0:
            return None

        if kind < 0.35:
            # Replace a snapshot with a full-length one of another request
            # with exposure left. Removing a snapshot can free time that only
            # a different target can use, as when the spacecraft waits on a
            # hidden target to observe it the moment it rises. Inserting
            # first then makes that snapshot late, pushing everything after
            # it later, and removing first loses science, so neither change
            # is kept on its own. Snapshots that idle time comes before are
            # the ones worth replacing, so they are picked more often.
            weights = [idle + self.ctx.step_size for idle in self._idle_before(current)]
            at = rng.choices(range(size), weights=weights)[0]
            others = [
                obsid
                for obsid in self._open_requests(current)
                if obsid != sequence[at].obsid
            ]
            if not others:
                return None
            changed = list(sequence)
            changed[at] = self._released(rng.choice(others))
            return changed, at, at, 0

        if kind < 0.55:
            # Release a snapshot from its earlier time and length.
            at = rng.randrange(size)
            if sequence[at] == self._released(sequence[at].obsid):
                return None
            changed = list(sequence)
            changed[at] = self._released(sequence[at].obsid)
            return changed, at, at, 0

        if kind < 0.65:
            # Remove a snapshot.
            at = rng.randrange(size)
            return [*sequence[:at], *sequence[at + 1 :]], at, at - 1, -1

        # Swap two nearby snapshots, or move one a few positions.
        i = rng.randrange(size)
        offset = rng.randint(-self.neighborhood, self.neighborhood)
        j = min(size - 1, max(0, i + offset))
        if i == j:
            return None
        lo, hi = min(i, j), max(i, j)
        changed = list(sequence)
        if kind < 0.85:
            changed[i], changed[j] = changed[j], changed[i]
        else:
            changed.insert(j, changed.pop(i))
        # Reordered snapshots no longer keep their earlier times.
        for k in range(lo, hi + 1):
            changed[k] = self._released(changed[k].obsid)
        return changed, lo, hi, 0

    def _open_requests(self, current: _Decoded) -> list[int]:
        """Return the requests the current plan leaves exposure to place."""
        return [
            obsid
            for obsid, request in self._by_obsid.items()
            if _wants_more(
                request, current.final.remaining[obsid], self._initial_remaining[obsid]
            )
        ]

    def _idle_before(self, current: _Decoded) -> list[float]:
        """Return, for each snapshot in the sequence, the idle seconds before it.

        That is the time from the end of the block before it to the start of
        its slew; zero for a snapshot that did not fit.
        """
        idle = []
        for before, after in zip(current.states, current.states[1:]):
            if after.cursor == before.cursor:
                # Not placed: _step returned the state unchanged.
                idle.append(0.0)
                continue
            block = after.timeline[after.cursor - 1]
            pred = (
                after.timeline[after.cursor - 2] if after.cursor > 1 else self._origin
            )
            start = pred.end if pred is not None else self.ctx.ustart
            idle.append(max(0.0, float(block.entry.begin) - start))
        return idle

    def _released(self, obsid: int) -> _Snapshot:
        """Return a full-length snapshot of ``obsid`` with no release time."""
        return _Snapshot(
            obsid=obsid, seconds=float(self._by_obsid[obsid].target.ss_max)
        )
