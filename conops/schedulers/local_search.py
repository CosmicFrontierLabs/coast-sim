"""Improve a priority-first plan by local search over its observation sequence.

A plan's science can be described by the order of its snapshots alone. A
decoder turns an order into a timeline by placing each snapshot at the earliest
time it fits after the one before, with the checks
:class:`~conops.schedulers.PriorityPlanner` applies, so every decoded timeline
executes as planned. Decoding in time order also closes the gaps that placing
requests in priority order leaves behind.

:class:`LocalSearchPlanner` starts from the priority-first plan and searches
for a better order by inserting, removing, swapping and moving snapshots,
keeping the best plan it finds within a time or iteration budget.
"""

import random
import time
from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from ..config import MissionConfig
from ..targets import Plan, Pointing
from .priority_planner import PriorityPlanner, _Block, _Request

Score = tuple[float, ...]
"""Merit-weighted science seconds per tier, highest tier first."""


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
    score: dict[int, float]
    """Merit-weighted science seconds by tier."""


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


class LocalSearchPlanner(PriorityPlanner):
    """Build a priority-first plan, then improve it by local search.

    The search changes the order of the plan's science snapshots and decodes
    each order into a timeline (see the module description). It tries:

    * inserting a snapshot of a request with exposure left to place,
    * releasing a snapshot from the time and length the priority-first plan
      gave it,
    * removing a snapshot,
    * swapping two nearby snapshots, and
    * moving a snapshot to a nearby position.

    A change is kept by late-acceptance hill climbing: when it is no worse than
    the current plan, or than the plan of ``history_length`` steps ago, so the
    search can cross plateaus. The objective is merit-weighted science time,
    compared tier by tier from the highest. A snapshot of a request with a
    deadline is worth less the later it starts: starting at the deadline
    rather than the start of the horizon costs ``earliness_weight`` of its
    value. The best plan found is returned, and never one worse than the
    priority-first plan.

    Ground passes and locked entries stay where the priority-first plan put
    them. Takes the same arguments as
    :class:`~conops.schedulers.PriorityPlanner`, plus:

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

    planner_name = "local_search"

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
        **options: object,
    ) -> None:
        super().__init__(config, targets, begin, end, **options)  # type: ignore[arg-type]
        if time_limit < 0:
            raise ValueError("time_limit must not be negative")
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
        self.initial_score: Score = ()
        """Objective of the priority-first plan."""
        self.start_score: Score = ()
        """Objective of the decoded starting sequence; equals ``initial_score``."""
        self.score: Score = ()
        """Objective of the returned plan."""

    def schedule(self) -> Plan:
        """Build the priority-first plan, then search for a better one."""
        self._reserve_fixed()
        fixed = [self._copy_block(block) for block in self.timeline]
        requests = self._requests()
        self._by_obsid = {int(r.target.obsid): r for r in requests}
        self._initial_remaining = {
            obsid: r.remaining for obsid, r in self._by_obsid.items()
        }
        self._tiers = sorted({r.merit.tier for r in requests}, reverse=True)
        fixed_entries = {id(block.entry) for block in self.timeline}

        self._place_requests(requests)
        greedy_timeline = list(self.timeline)
        greedy_score = self._timeline_score(
            [b for b in greedy_timeline if id(b.entry) not in fixed_entries]
        )
        self.initial_score = greedy_score

        self._fixed = fixed
        science = [b for b in greedy_timeline if id(b.entry) not in fixed_entries]
        # Released at their priority-first times, the snapshots decode to
        # the priority-first plan; moves release them one at a time.
        start = self._decode(
            [
                _Snapshot(
                    obsid=int(b.entry.obsid),
                    seconds=b.entry.collection_seconds_between(
                        b.entry.begin, b.entry.end
                    ),
                    release=float(b.entry.begin),
                )
                for b in science
            ]
        )
        self.start_score = self._objective(start.final)
        best = self._best = self._search(start)

        best_score = self._objective(best.final)
        if best_score >= greedy_score:
            self.score = best_score
            self._set_unplaced(best.final.remaining)
            return self._finish(best.final.timeline)
        self.score = greedy_score
        return self._finish(greedy_timeline)

    # ── Objective ────────────────────────────────────────────────────────

    def _objective(self, state: _State) -> Score:
        return tuple(state.score.get(tier, 0.0) for tier in self._tiers)

    def _timeline_score(self, science: Sequence[_Block]) -> Score:
        """Score a priority-first timeline's science blocks."""
        totals: dict[int, float] = {}
        for block in science:
            request = self._by_obsid.get(int(block.entry.obsid))
            if request is None:
                continue
            tier = request.merit.tier
            totals[tier] = totals.get(tier, 0.0) + self._contribution(request, block)
        return tuple(totals.get(tier, 0.0) for tier in self._tiers)

    def _contribution(self, request: _Request, block: _Block) -> float:
        """Return a snapshot's merit-weighted science, discounted for lateness."""
        entry = block.entry
        worth = request.merit.value * entry.collection_seconds_between(
            entry.begin, entry.end
        )
        deadline = request.target.deadline
        if deadline is None or self.earliness_weight == 0.0:
            return worth
        available = deadline - self.ctx.ustart
        if available <= 0.0 or entry.collection_begin is None:
            return worth
        delay = max(0.0, float(entry.collection_begin) - self.ctx.ustart)
        return worth * (1.0 - self.earliness_weight * min(1.0, delay / available))

    def _set_unplaced(self, remaining: dict[int, float]) -> None:
        self.unplaced = [
            request.target
            for obsid, request in self._by_obsid.items()
            if remaining[obsid] >= float(request.target.ss_min)
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
            score={},
        )

    def _decode(self, sequence: list[_Snapshot]) -> _Decoded:
        """Decode a whole sequence from the start."""
        states = [self._initial_state()]
        for snapshot in sequence:
            states.append(self._step(states[-1], snapshot))
        return _Decoded.model_construct(sequence=sequence, states=states)

    def _step(self, state: _State, snapshot: _Snapshot) -> _State:
        """Place the next snapshot at its earliest fit after the cursor."""
        obsid = snapshot.obsid
        request = self._by_obsid[obsid]
        target = request.target
        remaining = state.remaining[obsid]
        if remaining < float(target.ss_min):
            return state
        # _fit_in_gap sizes the snapshot from the request's remaining exposure.
        request.remaining = min(remaining, snapshot.seconds)
        timeline = state.timeline
        for index in range(state.cursor, len(timeline) + 1):
            pred = timeline[index - 1] if index > 0 else self._origin
            succ = timeline[index] if index < len(timeline) else None
            fit = self._fit_in_gap(request, pred, succ, snapshot.release)
            if fit is None:
                continue
            block, slew, succ_slew = fit
            self._apply_slew(block, slew)
            placed = [*timeline[:index], block]
            if succ is not None:
                assert succ_slew is not None
                moved = self._copy_block(succ)
                self._apply_slew(moved, succ_slew)
                placed.append(moved)
                placed.extend(timeline[index + 1 :])
            seconds = block.entry.collection_seconds_between(
                block.entry.begin, block.entry.end
            )
            score = dict(state.score)
            tier = request.merit.tier
            score[tier] = score.get(tier, 0.0) + self._contribution(request, block)
            left = dict(state.remaining)
            left[obsid] = remaining - seconds
            return _State.model_construct(
                timeline=placed, cursor=index + 1, remaining=left, score=score
            )
        return state

    @staticmethod
    def _same_state(a: _State, b: _State) -> bool:
        """Return whether decoding continues identically from two states."""
        if (
            a.remaining != b.remaining
            or len(a.timeline) - a.cursor != len(b.timeline) - b.cursor
        ):
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
            delta = {
                tier: here.score.get(tier, 0.0)
                + later.score.get(tier, 0.0)
                - there.score.get(tier, 0.0)
                for tier in {*here.score, *later.score, *there.score}
            }
            states.append(
                _State.model_construct(
                    timeline=[*prefix, *later.timeline[there.cursor :]],
                    cursor=later.cursor + offset,
                    remaining=later.remaining,
                    score=delta,
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
        """Return a changed sequence and the span it changed, or None."""
        sequence = current.sequence
        size = len(sequence)
        kind = rng.random()
        if kind < 0.3:
            # Insert a full-length snapshot of a request with exposure left.
            open_requests = [
                obsid
                for obsid, request in self._by_obsid.items()
                if current.final.remaining[obsid] >= float(request.target.ss_min)
            ]
            if not open_requests:
                return None
            obsid = rng.choice(open_requests)
            at = rng.randint(0, size)
            added = self._released(obsid)
            return [*sequence[:at], added, *sequence[at:]], at, at, 1
        if size == 0:
            return None
        if kind < 0.55:
            # Release a snapshot from its earlier time and length.
            at = rng.randrange(size)
            if sequence[at] == self._released(sequence[at].obsid):
                return None
            changed = list(sequence)
            changed[at] = self._released(sequence[at].obsid)
            return changed, at, at, 0
        if kind < 0.65:
            at = rng.randrange(size)
            return [*sequence[:at], *sequence[at + 1 :]], at, at - 1, -1
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

    def _released(self, obsid: int) -> _Snapshot:
        """Return a full-length snapshot of ``obsid`` with no release time."""
        return _Snapshot(
            obsid=obsid, seconds=float(self._by_obsid[obsid].target.ss_max)
        )
