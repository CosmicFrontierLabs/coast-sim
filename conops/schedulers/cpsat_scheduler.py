"""Plan with the OR-Tools CP-SAT constraint solver.

:class:`CpSatScheduler` splits the horizon into consecutive chunks and, for each,
lets CP-SAT choose which snapshots to observe and in what order. A candidate is
one snapshot of a request in one of its visibility windows (as many per window
as the exposure needs and the window holds), with an optional start and a
collection length between the request's ``ss_min`` and ``ss_max``. A circuit
through the chosen candidates sets their order. Each slew starts on the first
simulation step after the task before it ends, and once its target is visible,
as ACS requires. Ground passes are fixed tasks. The objective is the
merit-weighted, earliness-discounted science time that
:class:`~conops.schedulers.LocalSearchScheduler` maximizes.

The model approximates slews (each target's roll is chosen before solving), so
each chunk's order is decoded with the scheduler's exact checks before the next
chunk is solved from where it leaves the spacecraft. Every plan therefore
executes as planned.
"""

import time
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ..common.vector import quaternion_attitude_delta
from ..config import MissionConfig
from ..config.acs import scheduled_slew_time
from ..simulation.slew import Slew
from ..targets import Plan, Pointing
from .context import Attitude
from .local_search_scheduler import (
    LocalSearchScheduler,
    Score,
    _Decoded,
    _Snapshot,
    _State,
)
from .priority_scheduler import _Block, _Request, program_shares

if TYPE_CHECKING:
    from ortools.sat.python import cp_model

WindowKey = tuple[int, float]
"""A request's obsid and the Unix start time of one of its visibility windows."""

_SEED_RANGE = 2**31
"""Seeds CP-SAT accepts are 0 up to this, exclusive."""


class _PlannedSnapshot(NamedTuple):
    """Where the priority-first plan put a snapshot."""

    arrival: float
    """When its slew ends."""
    seconds: float
    """Its collection length."""


class _TaskHint(NamedTuple):
    """A task's place in the solver's starting solution."""

    slew_step: int
    """The step its slew starts on."""
    arrival: int
    """Its arrival, in seconds from the chunk's origin."""
    collection: int
    """Its collection length, in seconds."""


class _Task:
    """One node of a chunk's circuit: the start, a pass, or a candidate snapshot.

    Times are whole seconds from the chunk's origin, when the spacecraft is free
    to start the chunk's first slew.
    """

    def __init__(
        self,
        key: WindowKey | None,
        attitude: tuple[Attitude, Attitude],
        arrival: tuple[int, int],
        *,
        copy: int = 0,
        spacing: int | None = None,
        duration: int = 0,
        overhead: int = 0,
        collection: tuple[int, int] = (0, 0),
        window: tuple[int, int] = (0, 0),
        rate: float = 0.0,
        earliness: float = 0.0,
        offset: float = 0.0,
    ) -> None:
        self.key = key
        """Request and window of a candidate snapshot; None for the start and passes."""
        self.copy = copy
        """Which of the window's snapshots of the request this is, from 0."""
        self.spacing = spacing
        """For a request whose visits wait for its cadence, seconds from this
        snapshot's arrival plus its collection to the next visit's slew."""
        self.attitude_in, self.attitude_out = attitude
        self.arrival = arrival
        """Earliest and latest arrival."""
        self.duration = duration
        """Seconds from arrival to the end of a fixed task."""
        self.overhead = overhead
        """Setup, cleanup and handoff seconds of a candidate snapshot."""
        self.collection = collection
        """Shortest and longest collection of a candidate snapshot, in seconds."""
        self.window = window
        """The candidate's visibility window."""
        self.rate = rate
        """Objective value per second of collection."""
        self.earliness = earliness
        """Value per second of collection lost per second the arrival is delayed."""
        self.offset = offset
        """Value per second of collection added to the rate for an arrival at the
        chunk's origin, before the earliness discount; negative when the
        discount has already begun there."""

    @property
    def obsid(self) -> int | None:
        return self.key[0] if self.key is not None else None

    @property
    def optional(self) -> bool:
        """Whether this is a candidate the solver may leave out."""
        return self.key is not None

    @property
    def shortest(self) -> int:
        """Seconds from arrival to the end, at the shortest collection."""
        return self.duration + self.overhead + self.collection[0]


class _ChunkModel:
    """The CP-SAT model of one chunk.

    Task 0 is the start, where the spacecraft is at the chunk's origin. The
    solver chooses which candidates are present, a circuit from the start
    through them and the passes that orders them, and each task's slew step,
    arrival and collection length.
    """

    def __init__(
        self,
        tasks: Sequence[_Task],
        slews: np.ndarray,
        step: int,
        chunk_seconds: int,
        budget: dict[int, float],
    ) -> None:
        from ortools.sat.python import cp_model

        self.tasks = tasks
        self.slews = slews
        """Seconds of the scheduled slew from each task to each other task."""
        self.step = step
        self.chunk_seconds = chunk_seconds
        model = self.model = cp_model.CpModel()
        self.arrival = [
            model.new_int_var(*task.arrival, f"arrival{k}")
            for k, task in enumerate(tasks)
        ]
        self.present = [
            model.new_bool_var(f"present{k}")
            if task.optional
            else model.new_constant(1)
            for k, task in enumerate(tasks)
        ]
        self.collection = [
            model.new_int_var(*task.collection, f"collection{k}")
            for k, task in enumerate(tasks)
        ]
        # The step each task's slew starts on; chunks start on a step.
        last_step = max(task.arrival[1] for task in tasks) // step
        self.slew_step = [
            model.new_int_var(0, last_step, f"slew{k}") for k in range(len(tasks))
        ]
        self._add_snapshot_constraints()
        self._add_cadence_spacing()
        self._add_circuit()
        self._add_objective(budget)

    def _add_snapshot_constraints(self) -> None:
        """Keep each present candidate in its window and the chunk."""
        model, step = self.model, self.step
        arrival, present, collection = self.arrival, self.present, self.collection
        for k, task in enumerate(self.tasks):
            if task.optional:
                # Collection, cleanup and handoff finish inside the window.
                model.add(
                    arrival[k] + task.overhead + collection[k] <= task.window[1]
                ).only_enforce_if(present[k])
                # ACS starts a slew only once its target is visible.
                model.add(step * self.slew_step[k] >= task.window[0]).only_enforce_if(
                    present[k]
                )
                # The chunk holds the snapshots whose slews start in it.
                model.add(
                    step * self.slew_step[k] < self.chunk_seconds
                ).only_enforce_if(present[k])
            if task.copy > 0 and self.tasks[k - 1].key == task.key:
                # A window's snapshots of a request are used in order.
                model.add_implication(present[k], present[k - 1])
                model.add(
                    arrival[k] >= arrival[k - 1] + task.overhead + collection[k - 1]
                ).only_enforce_if(present[k])

    def _add_cadence_spacing(self) -> None:
        """Keep a cadence request's visits at least its cadence apart."""
        visits_of: dict[int, list[int]] = {}
        for k, task in enumerate(self.tasks):
            if task.spacing is not None and task.obsid is not None:
                visits_of.setdefault(task.obsid, []).append(k)
        for visits in visits_of.values():
            # Candidates are ranked, with the windows the priority-first plan
            # used first, so a later window can come before an earlier one.
            # Put the visits in time order: each waits for its cadence after
            # the collection before it ends.
            visits.sort(key=lambda k: (self.tasks[k].window[0], self.tasks[k].copy))
            for n, i in enumerate(visits):
                spacing = self.tasks[i].spacing
                assert spacing is not None
                for j in visits[n + 1 :]:
                    self.model.add(
                        self.step * self.slew_step[j]
                        >= self.arrival[i] + self.collection[i] + spacing
                    ).only_enforce_if([self.present[i], self.present[j]])

    def _add_circuit(self) -> None:
        """Order the tasks by a circuit from the start, timing each slew.

        An arc joins two tasks only if the second can be reached in time
        straight after the first.
        """
        model, step, tasks = self.model, self.step, self.tasks
        arrival, collection, slew_step = self.arrival, self.collection, self.slew_step
        # The earliest step a slew can leave each task, and reach a candidate.
        leaves = [_ceil(task.arrival[0] + task.shortest, step) for task in tasks]
        opens = [_ceil(task.window[0], step) if task.optional else 0 for task in tasks]
        arcs: list[tuple[int, int, cp_model.LiteralT]] = []
        self.arc_literals: dict[tuple[int, int], cp_model.IntVar] = {}
        # A candidate left out loops to itself.
        for k, task in enumerate(self.tasks):
            if task.optional:
                arcs.append((k, k, ~self.present[k]))
        self.empty = model.new_bool_var("empty")
        """Whether the chunk observes nothing: the start loops to itself."""
        arcs.append((0, 0, self.empty))
        for i, before in enumerate(tasks):
            for j in range(1, len(tasks)):
                if i == j:
                    continue
                earliest = max(leaves[i], opens[j])
                if earliest + int(self.slews[i, j]) > tasks[j].arrival[1]:
                    continue
                literal = model.new_bool_var(f"arc{i}_{j}")
                arcs.append((i, j, literal))
                self.arc_literals[(i, j)] = literal
                # The slew starts on the first step after the task before ends.
                model.add(
                    step * slew_step[j]
                    >= arrival[i] + before.duration + before.overhead + collection[i]
                ).only_enforce_if(literal)
                model.add(
                    arrival[j] >= step * slew_step[j] + int(self.slews[i, j])
                ).only_enforce_if(literal)
            if i != 0:
                closing = model.new_bool_var(f"arc{i}_0")
                arcs.append((i, 0, closing))
                self.arc_literals[(i, 0)] = closing
        # Arrival times rise along every arc, so the only circuit that skips
        # the start is the start's own loop, with nothing else present.
        model.add_circuit(arcs)

    def _add_objective(self, budget: dict[int, float]) -> None:
        """Maximize the value of the collection, within each request's budget."""
        model, present = self.model, self.present
        self.counted: dict[int, cp_model.IntVar] = {}
        """A candidate's collection if present, and 0 if not."""
        self.collection_arrival: dict[int, cp_model.IntVar] = {}
        """A deadline or cadence candidate's collection times its arrival, which
        its earliness discount follows; 0 if it is absent."""
        totals: dict[int, list[cp_model.IntVar]] = {}
        objective = []
        for k, task in enumerate(self.tasks):
            obsid = task.obsid
            if obsid is None:
                continue
            counted = self.counted[k] = model.new_int_var(
                0, task.collection[1], f"counted{k}"
            )
            model.add(counted == self.collection[k]).only_enforce_if(present[k])
            model.add(counted == 0).only_enforce_if(~present[k])
            totals.setdefault(obsid, []).append(counted)
            objective.append((task.rate + task.offset) * counted)
            if task.earliness > 0.0:
                # A late snapshot loses a fraction of its value, so a shorter
                # one loses less: the discount is per second of collection.
                # counted is 0 when the candidate is absent, and so is this.
                product = self.collection_arrival[k] = model.new_int_var(
                    0, task.collection[1] * task.arrival[1], f"collection_arrival{k}"
                )
                model.add_multiplication_equality(product, [counted, self.arrival[k]])
                objective.append(-task.earliness * product)
        for obsid, counted_list in totals.items():
            model.add(sum(counted_list) <= int(budget[obsid]))
        model.maximize(sum(objective))

    def add_hint(self, hint: dict[int, _TaskHint]) -> None:
        """Hint every variable, from the hinted tasks in time order.

        The solver then starts from the hint even when its time runs out
        before it improves on it. Tasks left out of the hint are hinted absent.
        """
        model = self.model
        chain = [0, *hint, 0]
        on_chain = set(zip(chain, chain[1:]))
        model.add_hint(self.empty, len(chain) == 2)
        for (i, j), literal in self.arc_literals.items():
            model.add_hint(literal, (i, j) in on_chain)
        for k, task in enumerate(self.tasks):
            first_step = _ceil(task.window[0], self.step) // self.step
            slew_step, arrival, collection = hint.get(
                k, (first_step, task.arrival[0], task.collection[0])
            )
            model.add_hint(self.slew_step[k], slew_step)
            model.add_hint(self.arrival[k], arrival)
            model.add_hint(self.collection[k], collection)
            if task.optional:
                model.add_hint(self.present[k], k in hint)
                model.add_hint(self.counted[k], collection if k in hint else 0)
                if k in self.collection_arrival:
                    model.add_hint(
                        self.collection_arrival[k],
                        collection * arrival if k in hint else 0,
                    )

    def solve(
        self, time_limit: float, workers: int, seed: int
    ) -> tuple[str, list[_Snapshot] | None]:
        """Solve, returning CP-SAT's status and the chosen snapshots in order.

        The snapshots are None if the solver found no solution in time.
        """
        from ortools.sat.python import cp_model

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = workers
        # CP-SAT's seed is a signed 32-bit integer; the search's seed, like
        # Python's, can be any integer.
        solver.parameters.random_seed = seed % _SEED_RANGE
        # Presolve can take longer than a short time limit leaves a chunk, and
        # the search does better with the time.
        solver.parameters.cp_model_presolve = False
        status = solver.solve(self.model)
        status_name = solver.status_name(status)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return status_name, None

        chosen = sorted(
            (solver.value(self.arrival[k]), k)
            for k, task in enumerate(self.tasks)
            if task.optional and solver.boolean_value(self.present[k])
        )
        snapshots = []
        for _, k in chosen:
            obsid = self.tasks[k].obsid
            assert obsid is not None
            seconds = float(solver.value(self.collection[k]))
            snapshots.append(_Snapshot(obsid=obsid, seconds=seconds))
        return status_name, snapshots


class CpSatScheduler(LocalSearchScheduler):
    """Plan chunk by chunk with CP-SAT, decoding each chunk exactly.

    1. The priority-first plan is built. Its snapshots in each chunk are
       the solver's starting hint.
    2. The horizon is split into ``chunk`` lengths. For each, CP-SAT chooses
       snapshots and their order within its share of ``solver_time_limit``,
       leaving room and exposure for the rest of the priority-first plan to
       follow. The order is decoded with the scheduler's exact checks and kept
       if it scores at least as well as the priority-first plan's snapshots
       for the chunk; otherwise, or if the solver finds no solution in time,
       those are kept instead.
    3. Any remaining ``time_limit`` is spent improving the result by local
       search (see :class:`~conops.schedulers.LocalSearchScheduler`).

    The plan returned is never worse than the priority-first plan;
    :attr:`solver_chunks_used` shows where the solver improved on it. Takes
    the arguments of :class:`~conops.schedulers.LocalSearchScheduler`
    (``time_limit`` defaults to 0, so no local search), plus:

    Args:
        solver_time_limit: Seconds CP-SAT may search, shared across chunks.
        chunk: Length of each chunk the horizon is solved in.
        workers: CP-SAT search workers. Use 1, with ``seed``, for runs that
            repeat exactly.
        max_candidates: Candidate snapshots in a chunk's model at most, those
            of the priority-first plan first, then in priority order.
    """

    scheduler_name = "cp_sat"

    def __init__(
        self,
        config: MissionConfig,
        targets: Sequence[Pointing],
        begin: datetime,
        end: datetime,
        *,
        solver_time_limit: float = 20.0,
        chunk: timedelta = timedelta(hours=3),
        workers: int = 8,
        max_candidates: int = 120,
        time_limit: float = 0.0,
        **options: object,
    ) -> None:
        super().__init__(
            config,
            targets,
            begin,
            end,
            time_limit=time_limit,
            **options,  # type: ignore[arg-type]
        )
        if solver_time_limit <= 0 or chunk <= timedelta(0):
            raise ValueError("solver_time_limit and chunk must be positive")
        if workers < 1 or max_candidates < 1:
            raise ValueError("workers and max_candidates must be positive")
        self.solver_time_limit = solver_time_limit
        self.chunk = chunk
        self.workers = workers
        self.max_candidates = max_candidates
        self.solver_statuses: list[str] = []
        """CP-SAT's final status for each chunk, such as OPTIMAL or FEASIBLE."""
        self.solver_chunks_used: list[bool] = []
        """For each chunk, whether the solver's snapshots were kept; if not,
        the priority-first plan's were, as they scored better."""
        self.solver_score: Score | None = None
        """Objective of the solver's plan after exact decoding."""
        self.solver_seconds = 0.0
        """Wall-clock time spent building, solving and decoding chunks."""

    def schedule(self) -> Plan:
        """Build the priority-first plan, solve chunk by chunk, and optionally polish."""
        start = self._start_from_priority_plan()
        began = time.perf_counter()
        solved = self._solve(start)
        self.solver_seconds = time.perf_counter() - began
        self.solver_score = self._objective(solved.final)
        candidate = solved if self.solver_score >= self.start_score else start
        best = self._best = self._search(candidate)
        return self._finish_best(best)

    # ── Chunks ───────────────────────────────────────────────────────────

    def _solve(self, start: _Decoded) -> _Decoded:
        """Solve and decode each chunk in turn."""
        _require_ortools()
        hints = [
            block
            for block in start.final.timeline
            if int(block.entry.obsid) in self._by_obsid and block.slew is not None
        ]
        chunk = self.chunk.total_seconds()
        chunks = max(1, int(np.ceil((self.ctx.uend - self.ctx.ustart) / chunk)))
        per_chunk = self.solver_time_limit / chunks
        self.solver_statuses = []
        self.solver_chunks_used = []

        states = [self._initial_state()]
        sequence: list[_Snapshot] = []
        chunk_end = self.ctx.ustart
        while chunk_end < self.ctx.uend:
            chunk_begin, chunk_end = chunk_end, min(chunk_end + chunk, self.ctx.uend)
            state = states[-1]
            # The chunk leaves the rest of the priority-first plan room to
            # follow it unchanged: it finishes before that plan's next slew and
            # leaves the exposure collected later to it. Its snapshots are kept
            # only if, followed by the rest of that plan, they score at least
            # as well as that plan does from here, so the result is never worse
            # than the priority-first plan.
            planned = self._released_between(start, chunk_begin, chunk_end)
            later = self._released_between(start, chunk_end, self.ctx.uend)
            budget = dict(state.remaining)
            for snapshot in later:
                budget[snapshot.obsid] -= snapshot.seconds
            finish = later[0].release if later else None
            solved = self._solve_chunk(
                state,
                chunk_end,
                hints,
                per_chunk,
                budget,
                finish if finish is not None else self.ctx.uend,
            )
            kept, used = planned, False
            if solved is not None and self._final_score(
                state, solved + later
            ) >= self._final_score(state, planned + later):
                kept, used = solved, True
            self.solver_chunks_used.append(used)
            for snapshot in kept:
                sequence.append(snapshot)
                states.append(self._step(states[-1], snapshot))
        return _Decoded.model_construct(sequence=sequence, states=states)

    @staticmethod
    def _released_between(start: _Decoded, begin: float, end: float) -> list[_Snapshot]:
        """The priority-first plan's snapshots whose slews start in ``[begin, end)``."""
        return [
            snapshot
            for snapshot in start.sequence
            if snapshot.release is not None and begin <= snapshot.release < end
        ]

    def _final_score(self, state: _State, sequence: Sequence[_Snapshot]) -> Score:
        """Objective after decoding ``sequence`` from ``state``."""
        for snapshot in sequence:
            state = self._step(state, snapshot)
        return self._objective(state)

    def _chunk_origin(self, state: _State) -> tuple[float, Attitude]:
        """Return when and from which attitude the next snapshot can start."""
        if state.cursor > 0:
            last = state.timeline[state.cursor - 1]
            return self.ctx.ceil_step(last.end), last.attitude_out
        if self._origin is not None:
            return self._origin.end, self._origin.attitude_out
        return self.ctx.ustart, self.ctx.initial_attitude(self.ctx.ustart)

    def _solve_chunk(
        self,
        state: _State,
        chunk_end: float,
        hints: Sequence[_Block],
        time_limit: float,
        budget: dict[int, float],
        finish: float,
    ) -> list[_Snapshot] | None:
        """Choose and order the snapshots that start in one chunk.

        ``budget`` is the most exposure the chunk may collect for each request,
        and every snapshot ends by ``finish``. Returns None if the solver found
        no solution in time.
        """
        origin_time, origin_attitude = self._chunk_origin(state)
        if origin_time >= chunk_end:
            return []
        planned = self._planned_in_chunk(hints, origin_time, chunk_end)
        tasks = [
            _Task(None, (origin_attitude, origin_attitude), (0, 0)),
            *self._pass_tasks(state, origin_time, chunk_end),
            *self._candidates(state, budget, origin_time, chunk_end, finish, planned),
        ]
        if not any(task.optional for task in tasks):
            return []

        slews = self._slews(tasks)
        chunk_seconds = int(round(chunk_end - origin_time))
        chunk = _ChunkModel(tasks, slews, self.ctx.step_size, chunk_seconds, budget)
        chunk.add_hint(
            self._hint(tasks, planned, origin_time, slews, budget, chunk_seconds)
        )
        status, snapshots = chunk.solve(time_limit, self.workers, self.seed)
        self.solver_statuses.append(status)
        return snapshots

    # ── Tasks ────────────────────────────────────────────────────────────

    def _pass_tasks(
        self, state: _State, origin_time: float, chunk_end: float
    ) -> list[_Task]:
        """Passes after the cursor that a chunk's snapshots must leave room for."""
        # A snapshot's slew starts in the chunk, so it ends at most a slew,
        # its setup and cleanup, and its longest collection after the chunk.
        reach = (
            chunk_end
            + Slew.duration_upper_bound(self.config.spacecraft_bus.attitude_control)
            + self.ctx.setup_seconds
            + self.ctx.post_collection_seconds
            + max(
                (float(r.target.ss_max) for r in self._by_obsid.values()),
                default=0.0,
            )
        )
        tasks = []
        for block in state.timeline[state.cursor :]:
            if block.ready >= reach:
                break
            ready = int(round(block.ready - origin_time))
            end = int(round(block.end - origin_time))
            if ready < 0:
                continue
            tasks.append(
                _Task(
                    None,
                    (block.attitude_in, block.attitude_out),
                    (ready, ready),
                    duration=max(0, end - ready),
                )
            )
        return tasks

    def _planned_in_chunk(
        self, hints: Sequence[_Block], origin_time: float, chunk_end: float
    ) -> dict[WindowKey, list[_PlannedSnapshot]]:
        """The priority-first plan's snapshots in the chunk, by request and window.

        A snapshot belongs to the chunk its slew starts in, as when the
        priority-first plan's snapshots are compared chunk by chunk. A long
        exposure can take several snapshots in one window; they are listed in
        time order.
        """
        planned: dict[WindowKey, list[_PlannedSnapshot]] = {}
        for block in hints:
            assert block.slew is not None
            if not origin_time <= float(block.slew.slewstart) < chunk_end:
                continue
            arrival = float(block.slew.slewend)
            entry = block.entry
            obsid = int(entry.obsid)
            seconds = float(entry.collection_end or 0.0) - float(
                entry.collection_begin or 0.0
            )
            for w0, w1 in self._by_obsid[obsid].windows:
                if w0 <= arrival < w1:
                    planned.setdefault((obsid, w0), []).append(
                        _PlannedSnapshot(arrival, seconds)
                    )
                    break
        return planned

    def _candidates(
        self,
        state: _State,
        budget: dict[int, float],
        origin_time: float,
        chunk_end: float,
        finish: float,
        planned: dict[WindowKey, list[_PlannedSnapshot]],
    ) -> list[_Task]:
        """Candidate snapshots of each request in each window open during the chunk.

        Candidates in windows the priority-first plan used come first, then the
        rest, each in priority order, up to ``max_candidates``.
        """
        ctx = self.ctx
        setup = ctx.setup_seconds
        overhead = int(np.ceil(setup + ctx.post_collection_seconds))
        weights = self._tier_weights()
        shares = program_shares(state.programs) if self._deficit_weight else {}
        ranked = sorted(
            self._by_obsid.values(), key=lambda r: r.merit.value_rank, reverse=True
        )
        candidates: list[tuple[int, int, _Task]] = []
        for rank, request in enumerate(ranked):
            target = request.target
            obsid = int(target.obsid)
            remaining = budget[obsid]
            # A snapshot must collect something, as in the priority-first
            # scheduler, even with an ss_min of zero; this also keeps the
            # snapshot counts below from dividing by zero.
            shortest = max(1, int(np.ceil(float(target.ss_min))))
            longest = int(np.floor(min(float(target.ss_max), remaining)))
            if longest < shortest:
                continue
            opens = self._opens(request, state, origin_time)
            spacing = None
            if request.cadence is not None:
                spacing = int(np.ceil(setup + request.cadence))
            for w0, w1 in request.windows:
                if w1 <= opens or w0 >= chunk_end:
                    continue
                window = (
                    int(np.ceil(max(w0, opens) - origin_time)),
                    int(np.floor(min(w1, ctx.uend, finish) - origin_time)),
                )
                # The slew must start in the chunk (see _ChunkModel); the
                # snapshot may arrive after it ends.
                latest = window[1] - overhead - shortest
                if target.deadline is not None:
                    latest = min(latest, int(target.deadline - origin_time - setup))
                if latest < window[0] or window[0] >= chunk_end - origin_time:
                    continue
                attitude = target.target_body_attitude(
                    ctx.instrument_roll(target, max(w0, origin_time))
                )
                rate, earliness, offset = self._candidate_value(
                    request, weights, shares, origin_time, opens
                )
                # As many snapshots as the exposure needs and the window holds.
                copies = min(
                    int(np.ceil(remaining / longest)),
                    max(1, (window[1] - window[0]) // (overhead + shortest)),
                )
                if spacing is not None:
                    copies = min(copies, 1 + (window[1] - window[0]) // spacing)
                priority = 0 if (obsid, w0) in planned else 1
                for copy in range(copies):
                    task = _Task(
                        (obsid, w0),
                        (attitude, attitude),
                        (window[0], latest),
                        copy=copy,
                        spacing=spacing,
                        overhead=overhead,
                        collection=(shortest, longest),
                        window=window,
                        rate=rate,
                        earliness=earliness,
                        offset=offset,
                    )
                    candidates.append((priority, rank, task))
        # Stable, so a window's snapshots stay together and in order.
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [task for _, _, task in candidates[: self.max_candidates]]

    def _opens(self, request: _Request, state: _State, origin_time: float) -> float:
        """Earliest arrival of the request's next snapshot.

        That is no earlier than the chunk's origin, its earliest start less
        setup, and, for a cadence request, a cadence after its last visit.
        """
        opens = origin_time
        target = request.target
        if target.earliest_start is not None:
            # Collection begins setup after the snapshot arrives.
            opens = max(opens, target.earliest_start - self.ctx.setup_seconds)
        if request.cadence is not None:
            last = state.last_visit.get(int(target.obsid))
            if last is not None:
                opens = max(opens, last + request.cadence)
        return opens

    def _candidate_value(
        self,
        request: _Request,
        weights: dict[int, float],
        shares: dict[str, float],
        origin_time: float,
        opens: float,
    ) -> tuple[float, float, float]:
        """A candidate's objective terms per second of collection: its rate,
        earliness and offset (see :class:`_Task`).

        The value counts the program shares delivered before the chunk. A
        candidate of a request with a deadline, or with a cadence, loses value
        the later it arrives, in proportion to the collection it gets.
        """
        ctx = self.ctx
        setup = ctx.setup_seconds
        target = request.target
        rate = weights[request.merit.tier] * self._value(request, shares)
        earliness = offset = 0.0
        if target.deadline is not None and self.earliness_weight > 0.0:
            released = max(ctx.ustart, target.earliest_start or ctx.ustart)
            available = target.deadline - released
            if available > 0:
                # Delay counts from the horizon start, or the earliest
                # start if later, at collection start, as in local search.
                earliness = rate * self.earliness_weight / available
                offset = -earliness * (origin_time + setup - released)
        if request.cadence is not None and self.earliness_weight > 0.0:
            # A visit is worth taking soon after it is due, so the
            # visits keep to the cadence: a visit a whole cadence late
            # loses earliness_weight of its value.
            late = rate * self.earliness_weight / request.cadence
            earliness += late
            offset -= late * (origin_time + setup - opens)
        return rate, earliness, offset

    def _hint(
        self,
        tasks: Sequence[_Task],
        planned: dict[WindowKey, list[_PlannedSnapshot]],
        origin_time: float,
        slews: np.ndarray,
        budget: dict[int, float],
        chunk_seconds: int,
    ) -> dict[int, _TaskHint]:
        """The priority-first plan's snapshots, and the passes, as a feasible hint.

        Returns a hint for each hinted task, in time order. The snapshots are
        repaired to satisfy the model: its slews are approximate and can be a
        little longer than the exact ones, so a snapshot is moved later where
        it needs to be, shortened to leave room for the next pass, and left
        out if it no longer fits. The solver's first solution is then at least
        as good as the hint.
        """
        step = self.ctx.step_size
        # Where the priority-first plan put each candidate, from the chunk's origin.
        wanted: dict[int, _PlannedSnapshot] = {}
        for k, task in enumerate(tasks):
            snapshots = planned.get(task.key) if task.key is not None else None
            if snapshots is not None and task.copy < len(snapshots):
                arrival, seconds = snapshots[task.copy]
                wanted[k] = _PlannedSnapshot(arrival - origin_time, seconds)
        passes = [k for k, task in enumerate(tasks) if k and not task.optional]
        order = sorted(
            [*wanted, *passes],
            key=lambda k: wanted[k].arrival if k in wanted else tasks[k].arrival[0],
        )
        exposure_left = {
            obsid: int(budget[obsid])
            for obsid in {task.obsid for task in tasks if task.obsid is not None}
        }
        hint: dict[int, _TaskHint] = {}
        due: dict[int, int] = {}
        """Earliest slew of each cadence request's next visit."""
        # The last task hinted, and when the spacecraft is free after it.
        previous, free = 0, 0
        for position, k in enumerate(order):
            task = tasks[k]
            obsid = task.obsid
            if obsid is None:
                # A pass stays where it is.
                slew_step = _ceil(free, step) // step
                hint[k] = _TaskHint(slew_step, task.arrival[0], 0)
                previous, free = k, task.arrival[0] + task.duration
                continue
            slew_start = _ceil(max(free, task.window[0], due.get(obsid, 0)), step)
            arrival = max(
                int(np.ceil(wanted[k].arrival)),
                slew_start + int(slews[previous, k]),
                task.arrival[0],
            )
            ends_by = task.window[1]
            next_pass = next((n for n in order[position + 1 :] if n in passes), None)
            if next_pass is not None:
                # The pass's slew starts on a step after this snapshot ends.
                latest_slew = tasks[next_pass].arrival[0] - int(slews[k, next_pass])
                ends_by = min(ends_by, latest_slew - latest_slew % step)
            seconds = min(
                max(int(wanted[k].seconds), task.collection[0]),
                task.collection[1],
                exposure_left[obsid],
                ends_by - arrival - task.overhead,
            )
            if (
                arrival > task.arrival[1]
                or seconds < task.collection[0]
                or slew_start >= chunk_seconds
            ):
                continue
            hint[k] = _TaskHint(slew_start // step, arrival, seconds)
            exposure_left[obsid] -= seconds
            if task.spacing is not None:
                due[obsid] = arrival + seconds + task.spacing
            previous, free = k, arrival + task.overhead + seconds
        return hint

    # ── Model inputs ─────────────────────────────────────────────────────

    def _tier_weights(self) -> dict[int, float]:
        """Weights that keep every tier above all the tiers below it."""
        weights: dict[int, float] = {}
        below = 0.0
        for tier in sorted(self._tiers):
            weights[tier] = below + 1.0
            tier_total = sum(
                self._value_bound(r) * self._initial_remaining[obsid]
                for obsid, r in self._by_obsid.items()
                if r.merit.tier == tier
            )
            below += weights[tier] * tier_total
        if not weights:
            return {}
        top = weights[max(self._tiers)]
        return {tier: weight / top for tier, weight in weights.items()}

    def _value_bound(self, request: _Request) -> float:
        """Most a second of the request's science can be worth."""
        if request.category.time_share is None:
            return request.steady_value
        return request.steady_value + self._deficit_weight

    def _slews(self, tasks: Sequence[_Task]) -> np.ndarray:
        """Seconds of the scheduled slew from each task to each other task."""
        acs = self.config.spacecraft_bus.attitude_control
        size = len(tasks)
        result = np.zeros((size, size), dtype=np.int64)
        cache: dict[tuple[Attitude, Attitude], int] = {}
        for i, before in enumerate(tasks):
            for j, after in enumerate(tasks):
                if i == j:
                    continue
                key = (before.attitude_out, after.attitude_in)
                seconds = cache.get(key)
                if seconds is None:
                    distance, axis = quaternion_attitude_delta(*key[0], *key[1])
                    seconds = scheduled_slew_time(acs.slew_time(distance, axis))
                    cache[key] = seconds
                result[i, j] = seconds
        return result


def _require_ortools() -> None:
    """Raise an ImportError that says how to install OR-Tools if it is missing."""
    try:
        from ortools.sat.python import cp_model  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "CpSatScheduler needs OR-Tools (the ortools package); install it "
            'with pip install "coast-sim[cpsat]", or use LocalSearchScheduler'
        ) from exc


def _ceil(seconds: int, step: int) -> int:
    """Round seconds up to a whole number of steps."""
    return -(-seconds // step) * step
