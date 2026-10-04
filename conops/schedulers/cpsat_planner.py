"""Plan with the OR-Tools CP-SAT constraint solver.

:class:`CpSatPlanner` splits the horizon into consecutive chunks and, for each,
lets CP-SAT choose which snapshots to observe and in what order. A candidate is
one request in one of its visibility windows, with an optional start and a
collection length between the request's ``ss_min`` and ``ss_max``. A circuit
through the chosen candidates sets their order, with the slew between them as
the gap they need; a slew must also start once the target is visible, as ACS
requires. Ground passes are fixed tasks. The objective is the merit-weighted,
earliness-discounted science time that
:class:`~conops.schedulers.LocalSearchPlanner` maximizes.

The model approximates slews (each target's roll is chosen before solving), so
each chunk's order is decoded with the planner's exact checks before the next
chunk is solved from where it leaves the spacecraft. Every plan therefore
executes as planned. Requires the optional ``ortools`` dependency
(``pip install coast-sim[cpsat]``).
"""

import time
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np

from ..common.vector import quaternion_attitude_delta
from ..config import MissionConfig
from ..config.acs import scheduled_slew_time
from ..targets import Plan, Pointing
from .context import Attitude
from .local_search import LocalSearchPlanner, Score, _Decoded, _Snapshot, _State
from .priority_planner import _Block

WindowKey = tuple[int, float]
"""A request's obsid and the Unix start time of one of its visibility windows."""


class _Task:
    """One node of a chunk's circuit: the start, a pass, or a candidate snapshot."""

    def __init__(
        self,
        key: WindowKey | None,
        attitude: tuple[Attitude, Attitude],
        arrival: tuple[int, int],
        *,
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
        self.attitude_in, self.attitude_out = attitude
        self.arrival = arrival
        """Earliest and latest arrival, in seconds from the chunk's origin."""
        self.duration = duration
        """Seconds from arrival to the end of a fixed task."""
        self.overhead = overhead
        """Setup, cleanup and handoff seconds of a candidate snapshot."""
        self.collection = collection
        """Shortest and longest collection of a candidate snapshot, in seconds."""
        self.window = window
        """The candidate's visibility window, in seconds from the chunk's origin."""
        self.rate = rate
        """Objective value per second of collection."""
        self.earliness = earliness
        """Objective value lost per second the arrival is delayed."""
        self.offset = offset
        """Objective value of observing the candidate at all, beyond the rate."""

    @property
    def obsid(self) -> int | None:
        return self.key[0] if self.key is not None else None

    @property
    def optional(self) -> bool:
        return self.key is not None

    @property
    def shortest(self) -> int:
        """Seconds from arrival to the end, at the shortest collection."""
        return self.duration + self.overhead + self.collection[0]


class CpSatPlanner(LocalSearchPlanner):
    """Plan chunk by chunk with CP-SAT, decoding each chunk exactly.

    1. The priority-first plan is built. It is the fallback, and its
       snapshots in each chunk are the solver's starting hint.
    2. The horizon is split into ``chunk`` lengths. For each, CP-SAT chooses
       snapshots and their order within its share of ``solver_time_limit``,
       and the order is decoded with the planner's exact checks.
    3. Any remaining ``time_limit`` is spent improving the result by local
       search (see :class:`~conops.schedulers.LocalSearchPlanner`).

    The best plan found is returned, never one worse than the priority-first
    plan. Takes the arguments of
    :class:`~conops.schedulers.LocalSearchPlanner` (``time_limit`` defaults to
    0, so no local search), plus:

    Args:
        solver_time_limit: Seconds CP-SAT may search, shared across chunks.
        chunk: Length of each chunk the horizon is solved in.
        workers: CP-SAT search workers. Use 1, with ``seed``, for runs that
            repeat exactly.
        max_candidates: Candidate snapshots in a chunk's model at most, those
            of the priority-first plan first, then in priority order.
    """

    planner_name = "cp_sat"

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
        try:
            from ortools.sat.python import cp_model  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "CpSatPlanner needs OR-Tools: install coast-sim[cpsat], or use "
                "LocalSearchPlanner"
            ) from exc

        hints = [
            block
            for block in start.final.timeline
            if int(block.entry.obsid) in self._by_obsid and block.slew is not None
        ]
        chunk = self.chunk.total_seconds()
        chunks = max(1, int(np.ceil((self.ctx.uend - self.ctx.ustart) / chunk)))
        per_chunk = self.solver_time_limit / chunks
        self.solver_statuses = []

        states = [self._initial_state()]
        sequence: list[_Snapshot] = []
        chunk_end = self.ctx.ustart
        while chunk_end < self.ctx.uend:
            chunk_end = min(chunk_end + chunk, self.ctx.uend)
            for snapshot in self._solve_chunk(states[-1], chunk_end, hints, per_chunk):
                sequence.append(snapshot)
                states.append(self._step(states[-1], snapshot))
        return _Decoded.model_construct(sequence=sequence, states=states)

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
    ) -> list[_Snapshot]:
        """Choose and order the snapshots that start in one chunk."""
        from ortools.sat.python import cp_model

        origin_time, origin_attitude = self._chunk_origin(state)
        if origin_time >= chunk_end:
            return []
        tasks = [_Task(None, (origin_attitude, origin_attitude), (0, 0))]
        tasks.extend(self._pass_tasks(state, origin_time, chunk_end))
        hinted = self._hinted_windows(hints, origin_time, chunk_end)
        tasks.extend(self._candidates(state, origin_time, chunk_end, hinted))
        if not any(task.optional for task in tasks):
            return []

        transition = self._transitions(tasks)
        model = cp_model.CpModel()
        arrival = [
            model.new_int_var(*task.arrival, f"arrival{k}")
            for k, task in enumerate(tasks)
        ]
        present = [
            model.new_bool_var(f"present{k}")
            if task.optional
            else model.new_constant(1)
            for k, task in enumerate(tasks)
        ]
        collection = [
            model.new_int_var(*task.collection, f"collection{k}")
            for k, task in enumerate(tasks)
        ]
        for k, task in enumerate(tasks):
            if task.optional:
                # Collection, cleanup and handoff finish inside the window.
                model.add(
                    arrival[k] + task.overhead + collection[k] <= task.window[1]
                ).only_enforce_if(present[k])

        arcs: list[tuple[int, int, cp_model.LiteralT]] = []
        arc_literals: dict[tuple[int, int], cp_model.IntVar] = {}
        for k, task in enumerate(tasks):
            if task.optional:
                arcs.append((k, k, ~present[k]))
        arcs.append((0, 0, model.new_bool_var("empty")))
        for i, before in enumerate(tasks):
            for j, after in enumerate(tasks):
                if i == j or j == 0:
                    continue
                gap = int(transition[i, j])
                if before.arrival[0] + before.shortest + gap > after.arrival[1]:
                    continue
                if after.optional and after.window[0] + gap > after.arrival[1]:
                    continue
                literal = model.new_bool_var(f"arc{i}_{j}")
                arcs.append((i, j, literal))
                arc_literals[(i, j)] = literal
                model.add(
                    arrival[j]
                    >= arrival[i]
                    + before.duration
                    + before.overhead
                    + collection[i]
                    + gap
                ).only_enforce_if(literal)
                if after.optional:
                    # ACS starts a slew only once its target is visible.
                    model.add(arrival[j] >= after.window[0] + gap).only_enforce_if(
                        literal
                    )
            if i != 0:
                closing = model.new_bool_var(f"arc{i}_0")
                arcs.append((i, 0, closing))
                arc_literals[(i, 0)] = closing
        # Arrival times rise along every arc, so the only circuit that skips
        # the start is the start's own loop, with nothing else present.
        model.add_circuit(arcs)

        totals: dict[int, list[cp_model.IntVar]] = {}
        objective = []
        for k, task in enumerate(tasks):
            obsid = task.obsid
            if obsid is None:
                continue
            counted = model.new_int_var(0, task.collection[1], f"counted{k}")
            model.add(counted == collection[k]).only_enforce_if(present[k])
            model.add(counted == 0).only_enforce_if(~present[k])
            totals.setdefault(obsid, []).append(counted)
            objective.append(task.rate * counted + task.offset * present[k])
            if task.earliness > 0.0:
                delayed = model.new_int_var(0, task.arrival[1], f"delay{k}")
                model.add(delayed == arrival[k]).only_enforce_if(present[k])
                model.add(delayed == 0).only_enforce_if(~present[k])
                objective.append(-task.earliness * delayed)
        for obsid, counted_list in totals.items():
            model.add(sum(counted_list) <= int(state.remaining[obsid]))
        model.maximize(sum(objective))

        hint = self._hint(tasks, hinted, origin_time)
        for k, task in enumerate(tasks):
            if task.optional:
                model.add_hint(present[k], k in hint)
        for k, when in hint.items():
            model.add_hint(arrival[k], when)
        chain = [0, *sorted(hint, key=lambda k: hint[k]), 0]
        for i, j in zip(chain, chain[1:]):
            hinted_arc = arc_literals.get((i, j))
            if hinted_arc is not None:
                model.add_hint(hinted_arc, True)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = self.workers
        solver.parameters.random_seed = self.seed
        status = solver.solve(model)
        self.solver_statuses.append(solver.status_name(status))
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []

        chosen = sorted(
            (solver.value(arrival[k]), k)
            for k, task in enumerate(tasks)
            if task.optional and solver.boolean_value(present[k])
        )
        snapshots = []
        for _, k in chosen:
            obsid = tasks[k].obsid
            assert obsid is not None
            seconds = float(solver.value(collection[k]))
            snapshots.append(_Snapshot(obsid=obsid, seconds=seconds))
        return snapshots

    # ── Candidates ───────────────────────────────────────────────────────

    def _pass_tasks(
        self, state: _State, origin_time: float, chunk_end: float
    ) -> list[_Task]:
        """Passes after the cursor that a chunk's snapshots must leave room for."""
        reach = chunk_end + max(
            (float(r.target.ss_max) for r in self._by_obsid.values()), default=0.0
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

    def _hinted_windows(
        self, hints: Sequence[_Block], origin_time: float, chunk_end: float
    ) -> dict[WindowKey, float]:
        """Arrival time of each priority-first snapshot arriving in the chunk."""
        hinted: dict[WindowKey, float] = {}
        for block in hints:
            assert block.slew is not None
            arrival = float(block.slew.slewend)
            if not origin_time <= arrival < chunk_end:
                continue
            obsid = int(block.entry.obsid)
            for w0, w1 in self._by_obsid[obsid].windows:
                if w0 <= arrival < w1:
                    hinted.setdefault((obsid, w0), arrival)
                    break
        return hinted

    def _candidates(
        self,
        state: _State,
        origin_time: float,
        chunk_end: float,
        hinted: dict[WindowKey, float],
    ) -> list[_Task]:
        """Candidate snapshots: each request in each window open during the chunk."""
        ctx = self.ctx
        setup = ctx.setup_seconds
        overhead = int(np.ceil(setup + ctx.post_collection_seconds))
        weights = self._tier_weights()
        ranked = sorted(
            self._by_obsid.values(), key=lambda r: r.merit.value_rank, reverse=True
        )
        candidates: list[tuple[int, int, _Task]] = []
        for rank, request in enumerate(ranked):
            target = request.target
            obsid = int(target.obsid)
            remaining = state.remaining[obsid]
            shortest = int(np.ceil(float(target.ss_min)))
            longest = int(np.floor(min(float(target.ss_max), remaining)))
            if longest < shortest:
                continue
            for w0, w1 in request.windows:
                if w1 <= origin_time or w0 >= chunk_end:
                    continue
                window = (
                    int(np.ceil(max(w0, origin_time) - origin_time)),
                    int(np.floor(min(w1, ctx.uend) - origin_time)),
                )
                latest = min(
                    window[1] - overhead - shortest,
                    int(np.floor(chunk_end - origin_time)),
                )
                if target.deadline is not None:
                    latest = min(latest, int(target.deadline - origin_time - setup))
                if latest < window[0]:
                    continue
                attitude = target.target_body_attitude(
                    ctx.instrument_roll(target, max(w0, origin_time))
                )
                scale = weights[request.merit.tier]
                rate = scale * request.merit.value
                earliness = offset = 0.0
                if target.deadline is not None and self.earliness_weight > 0.0:
                    available = target.deadline - ctx.ustart
                    if available > 0:
                        # Delay counts from the horizon start, at collection
                        # start, valued at the candidate's longest collection.
                        earliness = rate * longest * self.earliness_weight / available
                        offset = -earliness * (origin_time + setup - ctx.ustart)
                task = _Task(
                    (obsid, w0),
                    (attitude, attitude),
                    (window[0], latest),
                    overhead=overhead,
                    collection=(shortest, longest),
                    window=window,
                    rate=rate,
                    earliness=earliness,
                    offset=offset,
                )
                priority = 0 if (obsid, w0) in hinted else 1
                candidates.append((priority, rank, task))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [task for _, _, task in candidates[: self.max_candidates]]

    @staticmethod
    def _hint(
        tasks: Sequence[_Task], hinted: dict[WindowKey, float], origin_time: float
    ) -> dict[int, int]:
        """Hinted arrival, from the chunk's origin, of each hinted candidate."""
        hint: dict[int, int] = {}
        for k, task in enumerate(tasks):
            if task.key is None or task.key not in hinted:
                continue
            when = int(round(hinted[task.key] - origin_time))
            hint[k] = min(max(when, task.arrival[0]), task.arrival[1])
        return hint

    def _tier_weights(self) -> dict[int, float]:
        """Weights that keep every tier above all the tiers below it."""
        weights: dict[int, float] = {}
        below = 0.0
        for tier in sorted(self._tiers):
            weights[tier] = below + 1.0
            tier_total = sum(
                r.merit.value * self._initial_remaining[obsid]
                for obsid, r in self._by_obsid.items()
                if r.merit.tier == tier
            )
            below += weights[tier] * tier_total
        if not weights:
            return {}
        top = weights[max(self._tiers)]
        return {tier: weight / top for tier, weight in weights.items()}

    def _transitions(self, tasks: Sequence[_Task]) -> np.ndarray:
        """Seconds from the end of each task to arrival at each other task.

        The slew between the tasks' attitudes, plus a step because a slew
        starts on the first simulation step after the task before it ends.
        """
        acs = self.config.spacecraft_bus.attitude_control
        step = self.ctx.step_size
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
                result[i, j] = seconds + step
        return result
