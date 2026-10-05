"""Plan with the OR-Tools CP-SAT constraint solver.

:class:`CpSatPlanner` splits the horizon into consecutive chunks and, for each,
lets CP-SAT choose which snapshots to observe and in what order. A candidate is
one snapshot of a request in one of its visibility windows (as many per window
as the exposure needs and the window holds), with an optional start and a
collection length between the request's ``ss_min`` and ``ss_max``. A circuit
through the chosen candidates sets their order. Each slew starts on the first
simulation step after the task before it ends, and once its target is visible,
as ACS requires. Ground passes are fixed tasks. The objective is the merit-weighted,
earliness-discounted science time that
:class:`~conops.schedulers.LocalSearchPlanner` maximizes.

The model approximates slews (each target's roll is chosen before solving), so
each chunk's order is decoded with the planner's exact checks before the next
chunk is solved from where it leaves the spacecraft. Every plan therefore
executes as planned.
"""

import time
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np

from ..common.vector import quaternion_attitude_delta
from ..config import MissionConfig
from ..config.acs import scheduled_slew_time
from ..simulation.slew import Slew
from ..targets import Plan, Pointing
from .context import Attitude
from .local_search import LocalSearchPlanner, Score, _Decoded, _Snapshot, _State
from .priority_planner import _Block, _Request, program_shares

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

    1. The priority-first plan is built. Its snapshots in each chunk are
       the solver's starting hint.
    2. The horizon is split into ``chunk`` lengths. For each, CP-SAT chooses
       snapshots and their order within its share of ``solver_time_limit``,
       leaving room and exposure for the rest of the priority-first plan to
       follow. The order is decoded with the planner's exact checks and kept
       if it scores at least as well as the priority-first plan's snapshots
       for the chunk; otherwise, or if the solver finds no solution in time,
       those are kept instead.
    3. Any remaining ``time_limit`` is spent improving the result by local
       search (see :class:`~conops.schedulers.LocalSearchPlanner`).

    The plan returned is never worse than the priority-first plan;
    :attr:`solver_chunks_used` shows where the solver improved on it. Takes the arguments of
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
        try:
            from ortools.sat.python import cp_model  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "CpSatPlanner needs OR-Tools (the ortools package), a dependency "
                "of coast-sim; reinstall coast-sim, or use LocalSearchPlanner"
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

        self.solver_chunks_used = []

        states = [self._initial_state()]
        sequence: list[_Snapshot] = []
        chunk_end = self.ctx.ustart
        while chunk_end < self.ctx.uend:
            chunk_begin, chunk_end = chunk_end, min(chunk_end + chunk, self.ctx.uend)
            # The chunk leaves the rest of the priority-first plan room to
            # follow it unchanged: it finishes before that plan's next slew and
            # leaves the exposure collected later to it. Its snapshots are kept
            # only if, followed by the rest of that plan, they score at least
            # as well as that plan does from here, so the result is never worse
            # than the priority-first plan.
            later = self._released_between(start, chunk_end, self.ctx.uend)
            planned = self._released_between(start, chunk_begin, chunk_end)
            budget = dict(states[-1].remaining)
            for snapshot in later:
                budget[snapshot.obsid] -= snapshot.seconds
            finish = later[0].release if later else None
            solved = self._solve_chunk(
                states[-1], chunk_end, hints, per_chunk, budget, finish or self.ctx.uend
            )
            used = solved is not None and self._final_score(
                states[-1], solved + later
            ) >= self._final_score(states[-1], planned + later)
            self.solver_chunks_used.append(used)
            for snapshot in solved if used and solved is not None else planned:
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
        from ortools.sat.python import cp_model

        origin_time, origin_attitude = self._chunk_origin(state)
        if origin_time >= chunk_end:
            return []
        tasks = [_Task(None, (origin_attitude, origin_attitude), (0, 0))]
        tasks.extend(self._pass_tasks(state, origin_time, chunk_end))
        hinted = self._hinted_windows(hints, origin_time, chunk_end)
        tasks.extend(
            self._candidates(state, budget, origin_time, chunk_end, finish, hinted)
        )
        if not any(task.optional for task in tasks):
            return []

        slew = self._slews(tasks)
        chunk_seconds = int(round(chunk_end - origin_time))
        step = self.ctx.step_size
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
        # The step each task's slew starts on; chunks start on a step.
        last_step = max(task.arrival[1] for task in tasks) // step
        slew_step = [
            model.new_int_var(0, last_step, f"slew{k}") for k in range(len(tasks))
        ]
        for k, task in enumerate(tasks):
            if task.optional:
                # Collection, cleanup and handoff finish inside the window.
                model.add(
                    arrival[k] + task.overhead + collection[k] <= task.window[1]
                ).only_enforce_if(present[k])
                # ACS starts a slew only once its target is visible.
                model.add(step * slew_step[k] >= task.window[0]).only_enforce_if(
                    present[k]
                )
                # The chunk holds the snapshots whose slews start in it.
                model.add(step * slew_step[k] < chunk_seconds).only_enforce_if(
                    present[k]
                )
            if task.copy > 0 and tasks[k - 1].key == task.key:
                # A window's snapshots of a request are used in order.
                model.add_implication(present[k], present[k - 1])
                model.add(
                    arrival[k] >= arrival[k - 1] + task.overhead + collection[k - 1]
                ).only_enforce_if(present[k])
        spaced: dict[int, list[int]] = {}
        for k, task in enumerate(tasks):
            if task.spacing is not None and task.obsid is not None:
                spaced.setdefault(task.obsid, []).append(k)
        for visits in spaced.values():
            # Candidates are in time order, and each visit waits for its
            # cadence after the collection before it ends.
            for n, i in enumerate(visits):
                spacing = tasks[i].spacing
                assert spacing is not None
                for j in visits[n + 1 :]:
                    model.add(
                        step * slew_step[j] >= arrival[i] + collection[i] + spacing
                    ).only_enforce_if([present[i], present[j]])

        arcs: list[tuple[int, int, cp_model.LiteralT]] = []
        arc_literals: dict[tuple[int, int], cp_model.IntVar] = {}
        for k, task in enumerate(tasks):
            if task.optional:
                arcs.append((k, k, ~present[k]))
        empty = model.new_bool_var("empty")
        arcs.append((0, 0, empty))
        for i, before in enumerate(tasks):
            for j, after in enumerate(tasks):
                if i == j or j == 0:
                    continue
                earliest = _ceil(before.arrival[0] + before.shortest, step)
                if after.optional:
                    earliest = max(earliest, _ceil(after.window[0], step))
                if earliest + int(slew[i, j]) > after.arrival[1]:
                    continue
                literal = model.new_bool_var(f"arc{i}_{j}")
                arcs.append((i, j, literal))
                arc_literals[(i, j)] = literal
                # The slew starts on the first step after the task before ends.
                model.add(
                    step * slew_step[j]
                    >= arrival[i] + before.duration + before.overhead + collection[i]
                ).only_enforce_if(literal)
                model.add(
                    arrival[j] >= step * slew_step[j] + int(slew[i, j])
                ).only_enforce_if(literal)
            if i != 0:
                closing = model.new_bool_var(f"arc{i}_0")
                arcs.append((i, 0, closing))
                arc_literals[(i, 0)] = closing
        # Arrival times rise along every arc, so the only circuit that skips
        # the start is the start's own loop, with nothing else present.
        model.add_circuit(arcs)

        totals: dict[int, list[cp_model.IntVar]] = {}
        counted: dict[int, cp_model.IntVar] = {}
        delayed: dict[int, cp_model.IntVar] = {}
        objective = []
        for k, task in enumerate(tasks):
            obsid = task.obsid
            if obsid is None:
                continue
            counted[k] = model.new_int_var(0, task.collection[1], f"counted{k}")
            model.add(counted[k] == collection[k]).only_enforce_if(present[k])
            model.add(counted[k] == 0).only_enforce_if(~present[k])
            totals.setdefault(obsid, []).append(counted[k])
            objective.append(task.rate * counted[k] + task.offset * present[k])
            if task.earliness > 0.0:
                delayed[k] = model.new_int_var(0, task.arrival[1], f"delay{k}")
                model.add(delayed[k] == arrival[k]).only_enforce_if(present[k])
                model.add(delayed[k] == 0).only_enforce_if(~present[k])
                objective.append(-task.earliness * delayed[k])
        for obsid, counted_list in totals.items():
            model.add(sum(counted_list) <= int(budget[obsid]))
        model.maximize(sum(objective))

        # Hint every variable, so the solver starts from the priority-first
        # plan's snapshots even when its time runs out before it improves them.
        hint = self._hint(tasks, hinted, origin_time, slew, budget, chunk_seconds)
        chain = [0, *hint, 0]
        on_chain = set(zip(chain, chain[1:]))
        model.add_hint(empty, len(chain) == 2)
        for (i, j), literal in arc_literals.items():
            model.add_hint(literal, (i, j) in on_chain)
        for k, task in enumerate(tasks):
            first_step = _ceil(task.window[0], step) // step
            moved, start, length = hint.get(
                k, (first_step, task.arrival[0], task.collection[0])
            )
            model.add_hint(slew_step[k], moved)
            model.add_hint(arrival[k], start)
            model.add_hint(collection[k], length)
            if task.optional:
                model.add_hint(present[k], k in hint)
                model.add_hint(counted[k], length if k in hint else 0)
                if k in delayed:
                    model.add_hint(delayed[k], start if k in hint else 0)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = self.workers
        solver.parameters.random_seed = self.seed
        # Presolve can take longer than a short time limit leaves a chunk, and
        # the search does better with the time.
        solver.parameters.cp_model_presolve = False
        status = solver.solve(model)
        self.solver_statuses.append(solver.status_name(status))
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

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

    def _hinted_windows(
        self, hints: Sequence[_Block], origin_time: float, chunk_end: float
    ) -> dict[WindowKey, list[tuple[float, float]]]:
        """Arrival and collection seconds of the priority-first snapshots in the chunk.

        A snapshot belongs to the chunk its slew starts in, as when the
        priority-first plan's snapshots are compared chunk by chunk. A long
        exposure can take several snapshots in one window; they are listed in
        time order.
        """
        hinted: dict[WindowKey, list[tuple[float, float]]] = {}
        for block in hints:
            assert block.slew is not None
            if not origin_time <= float(block.slew.slewstart) < chunk_end:
                continue
            arrival = float(block.slew.slewend)
            obsid = int(block.entry.obsid)
            entry = block.entry
            seconds = float(entry.collection_end or 0.0) - float(
                entry.collection_begin or 0.0
            )
            for w0, w1 in self._by_obsid[obsid].windows:
                if w0 <= arrival < w1:
                    hinted.setdefault((obsid, w0), []).append((arrival, seconds))
                    break
        return hinted

    def _candidates(
        self,
        state: _State,
        budget: dict[int, float],
        origin_time: float,
        chunk_end: float,
        finish: float,
        hinted: dict[WindowKey, list[tuple[float, float]]],
    ) -> list[_Task]:
        """Candidate snapshots of each request in each window open during the chunk.

        A snapshot's value counts the program shares delivered before the
        chunk, and a request whose visits wait for its cadence has candidates
        only after its last visit's cadence has passed.
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
            shortest = int(np.ceil(float(target.ss_min)))
            longest = int(np.floor(min(float(target.ss_max), remaining)))
            if longest < shortest:
                continue
            opens = origin_time
            spacing = None
            if request.cadence is not None:
                spacing = int(np.ceil(setup + request.cadence))
                last = state.last_visit.get(obsid)
                if last is not None:
                    opens = max(opens, last + request.cadence)
            for w0, w1 in request.windows:
                if w1 <= opens or w0 >= chunk_end:
                    continue
                window = (
                    int(np.ceil(max(w0, opens) - origin_time)),
                    int(np.floor(min(w1, ctx.uend, finish) - origin_time)),
                )
                # The slew must start in the chunk (see _solve_chunk); the
                # snapshot may arrive after it ends.
                latest = window[1] - overhead - shortest
                if target.deadline is not None:
                    latest = min(latest, int(target.deadline - origin_time - setup))
                if latest < window[0] or window[0] >= chunk_end - origin_time:
                    continue
                attitude = target.target_body_attitude(
                    ctx.instrument_roll(target, max(w0, origin_time))
                )
                scale = weights[request.merit.tier]
                rate = scale * self._value(request, shares)
                earliness = offset = 0.0
                if target.deadline is not None and self.earliness_weight > 0.0:
                    available = target.deadline - ctx.ustart
                    if available > 0:
                        # Delay counts from the horizon start, at collection
                        # start, valued at the candidate's longest collection.
                        earliness = rate * longest * self.earliness_weight / available
                        offset = -earliness * (origin_time + setup - ctx.ustart)
                if request.cadence is not None and self.earliness_weight > 0.0:
                    # A visit is worth taking soon after it is due, so the
                    # visits keep to the cadence: a visit a whole cadence late
                    # loses earliness_weight of its value.
                    late = rate * longest * self.earliness_weight / request.cadence
                    earliness += late
                    offset -= late * (origin_time + setup - opens)
                # As many snapshots as the exposure needs and the window holds.
                copies = min(
                    int(np.ceil(remaining / longest)),
                    max(1, (window[1] - window[0]) // (overhead + shortest)),
                )
                if spacing is not None:
                    copies = min(copies, 1 + (window[1] - window[0]) // spacing)
                priority = 0 if (obsid, w0) in hinted else 1
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

    def _hint(
        self,
        tasks: Sequence[_Task],
        hinted: dict[WindowKey, list[tuple[float, float]]],
        origin_time: float,
        slew: np.ndarray,
        budget: dict[int, float],
        chunk_seconds: int,
    ) -> dict[int, tuple[int, int, int]]:
        """The priority-first plan's snapshots, and the passes, as a feasible hint.

        Returns the slew's start step, the arrival from the chunk's origin and
        the collection seconds of each hinted task, in time order. The
        snapshots are repaired to satisfy the model: its slews are approximate
        and can be a little longer than the exact ones, so a snapshot is moved
        later where it needs to be, shortened to leave room for the next pass,
        and left out if it no longer fits. The solver's first solution is then
        at least as good as the hint.
        """
        step = self.ctx.step_size
        wanted: dict[int, tuple[float, float]] = {}
        for k, task in enumerate(tasks):
            snapshots = hinted.get(task.key) if task.key is not None else None
            if snapshots is not None and task.copy < len(snapshots):
                arrival, seconds = snapshots[task.copy]
                wanted[k] = (arrival - origin_time, seconds)
        passes = [k for k, task in enumerate(tasks) if k and not task.optional]
        order = sorted(
            [*wanted, *passes],
            key=lambda k: wanted[k][0] if k in wanted else tasks[k].arrival[0],
        )
        left = {
            obsid: int(budget[obsid])
            for obsid in {task.obsid for task in tasks if task.obsid is not None}
        }
        hint: dict[int, tuple[int, int, int]] = {}
        due: dict[int, int] = {}
        """Earliest slew of each cadence request's next visit."""
        previous, free = 0, 0
        for position, k in enumerate(order):
            task = tasks[k]
            obsid = task.obsid
            opens = 0
            if obsid is not None:
                opens = max(task.window[0], due.get(obsid, 0))
            moved = _ceil(max(free, opens), step)
            if obsid is None:
                hint[k] = (moved // step, task.arrival[0], 0)
                previous, free = k, task.arrival[0] + task.duration
                continue
            start = max(
                int(np.ceil(wanted[k][0])),
                moved + int(slew[previous, k]),
                task.arrival[0],
            )
            finish = task.window[1]
            following = next((n for n in order[position + 1 :] if n in passes), None)
            if following is not None:
                # The pass's slew starts on a step after this snapshot ends.
                latest_slew = tasks[following].arrival[0] - int(slew[k, following])
                finish = min(finish, latest_slew - latest_slew % step)
            seconds = min(
                max(int(wanted[k][1]), task.collection[0]),
                task.collection[1],
                left[obsid],
                finish - start - task.overhead,
            )
            if (
                start > task.arrival[1]
                or seconds < task.collection[0]
                or moved >= chunk_seconds
            ):
                continue
            hint[k] = (moved // step, start, seconds)
            left[obsid] -= seconds
            if task.spacing is not None:
                due[obsid] = start + seconds + task.spacing
            previous, free = k, start + task.overhead + seconds
        return hint

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


def _ceil(seconds: int, step: int) -> int:
    """Round seconds up to a whole number of steps."""
    return -(-seconds // step) * step
