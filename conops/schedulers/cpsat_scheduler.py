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
:class:`~conops.schedulers.LocalSearchScheduler` maximizes. The onboard
recorder is in the model: each candidate's data goes on the recorder in the
segment, between contacts, where its collection starts, and every segment's
peak volume is kept within the planning limit, so the solver trades high-data
science against low-data science directly.

The model approximates slews (each target's roll is chosen before solving), so
each chunk's order is decoded with the scheduler's exact checks before the next
chunk is solved from where it leaves the spacecraft. Every plan therefore
executes as planned.
"""

import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ..common import ACSMode
from ..common.vector import quaternion_attitude_delta
from ..config import MissionConfig
from ..config.acs import scheduled_slew_time
from ..simulation.slew import Slew
from ..targets import Plan, Pointing
from ..targets.plan_entry import (
    extra_data_gb,
    instrument_modes_of,
    snapshot_min_seconds,
)
from .context import Attitude
from .local_search_scheduler import (
    LocalSearchScheduler,
    _Decoded,
    _Snapshot,
    _State,
)
from .priority_scheduler import _Block, _Request, program_shares
from .recorder import RecorderLedger

if TYPE_CHECKING:
    from ortools.sat.python import cp_model

WindowKey = tuple[int, float]
"""A request's obsid and the Unix start time of one of its visibility windows."""

SpanKey = tuple[int, int]
"""A request's obsid and the index of one of its spans of allocated time."""

_SEED_RANGE = 2**31
"""Seeds CP-SAT accepts are 0 up to this, exclusive."""

_KB_PER_GB = 1_000_000
"""The model counts data in kilobits, so that volumes are whole numbers:
whatever adds data is rounded up to a whole kilobit and whatever removes it
down, so the model never counts less data than the ledger does."""

_KB_ROUNDING = 1e-6
"""Kilobits within which a value is taken as the whole number it is next to,
so a rate that is whole in gigabits per second does not round up."""


def _kb_up(gb: float) -> int:
    return int(np.ceil(gb * _KB_PER_GB - _KB_ROUNDING))


def _kb_down(gb: float) -> int:
    return int(np.floor(gb * _KB_PER_GB + _KB_ROUNDING))


class _ChunkRecorder(NamedTuple):
    """The recorder over a chunk's segments, in kilobits and seconds from the origin.

    The segments are those from the one the chunk starts in to the end of the
    horizon: a candidate's data stays on the recorder, less what each contact
    removes, until the horizon ends, so every later peak counts.
    """

    edges: list[int]
    """Where each segment after the first starts: its contact's acquisition."""
    entering: int
    """The volume entering the first segment, after its contact."""
    accrual: list[int]
    """Housekeeping generated in each segment, outside its contact."""
    fixed: list[int]
    """Science already planned in each segment: before the chunk, locked, or
    the priority-first plan's snapshots after it."""
    removal: list[int]
    """What the contact at the start of each segment downlinks."""
    removal_accrual: list[int]
    """Housekeeping generated during that contact."""
    limit: int
    """The planning limit, rounded down and less a kilobit: the recorder is
    kept below the limit, not at it."""


class _Span(NamedTuple):
    """Allocated time a candidate snapshot can use, if it arrives in time."""

    key: SpanKey
    arrival: tuple[int, int]
    """Arrivals, from the chunk's origin, whose collection starts in the span."""
    rate: float
    """Objective value per allocated second, beyond the candidate's rate."""


class _ChunkBattery(NamedTuple):
    """The battery over a chunk's runs, in watt-seconds and seconds from the origin.

    The runs are the sunlit and eclipse runs from the one the chunk starts in
    to the end of the horizon. Energy in a run is linear in how long each
    candidate occupies it: the run's idle power for every second, replaced by
    the candidate's net power for the seconds it is there, and the slewing
    power for its slew in. The deficit at a run's end carries into the next,
    never below zero, and charging never takes it below the recharge floor
    or below where it entered the run already under the floor; it must stay
    within the limit at every run's end.
    """

    edges: list[int]
    """Where each run after the first starts."""
    sunlit: list[bool]
    """Whether each run is in sunlight."""
    entering: int
    """The deficit at the chunk's origin."""
    idle: list[int]
    """Watts drawn by idle time in each run: negative in a sunlit run with a
    charging attitude, where idle time charges."""
    slew: list[int]
    """Watts drawn by a slew in each run."""
    fixed: list[int]
    """Energy already planned in each run, relative to its idle power: the
    passes in the chunk and the priority-first plan's snapshots after it."""
    limit: int
    """The planning limit as a deficit."""
    floor: int
    """The deficit charging is not counted below."""
    horizon: int
    """Seconds from the origin to the end of the horizon."""


class _PlannedSnapshot(NamedTuple):
    """Where the priority-first plan put a snapshot."""

    arrival: float
    """When its slew ends."""
    seconds: float
    """Its collection length."""
    visit: int | None = None
    """Its visit window, for a target with visit windows."""


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
        setup: int = 0,
        collection: tuple[int, int] = (0, 0),
        window: tuple[int, int] = (0, 0),
        rate: float = 0.0,
        earliness: float = 0.0,
        offset: float = 0.0,
        late: float = 0.0,
        due: int = 0,
        cadence: int = 0,
        spans: Sequence[_Span] = (),
        data: tuple[int, int] = (0, 0),
        power: tuple[int, int] = (0, 0),
        visit: int | None = None,
    ) -> None:
        self.key = key
        """Request and window of a candidate snapshot; None for the start and passes."""
        self.visit = visit
        """The visit window of a candidate of a target with visit windows:
        at most one of a window's candidates is taken."""
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
        self.setup = setup
        """Seconds from a candidate's arrival to its collection, which places
        its data in a recorder segment."""
        self.collection = collection
        """Shortest and longest collection of a candidate snapshot, in seconds."""
        self.window = window
        """The candidate's visibility window, in which its slew starts."""
        self.rate = rate
        """Objective value per second of collection."""
        self.earliness = earliness
        """For a request with a deadline, value per second of collection lost
        per second the arrival is delayed."""
        self.offset = offset
        """Value per second of collection added to the rate for an arrival at the
        chunk's origin, before the earliness discount; negative when the
        discount has already begun there."""
        self.late = late
        """For a request whose visits wait for its cadence, value per second of
        collection lost per second the slew starts after the visit is due, for
        up to a cadence."""
        self.due = due
        """For such a request, when its first visit in the chunk is due."""
        self.cadence = cadence
        """For such a request, its cadence, in whole seconds."""
        self.spans = list(spans)
        """Allocated time the candidate's collection can count against."""
        self.data = data
        """Kilobits the snapshot puts on the recorder per second of collection,
        and once whatever its length."""
        self.power = power
        """Net watts the snapshot draws in sunlight, at its attitude, and in
        eclipse."""
        self.slew_in = 0
        """The shortest slew into the snapshot, in seconds."""

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
        allocated_left: Mapping[SpanKey, float] | None = None,
        recorder: _ChunkRecorder | None = None,
        battery: _ChunkBattery | None = None,
        category_caps: Sequence[tuple[int, Sequence[int]]] = (),
    ) -> None:
        from ortools.sat.python import cp_model

        self.tasks = tasks
        self.category_caps = list(category_caps)
        """For each category with a time budget: the seconds its candidates
        may collect in this chunk together, and the candidates' indices."""
        self.recorder = recorder
        self.battery = battery
        self.slews = slews
        """Seconds of the scheduled slew from each task to each other task."""
        self.step = step
        self.chunk_seconds = chunk_seconds
        self.allocated_left = dict(allocated_left or {})
        """Allocated seconds left in each span the candidates can use."""
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
        self.last_slew = step * last_step
        """The latest any slew can start."""
        self._add_snapshot_constraints()
        self._add_cadence_spacing()
        self._add_visit_windows()
        self._add_circuit()
        self._add_objective(budget)
        self._add_recorder()
        self._add_battery()

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

    def _add_visit_windows(self) -> None:
        """Take at most one candidate per visit window of a target with visit
        windows: a window that spans two visibility windows has a candidate
        in each."""
        by_visit: dict[tuple[int, int], list[int]] = {}
        for k, task in enumerate(self.tasks):
            if task.visit is not None and task.obsid is not None:
                by_visit.setdefault((task.obsid, task.visit), []).append(k)
        for candidates in by_visit.values():
            if len(candidates) > 1:
                self.model.add(sum(self.present[k] for k in candidates) <= 1)

    def _add_cadence_spacing(self) -> None:
        """Keep a cadence request's visits at least its cadence apart, and
        measure how late each visit is."""
        visits_of: dict[int, list[int]] = {}
        self.visits_of = visits_of
        """Each cadence request's candidates, in time order."""
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
        model, step = self.model, self.step
        self.next_due: dict[int, cp_model.IntVar] = {}
        """When the visit after a cadence candidate is due, if it is present."""
        self.due: dict[int, cp_model.IntVar] = {}
        """When a cadence candidate is due."""
        self.overdue: dict[int, cp_model.IntVar] = {}
        """At least the seconds a present cadence candidate's slew starts after
        it is due, and no less than 0."""
        self.lateness: dict[int, cp_model.IntVar] = {}
        """Its overdue seconds, up to a cadence; its discount follows this."""
        for visits in visits_of.values():
            if self.tasks[visits[0]].late == 0.0:
                continue
            for n, j in enumerate(visits):
                task = self.tasks[j]
                spacing = task.spacing
                assert spacing is not None
                next_due = self.next_due[j] = model.new_int_var(
                    0, _next_due_bound(task), f"next_due{j}"
                )
                model.add(
                    next_due == self.arrival[j] + self.collection[j] + spacing
                ).only_enforce_if(self.present[j])
                model.add(next_due == 0).only_enforce_if(~self.present[j])
                # A visit is due a cadence after the latest present visit
                # before it ends or, if none, when the request's next visit
                # is. An absent visit's next_due is 0, no later than task.due.
                earlier = [self.next_due[i] for i in visits[:n]]
                due = self.due[j] = model.new_int_var(
                    task.due,
                    max(
                        [
                            task.due,
                            *(_next_due_bound(self.tasks[i]) for i in visits[:n]),
                        ]
                    ),
                    f"due{j}",
                )
                model.add_max_equality(due, [task.due, *earlier])
                # Lateness up to a cadence costs value, so a visit after a
                # long gap is still worth taking.
                overdue = self.overdue[j] = model.new_int_var(
                    0, self.last_slew, f"overdue{j}"
                )
                model.add(overdue >= step * self.slew_step[j] - due).only_enforce_if(
                    self.present[j]
                )
                lateness = self.lateness[j] = model.new_int_var(
                    0, task.cadence, f"lateness{j}"
                )
                model.add_min_equality(lateness, [overdue, task.cadence])

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
        """A deadline candidate's collection times its arrival, which its
        earliness discount follows; 0 if it is absent."""
        self.collection_lateness: dict[int, cp_model.IntVar] = {}
        """A cadence candidate's collection times its lateness, which its
        discount follows; 0 if it is absent."""
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
            if k in self.lateness:
                # Likewise for a cadence visit's lateness.
                product = self.collection_lateness[k] = model.new_int_var(
                    0, task.collection[1] * task.cadence, f"collection_lateness{k}"
                )
                model.add_multiplication_equality(product, [counted, self.lateness[k]])
                objective.append(-task.late * product)
        for obsid, counted_list in totals.items():
            model.add(sum(counted_list) <= int(budget[obsid]))
        for cap, members in self.category_caps:
            # A category's time budget: its candidates share what is left.
            model.add(sum(self.counted[k] for k in members) <= cap)
        # Allocated seconds: collection that starts in a span of the request's
        # allocated time, up to what is left of it, is worth allocation_bonus
        # more.
        self.in_span: dict[tuple[int, int], cp_model.IntVar] = {}
        """Whether a candidate's collection starts in one of its spans."""
        self.span_seconds: dict[tuple[int, int], cp_model.IntVar] = {}
        """The seconds of that collection the span's allocated time counts."""
        used: dict[SpanKey, list[cp_model.IntVar]] = {}
        for k, task in enumerate(self.tasks):
            for n, span in enumerate(task.spans):
                within = self.in_span[(k, n)] = model.new_bool_var(f"in_span{k}_{n}")
                seconds = self.span_seconds[(k, n)] = model.new_int_var(
                    0, task.collection[1], f"span_seconds{k}_{n}"
                )
                model.add_implication(within, present[k])
                model.add(self.arrival[k] >= span.arrival[0]).only_enforce_if(within)
                model.add(self.arrival[k] <= span.arrival[1]).only_enforce_if(within)
                model.add(seconds <= self.counted[k])
                model.add(seconds == 0).only_enforce_if(~within)
                used.setdefault(span.key, []).append(seconds)
                objective.append(span.rate * seconds)
        for key, seconds_list in used.items():
            left = self.allocated_left.get(key, 0.0)
            if np.isfinite(left):
                model.add(sum(seconds_list) <= int(left))
        model.maximize(sum(objective))

    def _add_recorder(self) -> None:
        """Keep the recorder within its limit at the end of every segment.

        A candidate's data goes in the segment its collection starts in. Each
        segment's volume is what entered it after its contact, plus the
        housekeeping and science in it; what a contact cannot remove carries
        over. The carry-over is only bounded below, which is exact here: a
        larger carry-over only tightens the limit on the segments after it.
        """
        recorder = self.recorder
        if recorder is None:
            return
        from ortools.sat.python import cp_model

        model, present, counted = self.model, self.present, self.counted
        segments = len(recorder.accrual)
        starts = [-(2**40), *recorder.edges]
        ends = [*recorder.edges, 2**40]
        self.in_segment: dict[tuple[int, int], cp_model.IntVar] = {}
        """Whether a candidate's collection starts in a segment, where it could
        start in more than one."""
        volume: list[list[cp_model.LinearExprT]] = [[] for _ in range(segments)]
        for k, task in enumerate(self.tasks):
            if not task.optional or task.data == (0, 0):
                continue
            per_second, fixed = task.data
            # The snapshot's data, 0 when it is absent (counted is 0 then).
            total = per_second * counted[k] + fixed * present[k]
            lo = task.arrival[0] + task.setup
            hi = task.arrival[1] + task.setup
            possible = [s for s in range(segments) if starts[s] <= hi and lo < ends[s]]
            if len(possible) == 1:
                volume[possible[0]].append(total)
                continue
            bound = per_second * task.collection[1] + fixed
            literals = []
            for s in possible:
                literal = self.in_segment[(k, s)] = model.new_bool_var(
                    f"segment{k}_{s}"
                )
                literals.append(literal)
                begins = self.arrival[k] + task.setup
                if s > 0:
                    model.add(begins >= starts[s]).only_enforce_if(literal)
                if s < segments - 1:
                    model.add(begins < ends[s]).only_enforce_if(literal)
                part = model.new_int_var(0, bound, f"volume{k}_{s}")
                model.add(part == total).only_enforce_if(literal)
                model.add(part == 0).only_enforce_if(~literal)
                volume[s].append(part)
            model.add(sum(literals) == present[k])
        self.carry: list[cp_model.IntVar] = []
        """What each segment after the first starts with, after its contact."""
        level: cp_model.LinearExpr = cp_model.LinearExpr.constant(recorder.entering)
        for s in range(segments):
            if s > 0:
                carry = model.new_int_var(
                    0, recorder.limit + recorder.removal_accrual[s], f"carry{s}"
                )
                model.add(
                    carry >= level + recorder.removal_accrual[s] - recorder.removal[s]
                )
                self.carry.append(carry)
                level = carry
            level = cp_model.LinearExpr.sum(
                [level, recorder.accrual[s] + recorder.fixed[s], *volume[s]]
            )
            model.add(level <= recorder.limit)

    def _add_battery(self) -> None:
        """Keep the battery above its level at the end of every run.

        A candidate is in the run its arrival falls in, and what spills past
        that run's end is in the runs after, each for the seconds it holds
        of the candidate. Its energy is its net power less the
        run's idle power for the seconds it occupies in each, plus the slewing
        power less the idle power for its slew in; so a run's energy is its
        idle power for its whole length, plus what each candidate changes. In
        an eclipse run everything draws, so the run's end is its peak; in a
        sunlit run the deficit can rise and fall, and the decoder, which walks
        the plan exactly, has the last word. The carry-over between runs is
        only bounded below, which is exact: a larger carry-over only tightens
        the later limits. Where charging is credited only down to a recharge
        floor, the carry-over is also at least the smaller of the floor and
        the deficit entering the run, as the walk never goes below that.
        """
        battery = self.battery
        if battery is None:
            return
        from ortools.sat.python import cp_model

        model, present, counted = self.model, self.present, self.counted
        runs = len(battery.idle)
        starts = [-(2**40), *battery.edges]
        ends = [*battery.edges, 2**40]
        lengths = [
            min(ends[r], battery.horizon) - max(starts[r], 0) for r in range(runs)
        ]
        self.in_run: dict[tuple[int, int], cp_model.IntVar] = {}
        """Whether a candidate arrives in a run, where it could in more than one."""
        self.occupied: dict[int, cp_model.IntVar] = {}
        """Seconds a candidate occupies: its collection and overhead, 0 if absent."""
        energy: list[list[cp_model.LinearExprT]] = [[] for _ in range(runs)]
        for k, task in enumerate(self.tasks):
            if not task.optional:
                continue
            longest = task.collection[1] + task.overhead
            occupied = self.occupied[k] = model.new_int_var(0, longest, f"occupied{k}")
            model.add(occupied == counted[k] + task.overhead * present[k])
            lo, hi = task.arrival
            possible = [r for r in range(runs) if starts[r] <= hi and lo < ends[r]]
            literals = []
            # What spills past each run's end, the seconds of the candidate
            # after it: defined whatever run the candidate is in, so its
            # domain allows an arrival after the run's end, and 0 where the
            # candidate cannot reach past the end at all. A long candidate
            # spills through several runs: the seconds in each run after the
            # one it arrives in are what spills past the run before, less
            # what spills past the run itself.
            beyond: dict[int, cp_model.IntVar] = {}
            for b in range(runs - 1):
                if hi + longest - ends[b] <= 0:
                    continue
                spill = model.new_int_var(0, hi + longest - ends[b], f"spill{k}_{b}")
                model.add_max_equality(spill, [0, self.arrival[k] + occupied - ends[b]])
                beyond[b] = spill

            def rate(r: int) -> int:
                return (task.power[0] if battery.sunlit[r] else task.power[1]) - (
                    battery.idle[r]
                )

            for r in possible:
                slew_change = (battery.slew[r] - battery.idle[r]) * task.slew_in
                changes: dict[int, cp_model.LinearExprT] = {}
                inside: cp_model.LinearExpr = (
                    occupied - beyond[r] if r in beyond else occupied
                )
                changes[r] = rate(r) * inside + slew_change * present[k]
                bound = max(abs(rate(r)), 1) * longest + abs(slew_change)
                q = r + 1
                while q - 1 in beyond:
                    past: cp_model.LinearExpr | int = beyond[q] if q in beyond else 0
                    changes[q] = rate(q) * (beyond[q - 1] - past)
                    bound += abs(rate(q)) * longest
                    q += 1
                if len(possible) == 1:
                    for q, change in changes.items():
                        energy[q].append(change)
                    continue
                literal = self.in_run[(k, r)] = model.new_bool_var(f"run{k}_{r}")
                literals.append(literal)
                if r > 0:
                    model.add(self.arrival[k] >= starts[r]).only_enforce_if(literal)
                if r < runs - 1:
                    model.add(self.arrival[k] < ends[r]).only_enforce_if(literal)
                for q, change in changes.items():
                    part = model.new_int_var(-bound, bound, f"energy{k}_{r}_{q}")
                    model.add(part == change).only_enforce_if(literal)
                    model.add(part == 0).only_enforce_if(~literal)
                    energy[q].append(part)
            if len(possible) > 1:
                model.add(sum(literals) == present[k])
        self.level: list[cp_model.IntVar] = []
        """The deficit as each run after the first starts."""
        level: cp_model.LinearExpr = cp_model.LinearExpr.constant(battery.entering)
        entering: cp_model.LinearExprT = battery.entering
        for r in range(runs):
            level = cp_model.LinearExpr.sum(
                [level, battery.idle[r] * lengths[r] + battery.fixed[r], *energy[r]]
            )
            model.add(level <= battery.limit)
            if r < runs - 1:
                carry = model.new_int_var(0, battery.limit, f"level{r + 1}")
                model.add(carry >= level)
                if battery.floor > 0:
                    if isinstance(entering, int):
                        model.add(carry >= min(entering, battery.floor))
                    else:
                        kept = model.new_int_var(0, battery.limit, f"kept{r + 1}")
                        model.add_min_equality(kept, [entering, battery.floor])
                        model.add(carry >= kept)
                self.level.append(carry)
                level = carry
                entering = carry

    def _hint_battery(self, hint: dict[int, _TaskHint]) -> None:
        """Hint the run and carry-over variables from the hinted snapshots."""
        battery = self.battery
        if battery is None:
            return
        model = self.model
        runs = len(battery.idle)
        starts = [-(2**40), *battery.edges]
        ends = [*battery.edges, 2**40]
        lengths = [
            min(ends[r], battery.horizon) - max(starts[r], 0) for r in range(runs)
        ]
        energy = [battery.idle[r] * lengths[r] + battery.fixed[r] for r in range(runs)]
        for k, task in enumerate(self.tasks):
            if not task.optional:
                continue
            run = None
            occupied = 0
            if k in hint:
                arrival = hint[k].arrival
                run = sum(1 for edge in battery.edges if edge <= arrival)
                occupied = hint[k].collection + task.overhead
                energy[run] += (battery.slew[run] - battery.idle[run]) * task.slew_in
                # The seconds in each run, from the one it arrives in through
                # every run it spills into.
                beyond = [
                    max(0, arrival + occupied - ends[q]) if q < runs - 1 else 0
                    for q in range(runs)
                ]
                for q in range(run, runs):
                    inside = (occupied if q == run else beyond[q - 1]) - beyond[q]
                    net = task.power[0] if battery.sunlit[q] else task.power[1]
                    energy[q] += (net - battery.idle[q]) * inside
            model.add_hint(self.occupied[k], occupied)
            for r in range(runs):
                literal = self.in_run.get((k, r))
                if literal is not None:
                    model.add_hint(literal, r == run)
        level = battery.entering
        for r in range(runs - 1):
            level = max(level + energy[r], min(level, battery.floor), 0)
            model.add_hint(self.level[r], min(level, battery.limit))

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
        for visits in self.visits_of.values():
            if visits[0] not in self.due:
                continue
            due = self.tasks[visits[0]].due
            for k in visits:
                task = self.tasks[k]
                assert task.spacing is not None
                overdue = 0
                if k in hint:
                    slew_step, arrival, collection = hint[k]
                    overdue = max(0, self.step * slew_step - due)
                model.add_hint(self.due[k], due)
                model.add_hint(self.overdue[k], overdue)
                lateness = min(overdue, task.cadence)
                model.add_hint(self.lateness[k], lateness)
                if k in self.collection_lateness:
                    model.add_hint(
                        self.collection_lateness[k],
                        collection * lateness if k in hint else 0,
                    )
                next_due = arrival + collection + task.spacing if k in hint else 0
                model.add_hint(self.next_due[k], next_due)
                due = max(due, next_due)
        # The hinted snapshots use allocated time in time order, as decoding does.
        left = dict(self.allocated_left)
        for k in sorted(hint, key=lambda k: hint[k].arrival):
            _, arrival, collection = hint[k]
            for n, span in enumerate(self.tasks[k].spans):
                taken = 0
                if span.arrival[0] <= arrival <= span.arrival[1]:
                    taken = int(min(collection, left.get(span.key, 0.0)))
                    left[span.key] = left.get(span.key, 0.0) - taken
                model.add_hint(self.in_span[(k, n)], taken > 0)
                model.add_hint(self.span_seconds[(k, n)], taken)
        for (k, n), within in self.in_span.items():
            if k not in hint:
                model.add_hint(within, False)
                model.add_hint(self.span_seconds[(k, n)], 0)
        self._hint_recorder(hint)
        self._hint_battery(hint)

    def _hint_recorder(self, hint: dict[int, _TaskHint]) -> None:
        """Hint the segment and carry-over variables from the hinted snapshots."""
        recorder = self.recorder
        if recorder is None:
            return
        model = self.model
        segments = len(recorder.accrual)
        volume = list(recorder.fixed)
        for k, task in enumerate(self.tasks):
            if not task.optional or task.data == (0, 0):
                continue
            segment = None
            if k in hint:
                begins = hint[k].arrival + task.setup
                segment = sum(1 for edge in recorder.edges if edge <= begins)
                per_second, fixed = task.data
                volume[segment] += per_second * hint[k].collection + fixed
            for s in range(segments):
                literal = self.in_segment.get((k, s))
                if literal is not None:
                    model.add_hint(literal, s == segment)
        level = recorder.entering
        for s in range(segments):
            if s > 0:
                carry = max(
                    0, level + recorder.removal_accrual[s] - recorder.removal[s]
                )
                model.add_hint(self.carry[s - 1], carry)
                level = carry
            level += recorder.accrual[s] + volume[s]

    def solve(
        self, time_limit: float, workers: int, seed: int, origin_time: float
    ) -> tuple[str, list[_Snapshot] | None]:
        """Solve, returning CP-SAT's status and the chosen snapshots in order.

        The snapshots are None if the solver found no solution in time. A
        snapshot whose collection counts against allocated time is released
        (``origin_time`` plus its seconds from the origin) no earlier than its
        slew start, nor than the first arrival in the span: the model's slew
        may start early and wait at the target, but decoding collects as soon
        as the slew ends, and would collect before the span.
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
            release = None
            for n, span in enumerate(self.tasks[k].spans):
                if solver.value(self.span_seconds[(k, n)]) > 0:
                    slew = self.step * solver.value(self.slew_step[k])
                    release = origin_time + max(slew, _ceil(span.arrival[0], self.step))
            snapshots.append(_Snapshot(obsid=obsid, seconds=seconds, release=release))
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
        self.solver_score: float | None = None
        """Objective of the solver's plan after exact decoding."""
        self.solver_seconds = 0.0
        """Wall-clock time spent building, solving and decoding chunks."""

    def schedule(self) -> Plan:
        """Build the priority-first plan, solve chunk by chunk, and optionally
        polish, within the wheels' momentum where the configuration tracks it."""
        return self._plan_within_momentum(self._plan_once)

    def _plan_once(self) -> Plan:
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
                later,
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

    def _final_score(self, state: _State, sequence: Sequence[_Snapshot]) -> float:
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
        later: Sequence[_Snapshot] = (),
    ) -> list[_Snapshot] | None:
        """Choose and order the snapshots that start in one chunk.

        ``budget`` is the most exposure the chunk may collect for each request,
        every snapshot ends by ``finish``, and ``later`` are the priority-first
        plan's snapshots after the chunk, whose data the recorder must also
        hold. Returns None if the solver found no solution in time.
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
        for k, task in enumerate(tasks):
            if task.optional:
                # The shortest slew in: the model errs on the side of letting
                # the solver try, and the decoder, which is exact, decides.
                others = [int(x) for i, x in enumerate(slews[:, k]) if i != k]
                task.slew_in = min(others) if others else 0
        chunk_seconds = int(round(chunk_end - origin_time))
        recorder = (
            self._chunk_recorder(state, later, origin_time)
            if self._ledger.active
            else None
        )
        battery = (
            self._chunk_battery(state, later, origin_time, tasks)
            if self._battery.active
            else None
        )
        chunk = _ChunkModel(
            tasks,
            slews,
            self.ctx.step_size,
            chunk_seconds,
            budget,
            state.allocated_left,
            recorder,
            battery,
            self._category_caps(tasks, state),
        )
        chunk.add_hint(
            self._hint(tasks, planned, origin_time, slews, budget, chunk_seconds)
        )
        status, snapshots = chunk.solve(
            time_limit, self.workers, self.seed, origin_time
        )
        self.solver_statuses.append(status)
        return snapshots

    def _chunk_recorder(
        self, state: _State, later: Sequence[_Snapshot], origin_time: float
    ) -> _ChunkRecorder:
        """The recorder from the chunk's segment to the end of the horizon.

        What is already planned counts as fixed: the science placed before the
        chunk, locked entries, and the priority-first plan's snapshots after
        the chunk, in the segments their slews start in.
        """
        ledger: RecorderLedger = self._ledger
        volumes = list(state.volumes)
        for snapshot in later:
            assert snapshot.release is not None
            target = self._by_obsid[snapshot.obsid].target
            segment = ledger.segment(snapshot.release)
            volumes[segment] += ledger.science_volume(target, snapshot.seconds)
        first = ledger.segment(origin_time)
        levels = ledger.peaks(volumes)
        entering = levels[first] - ledger.accrual[first] - volumes[first]
        limit = ledger.limit
        assert limit is not None
        # What adds data rounds up and what removes it rounds down, so the
        # model's volumes are never below the ledger's, and a plan the model
        # keeps a kilobit under the limit passes the decoder's exact check.
        return _ChunkRecorder(
            edges=[
                int(round(bound - origin_time))
                for bound in ledger.bounds[first + 1 : -1]
            ],
            entering=_kb_up(entering),
            accrual=[_kb_up(x) for x in ledger.accrual[first:]],
            fixed=[_kb_up(x) for x in volumes[first:]],
            removal=[_kb_down(x) for x in ledger.removal[first:]],
            removal_accrual=[_kb_up(x) for x in ledger.removal_accrual[first:]],
            limit=_kb_down(limit) - 1,
        )

    # ── Tasks ────────────────────────────────────────────────────────────

    def _chunk_battery(
        self,
        state: _State,
        later: Sequence[_Snapshot],
        origin_time: float,
        tasks: Sequence[_Task],
    ) -> _ChunkBattery:
        """The battery from the chunk's run to the end of the horizon.

        What is already planned counts as fixed, relative to each run's idle
        power, exactly as the ledger counts it: the passes after the cursor
        and the priority-first plan's snapshots after the chunk.
        """
        ledger = self._battery
        limit = ledger.limit
        assert limit is not None
        bounds = [ledger.bounds[0]]
        sunlit = []
        for stretch in ledger.stretches:
            if stretch.begin > bounds[-1]:
                bounds.append(stretch.begin)
                sunlit.append(False)
            bounds.append(stretch.end)
            sunlit.append(True)
        if bounds[-1] < ledger.bounds[-1]:
            bounds.append(ledger.bounds[-1])
            sunlit.append(False)
        else:
            bounds.pop()
        # Runs from the one the chunk starts in.
        first = max(0, sum(1 for b in bounds[1:] if b <= origin_time))
        edges = [int(round(b - origin_time)) for b in bounds[first + 1 :]]
        sunlit = sunlit[first:]
        runs = len(sunlit)
        starts = [0, *edges]
        idle = []
        slew = []
        for r in range(runs):
            at = origin_time + starts[r]
            eclipse = not sunlit[r]
            idle_watts = int(np.ceil(ledger._watts(ACSMode.IDLE, eclipse)))
            lit = ledger.stretch_at(at) if sunlit[r] else None
            if lit is not None and lit.attitude is not None:
                charging = lit.charge[lit.charge > 0.0]
                if len(charging):
                    idle_watts = -int(np.floor(charging.mean()))
            idle.append(idle_watts)
            slew.append(int(np.ceil(ledger._watts(ACSMode.SLEWING, eclipse))))

        def run_of(utime: float) -> int:
            return min(
                sum(1 for edge in edges if origin_time + edge <= utime), runs - 1
            )

        # What is already planned, exactly as the ledger counts it, step by
        # step in the run of each step, relative to the run's idle power: the
        # passes after the cursor, and the priority-first plan's snapshots
        # released after the chunk, which the chunk must leave room for.
        fixed = [0] * runs
        step_size = self.ctx.step_size
        releases = {snapshot.release for snapshot in later}
        blocks = [
            *(b for b in state.timeline[state.cursor :] if b.is_pass),
            *(
                b
                for b in self._greedy_timeline
                if not b.is_pass
                and not b.is_charge
                and float(b.entry.begin) in releases
            ),
        ]
        # With a solar-array drive, as the walk counts them where the plan's
        # additions leave the drive; a block not in the plan at its own deltas.
        counted = ledger.walk_deltas(state.energy)
        for block in blocks:
            addition = self._block_addition(block)
            steps = ledger.steps(addition.begin, addition.end)
            for utime, delta in zip(steps, counted.get(addition, addition.deltas)):
                if utime < origin_time:
                    continue
                r = run_of(float(utime))
                fixed[r] += int(np.ceil(delta * 3600.0)) - idle[r] * step_size
        entering = ledger.level_at(state.energy, origin_time) * 3600.0
        return _ChunkBattery(
            edges=edges,
            sunlit=sunlit,
            entering=int(np.ceil(entering)),
            idle=idle,
            slew=slew,
            fixed=fixed,
            limit=int(np.floor(limit * 3600.0)),
            floor=int(np.ceil(ledger.recharge_floor * 3600.0)),
            horizon=int(round(ledger.bounds[-1] - origin_time)),
        )

    def _candidate_power(self, target: Pointing, when: float) -> tuple[int, int]:
        """Net watts a snapshot of ``target`` draws in sunlight, at the
        power-optimal roll for ``when``, and in eclipse; draws rounded up and
        generation down."""
        ledger = self._battery
        if not ledger.active:
            return (0, 0)
        attitude = target.target_body_attitude(self.ctx.instrument_roll(target, when))
        lit = next(
            (
                float(s)
                for s in ledger.grid
                if float(s) >= when and not ledger.in_eclipse(float(s))
            ),
            None,
        )
        generation = 0.0 if lit is None else ledger.generation_watts(attitude, lit)
        modes = target.effective_instrument_modes()
        sunlit = int(
            np.ceil(ledger._watts(ACSMode.SCIENCE, False, modes)) - np.floor(generation)
        )
        eclipse = int(np.ceil(ledger._watts(ACSMode.SCIENCE, True, modes)))
        return (sunlit, eclipse)

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
                        _PlannedSnapshot(arrival, seconds, entry.visit)
                    )
                    break
        return planned

    def _category_caps(
        self, tasks: Sequence[_Task], state: _State
    ) -> list[tuple[int, list[int]]]:
        """For each category with a time budget among the chunk's candidates:
        the seconds left of it, with what ``state`` has collected and
        planned, and the candidates' indices."""
        members: dict[str, list[int]] = {}
        budgets: dict[str, float] = {}
        for k, task in enumerate(tasks):
            if task.obsid is None:
                continue
            category = self._by_obsid[task.obsid].category
            if category.time_budget_seconds is None:
                continue
            members.setdefault(category.name, []).append(k)
            budgets[category.name] = category.time_budget_seconds
        return [
            (
                int(
                    np.floor(max(0.0, budgets[name] - state.categories.get(name, 0.0)))
                ),
                indices,
            )
            for name, indices in members.items()
        ]

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
        shares = program_shares(state.programs) if self._deficit_weight else {}
        ranked = sorted(
            self._by_obsid.values(), key=lambda r: r.merit.value, reverse=True
        )
        candidates: list[tuple[int, int, _Task]] = []
        for rank, request in enumerate(ranked):
            target = request.target
            obsid = int(target.obsid)
            timing = ctx.timing_for(obsid)
            setup = float(timing.setup_seconds)
            overhead = int(np.ceil(timing.total_seconds))
            remaining = budget[obsid]
            # A snapshot must collect something, as in the priority-first
            # scheduler, even with an ss_min of zero; this also keeps the
            # snapshot counts below from dividing by zero.
            shortest = int(np.ceil(snapshot_min_seconds(target)))
            if remaining < self._initial_remaining[obsid]:
                # Some of the exposure is placed already: the remainder may
                # be shorter than ss_min, as in the other planners.
                shortest = min(shortest, int(np.ceil(remaining)))
            shortest = max(1, shortest)
            longest = int(np.floor(min(float(target.ss_max), remaining)))
            if longest < shortest:
                continue
            slews_from, opens = self._opens(request, state, origin_time)
            spacing = cadence = None
            if request.cadence is not None:
                spacing = int(np.ceil(setup + request.cadence))
                cadence = int(np.ceil(request.cadence))
            # One slot of snapshots per request, or one per visit window
            # still to come for a target with visit windows, each with its
            # own earliest arrival and collection deadline.
            slots: list[tuple[float, float | None, int | None]] = [
                (opens, target.deadline, None)
            ]
            if request.timed:
                slots = [
                    (
                        max(opens, target.visit_windows[k].start - setup),
                        target.visit_windows[k].deadline,
                        k,
                    )
                    for k in request.open_visits(
                        state.last_visit.get(obsid), origin_time
                    )
                ]
            for w0, w1 in request.windows:
                if w0 >= chunk_end:
                    continue
                window_key = (obsid, w0)
                copy = 0
                for slot_opens, slot_deadline, visit in slots:
                    if w1 <= slot_opens:
                        continue
                    window = (
                        int(np.ceil(max(w0, slews_from) - origin_time)),
                        int(np.floor(min(w1, ctx.uend, finish) - origin_time)),
                    )
                    # The slew may start before the earliest start, which
                    # bounds only the arrival.
                    earliest = int(np.ceil(max(w0, slot_opens) - origin_time))
                    # The slew must start in the chunk (see _ChunkModel); the
                    # snapshot may arrive after it ends.
                    latest = window[1] - overhead - shortest
                    if slot_deadline is not None:
                        latest = min(latest, int(slot_deadline - origin_time - setup))
                    if latest < earliest or window[0] >= chunk_end - origin_time:
                        continue
                    attitude = target.target_body_attitude(
                        ctx.instrument_roll(target, max(w0, origin_time))
                    )
                    attitudes = (attitude, attitude)
                    task_overhead = overhead
                    if target.sequence is not None:
                        # A visit runs whole: its calibration, the slews
                        # between its steps and any step setup are part of
                        # its length, and it leaves the spacecraft at its
                        # last step.
                        when = max(w0, origin_time)
                        first = target.step_pointing(0)
                        last_step = target.step_pointing(len(target.sequence.steps) - 1)
                        attitudes = (
                            first.target_body_attitude(
                                ctx.instrument_roll(first, when)
                            ),
                            last_step.target_body_attitude(
                                ctx.instrument_roll(last_step, when)
                            ),
                        )
                        task_overhead = overhead + int(
                            np.ceil(
                                target.sequence_overhead_seconds(
                                    self.config.spacecraft_bus.attitude_control,
                                    self.config.payload.observation_timing,
                                )
                            )
                        )
                    bounds = (
                        (slot_opens + setup, slot_deadline)
                        if visit is not None
                        else (target.earliest_start, target.deadline)
                    )
                    rate, earliness, offset, late = self._candidate_value(
                        request, shares, origin_time, bounds
                    )
                    # Allocated time is worth allocation_bonus more, beyond the rate.
                    spans = self._spans(
                        state,
                        obsid,
                        origin_time,
                        (earliest, latest),
                        (self._allocated_factor - 1.0) * self._value(request, shares),
                    )
                    data = self._candidate_data(target)
                    power = self._candidate_power(target, max(w0, origin_time))
                    if visit is not None:
                        # One visit per window.
                        copies = 1
                    else:
                        # As many snapshots as the exposure needs and the
                        # window holds.
                        copies = min(
                            int(np.ceil(remaining / longest)),
                            max(1, (window[1] - earliest) // (overhead + shortest)),
                        )
                        if spacing is not None:
                            copies = min(copies, 1 + (window[1] - earliest) // spacing)
                        # A fixed task in the window can split the exposure
                        # into more, shorter snapshots; keep at least those of
                        # the priority-first plan, so the hint can reproduce it.
                        copies = max(copies, len(planned.get(window_key, ())))
                    priority = 0 if window_key in planned else 1
                    for _ in range(copies):
                        task = _Task(
                            window_key,
                            attitudes,
                            (earliest, latest),
                            copy=copy,
                            spacing=spacing,
                            overhead=task_overhead,
                            setup=int(np.ceil(setup)),
                            collection=(shortest, longest),
                            window=window,
                            rate=rate,
                            earliness=earliness,
                            offset=offset,
                            late=late,
                            due=int(np.ceil(opens - origin_time)),
                            cadence=cadence or 0,
                            spans=spans,
                            data=data,
                            power=power,
                            visit=visit,
                        )
                        copy += 1
                        candidates.append((priority, rank, task))
        # Stable, so a window's snapshots stay together and in order.
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [task for _, _, task in candidates[: self.max_candidates]]

    def _candidate_data(self, target: Pointing) -> tuple[int, int]:
        """Kilobits a snapshot of ``target`` generates per second, and once,
        each rounded up so the model never counts less than the ledger."""
        if not self._ledger.active:
            return (0, 0)
        modes = instrument_modes_of(target)
        once = self.config.science_generated(
            0.0, target.science_data, modes
        ) + extra_data_gb(target)
        per_second = (
            self.config.science_generated(1.0, target.science_data, modes) - once
        )
        return (_kb_up(per_second), _kb_up(once))

    def _opens(
        self, request: _Request, state: _State, origin_time: float
    ) -> tuple[float, float]:
        """Earliest slew and earliest arrival of the request's next snapshot.

        The slew starts no earlier than the chunk's origin and, for a cadence
        request, a cadence after its last visit ends. The snapshot also
        arrives no earlier than its earliest start less setup.
        """
        slews_from = origin_time
        target = request.target
        if request.cadence is not None:
            last = state.last_visit.get(int(target.obsid))
            if last is not None:
                slews_from = max(slews_from, last + request.cadence)
        opens = slews_from
        if target.earliest_start is not None:
            # Collection begins setup after the snapshot arrives.
            opens = max(
                opens, target.earliest_start - self.ctx.setup_for(int(target.obsid))
            )
        return slews_from, opens

    def _candidate_value(
        self,
        request: _Request,
        shares: dict[str, float],
        origin_time: float,
        bounds: tuple[float | None, float | None] | None = None,
    ) -> tuple[float, float, float, float]:
        """A candidate's objective terms per second of collection: its rate,
        earliness, offset and lateness discount (see :class:`_Task`).

        The value counts the program shares delivered before the chunk. A
        candidate of a request with a deadline loses value the later it
        arrives, and one with a cadence the later after its visit is due, in
        proportion to the collection it gets.
        """
        ctx = self.ctx
        target = request.target
        setup = ctx.setup_for(int(target.obsid))
        rate = self._value(request, shares)
        earliness = offset = late = 0.0
        earliest_start, deadline = (
            bounds if bounds is not None else (target.earliest_start, target.deadline)
        )
        if deadline is not None and self.earliness_weight > 0.0:
            released = max(ctx.ustart, earliest_start or ctx.ustart)
            available = deadline - released
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
        return rate, earliness, offset, late

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
            if snapshots is None:
                continue
            if task.visit is not None:
                # A visit's candidate takes the plan's snapshot in its window.
                match = next((s for s in snapshots if s.visit == task.visit), None)
                if match is not None:
                    wanted[k] = _PlannedSnapshot(
                        match.arrival - origin_time, match.seconds, match.visit
                    )
            elif task.copy < len(snapshots):
                arrival, seconds, _ = snapshots[task.copy]
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

    def _spans(
        self,
        state: _State,
        obsid: int,
        origin_time: float,
        arrival: tuple[int, int],
        rate: float,
    ) -> list[_Span]:
        """Spans of allocated time a candidate arriving in ``arrival`` can use."""
        if self.allocated is None or rate <= 0.0:
            return []
        setup = self.ctx.setup_for(obsid)
        spans = []
        for n, span in enumerate(self.allocated.get(obsid, [])):
            if state.allocated_left.get((obsid, n), 0.0) <= 0.0:
                continue
            # Collection begins setup after the snapshot arrives.
            lo = max(arrival[0], int(np.ceil(span.begin - setup - origin_time)))
            hi = min(arrival[1], int(np.ceil(span.end - setup - origin_time)) - 1)
            if lo <= hi:
                spans.append(_Span((obsid, n), (lo, hi), rate))
        return spans

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


def _next_due_bound(task: _Task) -> int:
    """The latest the visit after a cadence candidate can be due."""
    assert task.spacing is not None
    return task.arrival[1] + task.collection[1] + task.spacing


def _ceil(seconds: int, step: int) -> int:
    """Round seconds up to a whole number of steps."""
    return -(-seconds // step) * step
