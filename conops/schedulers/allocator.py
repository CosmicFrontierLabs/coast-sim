"""Long-range planning: decide on which day each request is observed.

:class:`LongRangeAllocator` plans a run of days to months at a coarse level,
deciding how much of each request's remaining exposure to collect in which bin
(a day by default). The short-term scheduling, rolling replans or dispatch, then
orders the observations within each bin, working the bin's allocated requests
first. Without it, short-term scheduling looks only hours ahead and spends each
day on the best targets available that day, so targets observable for only part
of the run, or due beyond its horizon, can miss their chance.
"""

import math
import time
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ..common.enums import ACSMode
from ..config import MissionConfig
from ..config.data_storage import RecorderView
from ..simulation.coverage import StationCoverage
from ..simulation.passes import Pass
from ..targets import Pointing
from ..targets.merit import MeritModel
from ..targets.plan_entry import snapshot_min_seconds
from .context import SchedulingContext
from .recorder import contact_downlink_gb


class AllocatedTime(NamedTuple):
    """Seconds of a request's exposure allocated to a span of time."""

    begin: float
    """Start of the span, in Unix seconds."""
    end: float
    """End of the span, in Unix seconds."""
    seconds: float
    """Exposure to collect in the span; collection starting in the span counts."""


class RecorderBudget(BaseModel):
    """One recorder's data budget, by bin, as an allocation counted it."""

    model_config = ConfigDict(frozen=True)

    downlink: list[float] = []
    """Gb each bin's ground contacts can take off the recorder."""
    housekeeping: list[float] = []
    """Gb each bin generates on the recorder on its own, at the housekeeping
    rates."""
    science: list[float] = []
    """Gb the science allocated to each bin generates on the recorder."""
    data_limit: float | None = None
    """Gb the recorder is kept within from bin to bin, its planning limit;
    None where it had no data budget."""
    start_volume_gb: float = 0.0
    """Gb on the recorder when the allocation starts."""

    def volumes(self) -> list[float]:
        """Gb carried on the recorder out of each bin: what the bin generates
        on it, housekeeping and allocated science, less what its contacts
        take off, and never below zero. Empty where it had no data budget,
        since nothing was counted."""
        if not self.downlink:
            return []
        return _carries(
            self.start_volume_gb, self.housekeeping, self.science, self.downlink
        )


_BUDGET_FIELDS = frozenset(RecorderBudget.model_fields)
"""The budget's fields as earlier allocations held them, for one recorder."""


class Allocation(BaseModel):
    """Seconds of exposure allocated to each request in each bin."""

    model_config = ConfigDict(frozen=True)

    bins: list[tuple[float, float]]
    """Start and end of each bin, in Unix seconds."""
    capacity: list[float]
    """Seconds of science each bin is expected to hold."""
    reserve: list[float]
    """Seconds of each bin's capacity held back for unplanned work, such as
    ToOs; planned requests are allocated only the rest."""
    seconds: dict[int, dict[int, float]]
    """Allocated seconds by obsid, then by bin index. Every request the
    allocator considered has an entry, even if nothing was allocated to it."""
    recorders: dict[str, RecorderBudget] = {}
    """Each recorder's data budget, by store name: ``default`` for a mission
    with one recorder, its stores' names for one with
    :class:`~conops.config.DataStorage`. A recorder without a planning limit,
    or that no request generates data on, has an empty budget."""

    @model_validator(mode="before")
    @classmethod
    def _legacy_budget(cls, data: object) -> object:
        """Take the one recorder's budget earlier allocations held in
        ``downlink``, ``housekeeping``, ``science``, ``data_limit`` and
        ``start_volume_gb`` as the ``default`` recorder's, so an allocation
        built or saved with that form keeps its budget."""
        if not isinstance(data, dict) or not _BUDGET_FIELDS & set(data):
            return data
        data = dict(data)
        budget = {name: data.pop(name) for name in _BUDGET_FIELDS if name in data}
        recorders = dict(data.get("recorders") or {})
        if "default" in recorders:
            raise ValueError(
                "give the budget either as recorders['default'] or as the "
                "downlink, housekeeping, science, data_limit and "
                "start_volume_gb fields, not both"
            )
        recorders["default"] = RecorderBudget(**budget)
        data["recorders"] = recorders
        return data

    @property
    def downlink(self) -> list[float]:
        """Gb each bin's ground contacts can take off the recorders, in all."""
        return _summed(r.downlink for r in self.recorders.values())

    @property
    def housekeeping(self) -> list[float]:
        """Gb each bin generates on its own, at the housekeeping rates, on
        the recorders in all."""
        return _summed(r.housekeeping for r in self.recorders.values())

    @property
    def science(self) -> list[float]:
        """Gb the science allocated to each bin generates, on the recorders
        in all."""
        return _summed(r.science for r in self.recorders.values())

    @property
    def data_limit(self) -> float | None:
        """Gb the recorders are kept within from bin to bin, their planning
        limits in all; None where no recorder had a data budget. Each
        recorder is kept within its own limit."""
        limits = [r.data_limit for r in self.recorders.values()]
        budgeted = [limit for limit in limits if limit is not None]
        return sum(budgeted) if budgeted else None

    @property
    def start_volume_gb(self) -> float:
        """Gb on the recorders, in all, when the allocation starts."""
        return sum((r.start_volume_gb for r in self.recorders.values()), 0.0)

    def recorder_volumes(self) -> list[float]:
        """Gb carried on the recorders, in all, out of each bin: what the bin
        generates, housekeeping and allocated science, less what its contacts
        take off, each recorder never below zero. Nothing where the allocation
        had no data budget, since nothing was counted."""
        volumes = _summed(r.volumes() for r in self.recorders.values())
        return volumes or [0.0] * len(self.bins)

    def bins_overlapping(self, begin: float, end: float) -> list[int]:
        """Indices of the bins that overlap ``[begin, end)``."""
        return [k for k, (b0, b1) in enumerate(self.bins) if b0 < end and begin < b1]

    def prefers(self, obsid: int, begin: float, end: float) -> bool:
        """Whether a request should be worked ahead of others during ``[begin, end)``.

        True if it has exposure allocated to a bin overlapping that time, or if
        the allocator never considered it, such as a Target of Opportunity
        submitted after the allocation was made.
        """
        allocated = self.seconds.get(obsid)
        if allocated is None:
            return True
        return any(
            allocated.get(k, 0.0) > 0.0 for k in self.bins_overlapping(begin, end)
        )

    def allocated(
        self,
        obsids: Sequence[int],
        begin: float,
        end: float,
        used: Mapping[int, Mapping[int, float]] | None = None,
    ) -> dict[int, list[AllocatedTime]]:
        """The time allocated to each of ``obsids`` during ``[begin, end)``, bin by bin.

        Each bin overlapping the span gives the request its seconds there, less
        any ``used`` already (by obsid, then bin index), such as science
        collected or committed in the bin since the allocation was made. A bin
        that ``end`` cuts short gives the share of those seconds that falls
        before ``end``, pro rata from ``begin`` or the bin's start. A request
        the allocator never considered, such as a Target of Opportunity
        submitted after the allocation was made, is allocated the whole span
        without limit.
        """
        spans: dict[int, list[AllocatedTime]] = {}
        for obsid in obsids:
            per_bin = self.seconds.get(obsid)
            if per_bin is None:
                spans[obsid] = [AllocatedTime(begin, end, math.inf)]
                continue
            spans[obsid] = []
            spent = used.get(obsid, {}) if used is not None else {}
            for k in self.bins_overlapping(begin, end):
                seconds = per_bin.get(k, 0.0) - spent.get(k, 0.0)
                if seconds <= 0.0:
                    continue
                b0, b1 = self.bins[k]
                lo, hi = max(b0, begin), min(b1, end)
                spans[obsid].append(
                    AllocatedTime(lo, hi, seconds * (hi - lo) / (b1 - lo))
                )
        return spans


class _Demand(BaseModel):
    """One request as the allocator sees it."""

    obsid: int
    value: float
    remaining: float
    ss_min: float
    deadline: float | None
    spread: bool
    """Whether its exposure should be spread across bins, as for a cadence."""
    visible: dict[int, float]
    """Seconds it is visible in each bin where it can be observed."""
    generated: dict[str, tuple[float, float]] = {}
    """Gb its target generates on each recorder, by store name: per second
    of collection, and per observation whatever its length, counted once in
    each bin it is allocated in."""


class _StoreBudget(NamedTuple):
    """The data one recorder can carry, by bin, as the allocator sees it."""

    limit: float | None
    """Gb the recorder is kept within from bin to bin; None for no budget."""
    start: float
    """Gb on the recorder when the allocation starts."""
    downlink: list[float]
    """Gb each bin's contacts take off it."""
    housekeeping: list[float]
    """Gb each bin generates on it on its own."""
    rates: dict[int, tuple[float, float]]
    """Each request's Gb per second of collection and per observation on it."""

    @property
    def active(self) -> bool:
        """Whether the budget limits what may be allocated: a limit, and a
        request that generates data on the recorder."""
        return self.limit is not None and any(
            rate > 0.0 or fixed > 0.0 for rate, fixed in self.rates.values()
        )

    def science(self, seconds: Mapping[int, Mapping[int, float]]) -> list[float]:
        """Gb the science in ``seconds`` (by obsid, then bin) generates in each
        bin; nothing where there is no budget."""
        if not self.downlink:
            return []
        generated = [0.0] * len(self.downlink)
        for obsid, per_bin in seconds.items():
            rate, fixed = self.rates.get(obsid, (0.0, 0.0))
            for k, part in per_bin.items():
                if part > 0.0:
                    generated[k] += rate * part + fixed
        return generated

    def room(self, science: Sequence[float], k: int) -> float:
        """The most data bin ``k`` can take with ``science`` already allocated
        by bin, keeping what every bin from it on carries within the limit."""
        assert self.limit is not None
        carries = _carries(self.start, self.housekeeping, science, self.downlink)
        # Work backwards through the carry recurrence. Spare downlink can
        # absorb more data even when the current carry has clipped to zero.
        allowed = self.limit
        for j in range(len(science) - 1, k - 1, -1):
            allowed = (
                min(self.limit, allowed)
                + self.downlink[j]
                - self.housekeeping[j]
                - science[j]
            )
        incoming = carries[k - 1] if k else self.start
        return allowed - incoming


class _Budget(NamedTuple):
    """The run's data budget: what each recorder can carry, by store name."""

    stores: dict[str, _StoreBudget]

    @property
    def active(self) -> bool:
        """Whether any recorder's budget limits what may be allocated."""
        return any(store.active for store in self.stores.values())

    def science(
        self, seconds: Mapping[int, Mapping[int, float]]
    ) -> dict[str, list[float]]:
        """Gb the science in ``seconds`` (by obsid, then bin) generates in
        each bin on each recorder with a budget."""
        return {
            name: store.science(seconds)
            for name, store in self.stores.items()
            if store.active
        }

    def data_seconds(
        self,
        demand: _Demand,
        science: Mapping[str, Sequence[float]],
        k: int,
        first: bool,
    ) -> float:
        """Seconds the data room in bin ``k`` allows the request there, on
        every recorder it generates data on, with ``science`` already
        allocated by recorder and bin."""
        most = math.inf
        for name, store in self.stores.items():
            if not store.active:
                continue
            rate, fixed = store.rates.get(demand.obsid, (0.0, 0.0))
            if rate <= 0.0 and fixed <= 0.0:
                continue
            left = store.room(science[name], k) - (fixed if first else 0.0)
            if left < 0.0:
                return 0.0
            if rate > 0.0:
                most = min(most, left / rate)
        return most

    def add(
        self,
        science: Mapping[str, list[float]],
        demand: _Demand,
        k: int,
        part: float,
        first: bool,
    ) -> None:
        """Count ``part`` seconds of the request in bin ``k`` on each recorder."""
        for name, store in self.stores.items():
            if store.active:
                rate, fixed = store.rates.get(demand.obsid, (0.0, 0.0))
                science[name][k] += rate * part + (fixed if first else 0.0)


def _summed(lists: Iterable[Sequence[float]]) -> list[float]:
    """The lists summed element by element, skipping empty ones; empty if
    all are."""
    total: list[float] = []
    for values in lists:
        if not values:
            continue
        if not total:
            total = list(values)
        else:
            total = [a + b for a, b in zip(total, values, strict=True)]
    return total


def _carries(
    start: float,
    housekeeping: Sequence[float],
    science: Sequence[float],
    downlink: Sequence[float],
) -> list[float]:
    """Gb carried on the recorder out of each bin, from ``start``: what the
    bin generates less what its contacts take off, never below zero."""
    carries = []
    volume = start
    for generated, own, removed in zip(science, housekeeping, downlink, strict=True):
        volume = max(0.0, volume + own + generated - removed)
        carries.append(volume)
    return carries


class LongRangeAllocator:
    """Allocate requests' remaining exposure to bins of a long run.

    For each request, the seconds its target is visible in each bin, from its
    earliest start to its deadline, come from the same roll-independent
    visibility windows the planners use. Each bin is expected to hold
    ``efficiency`` of its length in science; the rest goes to slews, setup and
    ground passes. Of that capacity, ``reserve`` is held back for work not yet
    known, such as Targets of Opportunity, so that when it arrives it fits
    without pushing planned requests to later bins. Requests passed as ``unplanned`` to
    :meth:`allocate` are allocated first and use the reserve before the rest.

    By default the allocation is solved as a mixed-integer program with HiGHS,
    which comes with OR-Tools, the optional ``cpsat`` extra. It maximizes merit-weighted seconds in two
    levels, unplanned requests first and then the rest, with the unplanned
    requests' allocation fixed while the rest are solved, subject to:

    * each request gets, in each bin, either nothing or at least its ``ss_min``,
      and no more than its target is visible there between its earliest
      start and its deadline;
    * no request gets more than its remaining exposure;
    * each bin's planned requests fit its capacity less the reserve, and all its
      requests fit its capacity;
    * a request with a cadence gets no more than an even share in any bin, so its
      visits are spread across the run;
    * the data the run generates fits what its ground contacts can downlink:
      each bin generates housekeeping and the science data of the requests
      allocated to it (each target's rate for its seconds, plus its fixed
      amount per observation once per bin), its contacts take off what they
      can, and what they cannot is carried to the next bin and must stay
      within the recorder's planning limit. There is no budget without a
      planning limit, or when no request generates data. A mission with
      named stores (:class:`~conops.config.DataStorage`) is budgeted store
      by store: each takes what is routed to it, its contacts take off its
      reserved share of the link, and each stays within its own limit.

    Requests with a deadline are worth slightly more in earlier bins. Keeping
    seconds in the bin the previous allocation gave them is worth 1% more, so
    that allocating again moves requests only for a real gain. As a tiebreak,
    the planned load is spread evenly over the bins, as far as visibility
    allows, worth a thousandth of the least merit per second, so the slack is
    spread across the run rather than left in a few bins. Each level starts
    from the greedy allocation of its requests into the room the level above
    leaves, so the solver only improves on it; ``solver="greedy"`` uses the
    greedy allocation on its own: requests in order of merit, each in the bins
    least contended by the requests still to come. If the solver finds no
    solution for a level in time, that level and any below it are allocated
    greedily.

    Args:
        config: Mission configuration, with an ephemeris on its constraint.
        begin: Start of the run.
        end: End of the run.
        bin_length: Length of each bin.
        efficiency: Fraction of each bin expected to hold science, in (0, 1].
        reserve: Fraction of that capacity held back for unplanned work, in
            [0, 1).
        solver: "milp" (the default), which needs OR-Tools, or "greedy",
            which does not.
        time_limit: Seconds the MILP solver may search per allocation.
        passes: The run's predicted ground passes, for the data budget, where
            the caller has them; predicted once otherwise, which takes a
            while for a long run.
        coverage: Link coverage for passes without spacecraft tracking,
            for the data budget, where the caller has it; built from the
            configuration otherwise.

    Raises:
        ImportError: If ``solver`` is "milp" and OR-Tools is not installed.
    """

    def __init__(
        self,
        config: MissionConfig,
        begin: datetime,
        end: datetime,
        *,
        bin_length: timedelta = timedelta(days=1),
        efficiency: float = 0.75,
        reserve: float = 0.1,
        solver: Literal["milp", "greedy"] = "milp",
        time_limit: float = 10.0,
        passes: Sequence[Pass] | None = None,
        coverage: StationCoverage | None = None,
    ) -> None:
        if bin_length <= timedelta(0):
            raise ValueError("bin_length must be positive")
        if not 0.0 < efficiency <= 1.0:
            raise ValueError("efficiency must be in (0, 1]")
        if not 0.0 <= reserve < 1.0:
            raise ValueError("reserve must be in [0, 1)")
        if solver not in ("milp", "greedy"):
            raise ValueError('solver must be "milp" or "greedy"')
        if time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if solver == "milp":
            _require_ortools()
        config.recorder_views(refresh=True)
        self.config = config
        self.begin = begin
        self.end = end
        self.bin_length = bin_length
        self.efficiency = efficiency
        self.reserve = reserve
        self.solver = solver
        self.time_limit = time_limit
        self.solver_status: str | None = None
        """The MILP solver's status for each stage of the last allocation,
        or "OPTIMAL" if every stage was solved to optimality."""
        self._previous: Allocation | None = None
        self.ctx = SchedulingContext(
            config, begin, end, passes=passes, coverage=coverage
        )
        self.merit_model = MeritModel(config)
        self._windows: dict[int, list[tuple[float, float]]] = {}
        length = bin_length.total_seconds()
        ustart, uend = begin.timestamp(), end.timestamp()
        count = max(1, int(-(-(uend - ustart) // length)))
        self.bins = [
            (ustart + k * length, min(ustart + (k + 1) * length, uend))
            for k in range(count)
        ]

    @property
    def passes(self) -> list[Pass] | None:
        """The run's bookings, shared with the context's link coverage."""
        return self.ctx.passes

    @passes.setter
    def passes(self, passes: Sequence[Pass] | None) -> None:
        self.ctx.passes = passes
        # Visibility windows can include cuts for the old coverage's repoints.
        self._windows.clear()

    def allocate(
        self,
        targets: Sequence[Pointing],
        start: float,
        reserved_seconds: Mapping[int, float] | None = None,
        unplanned: Collection[int] = (),
        start_volume_gb: float | Mapping[str, float] | None = None,
    ) -> Allocation:
        """Allocate the targets' remaining exposure from ``start`` to the end.

        Args:
            targets: Requests to allocate. Their remaining exposure is their
                ``exptime`` (or ``ss_max`` if unset), less ``reserved_seconds``.
            start: Unix time from which exposure can still be collected; bins
                before it get nothing and the bin containing it is shortened.
            reserved_seconds: Exposure, by obsid, already scheduled and not to
                be allocated again.
            unplanned: Obsids of requests that arrived unplanned, such as ToOs.
                They are allocated before the others and may use each bin's
                reserve.
            start_volume_gb: Gb on the recorder at ``start``, or on each
                store by name for a mission with named stores; the
                configured recorders' volumes if None.

        Raises:
            ValueError: If ``start_volume_gb`` does not name every store of
                a mission with named stores exactly once.
        """
        reserved = reserved_seconds or {}
        # Completion deficits count the science each program has collected
        # or has scheduled, as the schedulers' do.
        delivered: dict[str, float] = defaultdict(float)
        for target in targets:
            done = target.collected_seconds + reserved.get(int(target.obsid), 0.0)
            if done > 0.0:
                program = self.merit_model.category(target).program_name
                delivered[program] += done
        total = sum(delivered.values())
        shares = (
            {program: done / total for program, done in delivered.items()}
            if total > 0.0
            else {}
        )
        capacity = [
            max(0.0, b1 - max(b0, start)) * self.efficiency for b0, b1 in self.bins
        ]
        held = [c * self.reserve for c in capacity]
        demands = [
            demand
            for target in targets
            if (demand := self._demand(target, start, reserved, shares)) is not None
        ]
        budget = self._budget(start, demands, start_volume_gb)
        unplanned_set = set(unplanned)
        empty: dict[int, dict[int, float]] = {d.obsid: {} for d in demands}
        if self.solver == "greedy" or not demands:
            seconds = self._greedy(
                demands, capacity, held, unplanned_set, empty, budget
            )
        else:
            seconds, unsolved = self._solve(
                demands, capacity, held, unplanned_set, budget
            )
            if unsolved:
                # Levels the solver found no solution for go greedily into
                # the room the solved levels leave.
                seconds = self._greedy(
                    unsolved, capacity, held, unplanned_set, seconds, budget
                )
        science = budget.science(seconds)
        allocation = Allocation(
            bins=list(self.bins),
            capacity=capacity,
            reserve=held,
            seconds=seconds,
            recorders={
                name: RecorderBudget(
                    downlink=store.downlink,
                    housekeeping=store.housekeeping,
                    science=science.get(name, []),
                    data_limit=store.limit if store.active else None,
                    start_volume_gb=store.start,
                )
                for name, store in budget.stores.items()
            },
        )
        self._previous = allocation
        return allocation

    def _greedy(
        self,
        demands: Sequence[_Demand],
        capacity: Sequence[float],
        held: Sequence[float],
        unplanned_set: Collection[int],
        allocated: Mapping[int, Mapping[int, float]],
        budget: _Budget,
    ) -> dict[int, dict[int, float]]:
        """Add ``demands`` to ``allocated`` in priority order, each where it fits.

        Unplanned requests go first, then by value. A request with a
        cadence is spread evenly over the bins it is visible in; one with a
        deadline goes in its earliest bins; the rest go in the bins expected
        to be least full: what is already allocated there, plus what the
        requests not yet placed are expected to need. Where the data budget
        binds, a request gets in a bin no more seconds than the data room
        there allows, keeping every later bin's carry-over within the limit.
        """
        seconds = {obsid: dict(bins) for obsid, bins in allocated.items()}
        # Seconds allocated in each bin to unplanned requests, and to the rest.
        surprise = [0.0] * len(self.bins)
        planned = [0.0] * len(self.bins)
        for obsid, per_bin in seconds.items():
            load = surprise if obsid in unplanned_set else planned
            for k, part in per_bin.items():
                load[k] += part
        pressure = self._pressure(demands)
        science = budget.science(seconds)

        def room(k: int, is_unplanned: bool) -> float:
            if is_unplanned:
                return capacity[k] - surprise[k] - planned[k]
            # Planned requests leave the reserve, or as much of the bin as
            # unplanned requests have already taken, whichever is more.
            return capacity[k] - max(surprise[k], held[k]) - planned[k]

        def data_seconds(demand: _Demand, k: int, first: bool) -> float:
            """Seconds the data room in bin ``k`` allows the request there."""
            return budget.data_seconds(demand, science, k, first)

        def give(demand: _Demand, k: int, part: float) -> None:
            first = given.get(k, 0.0) <= 0.0
            given[k] = given.get(k, 0.0) + part
            load[k] += part
            budget.add(science, demand, k, part, first)

        order = sorted(
            demands,
            key=lambda d: (d.obsid not in unplanned_set, -d.value, d.obsid),
        )
        for demand in order:
            is_unplanned = demand.obsid in unplanned_set
            load = surprise if is_unplanned else planned
            # This request's own share no longer competes with it.
            for k, part in self._shares(demand).items():
                pressure[k] -= part
            need = demand.remaining
            # As the MILP: nothing in a bin, or at least a snapshot, and a
            # cadence request no more than an even share in any bin.
            smallest = min(demand.ss_min, need)
            bins = sorted(demand.visible)
            if demand.deadline is None and not demand.spread:
                bins.sort(
                    key=lambda k: ((load[k] + pressure[k]) / max(capacity[k], 1.0), k)
                )
            given = seconds.setdefault(demand.obsid, {})
            most = math.inf
            if demand.spread and bins:
                most = max(need / len(bins), demand.ss_min)
                for k in bins:
                    part = min(most, demand.visible[k], room(k, is_unplanned), need)
                    part = min(part, data_seconds(demand, k, given.get(k, 0.0) <= 0.0))
                    if part < smallest:
                        continue
                    give(demand, k, part)
                    need -= part
            for k in bins:
                if need < smallest:
                    break
                part = min(need, min(demand.visible[k], most) - given.get(k, 0.0))
                part = min(part, room(k, is_unplanned))
                part = min(part, data_seconds(demand, k, given.get(k, 0.0) <= 0.0))
                if part < smallest:
                    continue
                give(demand, k, part)
                need -= part
        return seconds

    def _solve(
        self,
        demands: Sequence[_Demand],
        capacity: Sequence[float],
        held: Sequence[float],
        unplanned: Collection[int],
        budget: _Budget,
    ) -> tuple[dict[int, dict[int, float]], list[_Demand]]:
        """Solve the allocation as a MILP with HiGHS, level by level.

        The levels are unplanned requests, then the rest. The unplanned
        requests' value is maximized first and their solution fixed, so they
        give up nothing for the rest. The rest also even out the bins' planned
        loads, as a tiebreak. Each level starts from
        the levels already solved with its own requests added greedily where
        they fit, which is feasible for it, so the solver only improves on the
        greedy allocation. Each level may use the time limit's remainder shared
        among the levels left.

        Returns the allocation of the levels solved, and the requests of the
        levels the solver found no solution for in time, if any.
        """
        from ortools.math_opt.python import (
            expressions,
            model_parameters,
            parameters,
            solve,
            variables,
        )
        from ortools.math_opt.python.model import Model
        from ortools.math_opt.python.result import TerminationReason

        model = Model(name="long-range allocation")
        seconds: dict[tuple[int, int], variables.Variable] = {}
        previous = self._previous.seconds if self._previous is not None else {}
        terms: dict[bool, list[variables.LinearTerm]] = {}
        planned_in: dict[int, list[variables.Variable]] = {}
        unplanned_in: dict[int, list[variables.Variable]] = {}
        planned_by_level: dict[bool, dict[int, list[variables.Variable]]]
        planned_by_level = {}
        # Each seconds variable's binary, and its stability variable with the
        # seconds kept from the previous allocation.
        used_of: dict[tuple[int, int], variables.Variable] = {}
        stays_of: dict[tuple[int, int], tuple[variables.Variable, float]] = {}
        level_of: dict[int, bool] = {}
        by_level: dict[bool, list[_Demand]] = {}
        limits: dict[int, tuple[float, float]] = {}
        last_bin = max(len(self.bins) - 1, 1)
        for demand in demands:
            remaining = float(int(demand.remaining))
            smallest = min(float(np.ceil(demand.ss_min)), remaining)
            level_of[demand.obsid] = demand.obsid in unplanned
            by_level.setdefault(level_of[demand.obsid], []).append(demand)
            limits[demand.obsid] = (smallest, remaining)
            even = None
            if demand.spread and demand.visible:
                even = max(demand.remaining / len(demand.visible), demand.ss_min)
            level = level_of[demand.obsid]
            own: list[variables.Variable] = []
            for k, seen in demand.visible.items():
                most = min(remaining, float(int(seen)))
                if even is not None:
                    most = min(most, float(int(even)))
                if most < smallest:
                    continue
                x = model.add_variable(lb=0.0, ub=most, name=f"x{demand.obsid}_{k}")
                used = model.add_binary_variable(name=f"y{demand.obsid}_{k}")
                # Nothing, or at least a minimum snapshot.
                model.add_linear_constraint(x <= most * used)
                model.add_linear_constraint(x >= smallest * used)
                seconds[(demand.obsid, k)] = x
                used_of[(demand.obsid, k)] = used
                own.append(x)
                (unplanned_in if level else planned_in).setdefault(k, []).append(x)
                if not level:
                    planned_by_level.setdefault(level, {}).setdefault(k, []).append(x)
                rate = demand.value
                if demand.deadline is not None:
                    # Earlier is slightly better for a request with a deadline.
                    rate *= 1.0 - 0.01 * k / last_bin
                level_terms = terms.setdefault(level, [])
                level_terms.append(variables.LinearTerm(x, rate))
                kept = float(previous.get(demand.obsid, {}).get(k, 0.0))
                if kept >= smallest:
                    # Staying where the previous allocation put it is worth 1%,
                    # so allocating again moves only for a real gain.
                    stays = model.add_variable(
                        lb=0.0, ub=kept, name=f"s{demand.obsid}_{k}"
                    )
                    model.add_linear_constraint(stays <= x)
                    stays_of[(demand.obsid, k)] = (stays, kept)
                    level_terms.append(variables.LinearTerm(stays, 0.01 * demand.value))
            if own:
                model.add_linear_constraint(expressions.fast_sum(own) <= remaining)
        for k in range(len(self.bins)):
            planned_x = planned_in.get(k, [])
            if planned_x:
                model.add_linear_constraint(
                    expressions.fast_sum(planned_x) <= capacity[k] - held[k]
                )
            every = planned_x + unplanned_in.get(k, [])
            if every:
                model.add_linear_constraint(expressions.fast_sum(every) <= capacity[k])

        # The data budget: what each bin carries on the recorder into the
        # next, its housekeeping and allocated science less its downlink, is
        # bounded below by the chain and above by the limit; a lower bound
        # is exact, since a larger carry-over only tightens the bins after.
        # Housekeeping alone can exceed the limit; a bin's bound is then
        # what it carries with no science, so the model stays feasible and
        # the science that fits elsewhere is still allocated.
        # Each recorder with a budget has its own chain.
        carries: dict[str, list[variables.Variable]] = {}
        for name, store in budget.stores.items():
            if not store.active:
                continue
            assert store.limit is not None
            unavoidable = _carries(
                store.start,
                store.housekeeping,
                [0.0] * len(self.bins),
                store.downlink,
            )
            generated_in: dict[int, list[variables.LinearTerm]] = {}
            for (obsid, k), x in seconds.items():
                rate, fixed = store.rates.get(obsid, (0.0, 0.0))
                if rate > 0.0:
                    generated_in.setdefault(k, []).append(variables.LinearTerm(x, rate))
                if fixed > 0.0:
                    generated_in.setdefault(k, []).append(
                        variables.LinearTerm(used_of[(obsid, k)], fixed)
                    )
            previous_carry: variables.Variable | float = store.start
            chain = carries[name] = []
            for k in range(len(self.bins)):
                carry = model.add_variable(
                    lb=0.0,
                    ub=max(store.limit, unavoidable[k]),
                    name=f"v{name}_{k}",
                )
                model.add_linear_constraint(
                    carry
                    >= previous_carry
                    + store.housekeeping[k]
                    - store.downlink[k]
                    + expressions.fast_sum(generated_in.get(k, []))
                )
                chain.append(carry)
                previous_carry = carry

        levels = sorted(terms, reverse=True)
        statuses: list[str] = []
        solved: dict[tuple[int, int], float] | None = None
        last_values: dict[variables.Variable, float] = {}
        done: set[bool] = set()
        # Each load-balancing deviation, its goal and the seconds it sums.
        deviations: list[
            tuple[variables.Variable, float, list[variables.Variable]]
        ] = []
        deadline = time.monotonic() + self.time_limit

        def start_for(level: bool) -> dict[variables.Variable, float]:
            """A feasible solution to start ``level`` from.

            The levels solved keep their solution exactly, so they still meet
            their held values; ``level``'s requests are added greedily where
            they fit, within each variable's bounds; the levels below get
            nothing yet.
            """
            base: dict[int, dict[int, float]] = {}
            for (obsid, k), value in (solved or {}).items():
                if level_of[obsid] in done and value > 0.0:
                    base.setdefault(obsid, {})[k] = value
            filled = self._greedy(
                by_level[level], capacity, held, unplanned, base, budget
            )
            own: dict[tuple[int, int], float] = {}
            for demand in by_level[level]:
                smallest, remaining = limits[demand.obsid]
                total = 0.0
                for k, value in sorted(filled.get(demand.obsid, {}).items()):
                    x = seconds.get((demand.obsid, k))
                    if x is None:
                        continue
                    part = min(value, x.upper_bound, remaining - total)
                    if part >= smallest:
                        own[(demand.obsid, k)] = part
                        total += part
            values: dict[variables.Variable, float] = {}
            for key, x in seconds.items():
                if level_of[key[0]] in done:
                    values[x] = last_values[x]
                    values[used_of[key]] = last_values[used_of[key]]
                    if key in stays_of:
                        stays = stays_of[key][0]
                        values[stays] = last_values[stays]
                    continue
                part = own.get(key, 0.0)
                values[x] = part
                values[used_of[key]] = 1.0 if part > 0.0 else 0.0
                if key in stays_of:
                    stays, kept = stays_of[key]
                    values[stays] = min(kept, part)
            for deviation, goal, summed in deviations:
                values[deviation] = abs(sum(values[x] for x in summed) - goal)
            if carries:
                started: dict[int, dict[int, float]] = {}
                for (obsid, k), x in seconds.items():
                    if values[x] > 0.0:
                        started.setdefault(obsid, {})[k] = values[x]
                for name, chain in carries.items():
                    store = budget.stores[name]
                    volumes = _carries(
                        store.start,
                        store.housekeeping,
                        store.science(started),
                        store.downlink,
                    )
                    for carry, value in zip(chain, volumes, strict=True):
                        values[carry] = min(value, carry.upper_bound)
            return values

        def run(
            level: bool, objective: variables.LinearSum, stages_left: int
        ) -> float | None:
            """Maximize ``objective`` for ``level``; return its value."""
            nonlocal solved
            model.maximize(objective)
            start = start_for(level)
            # Time the levels above did not use passes to the levels left.
            share = max(deadline - time.monotonic(), 0.0) / stages_left
            settings = parameters.SolveParameters(
                time_limit=timedelta(seconds=max(share, 0.1))
            )
            try:
                outcome = solve.solve(
                    model,
                    parameters.SolverType.HIGHS,
                    params=settings,
                    model_params=model_parameters.ModelSolveParameters(
                        solution_hints=[
                            model_parameters.SolutionHint(variable_values=start)
                        ]
                    ),
                )
            except RuntimeError:
                # HiGHS can reject a starting solution; solve without one.
                outcome = solve.solve(
                    model, parameters.SolverType.HIGHS, params=settings
                )
            reason = outcome.termination.reason
            optimal = reason == TerminationReason.OPTIMAL
            statuses.append("OPTIMAL" if optimal else "FEASIBLE")
            if not outcome.has_primal_feasible_solution():
                statuses[-1] = reason.name
                return None
            values = outcome.variable_values()
            last_values.clear()
            last_values.update(values)
            solved = {key: values[x] for key, x in seconds.items()}
            return outcome.objective_value()

        # Spread the planned load over the run: the planned seconds in each
        # bin are kept as close as visibility allows to the fill their demand
        # would give if spread evenly. It is a tiebreak in the planned level's
        # objective, worth a thousandth of its least merit per second, so it
        # never costs a second of science for less than a thousand seconds of
        # better balance, and below the stability bonus, so allocating again
        # still stays put. It is too small to steer the solver by itself in a
        # large allocation, so the greedy start is balanced too.
        balance: list[variables.LinearTerm] = []
        planned = by_level.get(False, [])
        planned_demand = sum(min(d.remaining, sum(d.visible.values())) for d in planned)
        rooms = sum(capacity[k] - held[k] for k in range(len(self.bins)))
        fill = min(1.0, planned_demand / max(rooms, 1.0))
        # Of the merits that are worth allocating: one a completion deficit
        # takes below zero would turn the penalty into a reward, unbounded.
        least = min(
            (t.coefficient for t in terms.get(False, []) if t.coefficient > 0.0),
            default=0.0,
        )
        for k, summed in planned_by_level.get(False, {}).items():
            goal = fill * (capacity[k] - held[k])
            deviation = model.add_variable(lb=0.0, name=f"dev{k}")
            load = expressions.fast_sum(summed)
            model.add_linear_constraint(deviation >= load - goal)
            model.add_linear_constraint(deviation >= goal - load)
            balance.append(variables.LinearTerm(deviation, -1e-3 * least))
            deviations.append((deviation, goal, list(summed)))
        for n, level in enumerate(levels):
            tiebreak = [] if level else balance
            objective = expressions.fast_sum([*terms[level], *tiebreak])
            if run(level, objective, len(levels) - n) is None:
                break
            done.add(level)
            # Fix this level's allocation while the levels below are solved.
            # Holding only its value would let them rearrange it, but a dense
            # constraint over every seconds variable of a large level leaves
            # HiGHS unable to solve the levels below.
            for key, x in seconds.items():
                if level_of[key[0]] == level:
                    x.lower_bound = x.upper_bound = last_values[x]
                    used = used_of[key]
                    used.lower_bound = used.upper_bound = round(last_values[used])
        self.solver_status = (
            "OPTIMAL"
            if statuses and set(statuses) == {"OPTIMAL"}
            else ",".join(statuses)
        )
        result: dict[int, dict[int, float]] = {d.obsid: {} for d in demands}
        # Requests with no seconds to allocate form no level and are done.
        unsolved = [d for d in demands if (d.obsid in unplanned) in set(levels) - done]
        if solved is None:
            return result, unsolved
        missing = {d.obsid for d in unsolved}
        for (obsid, k), value in solved.items():
            # Whole seconds, so the allocation is not cluttered by solver noise.
            if obsid not in missing and round(value) > 0:
                result[obsid][k] = float(round(value))
        return result, unsolved

    def _demand(
        self,
        target: Pointing,
        start: float,
        reserved: Mapping[int, float],
        shares: Mapping[str, float],
    ) -> _Demand | None:
        obsid = int(target.obsid)
        exptime = target.exptime if target.exptime is not None else target.ss_max
        remaining = float(exptime) - reserved.get(obsid, 0.0)
        if target.done or remaining < float(snapshot_min_seconds(target)):
            return None
        windows = self._windows.get(obsid)
        if windows is None:
            windows = self._windows[obsid] = self.ctx.visibility_windows(target)
        stop = target.deadline if target.deadline is not None else float("inf")
        opens = start
        if target.earliest_start is not None:
            opens = max(start, target.earliest_start)
        visible: dict[int, float] = {}
        for k, (b0, b1) in enumerate(self.bins):
            lo, hi = max(b0, opens), min(b1, stop)
            if hi <= lo:
                continue
            seen = sum(max(0.0, min(w1, hi) - max(w0, lo)) for w0, w1 in windows)
            if seen >= float(snapshot_min_seconds(target)):
                visible[k] = seen
        category = self.merit_model.category(target)
        merit = self.merit_model.value_terms(target, start, delivered_shares=shares)
        generated: dict[str, tuple[float, float]] = {}
        for name, view in self.config.recorder_views().items():
            fixed = view.science_generated(0.0, target.science_data)
            rate = view.science_generated(1.0, target.science_data) - fixed
            generated[name] = (max(0.0, rate), max(0.0, fixed))
        return _Demand(
            obsid=obsid,
            value=merit.value,
            remaining=remaining,
            ss_min=float(snapshot_min_seconds(target)),
            deadline=target.deadline,
            spread=category.cadence_seconds is not None
            and self.merit_model.cadence_weight > 0.0,
            visible=visible,
            generated=generated,
        )

    def _budget(
        self,
        start: float,
        demands: Sequence[_Demand],
        start_volume_gb: float | Mapping[str, float] | None,
    ) -> _Budget:
        """The run's data budget from ``start``, recorder by recorder: what
        each bin's contacts downlink from it and what it generates on its
        own, from the same pass prediction and rates the schedulers use. A
        contact straddling a bin boundary counts in each bin for the part
        inside it. A mission with named stores budgets each within its own
        limit and its own reserved share of the contacts."""
        views = self.config.recorder_views()
        if isinstance(start_volume_gb, Mapping) and set(start_volume_gb) != set(views):
            raise ValueError("start volumes must name every recorder exactly once")
        if (
            isinstance(start_volume_gb, (int, float))
            and self.config.data_storage is not None
        ):
            raise ValueError("named recorders need start volumes by store")
        stores: dict[str, _StoreBudget] = {}
        for name, view in views.items():
            volume = (
                start_volume_gb[name]
                if isinstance(start_volume_gb, Mapping)
                else start_volume_gb
                if start_volume_gb is not None
                else view.recorder.current_volume_gb
            )
            stores[name] = _StoreBudget(
                limit=view.recorder.planning_limit_gb,
                start=float(volume),
                downlink=[],
                housekeeping=[],
                rates={d.obsid: d.generated.get(name, (0.0, 0.0)) for d in demands},
            )
        if not any(store.active for store in stores.values()):
            # No budget: nothing is predicted or counted, as predicting a
            # run's passes is slow.
            return _Budget(stores)
        contacts: list[list[tuple[str, float]]] = [[] for _ in self.bins]
        contact_seconds = [0.0] * len(self.bins)
        coverage = self.ctx.coverage
        if coverage is None:
            if self.passes is None:
                self.passes = list(self.ctx.predict_passes())
            for gspass in self.passes:
                for k, (b0, b1) in enumerate(self.bins):
                    lo, hi = max(b0, start, gspass.begin), min(b1, gspass.end)
                    if hi > lo:
                        contacts[k].append((gspass.station, hi - lo))
                        contact_seconds[k] += hi - lo
        for name, store in stores.items():
            if not store.active:
                continue
            view = views[name]
            if coverage is not None:
                downlink = [
                    sum(
                        span.rate_mbps * (span.end - span.begin) / 1000.0
                        for span in (
                            coverage.recorder_spans(view, max(b0, start), b1)
                            if isinstance(view, RecorderView)
                            else coverage.spans(max(b0, start), b1)
                        )
                    )
                    for b0, b1 in self.bins
                ]
            else:
                downlink = [
                    sum(
                        contact_downlink_gb(view, station, seconds)
                        for station, seconds in k
                    )
                    for k in contacts
                ]
            housekeeping = []
            for k, (b0, b1) in enumerate(self.bins):
                seconds = max(0.0, b1 - max(b0, start))
                housekeeping.append(
                    view.housekeeping_generated(seconds - contact_seconds[k])
                    + view.housekeeping_generated(contact_seconds[k], mode=ACSMode.PASS)
                )
            stores[name] = store._replace(downlink=downlink, housekeeping=housekeeping)
        return _Budget(stores)

    @staticmethod
    def _shares(demand: _Demand) -> dict[int, float]:
        """How much of the request's exposure each bin might be asked for."""
        total = sum(demand.visible.values())
        if total <= 0.0:
            return {}
        return {
            k: demand.remaining * seen / total for k, seen in demand.visible.items()
        }

    def _pressure(self, demands: Sequence[_Demand]) -> list[float]:
        """Expected demand on each bin, if every request spread itself by visibility."""
        pressure = [0.0] * len(self.bins)
        for demand in demands:
            for k, part in self._shares(demand).items():
                pressure[k] += part
        return pressure


def _require_ortools() -> None:
    """Raise an ImportError that says how to install OR-Tools if it is missing."""
    try:
        from ortools.math_opt.python import solve  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'The "milp" allocation solver needs OR-Tools (the ortools package); '
            'install it with pip install "coast-sim[cpsat]", or use solver="greedy"'
        ) from exc
