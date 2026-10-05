"""Long-range planning: decide on which day each request is observed.

:class:`LongRangeAllocator` plans a run of days to months at a coarse level,
deciding how much of each request's remaining exposure to collect in which bin
(a day by default). The short-term scheduling, rolling replans or dispatch, then
orders the observations within each bin, working the bin's allocated requests
first. Without it, short-term scheduling looks only hours ahead and spends each
day on the best targets available that day, so targets observable for only part
of the run, or due beyond its horizon, can miss their chance.
"""

from collections.abc import Collection, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..config import MissionConfig
from ..targets import Pointing
from ..targets.merit import MeritModel
from .context import SchedulingContext


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

    def preferred(self, obsids: Sequence[int], begin: float, end: float) -> set[int]:
        """The obsids among ``obsids`` that :meth:`prefers` during ``[begin, end)``."""
        return {obsid for obsid in obsids if self.prefers(obsid, begin, end)}


class _Demand(BaseModel):
    """One request as the allocator sees it."""

    obsid: int
    tier: int
    value: float
    remaining: float
    ss_min: float
    deadline: float | None
    spread: bool
    """Whether its exposure should be spread across bins, as for a cadence."""
    visible: dict[int, float]
    """Seconds it is visible in each bin where it can be observed."""


class LongRangeAllocator:
    """Allocate requests' remaining exposure to bins of a long run.

    For each request, the seconds its target is visible in each bin, up to its
    deadline, come from the same roll-independent visibility windows the
    planners use. Each bin is expected to hold ``efficiency`` of its length in
    science; the rest goes to slews, setup and ground passes. Of that capacity,
    ``reserve`` is held back for work not yet known, such as Targets of
    Opportunity, so that when it arrives it fits without pushing planned
    requests to later bins. Requests passed as ``unplanned`` to
    :meth:`allocate` are allocated first and use the reserve before the rest.

    By default the allocation is solved as a mixed-integer program with OR-Tools
    CP-SAT. It maximizes merit-weighted seconds, with unplanned requests first
    and then tiers strictly ahead of lower tiers, subject to:

    * each request gets, in each bin, either nothing or at least its ``ss_min``,
      and no more than its target is visible there before its deadline;
    * no request gets more than its remaining exposure;
    * each bin's planned requests fit its capacity less the reserve, and all its
      requests fit its capacity;
    * a request with a cadence gets no more than an even share in any bin, so its
      visits are spread across the run.

    Requests with a deadline are worth slightly more in earlier bins. Keeping
    seconds in the bin the previous allocation gave them is worth slightly more,
    so that allocating again moves requests only for a real gain. The solver
    starts from a greedy allocation, which ``solver="greedy"`` uses on its own:
    requests in order of tier and merit, each in the bins least contended by
    the requests still to come.

    Args:
        config: Mission configuration, with an ephemeris on its constraint.
        begin: Start of the run.
        end: End of the run.
        bin_length: Length of each bin.
        efficiency: Fraction of each bin expected to hold science, in (0, 1].
        reserve: Fraction of that capacity held back for unplanned work, in
            [0, 1).
        solver: "milp" (the default) or "greedy".
        time_limit: Seconds the MILP solver may search per allocation.
        workers: CP-SAT search workers. Use 1 for allocations that repeat
            exactly.
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
        workers: int = 8,
    ) -> None:
        if bin_length <= timedelta(0):
            raise ValueError("bin_length must be positive")
        if not 0.0 < efficiency <= 1.0:
            raise ValueError("efficiency must be in (0, 1]")
        if not 0.0 <= reserve < 1.0:
            raise ValueError("reserve must be in [0, 1)")
        if solver not in ("milp", "greedy"):
            raise ValueError('solver must be "milp" or "greedy"')
        if time_limit <= 0 or workers < 1:
            raise ValueError("time_limit and workers must be positive")
        self.config = config
        self.begin = begin
        self.end = end
        self.bin_length = bin_length
        self.efficiency = efficiency
        self.reserve = reserve
        self.solver = solver
        self.time_limit = time_limit
        self.workers = workers
        self.solver_status: str | None = None
        """CP-SAT's status for the last allocation, such as OPTIMAL."""
        self._previous: Allocation | None = None
        self.ctx = SchedulingContext(config, begin, end)
        self.merit_model = MeritModel(config)
        self._windows: dict[int, list[tuple[float, float]]] = {}
        length = bin_length.total_seconds()
        ustart, uend = begin.timestamp(), end.timestamp()
        count = max(1, int(-(-(uend - ustart) // length)))
        self.bins = [
            (ustart + k * length, min(ustart + (k + 1) * length, uend))
            for k in range(count)
        ]

    def allocate(
        self,
        targets: Sequence[Pointing],
        start: float,
        reserved_seconds: Mapping[int, float] | None = None,
        unplanned: Collection[int] = (),
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
        """
        reserved = reserved_seconds or {}
        capacity = [
            max(0.0, b1 - max(b0, start)) * self.efficiency for b0, b1 in self.bins
        ]
        held = [c * self.reserve for c in capacity]
        demands = [
            demand
            for target in targets
            if (demand := self._demand(target, start, reserved)) is not None
        ]
        seconds: dict[int, dict[int, float]] = {d.obsid: {} for d in demands}
        # Seconds allocated in each bin to unplanned requests, and to the rest.
        surprise = [0.0] * len(self.bins)
        planned = [0.0] * len(self.bins)
        pressure = self._pressure(demands)
        unplanned_set = set(unplanned)

        def room(k: int, is_unplanned: bool) -> float:
            if is_unplanned:
                return capacity[k] - surprise[k] - planned[k]
            # Planned requests leave the reserve, or as much of the bin as
            # unplanned requests have already taken, whichever is more.
            return capacity[k] - max(surprise[k], held[k]) - planned[k]

        order = sorted(
            demands,
            key=lambda d: (d.obsid not in unplanned_set, -d.tier, -d.value, d.obsid),
        )
        # The greedy allocation: the result with solver="greedy", and the MILP
        # solver's starting point otherwise.
        for demand in order:
            is_unplanned = demand.obsid in unplanned_set
            load = surprise if is_unplanned else planned
            # This request's own share no longer competes with it.
            for k, part in self._shares(demand).items():
                pressure[k] -= part
            need = demand.remaining
            bins = sorted(demand.visible)
            if demand.deadline is None and not demand.spread:
                bins.sort(key=lambda k: (pressure[k] / max(capacity[k], 1.0), k))
            given = seconds[demand.obsid]
            if demand.spread and bins:
                even = max(need / len(bins), demand.ss_min)
                for k in bins:
                    part = min(even, demand.visible[k], room(k, is_unplanned), need)
                    if part < min(demand.ss_min, need):
                        continue
                    given[k] = given.get(k, 0.0) + part
                    load[k] += part
                    need -= part
            for k in bins:
                if need < demand.ss_min:
                    break
                part = min(need, demand.visible[k] - given.get(k, 0.0))
                part = min(part, room(k, is_unplanned))
                if part < min(demand.ss_min, need):
                    continue
                given[k] = given.get(k, 0.0) + part
                load[k] += part
                need -= part

        if self.solver == "milp" and demands:
            seconds = self._solve(demands, capacity, held, unplanned_set, seconds)
        allocation = Allocation(
            bins=list(self.bins), capacity=capacity, reserve=held, seconds=seconds
        )
        self._previous = allocation
        return allocation

    def _solve(
        self,
        demands: Sequence[_Demand],
        capacity: Sequence[float],
        held: Sequence[float],
        unplanned: Collection[int],
        greedy: dict[int, dict[int, float]],
    ) -> dict[int, dict[int, float]]:
        """Solve the allocation as a MILP, level by level, from the greedy one.

        Levels are unplanned requests, then each tier from the highest. Each
        level's value is maximized with the levels above held at their best, so
        no level gives up anything for the levels below it.
        """
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        seconds: dict[tuple[int, int], cp_model.IntVar] = {}
        bound: dict[tuple[int, int], int] = {}
        previous = self._previous.seconds if self._previous is not None else {}
        terms: dict[tuple[bool, int], list[cp_model.LinearExprT]] = {}
        planned_in: dict[int, list[cp_model.IntVar]] = {}
        unplanned_in: dict[int, list[cp_model.IntVar]] = {}
        last_bin = max(len(self.bins) - 1, 1)
        for demand in demands:
            remaining = int(demand.remaining)
            smallest = min(int(np.ceil(demand.ss_min)), remaining)
            even = None
            if demand.spread and demand.visible:
                even = int(max(demand.remaining / len(demand.visible), demand.ss_min))
            level = (demand.obsid in unplanned, demand.tier)
            own: list[cp_model.IntVar] = []
            for k, seen in demand.visible.items():
                most = min(remaining, int(seen))
                if even is not None:
                    most = min(most, even)
                if most < smallest:
                    continue
                x = model.new_int_var(0, most, f"x{demand.obsid}_{k}")
                used = model.new_bool_var(f"y{demand.obsid}_{k}")
                # Nothing, or at least a minimum snapshot.
                model.add(x <= most * used)
                model.add(x >= smallest * used)
                seconds[(demand.obsid, k)] = x
                bound[(demand.obsid, k)] = most
                own.append(x)
                (unplanned_in if level[0] else planned_in).setdefault(k, []).append(x)
                # Merit per second, in thousandths so the objective is exact.
                rate = 1000.0 * demand.value
                if demand.deadline is not None:
                    # Earlier is slightly better for a request with a deadline.
                    rate *= 1.0 - 0.01 * k / last_bin
                level_terms = terms.setdefault(level, [])
                level_terms.append(int(round(rate)) * x)
                kept = int(previous.get(demand.obsid, {}).get(k, 0.0))
                if kept >= smallest:
                    # Staying where the previous allocation put it is worth 1%,
                    # so allocating again moves only for a real gain.
                    stays = model.new_int_var(0, kept, f"s{demand.obsid}_{k}")
                    model.add(stays <= x)
                    level_terms.append(int(round(10.0 * demand.value)) * stays)
            if own:
                model.add(sum(own) <= remaining)
        for k in range(len(self.bins)):
            planned_x = planned_in.get(k, [])
            if planned_x:
                model.add(sum(planned_x) <= int(capacity[k] - held[k]))
            every = planned_x + unplanned_in.get(k, [])
            if every:
                model.add(sum(every) <= int(capacity[k]))

        hint = {
            key: min(max(int(greedy.get(key[0], {}).get(key[1], 0.0)), 0), bound[key])
            for key in seconds
        }
        levels = sorted(terms, reverse=True)
        statuses: list[str] = []
        solved: dict[tuple[int, int], int] | None = None
        for level in levels:
            objective = sum(terms[level])
            model.clear_hints()  # type: ignore[no-untyped-call]
            for key, x in seconds.items():
                model.add_hint(x, hint[key])
            model.maximize(objective)
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = self.time_limit / len(levels)
            solver.parameters.num_workers = self.workers
            solver.parameters.random_seed = self.config.random_seed or 0
            status = solver.solve(model)
            statuses.append(solver.status_name(status))
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                break
            solved = {key: int(solver.value(x)) for key, x in seconds.items()}
            hint = solved
            # Hold this level at what it achieved while the next is solved.
            model.add(objective >= int(round(solver.objective_value)))
        self.solver_status = (
            "OPTIMAL"
            if statuses and set(statuses) == {"OPTIMAL"}
            else ",".join(statuses)
        )
        if solved is None:
            return greedy
        result: dict[int, dict[int, float]] = {d.obsid: {} for d in demands}
        for (obsid, k), value in solved.items():
            if value > 0:
                result[obsid][k] = float(value)
        return result

    def _demand(
        self, target: Pointing, start: float, reserved: Mapping[int, float]
    ) -> _Demand | None:
        obsid = int(target.obsid)
        exptime = target.exptime if target.exptime is not None else target.ss_max
        remaining = float(exptime) - reserved.get(obsid, 0.0)
        if target.done or remaining < float(target.ss_min):
            return None
        windows = self._windows.get(obsid)
        if windows is None:
            windows = self._windows[obsid] = self.ctx.visibility_windows(target)
        stop = target.deadline if target.deadline is not None else float("inf")
        visible: dict[int, float] = {}
        for k, (b0, b1) in enumerate(self.bins):
            lo, hi = max(b0, start), min(b1, stop)
            if hi <= lo:
                continue
            seen = sum(max(0.0, min(w1, hi) - max(w0, lo)) for w0, w1 in windows)
            if seen >= float(target.ss_min):
                visible[k] = seen
        category = self.merit_model.category(target)
        merit = self.merit_model.value_terms(target, start, base=float(target.fom))
        return _Demand(
            obsid=obsid,
            tier=category.tier,
            value=merit.value,
            remaining=remaining,
            ss_min=float(target.ss_min),
            deadline=target.deadline,
            spread=category.cadence_seconds is not None
            and self.merit_model.cadence_weight > 0.0,
            visible=visible,
        )

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
