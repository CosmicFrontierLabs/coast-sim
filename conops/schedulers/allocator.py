"""Long-range planning: decide on which day each request is observed.

:class:`LongRangeAllocator` plans a run of days to months at a coarse level,
deciding how much of each request's remaining exposure to collect in which bin
(a day by default). The short-term scheduling, rolling replans or dispatch, then
orders the observations within each bin, working the bin's allocated requests
first. Without it, short-term scheduling looks only hours ahead and spends each
day on the best targets available that day, so targets observable for only part
of the run, or due beyond its horizon, can miss their chance.
"""

import time
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

    For each request, the seconds its target is visible in each bin, from its
    earliest start to its deadline, come from the same roll-independent
    visibility windows the planners use. Each bin is expected to hold
    ``efficiency`` of its length in science; the rest goes to slews, setup and
    ground passes. Of that capacity, ``reserve`` is held back for work not yet
    known, such as Targets of Opportunity, so that when it arrives it fits
    without pushing planned requests to later bins. Requests passed as ``unplanned`` to
    :meth:`allocate` are allocated first and use the reserve before the rest.

    By default the allocation is solved as a mixed-integer program with HiGHS
    (bundled with OR-Tools). It maximizes merit-weighted seconds, one level at
    a time, unplanned requests first and then each tier, each with the levels
    above fixed at their solution, subject to:

    * each request gets, in each bin, either nothing or at least its ``ss_min``,
      and no more than its target is visible there between its earliest
      start and its deadline;
    * no request gets more than its remaining exposure;
    * each bin's planned requests fit its capacity less the reserve, and all its
      requests fit its capacity;
    * a request with a cadence gets no more than an even share in any bin, so its
      visits are spread across the run.

    Requests with a deadline are worth slightly more in earlier bins. Keeping
    seconds in the bin the previous allocation gave them is worth 1% more, so
    that allocating again moves requests only for a real gain. As a tiebreak in
    each level, the planned load of the level and those above it is spread
    evenly over the bins, as far as visibility allows, worth a thousandth of the
    level's least merit per second, so the slack is spread across the run
    rather than left in a few bins. Each level
    starts from the greedy allocation of its requests into the room the levels
    above leave, so the solver only improves on it; ``solver="greedy"`` uses
    the greedy allocation on its own: requests in order of tier and merit, each
    in the bins least contended by the requests still to come. If the solver
    finds no solution for a level in time, that level and those below it are
    allocated greedily.

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
        unplanned_set = set(unplanned)
        empty: dict[int, dict[int, float]] = {d.obsid: {} for d in demands}
        if self.solver == "greedy" or not demands:
            seconds = self._greedy(demands, capacity, held, unplanned_set, empty)
        else:
            seconds, unsolved = self._solve(demands, capacity, held, unplanned_set)
            if unsolved:
                # Levels the solver found no solution for go greedily into
                # the room the solved levels leave.
                seconds = self._greedy(unsolved, capacity, held, unplanned_set, seconds)
        allocation = Allocation(
            bins=list(self.bins), capacity=capacity, reserve=held, seconds=seconds
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
    ) -> dict[int, dict[int, float]]:
        """Add ``demands`` to ``allocated`` in priority order, each where it fits.

        Unplanned requests go first, then by tier and value. A request with a
        cadence is spread evenly over the bins it is visible in; one with a
        deadline goes in its earliest bins; the rest go in the bins the
        requests not yet placed are expected to need least.
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
            given = seconds.setdefault(demand.obsid, {})
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
        return seconds

    def _solve(
        self,
        demands: Sequence[_Demand],
        capacity: Sequence[float],
        held: Sequence[float],
        unplanned: Collection[int],
    ) -> tuple[dict[int, dict[int, float]], list[_Demand]]:
        """Solve the allocation as a MILP with HiGHS, level by level.

        Levels are unplanned requests, then each tier from the highest. Each
        level's value is maximized with the levels above fixed at their
        solution, so no level gives up anything for the levels below it. Each
        planned level also evens out the bins' planned loads, as a tiebreak. Each level starts from
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
        terms: dict[tuple[bool, int], list[variables.LinearTerm]] = {}
        planned_in: dict[int, list[variables.Variable]] = {}
        unplanned_in: dict[int, list[variables.Variable]] = {}
        planned_by_level: dict[tuple[bool, int], dict[int, list[variables.Variable]]]
        planned_by_level = {}
        # Each seconds variable's binary, and its stability variable with the
        # seconds kept from the previous allocation.
        used_of: dict[tuple[int, int], variables.Variable] = {}
        stays_of: dict[tuple[int, int], tuple[variables.Variable, float]] = {}
        level_of: dict[int, tuple[bool, int]] = {}
        by_level: dict[tuple[bool, int], list[_Demand]] = {}
        limits: dict[int, tuple[float, float]] = {}
        last_bin = max(len(self.bins) - 1, 1)
        for demand in demands:
            remaining = float(int(demand.remaining))
            smallest = min(float(np.ceil(demand.ss_min)), remaining)
            level_of[demand.obsid] = (demand.obsid in unplanned, demand.tier)
            by_level.setdefault(level_of[demand.obsid], []).append(demand)
            limits[demand.obsid] = (smallest, remaining)
            even = None
            if demand.spread and demand.visible:
                even = max(demand.remaining / len(demand.visible), demand.ss_min)
            level = (demand.obsid in unplanned, demand.tier)
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
                (unplanned_in if level[0] else planned_in).setdefault(k, []).append(x)
                if not level[0]:
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

        levels = sorted(terms, reverse=True)
        statuses: list[str] = []
        solved: dict[tuple[int, int], float] | None = None
        last_values: dict[variables.Variable, float] = {}
        done: set[tuple[bool, int]] = set()
        # Each load-balancing deviation, its goal and the seconds it sums.
        deviations: list[
            tuple[variables.Variable, float, list[variables.Variable]]
        ] = []
        deadline = time.monotonic() + self.time_limit

        def start_for(level: tuple[bool, int]) -> dict[variables.Variable, float]:
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
            filled = self._greedy(by_level[level], capacity, held, unplanned, base)
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
            return values

        def run(
            level: tuple[bool, int], objective: variables.LinearSum, stages_left: int
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

        # Spread the planned load over the run, level by level: with each
        # planned level, the seconds it and the planned levels above it have in
        # each bin are kept as close as visibility allows to the fill their
        # demand would give if spread evenly, so each level evens out what the
        # levels above leave. It is a tiebreak in the level's objective, worth a
        # thousandth of the level's least merit per second, so it never costs a
        # second of science for less than a thousand seconds of better balance,
        # and below the stability bonus, so allocating again still stays put.
        balance: dict[tuple[bool, int], list[variables.LinearTerm]] = {}
        rooms = sum(capacity[k] - held[k] for k in range(len(self.bins)))
        above: dict[int, list[variables.Variable]] = {}
        demand_above = 0.0
        for level in levels:
            if level[0]:
                continue
            for k, summed in planned_by_level.get(level, {}).items():
                above.setdefault(k, []).extend(summed)
            demand_above += sum(
                min(d.remaining, sum(d.visible.values())) for d in by_level[level]
            )
            fill = min(1.0, demand_above / max(rooms, 1.0))
            least = min(term.coefficient for term in terms[level])
            for k, summed in above.items():
                goal = fill * (capacity[k] - held[k])
                deviation = model.add_variable(lb=0.0, name=f"dev{level[1]}_{k}")
                load = expressions.fast_sum(summed)
                model.add_linear_constraint(deviation >= load - goal)
                model.add_linear_constraint(deviation >= goal - load)
                balance.setdefault(level, []).append(
                    variables.LinearTerm(deviation, -1e-3 * least)
                )
                deviations.append((deviation, goal, list(summed)))
        for n, level in enumerate(levels):
            objective = expressions.fast_sum([*terms[level], *balance.get(level, [])])
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
        unsolved = [
            d for d in demands if (d.obsid in unplanned, d.tier) in set(levels) - done
        ]
        if solved is None:
            return result, unsolved
        missing = {d.obsid for d in unsolved}
        for (obsid, k), value in solved.items():
            # Whole seconds, so the allocation is not cluttered by solver noise.
            if obsid not in missing and round(value) > 0:
                result[obsid][k] = float(round(value))
        return result, unsolved

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
        opens = start
        if target.earliest_start is not None:
            opens = max(start, target.earliest_start)
        visible: dict[int, float] = {}
        for k, (b0, b1) in enumerate(self.bins):
            lo, hi = max(b0, opens), min(b1, stop)
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
