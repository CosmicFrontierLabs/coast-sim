"""Long-range allocation and how it steers schedulers and simulations."""

import builtins
import importlib.util
import math
from collections.abc import Callable
from datetime import timedelta
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from conops import QueueDITL, RollingHorizonDITL, create_ditl
from conops.common import ObsType
from conops.config import (
    AllocationSettings,
    MissionConfig,
    SchedulerConfig,
    SchedulerMode,
)
from conops.config.observation_categories import ObservationCategory
from conops.ditl.factory import queue_targets
from conops.schedulers import (
    AllocatedTime,
    Allocation,
    CpSatScheduler,
    LocalSearchScheduler,
    LongRangeAllocator,
    PriorityScheduler,
    SchedulingContext,
)
from conops.targets import Plan, Pointing, TargetQueue

from .planning_scenario import BEGIN, CONSTRAINED, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None
needs_ortools = pytest.mark.skipif(not HAS_ORTOOLS, reason="needs OR-Tools")


class TestAllocation:
    ALLOCATION = Allocation(
        bins=[(T0, T0 + HOUR), (T0 + HOUR, T0 + 2 * HOUR)],
        capacity=[2700.0, 2700.0],
        reserve=[270.0, 270.0],
        seconds={1: {0: 600.0}, 2: {1: 600.0}, 3: {}},
    )

    def test_prefers_requests_allocated_to_the_time(self) -> None:
        allocation = self.ALLOCATION

        assert allocation.prefers(1, T0, T0 + HOUR)
        assert not allocation.prefers(2, T0, T0 + HOUR)
        assert allocation.prefers(2, T0 + 30 * MIN, T0 + 90 * MIN)
        assert not allocation.prefers(3, T0, T0 + 2 * HOUR)

    def test_prefers_requests_it_never_considered(self) -> None:
        """Such as a ToO submitted after the allocation was made."""
        assert self.ALLOCATION.prefers(99, T0, T0 + HOUR)

    def test_allocated_time_is_given_bin_by_bin(self) -> None:
        allocated = self.ALLOCATION.allocated([1, 2, 3], T0, T0 + 2 * HOUR)

        assert allocated == {
            1: [AllocatedTime(T0, T0 + HOUR, 600.0)],
            2: [AllocatedTime(T0 + HOUR, T0 + 2 * HOUR, 600.0)],
            3: [],
        }

    def test_a_bin_cut_short_gives_its_share(self) -> None:
        """From the start, or the bin's start, to the end of the span."""
        allocated = self.ALLOCATION.allocated([1, 2], T0 + 30 * MIN, T0 + 75 * MIN)

        # All of bin 0 from the span's start is in the span; a quarter of bin 1.
        assert allocated[1] == [AllocatedTime(T0 + 30 * MIN, T0 + HOUR, 600.0)]
        assert allocated[2] == [AllocatedTime(T0 + HOUR, T0 + 75 * MIN, 150.0)]

    def test_used_seconds_come_off_their_bin(self) -> None:
        allocated = self.ALLOCATION.allocated(
            [1, 2], T0, T0 + 2 * HOUR, used={1: {0: 200.0}, 2: {1: 600.0}}
        )

        assert allocated[1] == [AllocatedTime(T0, T0 + HOUR, 400.0)]
        assert allocated[2] == []

    def test_a_request_it_never_considered_has_the_whole_span(self) -> None:
        (span,) = self.ALLOCATION.allocated([99], T0, T0 + HOUR)[99]

        assert (span.begin, span.end) == (T0, T0 + HOUR)
        assert math.isinf(span.seconds)


# ── A week in which when each request can be observed changes ───────────

DAYS = 7
EDGE = 30
"""Targets in each of the Early and Late programs."""


def _along_sun_path(
    config: MissionConfig, count: int, elongations: tuple[float, float], seed: int
) -> list[tuple[float, float]]:
    """Positions along the Sun's path over the run, at ``elongations`` from it.

    Positive elongations lie ahead of the Sun, which closes in on them; each
    position is offset a few degrees across the path.
    """
    ephem = config.constraint.ephem
    assert ephem is not None

    def unit(ra: float, dec: float) -> np.ndarray:
        r, d = np.radians(ra), np.radians(dec)
        return np.array([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])

    first = unit(float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0]))
    last = unit(float(ephem.sun_ra_deg[-1]), float(ephem.sun_dec_deg[-1]))
    normal = np.cross(first, last)
    normal /= np.linalg.norm(normal)
    ahead = np.cross(normal, first)
    rng = np.random.default_rng(seed)
    positions = []
    for _ in range(count):
        angle = np.radians(rng.uniform(*elongations))
        across = np.radians(rng.uniform(-4.0, 4.0))
        vector = np.cos(angle) * first + np.sin(angle) * ahead
        vector = np.cos(across) * vector + np.sin(across) * normal
        vector /= np.linalg.norm(vector)
        ra = float(np.degrees(np.arctan2(vector[1], vector[0])) % 360.0)
        dec = float(np.degrees(np.arcsin(np.clip(vector[2], -1.0, 1.0))))
        positions.append((ra, dec))
    return positions


def _week_targets(config: MissionConfig, seed: int = 1234) -> list[Pointing]:
    """300 random targets, half due within the week, oversubscribing it, and
    two programs by the Sun's 45-degree avoidance zone: Early targets just
    ahead of it, which it covers as the week goes on, and Late ones behind."""
    rng = np.random.default_rng(seed)
    targets = []
    for k in range(300):
        value = float(rng.integers(10, 100))
        deadline = None
        if rng.random() < 0.5:
            deadline = T0 + float(rng.uniform(12.0, DAYS * 24 - 6.0)) * HOUR
        target = Pointing(
            config=config,
            ra=float(rng.uniform(0, 360)),
            dec=float(np.degrees(np.arcsin(rng.uniform(-1, 1)))),
            obsid=10000 + k,
            name=f"t{10000 + k}",
            merit=value,
            ss_min=300,
            ss_max=1200,
            deadline=deadline,
        )
        target.exptime = int(rng.choice((600, 1200, 2400)))
        targets.append(target)
    for first_obsid, elongations, group_seed in (
        (40000, (46.0, 52.0), seed + 5),
        (50000, (-44.0, -38.0), seed + 6),
    ):
        group_rng = np.random.default_rng(group_seed)
        for k, (ra, dec) in enumerate(
            _along_sun_path(config, EDGE, elongations, group_seed)
        ):
            value = float(group_rng.integers(40, 70))
            target = Pointing(
                config=config,
                ra=ra,
                dec=dec,
                obsid=first_obsid + k,
                name=f"t{first_obsid + k}",
                merit=value,
                ss_min=300,
                ss_max=1200,
            )
            target.exptime = 2400
            targets.append(target)
    return targets


def _week_config() -> MissionConfig:
    return _config(
        DAYS * 24,
        categories=[
            ObservationCategory(name="Early", obsid_min=40000, obsid_max=40000 + EDGE),
            ObservationCategory(name="Late", obsid_min=50000, obsid_max=50000 + EDGE),
        ],
    )


Week = tuple[MissionConfig, list[Pointing], LongRangeAllocator, Allocation]


@pytest.fixture(scope="module")
def week() -> Week:
    """The week allocated with the default solver, the MILP."""
    if not HAS_ORTOOLS:
        pytest.skip("needs OR-Tools")
    config = _week_config()
    targets = _week_targets(config)
    allocator = LongRangeAllocator(config, BEGIN, BEGIN + timedelta(days=DAYS))
    return config, targets, allocator, allocator.allocate(targets, T0)


def _fresh(week: Week, **options: object) -> Week:
    """A new allocator for the week and its first allocation.

    Allocations build on the previous one, so tests that allocate twice need
    their own allocator.
    """
    config, targets, _, _ = week
    allocator = LongRangeAllocator(
        config,
        BEGIN,
        BEGIN + timedelta(days=DAYS),
        **options,  # type: ignore[arg-type]
    )
    return config, targets, allocator, allocator.allocate(targets, T0)


class TestLongRangeAllocator:
    def test_bins_and_capacity(self, week: Week) -> None:
        _, _, _, allocation = week

        assert len(allocation.bins) == DAYS
        assert allocation.capacity == pytest.approx([0.75 * 24 * HOUR] * DAYS)
        assert allocation.reserve == pytest.approx([0.1 * 0.75 * 24 * HOUR] * DAYS)

    def test_never_overfills_a_bin_or_a_request(self, week: Week) -> None:
        _, targets, _, allocation = week
        load = [0.0] * len(allocation.bins)
        for target in targets:
            given = allocation.seconds[int(target.obsid)]
            assert sum(given.values()) <= float(target.exptime or 0) + 1e-6
            for k, seconds in given.items():
                load[k] += seconds

        # Nothing here arrived unplanned, so the reserve stays free.
        assert all(
            used <= cap - held + 1e-6
            for used, cap, held in zip(load, allocation.capacity, allocation.reserve)
        )

    def test_allocates_only_where_and_before_a_request_can_be_observed(
        self, week: Week
    ) -> None:
        _, targets, allocator, allocation = week
        for target in targets:
            visible = allocator._windows[int(target.obsid)]
            for k in allocation.seconds[int(target.obsid)]:
                b0, b1 = allocation.bins[k]
                assert any(w0 < b1 and b0 < w1 for w0, w1 in visible)
                if target.deadline is not None:
                    assert b0 < target.deadline

    def test_targets_the_sun_covers_are_allocated_before_it_does(
        self, week: Week
    ) -> None:
        """The Early targets are visible less as the week goes on."""
        _, targets, _, allocation = week
        early = [t for t in targets if 40000 <= int(t.obsid) < 50000]
        allocated = sum(sum(allocation.seconds[int(t.obsid)].values()) for t in early)

        # Nearly all, despite the oversubscribed week and the reserve held
        # back from every day.
        assert allocated >= 0.9 * sum(float(t.exptime or 0) for t in early)

    def test_allocates_from_the_start_time_and_minus_reserved(self, week: Week) -> None:
        _, targets, allocator, _ = week
        later = T0 + 36 * HOUR
        first = targets[0]
        reserved = {int(first.obsid): float(first.exptime or 0)}

        allocation = allocator.allocate(targets, later, reserved)

        assert allocation.capacity[0] == 0.0
        assert allocation.capacity[1] == pytest.approx(0.75 * 12 * HOUR)
        assert int(first.obsid) not in allocation.seconds

    @needs_ortools
    def test_allocates_nothing_before_the_earliest_start(self) -> None:
        config = _config(4)
        target = _target(
            config,
            1,
            105.0,
            10.0,
            minutes=60,
            snapshot=20,
            earliest_start=T0 + 2 * HOUR,
        )
        allocator = LongRangeAllocator(
            config,
            BEGIN,
            BEGIN + timedelta(hours=4),
            bin_length=timedelta(hours=1),
        )

        allocation = allocator.allocate([target], T0)

        assert set(allocation.seconds[1]) <= {2, 3}
        assert sum(allocation.seconds[1].values()) == pytest.approx(60 * MIN)

    def test_requests_are_valued_by_their_merit(self) -> None:
        """Not by the legacy fom, which is not kept in step with merit."""
        config = _config(2)
        target = Pointing(
            config=config, ra=105.0, dec=10.0, obsid=1, merit=30.0, ss_min=300
        )
        target.exptime = 1200
        assert target.fom != target.merit
        allocator = LongRangeAllocator(
            config, BEGIN, BEGIN + timedelta(hours=2), solver="greedy"
        )

        demand = allocator._demand(target, T0, {})

        assert demand is not None
        assert demand.value == pytest.approx(30.0)

    @pytest.mark.parametrize(
        "options",
        [
            {"bin_length": timedelta(0)},
            {"efficiency": 0.0},
            {"reserve": -0.1},
            {"reserve": 1.0},
            {"solver": "lp"},
            {"time_limit": 0.0},
        ],
    )
    def test_rejects_invalid_settings(self, options: dict[str, object]) -> None:
        config = _config(2)

        with pytest.raises(ValueError):
            LongRangeAllocator(
                config,
                BEGIN,
                BEGIN + timedelta(hours=2),
                **{"solver": "greedy", **options},  # type: ignore[arg-type]
            )


def _without_ortools() -> Callable[..., ModuleType]:
    """An import that fails for OR-Tools, as if it were not installed."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name == "ortools" or name.startswith("ortools."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    return fake_import


class TestOptionalOrTools:
    def test_the_milp_solver_needs_ortools(self) -> None:
        config = _config(2)

        with patch.object(builtins, "__import__", _without_ortools()):
            with pytest.raises(ImportError, match=r"coast-sim\[cpsat\].*greedy"):
                LongRangeAllocator(config, BEGIN, BEGIN + timedelta(hours=2))

    def test_the_greedy_solver_does_not(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, minutes=20)

        with patch.object(builtins, "__import__", _without_ortools()):
            allocator = LongRangeAllocator(
                config, BEGIN, BEGIN + timedelta(hours=2), solver="greedy"
            )
            allocation = allocator.allocate([target], T0)

        assert sum(allocation.seconds[1].values()) == pytest.approx(20 * MIN)


@needs_ortools
class TestSolver:
    def test_a_level_without_a_solution_is_allocated_greedily(self) -> None:
        """If the solver finds nothing for a lower tier, it still gets time."""
        from types import SimpleNamespace

        from ortools.math_opt.python import solve
        from ortools.math_opt.python.result import TerminationReason

        categories = [
            ObservationCategory(name="Low", obsid_min=100, obsid_max=200, tier=-1)
        ]
        config = _config(4, categories=categories)
        science = _target(config, 1, 105.0, 10.0, minutes=60, snapshot=20)
        filler = _target(config, 100, 110.0, 10.0, minutes=40, snapshot=20)
        allocator = LongRangeAllocator(
            config, BEGIN, BEGIN + timedelta(hours=4), bin_length=timedelta(hours=1)
        )
        real = solve.solve
        calls = []

        def second_finds_nothing(*args: object, **kwargs: object) -> object:
            calls.append(1)
            if len(calls) == 2:
                return SimpleNamespace(
                    termination=SimpleNamespace(
                        reason=TerminationReason.NO_SOLUTION_FOUND
                    ),
                    has_primal_feasible_solution=lambda: False,
                )
            return real(*args, **kwargs)  # type: ignore[arg-type]

        with patch.object(solve, "solve", second_finds_nothing):
            allocation = allocator.allocate([science, filler], T0)

        assert allocator.solver_status == "OPTIMAL,NO_SOLUTION_FOUND"
        assert sum(allocation.seconds[1].values()) == pytest.approx(60 * MIN)
        # Greedily, into the room the science left: at least a snapshot.
        assert sum(allocation.seconds[100].values()) >= 20 * MIN

    def test_every_level_is_solved_from_the_level_before(self) -> None:
        categories = [
            ObservationCategory(name="Low", obsid_min=100, obsid_max=200, tier=-1)
        ]
        config = _config(4, categories=categories)
        targets = [
            _target(config, 1, 105.0, 10.0, minutes=60, snapshot=20),
            _target(config, 100, 110.0, 10.0, minutes=60, snapshot=20),
        ]
        allocator = LongRangeAllocator(
            config, BEGIN, BEGIN + timedelta(hours=4), bin_length=timedelta(hours=1)
        )

        allocation = allocator.allocate(targets, T0)

        assert allocator.solver_status == "OPTIMAL"
        assert sum(allocation.seconds[100].values()) == pytest.approx(60 * MIN)

    def test_solves_the_week_to_optimality_by_default(self, week: Week) -> None:
        _, _, allocator, _ = week

        assert allocator.solver == "milp"
        assert allocator.solver_status == "OPTIMAL"

    def test_does_at_least_as_well_as_greedy(self, week: Week) -> None:
        _, targets, _, milp = week
        _, _, _, greedy = _fresh(week, solver="greedy")
        merit = {int(t.obsid): float(t.merit) for t in targets}

        def worth(allocation: Allocation) -> float:
            return sum(
                merit[obsid] * sum(per_bin.values())
                for obsid, per_bin in allocation.seconds.items()
            )

        assert worth(milp) >= worth(greedy)

    def test_never_allocates_less_than_a_snapshot(self, week: Week) -> None:
        _, targets, _, allocation = week
        for target in targets:
            for seconds in allocation.seconds[int(target.obsid)].values():
                assert seconds >= min(float(target.ss_min), float(target.exptime or 0))

    def test_allocating_again_keeps_the_allocation(self, week: Week) -> None:
        _, targets, allocator, first = _fresh(week)

        again = allocator.allocate(targets, T0)

        assert again.seconds == first.seconds


@needs_ortools
class TestReserve:
    """Capacity held back so unplanned work fits without displacing the plan."""

    @staticmethod
    def _too(config: MissionConfig, seconds: int) -> Pointing:
        too = Pointing(
            config=config,
            ra=105.0,
            dec=10.0,
            obsid=1_000_001,
            name="ToO",
            merit=500.0,
            ss_min=300,
            ss_max=seconds,
        )
        too.exptime = seconds
        return too

    def test_a_too_within_the_reserve_leaves_the_plan_alone(self, week: Week) -> None:
        config, targets, allocator, before = _fresh(week)
        assert HOUR < before.reserve[0]
        too = self._too(config, HOUR)

        after = allocator.allocate([*targets, too], T0, unplanned={too.obsid})

        assert sum(after.seconds[too.obsid].values()) == pytest.approx(HOUR)
        for target in targets:
            obsid = int(target.obsid)
            assert after.seconds[obsid] == before.seconds[obsid]

    def test_a_too_larger_than_the_reserve_takes_regular_capacity(
        self, week: Week
    ) -> None:
        config, targets, allocator, before = _fresh(week)
        too = self._too(config, int(before.reserve[0] + HOUR))

        after = allocator.allocate([*targets, too], T0, unplanned={too.obsid})

        assert sum(after.seconds[too.obsid].values()) == pytest.approx(
            before.reserve[0] + HOUR
        )

    def test_without_a_reserve_a_too_displaces_planned_requests(
        self, week: Week
    ) -> None:
        config, targets, allocator, before = _fresh(week, reserve=0.0)
        too = self._too(config, 2 * HOUR)

        after = allocator.allocate([*targets, too], T0, unplanned={too.obsid})

        assert any(
            after.seconds[int(t.obsid)] != before.seconds[int(t.obsid)] for t in targets
        )

    def test_unplanned_requests_go_first_whatever_their_merit(self, week: Week) -> None:
        config, targets, allocator, _ = _fresh(week)
        too = self._too(config, 2 * HOUR)
        too.merit = 1.0

        after = allocator.allocate([*targets, too], T0, unplanned={too.obsid})

        assert sum(after.seconds[too.obsid].values()) == pytest.approx(2 * HOUR)


def _queue(ditl: QueueDITL) -> TargetQueue:
    assert isinstance(ditl.queue, TargetQueue)
    return ditl.queue


class TestDispatchPreference:
    @pytest.mark.parametrize(
        ("strictness", "chosen"), [("strict", 100), ("tier", 1), ("weighted", 1)]
    )
    def test_dispatch_follows_the_strictness(
        self, strictness: str, chosen: int
    ) -> None:
        """Filler (tier -1) allocated now; tier-0 science that is not."""
        categories = [
            ObservationCategory(name="Filler", obsid_min=100, obsid_max=200, tier=-1)
        ]
        config = _config(2, categories=categories)
        ditl = QueueDITL(config=config, begin=BEGIN, end=BEGIN + timedelta(hours=2))
        queue_targets(
            ditl,
            [
                _target(config, 100, 105.0, 10.0, merit=10),
                _target(config, 1, 110.0, 10.0, merit=90),
            ],
        )
        queue = _queue(ditl)
        queue.prefer = lambda obsid: obsid == 100
        queue.allocation_strictness = strictness  # type: ignore[assignment]

        target = queue.get(105.0, 10.0, T0 + 10 * MIN)

        assert target is not None and int(target.obsid) == chosen

    def test_dispatch_weighted_bonus_is_a_merit_term(self) -> None:
        config = _config(2)
        ditl = QueueDITL(config=config, begin=BEGIN, end=BEGIN + timedelta(hours=2))
        queue_targets(
            ditl,
            [
                _target(config, 1, 105.0, 10.0, merit=40),
                _target(config, 2, 110.0, 10.0, merit=50),
            ],
        )
        queue = _queue(ditl)
        queue.prefer = lambda obsid: obsid == 1
        queue.allocation_strictness = "weighted"
        queue.allocation_bonus = 0.5

        target = queue.get(105.0, 10.0, T0 + 10 * MIN)

        assert target is not None and int(target.obsid) == 1
        assert target.merit_breakdown is not None
        assert target.merit_breakdown.allocation == pytest.approx(20.0)
        assert target.merit == pytest.approx(60.0)

    def test_dispatch_prefers_within_a_tier(self) -> None:
        config = _config(2)
        ditl = QueueDITL(config=config, begin=BEGIN, end=BEGIN + timedelta(hours=2))
        queue_targets(
            ditl,
            [
                _target(config, 1, 105.0, 10.0, merit=10),
                _target(config, 2, 110.0, 10.0, merit=90),
            ],
        )
        queue = _queue(ditl)
        queue.prefer = lambda obsid: obsid == 1

        chosen = queue.get(105.0, 10.0, T0 + 10 * MIN)

        assert chosen is not None and int(chosen.obsid) == 1


def _science(plan: Plan) -> list[tuple[int, float]]:
    return [
        (int(e.obsid), float(e.collection_begin))
        for e in plan
        if e.obstype != ObsType.GSP and e.collection_begin is not None
    ]


class TestAllocatedTime:
    """Schedulers plan a request's allocated seconds in the time they are allocated to."""

    SECOND_HOUR = AllocatedTime(T0 + HOUR, T0 + 2 * HOUR, 20 * MIN)

    @pytest.mark.parametrize(
        ("scheduler", "options"),
        [
            (PriorityScheduler, {}),
            (LocalSearchScheduler, {"max_iterations": 200, "seed": 1}),
            pytest.param(
                CpSatScheduler,
                {"solver_time_limit": 5.0, "workers": 1, "seed": 1},
                marks=needs_ortools,
            ),
        ],
    )
    def test_allocated_seconds_are_planned_in_their_time(
        self, scheduler: type[PriorityScheduler], options: dict[str, object]
    ) -> None:
        """Allocated to the second hour, a low-merit request waits for it."""
        config = _config(3)
        later = _target(config, 1, 105.0, 10.0, merit=10)
        now = _target(config, 2, 110.0, 10.0, merit=90)

        plan = scheduler(
            config,
            [later, now],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={1: [self.SECOND_HOUR], 2: []},
            **options,  # type: ignore[arg-type]
        ).schedule()

        science = _science(plan)
        assert science[0][0] == 2
        (begin,) = (begin for obsid, begin in science if obsid == 1)
        assert T0 + HOUR <= begin < T0 + 2 * HOUR

    def test_allocated_time_outranks_unallocated_work_in_higher_tiers(self) -> None:
        """The allocation has weighed the tiers; working ahead uses what is left."""
        assert [obsid for obsid, _ in self._filler_and_ahead("strict")] == [100, 1]

    @staticmethod
    def _filler_and_ahead(
        strictness: str, bonus: float = 0.5
    ) -> list[tuple[int, float]]:
        """Filler (tier -1) allocated the first hour; tier-0 work to do ahead."""
        categories = [
            ObservationCategory(name="Filler", obsid_min=100, obsid_max=200, tier=-1)
        ]
        config = _config(3, categories=categories)
        filler = _target(config, 100, 105.0, 10.0, merit=10)
        ahead = _target(config, 1, 110.0, 10.0, merit=90)
        plan = PriorityScheduler(
            config,
            [filler, ahead],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={100: [AllocatedTime(T0, T0 + HOUR, 20 * MIN)], 1: []},
            allocation_strictness=strictness,  # type: ignore[arg-type]
            allocation_bonus=bonus,
        ).schedule()
        return _science(plan)

    def test_tier_strictness_lets_a_higher_tier_take_allocated_time(self) -> None:
        assert [obsid for obsid, _ in self._filler_and_ahead("tier")] == [1, 100]

    @staticmethod
    def _low_allocated_high_ahead(bonus: float) -> list[tuple[int, float]]:
        """In one tier: merit 40 allocated the first hour, merit 50 unallocated."""
        config = _config(3)
        allocated = _target(config, 1, 105.0, 10.0, merit=40)
        ahead = _target(config, 2, 110.0, 10.0, merit=50)
        plan = PriorityScheduler(
            config,
            [allocated, ahead],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={1: [AllocatedTime(T0, T0 + HOUR, 20 * MIN)], 2: []},
            allocation_strictness="weighted",
            allocation_bonus=bonus,
        ).schedule()
        return _science(plan)

    def test_weighted_strictness_trades_the_bonus_against_merit(self) -> None:
        # 40 * 1.1 < 50: merit wins; 40 * 1.5 > 50: the allocation wins.
        assert [o for o, _ in self._low_allocated_high_ahead(0.1)] == [2, 1]
        assert [o for o, _ in self._low_allocated_high_ahead(0.5)] == [1, 2]

    def test_weighted_local_search_scores_the_bonus(self) -> None:
        config = _config(3)
        target = _target(config, 1, 105.0, 10.0, merit=10)
        scheduler = LocalSearchScheduler(
            config,
            [target],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={1: [self.SECOND_HOUR]},
            allocation_strictness="weighted",
            allocation_bonus=0.5,
            max_iterations=0,
        )

        scheduler.schedule()

        # One tier; the 20 allocated minutes are worth 1.5 times their merit.
        assert scheduler.score == (pytest.approx(1.5 * 10 * 20 * MIN),)

    @pytest.mark.parametrize(
        "options",
        [{"allocation_strictness": "loose"}, {"allocation_bonus": -0.1}],
    )
    def test_rejects_invalid_strictness(self, options: dict[str, object]) -> None:
        config = _config(2)
        with pytest.raises(ValueError):
            PriorityScheduler(
                config,
                [],
                BEGIN,
                BEGIN + timedelta(hours=2),
                **options,  # type: ignore[arg-type]
            )

    def test_exposure_beyond_the_allocation_is_planned_like_the_rest(self) -> None:
        config = _config(3)
        later = _target(config, 1, 105.0, 10.0, merit=10, minutes=40, snapshot=20)

        plan = PriorityScheduler(
            config,
            [later],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={1: [self.SECOND_HOUR]},
        ).schedule()

        begins = sorted(begin for _, begin in _science(plan))
        # The unallocated 20 minutes work ahead; the allocated 20 wait.
        assert len(begins) == 2
        assert begins[0] < T0 + HOUR <= begins[1] < T0 + 2 * HOUR

    def test_allocated_seconds_too_few_for_a_snapshot_are_not_lost(self) -> None:
        """Ten allocated minutes cannot hold a twenty-minute snapshot."""
        config = _config(3)
        target = _target(config, 1, 105.0, 10.0, minutes=20)
        too_short = AllocatedTime(T0 + HOUR, T0 + HOUR + 10 * MIN, 10 * MIN)

        scheduler = PriorityScheduler(
            config,
            [target],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={1: [too_short]},
        )
        plan = scheduler.schedule()

        assert len(_science(plan)) == 1
        assert scheduler.unplaced == []

    def test_allocated_seconds_that_do_not_fit_their_span_are_not_lost(self) -> None:
        """Allocated to a span the target is hidden throughout, a snapshot's
        worth of seconds is planned with the rest of its exposure."""
        hours = 6
        config = _config(hours)
        # Hidden for about 40 minutes of every orbit.
        target = _target(config, 1, *CONSTRAINED[0], minutes=20)
        windows = SchedulingContext(
            config, BEGIN, BEGIN + timedelta(hours=hours)
        ).visibility_windows(target)
        hidden = next(
            (w1, w2)
            for (_, w1), (w2, _) in zip(windows, windows[1:])
            if w2 - w1 >= 30 * MIN
        )

        scheduler = PriorityScheduler(
            config,
            [target],
            BEGIN,
            BEGIN + timedelta(hours=hours),
            allocated={1: [AllocatedTime(hidden[0], hidden[1], 20 * MIN)]},
        )
        plan = scheduler.schedule()

        assert len(_science(plan)) == 1
        assert scheduler.unplaced == []

    def test_local_search_scores_allocated_seconds_a_tier_up(self) -> None:
        config = _config(3)
        later = _target(config, 1, 105.0, 10.0, merit=10)
        scheduler = LocalSearchScheduler(
            config,
            [later],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={1: [self.SECOND_HOUR]},
            max_iterations=0,
        )

        scheduler.schedule()

        # Tiers 1 (allocated) and 0; all 20 minutes count as allocated.
        assert scheduler.score[0] == pytest.approx(10 * 20 * MIN)
        assert scheduler.score[1] == 0.0

    @needs_ortools
    def test_cp_sat_hint_reproduces_the_plan_with_allocated_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Held to its hint, the solver returns the priority-first plan, with
        the allocated seconds counted as decoding counts them."""
        from ortools.sat.python import cp_model

        solve = cp_model.CpSolver.solve

        def hinted_only(solver: cp_model.CpSolver, model: cp_model.CpModel) -> object:
            solver.parameters.fix_variables_to_their_hinted_value = True
            return solve(solver, model)

        monkeypatch.setattr(cp_model.CpSolver, "solve", hinted_only)
        config = _config(3)
        targets = [
            _target(config, 1 + i, ra, dec, merit=90 - 10 * i, minutes=40, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH)
        ]
        scheduler = CpSatScheduler(
            config,
            targets,
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={
                3: [AllocatedTime(T0, T0 + HOUR, 20 * MIN)],
                5: [AllocatedTime(T0 + HOUR, T0 + 2 * HOUR, 40 * MIN)],
            },
            solver_time_limit=5.0,
            workers=1,
            seed=1,
        )

        scheduler.schedule()

        assert set(scheduler.solver_statuses) == {"OPTIMAL"}
        assert all(scheduler.solver_chunks_used)
        assert scheduler.solver_score == scheduler.start_score


def _targets(config: MissionConfig) -> list[Pointing]:
    return [
        _target(config, 100 + i, ra, dec, merit=90 - 10 * i, minutes=40)
        for i, (ra, dec) in enumerate(PATCH)
    ]


class TestSimulations:
    HOURS = 4
    BIN = timedelta(hours=1)

    def _allocator(self, config: MissionConfig) -> LongRangeAllocator:
        return LongRangeAllocator(
            config,
            BEGIN,
            BEGIN + timedelta(hours=self.HOURS),
            bin_length=self.BIN,
            solver="greedy",
        )

    def test_rolling_replans_with_an_allocation(self) -> None:
        config = _config(self.HOURS)
        ditl = RollingHorizonDITL(
            config,
            _targets(config),
            begin=BEGIN,
            end=BEGIN + timedelta(hours=self.HOURS),
            horizon=timedelta(hours=2),
            replan_interval=timedelta(hours=1),
            allocator=self._allocator(config),
        )
        ditl.step_size = 60

        assert ditl.calc()

        assert ditl.allocation is not None
        assert ditl.validate_plan_matches_execution() == []

    def test_rolling_follows_a_fixed_allocation(self) -> None:
        """Each replan counts what the bin has collected against its seconds."""
        config = _config(self.HOURS)
        targets = _targets(config)
        allocation = self._allocator(config).allocate(targets, T0)
        ditl = RollingHorizonDITL(
            config,
            targets,
            begin=BEGIN,
            end=BEGIN + timedelta(hours=self.HOURS),
            horizon=timedelta(hours=2),
            replan_interval=timedelta(minutes=30),
            allocation=allocation,
        )
        ditl.step_size = 60
        given: list[dict[int, list[AllocatedTime]]] = []
        scheduler = ditl.scheduler

        def recording(*args: object, **kwargs: object) -> PriorityScheduler:
            given.append(kwargs["allocated"])  # type: ignore[arg-type]
            return scheduler(*args, **kwargs)  # type: ignore[arg-type]

        ditl.scheduler = recording  # type: ignore[assignment]

        assert ditl.calc()

        assert ditl.allocation is allocation
        assert ditl.validate_plan_matches_execution() == []
        # Within the first bin, a later replan has less of it left to plan.
        first, later = given[0], given[1]
        first_bin = sum(
            span.seconds
            for spans in first.values()
            for span in spans
            if span.begin < T0 + HOUR
        )
        later_bin = sum(
            span.seconds
            for spans in later.values()
            for span in spans
            if span.begin < T0 + HOUR
        )
        assert later_bin < first_bin

    def test_rolling_takes_an_allocator_or_an_allocation(self) -> None:
        config = _config(self.HOURS)
        targets = _targets(config)
        allocator = self._allocator(config)
        with pytest.raises(ValueError, match="allocator or allocation"):
            RollingHorizonDITL(
                config,
                targets,
                begin=BEGIN,
                end=BEGIN + timedelta(hours=self.HOURS),
                allocator=allocator,
                allocation=allocator.allocate(targets, T0),
            )

    def test_dispatch_allocates_again_each_bin(self) -> None:
        config = _config(self.HOURS)
        allocator = self._allocator(config)
        calls: list[float] = []
        allocate = allocator.allocate

        def counted(*args: object, **kwargs: object) -> Allocation:
            calls.append(args[1])  # type: ignore[arg-type]
            return allocate(*args, **kwargs)  # type: ignore[arg-type]

        allocator.allocate = counted  # type: ignore[method-assign]
        ditl = QueueDITL(
            config=config,
            begin=BEGIN,
            end=BEGIN + timedelta(hours=self.HOURS),
            allocator=allocator,
        )
        queue_targets(ditl, _targets(config))

        ditl.calc()

        assert calls == [T0 + k * HOUR for k in range(self.HOURS)]
        assert _queue(ditl).prefer is not None

    def test_dispatch_needs_a_target_queue(self) -> None:
        config = _config(self.HOURS)

        class Policy:
            targets: list[Pointing] = []
            log = None

        with pytest.raises(TypeError, match="TargetQueue"):
            QueueDITL(
                config=config,
                queue=Policy(),  # type: ignore[arg-type]
                allocator=self._allocator(config),
            )


class TestConfiguration:
    def test_create_ditl_gives_dispatch_and_rolling_an_allocator(self) -> None:
        for mode in (SchedulerMode.DISPATCH, SchedulerMode.ROLLING):
            config = _config(2)
            config.scheduler = SchedulerConfig(
                mode=mode,
                allocation=AllocationSettings(
                    bin_seconds=HOUR,
                    efficiency=0.5,
                    reserve=0.2,
                    solver="greedy",
                    time_limit_seconds=3.0,
                ),
            )

            ditl = create_ditl(config, _targets(config))

            assert isinstance(ditl, QueueDITL | RollingHorizonDITL)
            allocator = ditl.allocator
            assert allocator is not None
            assert allocator.bin_length == timedelta(hours=1)
            assert allocator.efficiency == 0.5
            assert allocator.reserve == 0.2
            assert allocator.solver == "greedy"
            assert allocator.time_limit == 3.0

    def test_create_ditl_passes_the_strictness(self) -> None:
        for mode in (SchedulerMode.DISPATCH, SchedulerMode.ROLLING):
            config = _config(2)
            config.scheduler = SchedulerConfig(
                mode=mode,
                allocation=AllocationSettings(
                    solver="greedy", strictness="weighted", bonus=0.25
                ),
            )

            ditl = create_ditl(config, _targets(config))

            if isinstance(ditl, QueueDITL):
                assert _queue(ditl).allocation_strictness == "weighted"
                assert _queue(ditl).allocation_bonus == 0.25
            else:
                assert isinstance(ditl, RollingHorizonDITL)
                assert ditl.allocation_strictness == "weighted"
                assert ditl.allocation_bonus == 0.25

    def test_strictness_is_strict_by_default(self) -> None:
        settings = AllocationSettings()

        assert settings.strictness == "strict"
        with pytest.raises(ValidationError):
            AllocationSettings(strictness="loose")  # type: ignore[arg-type]

    def test_planned_mode_rejects_an_allocation(self) -> None:
        with pytest.raises(ValidationError, match="allocation"):
            SchedulerConfig(mode=SchedulerMode.PLANNED, allocation=AllocationSettings())

    def test_no_allocation_by_default(self) -> None:
        config = _config(2)

        ditl = create_ditl(config, _targets(config))

        assert isinstance(ditl, QueueDITL)
        assert ditl.allocator is None

    def test_settings_round_trip_through_yaml(self, tmp_path: object) -> None:
        from pathlib import Path

        config = _config(2)
        config.scheduler = SchedulerConfig(
            mode=SchedulerMode.ROLLING,
            allocation=AllocationSettings(
                bin_seconds=2 * HOUR, solver="greedy", strictness="tier", bonus=0.2
            ),
        )
        path = Path(str(tmp_path)) / "config.yaml"

        config.to_yaml_file(path)
        loaded = MissionConfig.from_yaml_file(path)

        assert loaded.scheduler.allocation == config.scheduler.allocation
