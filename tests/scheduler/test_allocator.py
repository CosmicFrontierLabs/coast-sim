"""Long-range allocation and how it steers planners and simulations."""

import math
from datetime import timedelta
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from conops import QueueDITL, RollingHorizonDITL, create_ditl
from conops.benchmark.scenarios import long_range
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
    CpSatPlanner,
    LocalSearchPlanner,
    LongRangeAllocator,
    PriorityPlanner,
)
from conops.targets import Pointing

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

TLE = "examples/example.tle"


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

    def test_preferred_picks_from_the_obsids_given(self) -> None:
        assert self.ALLOCATION.preferred([1, 2, 3, 99], T0, T0 + HOUR) == {1, 99}

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


@pytest.fixture(scope="module")
def week() -> tuple[MissionConfig, list[Pointing], LongRangeAllocator, Allocation]:
    scenario = long_range(TLE)
    config = scenario.make_config()
    targets = scenario.make_targets(config)
    allocator = LongRangeAllocator(config, scenario.begin, scenario.end)
    return config, targets, allocator, allocator.allocate(targets, T0)


class TestLongRangeAllocator:
    def test_bins_and_capacity(self, week: tuple[object, ...]) -> None:
        _, _, allocator, allocation = week
        assert isinstance(allocation, Allocation)

        assert len(allocation.bins) == 7
        assert allocation.capacity == pytest.approx([0.75 * 24 * HOUR] * 7)
        assert allocation.reserve == pytest.approx([0.1 * 0.75 * 24 * HOUR] * 7)

    def test_never_overfills_a_bin_or_a_request(
        self, week: tuple[MissionConfig, list[Pointing], object, Allocation]
    ) -> None:
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
        self,
        week: tuple[MissionConfig, list[Pointing], LongRangeAllocator, Allocation],
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
        self, week: tuple[MissionConfig, list[Pointing], object, Allocation]
    ) -> None:
        """The scenario's Early targets are visible less as the week goes on."""
        _, targets, _, allocation = week
        early = [t for t in targets if 40000 <= int(t.obsid) < 50000]
        allocated = sum(sum(allocation.seconds[int(t.obsid)].values()) for t in early)

        # Nearly all, despite the oversubscribed week and the reserve held
        # back from every day.
        assert allocated >= 0.9 * sum(float(t.exptime or 0) for t in early)

    def test_allocates_from_the_start_time_and_minus_reserved(
        self,
        week: tuple[MissionConfig, list[Pointing], LongRangeAllocator, Allocation],
    ) -> None:
        _, targets, allocator, _ = week
        later = T0 + 36 * HOUR
        first = targets[0]
        reserved = {int(first.obsid): float(first.exptime or 0)}

        allocation = allocator.allocate(targets, later, reserved)

        assert allocation.capacity[0] == 0.0
        assert allocation.capacity[1] == pytest.approx(0.75 * 12 * HOUR)
        assert int(first.obsid) not in allocation.seconds

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
            config, BEGIN, BEGIN + timedelta(hours=4), bin_length=timedelta(hours=1)
        )

        allocation = allocator.allocate([target], T0)

        assert set(allocation.seconds[1]) <= {2, 3}
        assert sum(allocation.seconds[1].values()) == pytest.approx(60 * MIN)

    def test_dispatch_queue_keeps_the_earliest_start(self) -> None:
        config = _config(2)
        ditl = QueueDITL(config=config, begin=BEGIN, end=BEGIN + timedelta(hours=2))
        queue_targets(ditl, [_target(config, 1, 105.0, 10.0, earliest_start=T0 + HOUR)])

        assert ditl.queue.get(105.0, 10.0, T0 + 10 * MIN) is None
        assert ditl.queue.targets[0].earliest_start == T0 + HOUR

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
                **options,  # type: ignore[arg-type]
            )


def _fresh(
    week: tuple[MissionConfig, list[Pointing], object, object], **options: object
) -> tuple[MissionConfig, list[Pointing], LongRangeAllocator, Allocation]:
    """A new allocator for the week and its first allocation.

    Allocations build on the previous one, so tests that allocate twice need
    their own allocator.
    """
    config, targets, _, _ = week
    scenario = long_range(TLE)
    allocator = LongRangeAllocator(
        config,
        scenario.begin,
        scenario.end,
        **options,  # type: ignore[arg-type]
    )
    return config, targets, allocator, allocator.allocate(targets, T0)


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

    def test_solves_the_week_to_optimality_by_default(
        self, week: tuple[MissionConfig, list[Pointing], LongRangeAllocator, object]
    ) -> None:
        _, _, allocator, _ = week

        assert allocator.solver == "milp"
        assert allocator.solver_status == "OPTIMAL"

    def test_does_at_least_as_well_as_greedy(
        self, week: tuple[MissionConfig, list[Pointing], object, Allocation]
    ) -> None:
        _, targets, _, milp = week
        _, _, _, greedy = _fresh(week, solver="greedy")
        merit = {int(t.obsid): float(t.fom) for t in targets}

        def worth(allocation: Allocation) -> float:
            return sum(
                merit[obsid] * sum(per_bin.values())
                for obsid, per_bin in allocation.seconds.items()
            )

        assert worth(milp) >= worth(greedy)

    def test_never_allocates_less_than_a_snapshot(
        self, week: tuple[MissionConfig, list[Pointing], object, Allocation]
    ) -> None:
        _, targets, _, allocation = week
        for target in targets:
            for seconds in allocation.seconds[int(target.obsid)].values():
                assert seconds >= min(float(target.ss_min), float(target.exptime or 0))

    def test_allocating_again_keeps_the_allocation(
        self, week: tuple[MissionConfig, list[Pointing], object, object]
    ) -> None:
        _, targets, allocator, first = _fresh(week)

        again = allocator.allocate(targets, T0)

        assert again.seconds == first.seconds


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
            fom=500.0,
            ss_min=300,
            ss_max=seconds,
        )
        too.exptime = seconds
        return too

    def test_a_too_within_the_reserve_leaves_the_plan_alone(
        self,
        week: tuple[MissionConfig, list[Pointing], LongRangeAllocator, Allocation],
    ) -> None:
        config, targets, allocator, before = _fresh(week)
        assert HOUR < before.reserve[0]
        too = self._too(config, HOUR)

        after = allocator.allocate([*targets, too], T0, unplanned={too.obsid})

        assert sum(after.seconds[too.obsid].values()) == pytest.approx(HOUR)
        for target in targets:
            obsid = int(target.obsid)
            assert after.seconds[obsid] == before.seconds[obsid]

    def test_a_too_larger_than_the_reserve_takes_regular_capacity(
        self,
        week: tuple[MissionConfig, list[Pointing], LongRangeAllocator, Allocation],
    ) -> None:
        config, targets, allocator, before = _fresh(week)
        too = self._too(config, int(before.reserve[0] + HOUR))

        after = allocator.allocate([*targets, too], T0, unplanned={too.obsid})

        assert sum(after.seconds[too.obsid].values()) == pytest.approx(
            before.reserve[0] + HOUR
        )

    def test_without_a_reserve_a_too_displaces_planned_requests(
        self, week: tuple[MissionConfig, list[Pointing], object, object]
    ) -> None:
        config, targets, allocator, before = _fresh(week, reserve=0.0)
        too = self._too(config, 2 * HOUR)

        after = allocator.allocate([*targets, too], T0, unplanned={too.obsid})

        assert any(
            after.seconds[int(t.obsid)] != before.seconds[int(t.obsid)] for t in targets
        )

    def test_unplanned_requests_go_first_whatever_their_merit(
        self,
        week: tuple[MissionConfig, list[Pointing], LongRangeAllocator, Allocation],
    ) -> None:
        config, targets, allocator, _ = _fresh(week)
        too = self._too(config, 2 * HOUR)
        too.merit = too.fom = 1.0

        after = allocator.allocate([*targets, too], T0, unplanned={too.obsid})

        assert sum(after.seconds[too.obsid].values()) == pytest.approx(2 * HOUR)


class TestPreferred:
    def test_preferred_requests_go_first_within_their_tier(self) -> None:
        config = _config(2)
        low = _target(config, 1, 105.0, 10.0, merit=10)
        high = _target(config, 2, 110.0, 10.0, merit=90)

        plan = PriorityPlanner(
            config, [low, high], BEGIN, BEGIN + timedelta(hours=2), preferred={1}
        ).schedule()

        assert [e.obsid for e in plan if e.obstype != ObsType.GSP] == [1, 2]

    def test_tiers_still_come_first(self) -> None:
        categories = [ObservationCategory(name="Cal", obsid_min=2, obsid_max=3, tier=1)]
        config = _config(2, categories=categories)
        science = _target(config, 1, 105.0, 10.0, merit=90)
        calibration = _target(config, 2, 110.0, 10.0, merit=1)

        plan = PriorityPlanner(
            config,
            [science, calibration],
            BEGIN,
            BEGIN + timedelta(hours=2),
            preferred={1},
        ).schedule()

        assert [e.obsid for e in plan if e.obstype != ObsType.GSP] == [2, 1]

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
        ditl.queue.prefer = lambda obsid: obsid == 100  # type: ignore[attr-defined]
        ditl.queue.allocation_strictness = strictness  # type: ignore[attr-defined]

        target = ditl.queue.get(105.0, 10.0, T0 + 10 * MIN)

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
        ditl.queue.prefer = lambda obsid: obsid == 1  # type: ignore[attr-defined]
        ditl.queue.allocation_strictness = "weighted"  # type: ignore[attr-defined]
        ditl.queue.allocation_bonus = 0.5  # type: ignore[attr-defined]

        target = ditl.queue.get(105.0, 10.0, T0 + 10 * MIN)

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
        ditl.queue.prefer = lambda obsid: obsid == 1  # type: ignore[attr-defined]

        chosen = ditl.queue.get(105.0, 10.0, T0 + 10 * MIN)

        assert chosen is not None and int(chosen.obsid) == 1


class TestAllocatedTime:
    """Planners plan a request's allocated seconds in the time they are allocated to."""

    SECOND_HOUR = AllocatedTime(T0 + HOUR, T0 + 2 * HOUR, 20 * MIN)

    @staticmethod
    def _science(plan: object) -> list[tuple[int, float]]:
        return [
            (int(e.obsid), float(e.collection_begin))
            for e in plan  # type: ignore[attr-defined]
            if e.obstype != ObsType.GSP and e.collection_begin is not None
        ]

    @pytest.mark.parametrize(
        ("planner", "options"),
        [
            (PriorityPlanner, {}),
            (LocalSearchPlanner, {"max_iterations": 200, "seed": 1}),
            (CpSatPlanner, {"solver_time_limit": 5.0, "workers": 1, "seed": 1}),
        ],
    )
    def test_allocated_seconds_are_planned_in_their_time(
        self, planner: type[PriorityPlanner], options: dict[str, object]
    ) -> None:
        """Allocated to the second hour, a low-merit request waits for it."""
        config = _config(3)
        later = _target(config, 1, 105.0, 10.0, merit=10)
        now = _target(config, 2, 110.0, 10.0, merit=90)

        plan = planner(
            config,
            [later, now],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={1: [self.SECOND_HOUR], 2: []},
            **options,  # type: ignore[arg-type]
        ).schedule()

        science = self._science(plan)
        assert science[0][0] == 2
        (begin,) = (begin for obsid, begin in science if obsid == 1)
        assert T0 + HOUR <= begin < T0 + 2 * HOUR

    def test_allocated_time_outranks_unallocated_work_in_higher_tiers(self) -> None:
        """The allocation has weighed the tiers; working ahead uses what is left."""
        categories = [
            ObservationCategory(name="Filler", obsid_min=100, obsid_max=200, tier=-1)
        ]
        config = _config(3, categories=categories)
        filler = _target(config, 100, 105.0, 10.0, merit=10)
        ahead = _target(config, 1, 110.0, 10.0, merit=90)
        first_hour = AllocatedTime(T0, T0 + HOUR, 20 * MIN)

        plan = PriorityPlanner(
            config,
            [filler, ahead],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={100: [first_hour], 1: []},
        ).schedule()

        assert [obsid for obsid, _ in self._science(plan)] == [100, 1]

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
        plan = PriorityPlanner(
            config,
            [filler, ahead],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={100: [AllocatedTime(T0, T0 + HOUR, 20 * MIN)], 1: []},
            allocation_strictness=strictness,  # type: ignore[arg-type]
            allocation_bonus=bonus,
        ).schedule()
        return TestAllocatedTime._science(plan)

    def test_tier_strictness_lets_a_higher_tier_take_allocated_time(self) -> None:
        assert [obsid for obsid, _ in self._filler_and_ahead("tier")] == [1, 100]

    @staticmethod
    def _low_allocated_high_ahead(bonus: float) -> list[tuple[int, float]]:
        """In one tier: merit 40 allocated the first hour, merit 50 unallocated."""
        config = _config(3)
        allocated = _target(config, 1, 105.0, 10.0, merit=40)
        ahead = _target(config, 2, 110.0, 10.0, merit=50)
        plan = PriorityPlanner(
            config,
            [allocated, ahead],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={1: [AllocatedTime(T0, T0 + HOUR, 20 * MIN)], 2: []},
            allocation_strictness="weighted",
            allocation_bonus=bonus,
        ).schedule()
        return TestAllocatedTime._science(plan)

    def test_weighted_strictness_trades_the_bonus_against_merit(self) -> None:
        # 40 * 1.1 < 50: merit wins; 40 * 1.5 > 50: the allocation wins.
        assert [o for o, _ in self._low_allocated_high_ahead(0.1)] == [2, 1]
        assert [o for o, _ in self._low_allocated_high_ahead(0.5)] == [1, 2]

    def test_weighted_local_search_scores_the_bonus(self) -> None:
        config = _config(3)
        target = _target(config, 1, 105.0, 10.0, merit=10)
        planner = LocalSearchPlanner(
            config,
            [target],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={1: [self.SECOND_HOUR]},
            allocation_strictness="weighted",
            allocation_bonus=0.5,
            max_iterations=0,
        )

        planner.schedule()

        # One tier; the 20 allocated minutes are worth 1.5 times their merit.
        assert planner.score == (pytest.approx(1.5 * 10 * 20 * MIN),)

    @pytest.mark.parametrize(
        "options",
        [{"allocation_strictness": "loose"}, {"allocation_bonus": -0.1}],
    )
    def test_rejects_invalid_strictness(self, options: dict[str, object]) -> None:
        config = _config(2)
        with pytest.raises(ValueError):
            PriorityPlanner(
                config,
                [],
                BEGIN,
                BEGIN + timedelta(hours=2),
                **options,  # type: ignore[arg-type]
            )

    def test_preferred_alone_would_take_the_first_slot(self) -> None:
        """Preferring across the whole horizon ignores when it was allocated."""
        config = _config(3)
        later = _target(config, 1, 105.0, 10.0, merit=10)
        now = _target(config, 2, 110.0, 10.0, merit=90)

        plan = PriorityPlanner(
            config, [later, now], BEGIN, BEGIN + timedelta(hours=3), preferred={1}
        ).schedule()

        assert self._science(plan)[0][0] == 1

    def test_exposure_beyond_the_allocation_is_planned_like_the_rest(self) -> None:
        config = _config(3)
        later = _target(config, 1, 105.0, 10.0, merit=10, minutes=40, snapshot=20)

        plan = PriorityPlanner(
            config,
            [later],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={1: [self.SECOND_HOUR]},
        ).schedule()

        begins = sorted(begin for _, begin in self._science(plan))
        # The unallocated 20 minutes work ahead; the allocated 20 wait.
        assert len(begins) == 2
        assert begins[0] < T0 + HOUR <= begins[1] < T0 + 2 * HOUR

    def test_allocated_seconds_that_do_not_fit_are_not_lost(self) -> None:
        """Ten allocated minutes cannot hold a twenty-minute snapshot."""
        config = _config(3)
        target = _target(config, 1, 105.0, 10.0, minutes=20)
        too_short = AllocatedTime(T0 + HOUR, T0 + HOUR + 10 * MIN, 10 * MIN)

        planner = PriorityPlanner(
            config,
            [target],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={1: [too_short]},
        )
        plan = planner.schedule()

        assert len(self._science(plan)) == 1
        assert planner.unplaced == []

    def test_local_search_scores_allocated_seconds_a_tier_up(self) -> None:
        config = _config(3)
        later = _target(config, 1, 105.0, 10.0, merit=10)
        planner = LocalSearchPlanner(
            config,
            [later],
            BEGIN,
            BEGIN + timedelta(hours=3),
            allocated={1: [self.SECOND_HOUR]},
            max_iterations=0,
        )

        planner.schedule()

        # Tiers 1 (allocated) and 0; all 20 minutes count as allocated.
        assert planner.score[0] == pytest.approx(10 * 20 * MIN)
        assert planner.score[1] == 0.0

    def test_not_with_preferred(self) -> None:
        config = _config(2)
        with pytest.raises(ValueError, match="preferred or allocated"):
            PriorityPlanner(
                config,
                [],
                BEGIN,
                BEGIN + timedelta(hours=2),
                preferred={1},
                allocated={},
            )


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
            config, BEGIN, BEGIN + timedelta(hours=self.HOURS), bin_length=self.BIN
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
        planner = ditl.planner

        def recording(*args: object, **kwargs: object) -> PriorityPlanner:
            given.append(kwargs["allocated"])  # type: ignore[arg-type]
            return planner(*args, **kwargs)  # type: ignore[arg-type]

        ditl.planner = recording  # type: ignore[assignment]

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
        assert ditl.queue.prefer is not None  # type: ignore[attr-defined]

    def test_dispatch_needs_a_queue_that_takes_preferences(self) -> None:
        config = _config(self.HOURS)

        class Policy:
            targets: list[Pointing] = []
            log = None

        with pytest.raises(TypeError, match="prefer"):
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
                allocation=AllocationSettings(strictness="weighted", bonus=0.25),
            )

            ditl = create_ditl(config, _targets(config))

            if isinstance(ditl, QueueDITL):
                assert ditl.queue.allocation_strictness == "weighted"  # type: ignore[attr-defined]
                assert ditl.queue.allocation_bonus == 0.25  # type: ignore[attr-defined]
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
