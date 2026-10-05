"""Long-range allocation and how it steers planners and simulations."""

from datetime import timedelta

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
from conops.schedulers import Allocation, LongRangeAllocator, PriorityPlanner
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

    @pytest.mark.parametrize(
        "options",
        [
            {"bin_length": timedelta(0)},
            {"efficiency": 0.0},
            {"reserve": -0.1},
            {"reserve": 1.0},
            {"solver": "lp"},
            {"time_limit": 0.0},
            {"workers": 0},
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

    def test_planned_mode_rejects_an_allocation(self) -> None:
        with pytest.raises(ValidationError, match="allocation"):
            SchedulerConfig(mode=SchedulerMode.PLANNED, allocation=AllocationSettings())

    def test_no_allocation_by_default(self) -> None:
        config = _config(2)

        ditl = create_ditl(config, _targets(config))

        assert isinstance(ditl, QueueDITL)
        assert ditl.allocator is None
