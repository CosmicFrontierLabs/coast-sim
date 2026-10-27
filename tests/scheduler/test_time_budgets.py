"""A category's time budget: a cap over the run on the science its targets
collect, honoured by dispatch, the planners and the allocator, and reported
in the category summary."""

from datetime import timedelta

import pytest

from conops import QueueDITL, create_ditl
from conops.common import ObsType
from conops.config.observation_categories import ObservationCategory
from conops.schedulers import (
    LocalSearchScheduler,
    LongRangeAllocator,
    PriorityScheduler,
)
from conops.targets import ObservationSequence, Pointing, SequenceStep

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 6
BUDGET = 50 * MIN
CAPPED = ObservationCategory(
    name="Capped", obsid_min=1, obsid_max=10, time_budget_seconds=BUDGET
)


def _scenario(*, collected_before: float = 0.0):  # type: ignore[no-untyped-def]
    """Three capped targets of an hour each that outrank a filler."""
    config = _config(HOURS, categories=[CAPPED])
    capped = [
        _target(config, k, *PATCH[k], merit=100 - k, minutes=60, snapshot=20, ss_min=5)
        for k in range(1, 4)
    ]
    if collected_before:
        capped[0].record_collection(T0 - HOUR, collected_before)
    filler = _target(config, 100, *PATCH[4], merit=10, minutes=300, snapshot=20)
    return config, [*capped, filler]


def _capped_seconds(plan) -> float:  # type: ignore[no-untyped-def]
    return sum(
        e.collection_seconds_between(float(e.begin), float(e.end))
        for e in plan
        if e.obstype not in (ObsType.GSP, ObsType.ACTIVITY) and 1 <= int(e.obsid) < 10
    )


class TestDispatch:
    def test_the_category_stops_at_its_budget(self) -> None:
        config, targets = _scenario()
        ditl = create_ditl(config, targets, BEGIN, BEGIN + timedelta(hours=HOURS))
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = 60
        assert ditl.calc()
        assert BUDGET - 60 <= _capped_seconds(ditl.plan) <= BUDGET + 1e-6
        # The filler takes over once the budget is used; the capped targets
        # keep their exposure, unobserved.
        assert any(int(e.obsid) == 100 for e in ditl.plan)
        assert any(
            e.exposure < 20 * MIN
            for e in ditl.plan
            if e.obstype == ObsType.AT and 1 <= int(e.obsid) < 10
        ), "the last capped observation is sized to the budget left"
        row = next(r for r in ditl.summary_by_category() if r.category == "Capped")
        assert row.budget_hours == pytest.approx(BUDGET / 3600)
        assert row.budget_used == pytest.approx(_capped_seconds(ditl.plan) / BUDGET)

    def test_science_collected_before_the_run_counts(self) -> None:
        config, targets = _scenario(collected_before=30 * MIN)
        ditl = create_ditl(config, targets, BEGIN, BEGIN + timedelta(hours=HOURS))
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = 60
        assert ditl.calc()
        assert (
            BUDGET - 30 * MIN - 60
            <= _capped_seconds(ditl.plan)
            <= BUDGET - 30 * MIN + 1e-6
        )

    def test_a_visit_is_taken_only_whole(self) -> None:
        config = _config(HOURS, categories=[CAPPED])
        visit = Pointing(
            config=config,
            obsid=1,
            name="visit",
            ra=PATCH[0][0],
            dec=PATCH[0][1],
            merit=100.0,
            sequence=ObservationSequence(
                steps=[SequenceStep(name="s", duration_seconds=30 * MIN)]
            ),
            visits=3,
        )
        filler = _target(config, 100, *PATCH[4], merit=10, minutes=300, snapshot=20)
        ditl = create_ditl(
            config, [visit, filler], BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = 60
        assert ditl.calc()
        visits = [e for e in ditl.plan if int(e.obsid) == 1 and e.step is not None]
        assert len(visits) == 1, "a second visit would not fit the budget whole"


class TestPlanners:
    @pytest.mark.parametrize("scheduler", [PriorityScheduler, LocalSearchScheduler])
    def test_the_plan_caps_the_category(self, scheduler) -> None:  # type: ignore[no-untyped-def]
        config, targets = _scenario()
        options = {"time_limit": 1.0} if scheduler is LocalSearchScheduler else {}
        built = scheduler(
            config, targets, BEGIN, BEGIN + timedelta(hours=HOURS), **options
        )
        plan = built.schedule()
        assert BUDGET - 60 <= _capped_seconds(plan) <= BUDGET + 1e-6
        assert any(int(e.obsid) == 100 for e in plan)
        if scheduler is PriorityScheduler:
            assert any("time budget leaves" in e.description for e in built.log.events)

    def test_cp_sat_caps_the_category(self) -> None:
        pytest.importorskip("ortools")
        from conops.schedulers import CpSatScheduler

        config, targets = _scenario()
        plan = CpSatScheduler(
            config,
            targets,
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            solver_time_limit=2.0,
        ).schedule()
        assert _capped_seconds(plan) <= BUDGET + 1e-6
        assert any(int(e.obsid) == 100 for e in plan)


class TestAllocator:
    @pytest.mark.parametrize("solver", ["greedy", "milp"])
    def test_the_allocation_caps_the_category(self, solver: str) -> None:
        if solver == "milp":
            pytest.importorskip("ortools")
        config = _config(48, categories=[CAPPED])
        targets = [
            _target(config, k, *PATCH[k], merit=100 - k, minutes=120, snapshot=20)
            for k in range(1, 4)
        ]
        targets.append(
            _target(config, 100, *PATCH[4], merit=10, minutes=600, snapshot=20)
        )
        allocator = LongRangeAllocator(
            config, BEGIN, BEGIN + timedelta(days=2), solver=solver
        )
        allocation = allocator.allocate(targets, T0)
        capped = sum(
            sum(per_bin.values())
            for obsid, per_bin in allocation.seconds.items()
            if 1 <= obsid < 10
        )
        assert 0 < capped <= BUDGET + 1e-6
        assert sum(allocation.seconds.get(100, {}).values()) > 0
