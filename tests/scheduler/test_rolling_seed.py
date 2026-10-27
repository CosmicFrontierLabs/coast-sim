"""A rolling run seeded by a start plan: each replan is given the entries of
the plan inside its horizon."""

from datetime import timedelta

import pytest

from conops import create_ditl
from conops.config import MissionConfig
from conops.config.scheduler import SchedulerKind, SchedulerMode
from conops.ditl import RollingHorizonDITL
from conops.ditl.queue_ditl import QueueDITL
from conops.schedulers import LocalSearchScheduler, PriorityScheduler
from conops.targets import Plan, Pointing

from .planning_scenario import BEGIN, PATCH
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 12
END = BEGIN + timedelta(hours=HOURS)
STEP = 60


def _targets(config: MissionConfig) -> list[Pointing]:
    return [
        _target(config, 1 + k, *PATCH[k], merit=90.0 - 10 * k, minutes=120, snapshot=30)
        for k in range(4)
    ]


def _dispatch_plan() -> Plan:
    config = _config(HOURS)
    ditl = create_ditl(config, _targets(config), BEGIN, END)
    assert isinstance(ditl, QueueDITL)
    ditl.step_size = STEP
    assert ditl.calc()
    return ditl.plan


class _Recording(LocalSearchScheduler):
    """A local-search scheduler that keeps the start plan each replan gave it."""

    seeds: list[Plan | None] = []
    windows: list[tuple[float, float]] = []

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        type(self).seeds.append(kwargs.get("start_plan"))
        begin, end = args[2], args[3]
        type(self).windows.append((begin.timestamp(), end.timestamp()))
        super().__init__(*args, **kwargs)


@pytest.fixture(autouse=True)
def _clear_recording() -> None:
    _Recording.seeds = []
    _Recording.windows = []


class TestSeeding:
    def test_each_replan_gets_the_entries_inside_its_horizon(self) -> None:
        plan = _dispatch_plan()
        assert len(plan) > 1
        config = _config(HOURS)
        rolling = RollingHorizonDITL(
            config,
            _targets(config),
            begin=BEGIN,
            end=END,
            horizon=timedelta(hours=6),
            replan_interval=timedelta(hours=3),
            scheduler=_Recording,
            scheduler_options={"time_limit": 0.5},
            start_plan=plan,
        )
        rolling.step_size = STEP
        assert rolling.calc()
        assert len(rolling.replans) >= 2
        assert all(
            r.seed_entries == len(s or [])
            for r, s in zip(rolling.replans, _Recording.seeds, strict=True)
        )
        assert rolling.replans[0].seed_entries > 0
        for seed, (start, end) in zip(
            _Recording.seeds, _Recording.windows, strict=True
        ):
            if seed is None:
                continue
            for entry in seed:
                assert float(entry.end) > start and float(entry.begin) < end
        # The whole plan was not handed over: a later replan sees fewer entries.
        assert _Recording.seeds[-1] is None or len(_Recording.seeds[-1]) < len(plan)

    def test_without_a_start_plan_nothing_is_seeded(self) -> None:
        config = _config(HOURS)
        rolling = RollingHorizonDITL(
            config,
            _targets(config),
            begin=BEGIN,
            end=END,
            horizon=timedelta(hours=6),
            replan_interval=timedelta(hours=6),
            scheduler=_Recording,
            scheduler_options={"time_limit": 0.5},
        )
        rolling.step_size = STEP
        assert rolling.calc()
        assert all(s is None for s in _Recording.seeds)
        assert all(r.seed_entries == 0 for r in rolling.replans)

    def test_a_priority_scheduler_is_not_seeded(self) -> None:
        plan = _dispatch_plan()
        config = _config(HOURS)
        rolling = RollingHorizonDITL(
            config,
            _targets(config),
            begin=BEGIN,
            end=END,
            horizon=timedelta(hours=6),
            replan_interval=timedelta(hours=6),
            scheduler=PriorityScheduler,
            start_plan=plan,
        )
        rolling.step_size = STEP
        assert rolling.calc()
        assert all(r.seed_entries == 0 for r in rolling.replans)


class TestFactory:
    def test_rolling_and_planned_take_the_plan_and_dispatch_refuses_it(self) -> None:
        plan = _dispatch_plan()
        config = _config(HOURS)
        config.scheduler.mode = SchedulerMode.ROLLING
        config.scheduler.kind = SchedulerKind.LOCAL_SEARCH
        rolling = create_ditl(config, _targets(config), BEGIN, END, start_plan=plan)
        assert isinstance(rolling, RollingHorizonDITL) and rolling.start_plan is plan
        planned_config = _config(HOURS)
        planned_config.scheduler.mode = SchedulerMode.PLANNED
        planned_config.scheduler.kind = SchedulerKind.LOCAL_SEARCH
        planned = create_ditl(
            planned_config, _targets(planned_config), BEGIN, END, start_plan=plan
        )
        assert planned.plan is not None
        dispatch_config = _config(HOURS)
        with pytest.raises(ValueError, match="dispatch takes no start plan"):
            create_ditl(
                dispatch_config, _targets(dispatch_config), BEGIN, END, start_plan=plan
            )
