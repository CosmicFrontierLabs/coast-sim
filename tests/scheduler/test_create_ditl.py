"""create_ditl builds the simulation a configuration's scheduler section selects."""

import importlib.util
from datetime import timedelta

import pytest

from conops import DITL, QueueDITL, RollingHorizonDITL, create_ditl
from conops.config import (
    MissionConfig,
    ReplanSettings,
    SchedulerConfig,
    SchedulerKind,
    SchedulerMode,
)
from conops.schedulers import SCHEDULERS, LocalSearchScheduler, PriorityScheduler
from conops.targets import Pointing

from .planning_scenario import BEGIN, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 2
END = BEGIN + timedelta(hours=HOURS)


def _setup(scheduler: SchedulerConfig) -> tuple[MissionConfig, list[Pointing]]:
    config = _config(HOURS)
    config.scheduler = scheduler
    targets = [
        _target(config, 100 + i, ra, dec, merit=90 - 10 * i, minutes=40)
        for i, (ra, dec) in enumerate(PATCH)
    ]
    return config, targets


def _run(ditl: QueueDITL | DITL | RollingHorizonDITL) -> None:
    assert ditl.calc()
    assert ditl.validate_plan_matches_execution() == []


class TestDispatch:
    def test_queues_every_target_with_its_settings(self) -> None:
        config, targets = _setup(SchedulerConfig())
        targets[0].deadline = T0 + 30 * MIN
        targets[1].earliest_start = T0 + 20 * MIN
        # Merit is what counts, not the legacy fom, which is not kept in step.
        targets[2].merit = 33.0

        ditl = create_ditl(config, targets, BEGIN, END)

        assert isinstance(ditl, QueueDITL)
        queued = {int(t.obsid): t for t in ditl.queue.targets}
        assert set(queued) == {int(t.obsid) for t in targets}
        assert [queued[int(t.obsid)].merit for t in targets] == [
            t.merit for t in targets
        ]
        assert queued[102].merit == 33.0
        assert queued[100].deadline == T0 + 30 * MIN
        assert queued[101].earliest_start == T0 + 20 * MIN
        assert queued[100].exptime == targets[0].exptime
        _run(ditl)


class TestPlanned:
    @pytest.mark.parametrize(
        ("scheduler", "name"),
        [
            (SchedulerConfig(mode=SchedulerMode.PLANNED), "priority"),
            (
                SchedulerConfig(
                    mode=SchedulerMode.PLANNED,
                    kind=SchedulerKind.LOCAL_SEARCH,
                    time_limit=600,
                    max_iterations=50,
                ),
                "local_search",
            ),
        ],
    )
    def test_builds_and_executes_the_configured_plan(
        self, scheduler: SchedulerConfig, name: str
    ) -> None:
        config, targets = _setup(scheduler)

        ditl = create_ditl(config, targets, BEGIN, END)

        assert isinstance(ditl, DITL)
        assert ditl.plan.metadata is not None
        assert ditl.plan.metadata["scheduler"] == name
        assert len(ditl.plan) > 0
        _run(ditl)

    @pytest.mark.skipif(
        importlib.util.find_spec("ortools") is None, reason="needs OR-Tools"
    )
    def test_cp_sat_scheduler(self) -> None:
        config, targets = _setup(
            SchedulerConfig(
                mode=SchedulerMode.PLANNED,
                kind=SchedulerKind.CP_SAT,
                solver_time_limit=3,
                workers=1,
                seed=1,
            )
        )

        ditl = create_ditl(config, targets, BEGIN, END)

        assert ditl.plan.metadata is not None
        assert ditl.plan.metadata["scheduler"] == "cp_sat"
        _run(ditl)

    def test_without_passes_the_plan_has_none(self) -> None:
        config, targets = _setup(
            SchedulerConfig(mode=SchedulerMode.PLANNED, include_passes=False)
        )

        ditl = create_ditl(config, targets, BEGIN, END)

        assert all(e.obstype.name != "GSP" for e in ditl.plan)
        _run(ditl)


class TestRolling:
    def test_passes_the_replanning_settings_through(self) -> None:
        config, targets = _setup(
            SchedulerConfig(
                mode=SchedulerMode.ROLLING,
                kind=SchedulerKind.LOCAL_SEARCH,
                include_passes=False,
                time_limit=600,
                max_iterations=20,
                replanning=ReplanSettings(
                    horizon_seconds=5400,
                    replan_interval_seconds=1800,
                    commit_lead_time_seconds=600,
                    rapid_replans=False,
                    allow_interrupts=False,
                ),
            )
        )

        ditl = create_ditl(config, targets, BEGIN, END)

        assert isinstance(ditl, RollingHorizonDITL)
        assert ditl.horizon == timedelta(minutes=90)
        assert ditl.replan_interval == timedelta(minutes=30)
        assert ditl.commit_lead_time == timedelta(minutes=10)
        assert not ditl.rapid_replans and not ditl.allow_interrupts
        assert not ditl.include_passes
        assert ditl.scheduler is LocalSearchScheduler
        assert ditl.scheduler_options == {
            "successor_retries": 3,
            "time_limit": 600,
            "max_iterations": 20,
        }
        _run(ditl)
        assert len(ditl.replans) == 4


class TestDefaults:
    def test_horizon_defaults_to_the_ephemeris(self) -> None:
        config, targets = _setup(SchedulerConfig(mode=SchedulerMode.PLANNED))

        ditl = create_ditl(config, targets)

        assert ditl.begin == BEGIN
        assert ditl.end == END

    def test_requires_an_ephemeris(self) -> None:
        config, targets = _setup(SchedulerConfig())
        config.constraint.ephem = None

        with pytest.raises(ValueError, match="ephemeris"):
            create_ditl(config, targets, BEGIN, END)


def test_registry_covers_every_scheduler_kind() -> None:
    assert set(SCHEDULERS) == set(SchedulerKind)
    assert SCHEDULERS[SchedulerKind.PRIORITY] is PriorityScheduler
