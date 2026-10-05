"""create_ditl builds the simulation a configuration's scheduler section selects."""

import importlib.util
from datetime import timedelta

import pytest

from conops import DITL, QueueDITL, RollingHorizonDITL, create_ditl
from conops.benchmark import BenchmarkScenario, configured, run_benchmark
from conops.config import (
    MissionConfig,
    PlannerKind,
    PlannerSettings,
    ReplanSettings,
    SchedulerConfig,
    SchedulerMode,
)
from conops.schedulers import PLANNERS, LocalSearchPlanner, PriorityPlanner
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
    def test_queues_every_target(self) -> None:
        config, targets = _setup(SchedulerConfig())
        targets[0].deadline = T0 + 30 * MIN

        ditl = create_ditl(config, targets, BEGIN, END)

        assert isinstance(ditl, QueueDITL)
        queued = {int(t.obsid): t for t in ditl.queue.targets}
        assert set(queued) == {int(t.obsid) for t in targets}
        assert queued[100].deadline == T0 + 30 * MIN
        assert queued[100].exptime == targets[0].exptime
        _run(ditl)


class TestPlanned:
    @pytest.mark.parametrize(
        ("planner", "name"),
        [
            (PlannerSettings(), "priority"),
            (
                PlannerSettings(
                    kind=PlannerKind.LOCAL_SEARCH, time_limit=600, max_iterations=50
                ),
                "local_search",
            ),
        ],
    )
    def test_builds_and_executes_the_configured_plan(
        self, planner: PlannerSettings, name: str
    ) -> None:
        config, targets = _setup(
            SchedulerConfig(mode=SchedulerMode.PLANNED, planner=planner)
        )

        ditl = create_ditl(config, targets, BEGIN, END)

        assert isinstance(ditl, DITL)
        assert ditl.plan.metadata is not None
        assert ditl.plan.metadata["planner"] == name
        assert len(ditl.plan) > 0
        _run(ditl)

    @pytest.mark.skipif(
        importlib.util.find_spec("ortools") is None, reason="needs OR-Tools"
    )
    def test_cp_sat_planner(self) -> None:
        config, targets = _setup(
            SchedulerConfig(
                mode=SchedulerMode.PLANNED,
                planner=PlannerSettings(
                    kind=PlannerKind.CP_SAT, solver_time_limit=3, workers=1, seed=1
                ),
            )
        )

        ditl = create_ditl(config, targets, BEGIN, END)

        assert ditl.plan.metadata is not None
        assert ditl.plan.metadata["planner"] == "cp_sat"
        _run(ditl)

    def test_passes_successor_retries_to_the_planner(self) -> None:
        settings = PlannerSettings(successor_retries=2)

        assert settings.options()["successor_retries"] == 2
        config, targets = _setup(
            SchedulerConfig(mode=SchedulerMode.PLANNED, planner=settings)
        )
        _run(create_ditl(config, targets, BEGIN, END))


class TestRolling:
    def test_passes_the_replanning_settings_through(self) -> None:
        config, targets = _setup(
            SchedulerConfig(
                mode=SchedulerMode.ROLLING,
                planner=PlannerSettings(
                    kind=PlannerKind.LOCAL_SEARCH,
                    include_passes=False,
                    time_limit=600,
                    max_iterations=20,
                ),
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
        assert ditl.planner is LocalSearchPlanner
        assert ditl.planner_options == {
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


def test_registry_covers_every_planner_kind() -> None:
    assert set(PLANNERS) == set(PlannerKind)
    assert PLANNERS[PlannerKind.PRIORITY] is PriorityPlanner


def test_benchmark_runs_the_configured_scheduler() -> None:
    def make_config() -> MissionConfig:
        config, _ = _setup(SchedulerConfig(mode=SchedulerMode.PLANNED))
        return config

    scenario = BenchmarkScenario(
        name="configured",
        begin=BEGIN,
        end=END,
        make_config=make_config,
        make_targets=lambda config: [
            _target(config, 100 + i, ra, dec, minutes=40)
            for i, (ra, dec) in enumerate(PATCH)
        ],
    )

    (result,) = run_benchmark(scenario, [configured()])

    assert result.error is None
    assert result.science_hours > 0
    assert result.planning_seconds is not None
    assert result.mismatches == 0
