"""Planners follow the dynamic merit terms: cadence and completion deficit."""

from collections.abc import Mapping, Sequence
from datetime import timedelta

import pytest

from conops import DITL
from conops.common import ObsType
from conops.config import MissionConfig, TargetConfig
from conops.config.observation_categories import ObservationCategory
from conops.schedulers import LocalSearchPlanner, PriorityPlanner
from conops.targets import Plan, Pointing

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 4
MONITOR = ObservationCategory(
    name="Monitor", obsid_min=1, obsid_max=2, cadence_seconds=HOUR
)
PROGRAMS = [
    ObservationCategory(name="A", obsid_min=100, obsid_max=200, time_share=0.5),
    ObservationCategory(name="B", obsid_min=200, obsid_max=300, time_share=0.5),
]


def _monitor_config(cadence_weight: float) -> MissionConfig:
    return _config(
        HOURS,
        weights=TargetConfig(cadence_weight=cadence_weight),
        categories=[MONITOR],
    )


def _monitored(config: MissionConfig) -> list[Pointing]:
    """A cadence target wanting three ten-minute visits, and filler."""
    monitor = _target(config, 1, *PATCH[0], merit=60, minutes=30, snapshot=10)
    filler = [
        _target(config, 10 + i, ra, dec, merit=40, minutes=60, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH[1:])
    ]
    return [monitor, *filler]


def _programs_config(deficit_weight: float) -> MissionConfig:
    return _config(
        HOURS,
        weights=TargetConfig(completion_deficit_weight=deficit_weight),
        categories=PROGRAMS,
    )


def _programs(config: MissionConfig) -> list[Pointing]:
    """Program A outranks B, and either could fill the horizon alone."""
    a = [
        _target(config, 100 + i, ra, dec, merit=100, minutes=120, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH[:3])
    ]
    b = [
        _target(config, 200 + i, ra, dec, merit=60, minutes=120, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH[2:])
    ]
    return [*a, *b]


def _plan(
    planner: type[PriorityPlanner],
    config: MissionConfig,
    targets: Sequence[Pointing],
    **options: float | Mapping[int, float],
) -> tuple[PriorityPlanner, Plan]:
    if planner is LocalSearchPlanner:
        options = {"time_limit": 0.0, "max_iterations": 300, "seed": 1, **options}
    built = planner(
        config,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=HOURS),
        **options,  # type: ignore[arg-type]
    )
    return built, built.schedule()


def _visits(plan: Plan, obsid: int) -> list[tuple[float, float]]:
    return sorted(
        (float(e.collection_begin), float(e.collection_end))
        for e in plan
        if e.obsid == obsid
        and e.collection_begin is not None
        and e.collection_end is not None
    )


def _share(plan: Plan, low: int, high: int) -> float:
    seconds = {True: 0.0, False: 0.0}
    for e in plan:
        if e.obstype == ObsType.GSP or e.collection_begin is None:
            continue
        assert e.collection_end is not None
        span = float(e.collection_end) - float(e.collection_begin)
        seconds[low <= e.obsid < high] += span
    total = seconds[True] + seconds[False]
    return seconds[True] / total if total else 0.0


def _executes_exactly(config: MissionConfig, plan: Plan) -> bool:
    ditl = DITL(
        config=config,
        ephem=config.constraint.ephem,
        plan=Plan.model_validate_json(plan.model_dump_json()),
        begin=BEGIN,
        end=BEGIN + timedelta(hours=HOURS),
    )
    ditl.step_size = 60
    assert ditl.calc()
    return ditl.validate_plan_matches_execution() == []


PLANNERS = [PriorityPlanner, LocalSearchPlanner]


@pytest.mark.parametrize("planner", PLANNERS)
class TestCadence:
    def test_visits_wait_for_the_cadence(self, planner: type[PriorityPlanner]) -> None:
        config = _monitor_config(cadence_weight=50)

        _, plan = _plan(planner, config, _monitored(config))

        visits = _visits(plan, 1)
        assert len(visits) == 3
        for (_, end), (begin, _) in zip(visits, visits[1:]):
            assert begin >= end + HOUR
        assert _executes_exactly(config, plan)

    def test_without_a_cadence_weight_visits_are_not_spaced(
        self, planner: type[PriorityPlanner]
    ) -> None:
        config = _monitor_config(cadence_weight=0)

        _, plan = _plan(planner, config, _monitored(config))

        visits = _visits(plan, 1)
        assert any(
            begin < end + HOUR for (_, end), (begin, _) in zip(visits, visits[1:])
        )

    def test_the_last_collected_visit_counts(
        self, planner: type[PriorityPlanner]
    ) -> None:
        config = _monitor_config(cadence_weight=50)
        targets = _monitored(config)
        targets[0].record_collection(T0 - 30 * MIN, 60.0)

        _, plan = _plan(planner, config, targets)

        first, _ = _visits(plan, 1)[0]
        assert first >= T0 + 30 * MIN


@pytest.mark.parametrize("planner", PLANNERS)
class TestCompletionDeficit:
    def test_shares_balance_programs(self, planner: type[PriorityPlanner]) -> None:
        plain_config = _programs_config(deficit_weight=0)
        balanced_config = _programs_config(deficit_weight=100)

        _, plain = _plan(planner, plain_config, _programs(plain_config))
        _, balanced = _plan(planner, balanced_config, _programs(balanced_config))

        assert _share(balanced, 200, 300) > _share(plain, 200, 300) + 0.2
        assert _executes_exactly(balanced_config, balanced)

    def test_collected_science_counts_towards_shares(
        self, planner: type[PriorityPlanner]
    ) -> None:
        config = _programs_config(deficit_weight=100)
        targets = _programs(config)
        targets[0].record_collection(T0 - HOUR, 10 * HOUR)

        _, plan = _plan(planner, config, targets)

        assert _share(plan, 200, 300) > 0.8


class TestLocalSearchObjective:
    def test_decoded_start_scores_as_the_priority_first_plan(self) -> None:
        config = _config(
            HOURS,
            weights=TargetConfig(cadence_weight=50, completion_deficit_weight=100),
            categories=[MONITOR, *PROGRAMS],
        )
        targets = [*_monitored(config), *_programs(config)]

        planner, _ = _plan(LocalSearchPlanner, config, targets)

        assert isinstance(planner, LocalSearchPlanner)
        assert planner.start_score == pytest.approx(planner.initial_score)
        assert planner.score >= planner.initial_score


class TestCpSat:
    """CP-SAT follows the same terms; its hint must stay feasible with them."""

    @pytest.fixture(autouse=True)
    def _ortools(self) -> None:
        pytest.importorskip("ortools")

    @staticmethod
    def _cp_sat(config: MissionConfig, targets: Sequence[Pointing]) -> Plan:
        from conops.schedulers import CpSatPlanner

        _, plan = _plan(
            CpSatPlanner, config, targets, solver_time_limit=3.0, workers=1, seed=1
        )
        return plan

    def test_visits_wait_for_the_cadence(self) -> None:
        config = _monitor_config(cadence_weight=50)

        plan = self._cp_sat(config, _monitored(config))

        visits = _visits(plan, 1)
        assert len(visits) == 3
        for (_, end), (begin, _) in zip(visits, visits[1:]):
            assert begin >= end + HOUR
        assert _executes_exactly(config, plan)

    def test_shares_balance_programs(self) -> None:
        config = _programs_config(deficit_weight=100)

        plan = self._cp_sat(config, _programs(config))

        assert _share(plan, 200, 300) > 0.2
        assert _executes_exactly(config, plan)

    def test_hint_is_feasible_with_dynamic_terms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ortools.sat.python import cp_model, cp_model_helper

        from conops.schedulers import CpSatPlanner

        solve = cp_model.CpSolver.solve

        def hinted_only(
            solver: cp_model.CpSolver, model: cp_model.CpModel
        ) -> cp_model_helper.CpSolverStatus:
            solver.parameters.fix_variables_to_their_hinted_value = True
            return solve(solver, model)

        monkeypatch.setattr(cp_model.CpSolver, "solve", hinted_only)
        config = _config(
            HOURS,
            weights=TargetConfig(cadence_weight=50, completion_deficit_weight=100),
            categories=[MONITOR, *PROGRAMS],
        )
        planner = CpSatPlanner(
            config,
            [*_monitored(config), *_programs(config)],
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            chunk=timedelta(hours=1),
            workers=1,
            seed=1,
        )

        planner.schedule()

        assert set(planner.solver_statuses) == {"OPTIMAL"}
        assert planner.solver_score == planner.start_score


@pytest.mark.parametrize("planner", PLANNERS)
class TestCommittedObservations:
    """Observations committed outside the plan count as visits and shares."""

    def test_a_committed_visit_delays_the_next(
        self, planner: type[PriorityPlanner]
    ) -> None:
        config = _monitor_config(cadence_weight=50)

        _, plan = _plan(
            planner,
            config,
            _monitored(config),
            reserved_visits={1: T0 + 20 * MIN},
        )

        first, _ = _visits(plan, 1)[0]
        assert first >= T0 + 80 * MIN

    def test_committed_seconds_count_towards_shares(
        self, planner: type[PriorityPlanner]
    ) -> None:
        plain_config = _programs_config(deficit_weight=100)
        reserved_config = _programs_config(deficit_weight=100)

        _, plain = _plan(planner, plain_config, _programs(plain_config))
        _, reserved = _plan(
            planner,
            reserved_config,
            _programs(reserved_config),
            reserved_seconds={100: HOUR},
        )

        # An hour of A already committed: the plan gives B more.
        assert _share(reserved, 200, 300) > _share(plain, 200, 300) + 0.05


def _rolling_cadence(monkeypatch: pytest.MonkeyPatch | None) -> list[float]:
    """Run a monitored target under rolling replanning; return its visit gaps."""
    from conops.benchmark.metrics import visit_starts
    from conops.ditl import RollingHorizonDITL

    if monkeypatch is not None:
        monkeypatch.setattr(
            RollingHorizonDITL, "_reserved_visits", staticmethod(lambda *_: {})
        )
    config = _monitor_config(cadence_weight=50)
    ditl = RollingHorizonDITL(
        config,
        _monitored(config),
        begin=BEGIN,
        end=BEGIN + timedelta(hours=HOURS),
        horizon=timedelta(hours=2),
        replan_interval=timedelta(minutes=30),
        # Long enough that a replan finds a visit committed but not collected.
        commit_lead_time=timedelta(minutes=30),
    )
    ditl.step_size = 60
    assert ditl.calc()
    records = [
        (int(r.obsid), r.timestamp.timestamp(), r.collection_seconds or 0.0)
        for r in ditl.telemetry.housekeeping
        if r.obsid is not None
    ]
    starts = visit_starts(records, 60.0).get(1, [])
    return [b - a for a, b in zip(starts, starts[1:])]


class TestRollingCadence:
    def test_replans_keep_visits_apart(self) -> None:
        gaps = _rolling_cadence(None)

        assert gaps
        assert min(gaps) >= HOUR

    def test_without_committed_visits_a_replan_revisits_early(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guards the test above: it must fail if committed visits are ignored."""
        assert min(_rolling_cadence(monkeypatch)) < HOUR
