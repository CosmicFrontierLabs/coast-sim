"""CpSatPlanner: solver plans must execute exactly and never lose to priority-first."""

import sys
from datetime import timedelta

import pytest

pytest.importorskip("ortools")

from ortools.sat.python import cp_model, cp_model_helper  # noqa: E402

from conops import DITL  # noqa: E402
from conops.common import ObsType  # noqa: E402
from conops.ditl import RollingHorizonDITL  # noqa: E402
from conops.schedulers import CpSatPlanner  # noqa: E402
from conops.targets import Plan, Pointing  # noqa: E402

from .planning_scenario import BEGIN, CONSTRAINED, HOUR, MIN, PATCH, T0  # noqa: E402
from .planning_scenario import make_config as _config  # noqa: E402
from .planning_scenario import make_target as _target  # noqa: E402


def _targets(hours: int, *, stations: bool = False) -> list[Pointing]:
    config = _config(hours, stations=stations)
    return [
        _target(config, 100 + i, ra, dec, merit=90 - 7 * i, minutes=60, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH + CONSTRAINED)
    ]


def _planner(
    targets: list[Pointing], hours: int, **options: float | int | timedelta
) -> CpSatPlanner:
    config = targets[0].config
    assert config is not None
    settings: dict[str, float | int | timedelta] = {
        "solver_time_limit": 5.0,
        "workers": 1,
        "seed": 1,
    }
    settings.update(options)
    return CpSatPlanner(
        config,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=hours),
        **settings,  # type: ignore[arg-type]
    )


def _execute(targets: list[Pointing], plan: Plan, hours: int) -> DITL:
    config = targets[0].config
    assert config is not None
    ditl = DITL(
        config=config,
        ephem=config.constraint.ephem,
        plan=Plan.model_validate_json(plan.model_dump_json()),
        begin=BEGIN,
        end=BEGIN + timedelta(hours=hours),
    )
    ditl.step_size = 60
    assert ditl.calc()
    return ditl


def _science(plan: Plan) -> list[int]:
    return [int(e.obsid) for e in plan if e.obstype != ObsType.GSP]


def _collected(plan: Plan) -> float:
    return sum(
        float(e.collection_end) - float(e.collection_begin)
        for e in plan
        if e.collection_begin is not None and e.collection_end is not None
    )


class TestPlans:
    def test_never_worse_than_priority_first_and_executes_exactly(self) -> None:
        targets = _targets(4)
        planner = _planner(targets, 4)

        plan = planner.schedule()
        ditl = _execute(targets, plan, 4)

        assert planner.score >= planner.initial_score
        assert planner.solver_statuses
        assert set(planner.solver_statuses) <= {"OPTIMAL", "FEASIBLE"}
        assert ditl.validate_plan_matches_execution() == []
        executed = sum(hk.collection_seconds for hk in ditl.telemetry.housekeeping)
        assert executed == pytest.approx(_collected(plan))

    def test_solves_in_several_chunks(self) -> None:
        targets = _targets(3)
        planner = _planner(targets, 3, chunk=timedelta(hours=1))

        plan = planner.schedule()

        assert len(planner.solver_statuses) == 3
        assert len(planner.solver_chunks_used) == 3
        assert planner.solver_score is not None
        assert planner.solver_score >= planner.start_score
        assert _execute(targets, plan, 3).validate_plan_matches_execution() == []

    def test_plan_with_ground_passes_executes_exactly(self) -> None:
        targets = _targets(12, stations=True)
        planner = _planner(targets, 12, solver_time_limit=8.0)

        plan = planner.schedule()

        assert any(entry.obstype == ObsType.GSP for entry in plan)
        assert _execute(targets, plan, 12).validate_plan_matches_execution() == []

    def test_inputs_are_not_modified(self) -> None:
        targets = _targets(3)

        _planner(targets, 3).schedule()

        assert all(t.exptime == 60 * MIN for t in targets)


class TestHint:
    def test_hint_is_feasible_and_reproduces_the_priority_first_plan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Held to its hint, the solver must return the priority-first plan.

        The hint covers each slew starting on a step, passes, and several
        snapshots of one request in a window, so any mismatch between the
        model and the decoder makes a chunk infeasible or lose value.
        """
        solve = cp_model.CpSolver.solve

        def hinted_only(
            solver: cp_model.CpSolver, model: cp_model.CpModel
        ) -> cp_model_helper.CpSolverStatus:
            solver.parameters.fix_variables_to_their_hinted_value = True
            return solve(solver, model)

        monkeypatch.setattr(cp_model.CpSolver, "solve", hinted_only)
        targets = _targets(12, stations=True)
        planner = _planner(targets, 12)

        planner.schedule()

        assert set(planner.solver_statuses) == {"OPTIMAL"}
        assert all(planner.solver_chunks_used)
        assert planner.solver_score == planner.start_score

    def test_no_solution_in_time_keeps_the_priority_first_plan(self) -> None:
        targets = _targets(6)
        planner = _planner(targets, 6, solver_time_limit=1e-6)

        plan = planner.schedule()

        assert "UNKNOWN" in planner.solver_statuses
        assert not any(planner.solver_chunks_used)
        assert planner.solver_score == planner.start_score
        assert _execute(targets, plan, 6).validate_plan_matches_execution() == []


class TestSeveralSnapshotsInAWindow:
    def test_solver_collects_an_exposure_longer_than_one_snapshot(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, minutes=60, snapshot=20)
        planner = CpSatPlanner(
            config,
            [target],
            BEGIN,
            BEGIN + timedelta(hours=2),
            solver_time_limit=5.0,
            workers=1,
            seed=1,
        )

        plan = planner.schedule()

        assert planner.solver_chunks_used == [True]
        assert _science(plan) == [1, 1, 1]
        assert _collected(plan) == pytest.approx(60 * MIN)


class TestShortWindowRequest:
    def test_solver_fits_the_request_priority_order_lost(self) -> None:
        """Priority order fills B's only slot with A; the solver fits both."""
        config = _config(3)
        flexible = _target(config, 1, 105.0, 10.0, merit=100, minutes=60)
        urgent = _target(
            config, 2, 110.0, 10.0, merit=70, minutes=30, deadline=T0 + 30 * MIN
        )

        planner = CpSatPlanner(
            config,
            [flexible, urgent],
            BEGIN,
            BEGIN + timedelta(hours=3),
            solver_time_limit=5.0,
            workers=1,
            seed=1,
        )
        plan = planner.schedule()

        assert _science(plan) == [2, 1]
        assert planner.solver_score is not None
        assert planner.solver_score > planner.initial_score
        urgent_entry = next(e for e in plan if e.obsid == 2)
        assert urgent_entry.collection_begin is not None
        assert urgent_entry.collection_begin <= T0 + 30 * MIN


class TestEarliestStart:
    def test_solver_keeps_the_earliest_start(self) -> None:
        config = _config(3)
        flexible = _target(config, 1, 105.0, 10.0, merit=100, minutes=60)
        later = _target(
            config,
            2,
            110.0,
            10.0,
            merit=70,
            minutes=30,
            earliest_start=T0 + HOUR,
            deadline=T0 + 90 * MIN,
        )

        planner = CpSatPlanner(
            config,
            [flexible, later],
            BEGIN,
            BEGIN + timedelta(hours=3),
            solver_time_limit=5.0,
            workers=1,
            seed=1,
        )
        plan = planner.schedule()

        entry = next(e for e in plan if e.obsid == 2)
        assert entry.collection_begin is not None
        assert T0 + HOUR <= entry.collection_begin <= T0 + 90 * MIN


class TestRollingHorizon:
    def test_rolling_replanning_uses_the_cp_sat_planner(self) -> None:
        targets = _targets(3)
        ditl = RollingHorizonDITL(
            targets[0].config,  # type: ignore[arg-type]
            targets,
            begin=BEGIN,
            end=BEGIN + timedelta(hours=3),
            replan_interval=timedelta(hours=1),
            planner=CpSatPlanner,
            planner_options={"solver_time_limit": 3.0, "workers": 1, "seed": 1},
        )
        ditl.step_size = 60

        assert ditl.calc()

        assert len(ditl.replans) == 3
        assert ditl.validate_plan_matches_execution() == []


class TestArguments:
    @pytest.mark.parametrize(
        "options",
        [
            {"solver_time_limit": 0.0},
            {"chunk": timedelta(0)},
            {"workers": 0},
            {"max_candidates": 0},
        ],
    )
    def test_rejects_invalid_settings(
        self, options: dict[str, float | int | timedelta]
    ) -> None:
        config = _config(1)

        with pytest.raises(ValueError):
            CpSatPlanner(
                config,
                [],
                BEGIN,
                BEGIN + timedelta(hours=1),
                **options,  # type: ignore[arg-type]
            )

    def test_missing_ortools_names_the_package(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        targets = _targets(1)
        monkeypatch.setitem(sys.modules, "ortools.sat.python", None)

        with pytest.raises(ImportError, match="ortools"):
            _planner(targets, 1).schedule()
