"""LocalSearchPlanner: improve priority-first plans without breaking execution."""

from datetime import timedelta

import pytest

from conops import DITL
from conops.common import ObsType
from conops.ditl import RollingHorizonDITL
from conops.schedulers import LocalSearchPlanner, PriorityPlanner
from conops.targets import Plan, Pointing

from .planning_scenario import BEGIN, CONSTRAINED, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target


def _targets(hours: int) -> list[Pointing]:
    config = _config(hours)
    return [
        _target(config, 100 + i, ra, dec, merit=90 - 7 * i, minutes=60, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH + CONSTRAINED)
    ]


def _science(plan: Plan) -> list[tuple[int, float, float]]:
    return [
        (int(e.obsid), float(e.begin), float(e.end))
        for e in plan
        if e.obstype != ObsType.GSP
    ]


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


def _planner(
    targets: list[Pointing], hours: int, iterations: int
) -> LocalSearchPlanner:
    config = targets[0].config
    assert config is not None
    return LocalSearchPlanner(
        config,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=hours),
        time_limit=600,
        max_iterations=iterations,
        seed=1,
    )


class TestStartingPoint:
    def test_without_search_the_plan_is_the_priority_first_plan(self) -> None:
        targets = _targets(4)
        greedy = PriorityPlanner(
            targets[0].config,  # type: ignore[arg-type]
            targets,
            BEGIN,
            BEGIN + timedelta(hours=4),
        ).schedule()

        planner = _planner(targets, 4, iterations=0)
        plan = planner.schedule()

        assert planner.start_score == planner.initial_score
        assert _science(plan) == _science(greedy)


@pytest.fixture(scope="module")
def searched() -> tuple[LocalSearchPlanner, Plan, list[Pointing]]:
    targets = _targets(4)
    planner = _planner(targets, 4, iterations=400)
    return planner, planner.schedule(), targets


class TestSearch:
    def test_never_worse_than_priority_first(
        self, searched: tuple[LocalSearchPlanner, Plan, list[Pointing]]
    ) -> None:
        planner, _, _ = searched

        assert planner.score >= planner.initial_score
        assert planner.iterations == 400

    def test_improved_plan_executes_exactly_as_planned(
        self, searched: tuple[LocalSearchPlanner, Plan, list[Pointing]]
    ) -> None:
        _, plan, targets = searched

        ditl = _execute(targets, plan, 4)

        assert ditl.validate_plan_matches_execution() == []
        planned = sum(
            float(e.collection_end) - float(e.collection_begin)
            for e in plan
            if e.collection_begin is not None and e.collection_end is not None
        )
        executed = sum(hk.collection_seconds for hk in ditl.telemetry.housekeeping)
        assert executed == pytest.approx(planned)

    def test_incremental_decoding_matches_a_full_decode(self) -> None:
        targets = _targets(4)
        planner = _planner(targets, 4, iterations=200)
        planner.schedule()
        best = planner._best

        redecoded = planner._decode(best.sequence)

        assert planner._objective(redecoded.final) == planner._objective(best.final)
        assert [
            (b.entry.obsid, b.entry.begin, b.end) for b in redecoded.final.timeline
        ] == [(b.entry.obsid, b.entry.begin, b.end) for b in best.final.timeline]

    def test_same_seed_and_budget_give_the_same_plan(self) -> None:
        first = _planner(_targets(3), 3, iterations=150).schedule()
        second = _planner(_targets(3), 3, iterations=150).schedule()

        assert _science(first) == _science(second)

    def test_inputs_are_not_modified(self) -> None:
        targets = _targets(3)

        _planner(targets, 3, iterations=100).schedule()

        assert all(t.exptime == 60 * MIN for t in targets)


class TestShortWindowRequest:
    def test_search_fits_the_request_priority_order_lost(self) -> None:
        """Priority order fills B's only slot with A; the search fits both."""
        config = _config(3)
        flexible = _target(config, 1, 105.0, 10.0, merit=100, minutes=60)
        urgent = _target(
            config, 2, 110.0, 10.0, merit=70, minutes=30, deadline=T0 + 30 * MIN
        )

        planner = LocalSearchPlanner(
            config,
            [flexible, urgent],
            BEGIN,
            BEGIN + timedelta(hours=3),
            max_iterations=200,
            seed=1,
        )
        plan = planner.schedule()

        assert [obsid for obsid, _, _ in _science(plan)] == [2, 1]
        assert planner.unplaced == []
        assert planner.score > planner.initial_score


class TestEarliness:
    def test_deadline_request_moves_ahead_at_no_cost_in_science(self) -> None:
        """Equal science either way, so the deadline request should start first."""
        config = _config(3)
        filler = _target(config, 1, 105.0, 10.0, merit=60, minutes=20)
        deadline = _target(
            config, 2, 110.0, 10.0, merit=50, minutes=20, deadline=T0 + 2 * 3600
        )

        planner = LocalSearchPlanner(
            config,
            [filler, deadline],
            BEGIN,
            BEGIN + timedelta(hours=3),
            max_iterations=300,
            seed=1,
        )
        plan = planner.schedule()

        assert [obsid for obsid, _, _ in _science(plan)] == [2, 1]
        assert planner.score > planner.initial_score

    def test_late_snapshot_still_counts(self) -> None:
        """A snapshot starting at its deadline keeps part of its value."""
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, merit=50, deadline=T0 + 3600)
        planner = LocalSearchPlanner(
            config,
            [target],
            BEGIN,
            BEGIN + timedelta(hours=2),
            max_iterations=0,
            earliness_weight=1.0,
        )

        planner.schedule()

        assert planner.score[0] > 0


class TestArguments:
    @pytest.mark.parametrize(
        "options",
        [
            {"time_limit": -1.0},
            {"neighborhood": 0},
            {"history_length": 0},
            {"earliness_weight": 1.5},
        ],
    )
    def test_rejects_invalid_search_settings(
        self, options: dict[str, float | int]
    ) -> None:
        config = _config(1)

        with pytest.raises(ValueError):
            LocalSearchPlanner(
                config,
                [],
                BEGIN,
                BEGIN + timedelta(hours=1),
                **options,  # type: ignore[arg-type]
            )


class TestRollingHorizon:
    def test_rolling_replanning_uses_the_local_search_planner(self) -> None:
        targets = _targets(3)
        ditl = RollingHorizonDITL(
            targets[0].config,  # type: ignore[arg-type]
            targets,
            begin=BEGIN,
            end=BEGIN + timedelta(hours=3),
            replan_interval=timedelta(hours=1),
            planner=LocalSearchPlanner,
            planner_options={"time_limit": 600, "max_iterations": 100, "seed": 1},
        )
        ditl.step_size = 60

        assert ditl.calc()

        assert len(ditl.replans) == 3
        assert ditl.validate_plan_matches_execution() == []
