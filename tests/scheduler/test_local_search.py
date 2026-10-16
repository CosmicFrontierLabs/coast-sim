"""LocalSearchPlanner: improve priority-first plans without breaking execution."""

from datetime import timedelta

import numpy as np
import pytest

from conops import DITL
from conops.common import ObsType
from conops.config import MissionConfig, TargetConfig
from conops.config.observation_categories import ObservationCategory
from conops.ditl import RollingHorizonDITL
from conops.schedulers import LocalSearchPlanner, PriorityPlanner
from conops.schedulers.priority_planner import program_shares
from conops.targets import Plan, PlanEntry, Pointing

from .planning_scenario import BEGIN, CONSTRAINED, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

MONITOR = ObservationCategory(
    name="Monitor", obsid_min=1, obsid_max=2, cadence_seconds=HOUR
)


def _targets(hours: int, config: MissionConfig | None = None) -> list[Pointing]:
    config = config if config is not None else _config(hours)
    return [
        _target(config, 100 + i, ra, dec, merit=90 - 7 * i, minutes=60, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH + CONSTRAINED)
    ]


def _random_targets(config: MissionConfig) -> list[Pointing]:
    """Targets whose snapshots straddle Earth occultations, so the priority-first
    planner moves snapshots later to fill the time a target is hidden."""
    rng = np.random.default_rng(5)
    return [
        _target(
            config,
            100 + i,
            rng.uniform(0, 360),
            rng.uniform(-90, 90),
            merit=rng.uniform(10, 100),
            minutes=2000 / 60,
            snapshot=1000 / 60,
            ss_min=5,
        )
        for i in range(40)
    ]


def _monitored(config: MissionConfig) -> list[Pointing]:
    """A cadence target wanting three ten-minute visits, and filler."""
    monitor = _target(config, 1, *PATCH[0], merit=60, minutes=30, snapshot=10)
    filler = [
        _target(config, 10 + i, ra, dec, merit=40, minutes=60, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH[1:])
    ]
    return [monitor, *filler]


def _science(plan: Plan) -> list[tuple[int, float, float]]:
    return [
        (int(e.obsid), float(e.begin), float(e.end))
        for e in plan
        if e.obstype != ObsType.GSP
    ]


def _visits(plan: Plan, obsid: int) -> list[tuple[float, float]]:
    return sorted(
        (float(e.collection_begin), float(e.collection_end))
        for e in plan
        if e.obsid == obsid
        and e.collection_begin is not None
        and e.collection_end is not None
    )


def _collection_seconds(entry: PlanEntry) -> float:
    return entry.collection_seconds_between(entry.begin, entry.end)


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
    targets: list[Pointing],
    hours: int,
    iterations: int,
    **options: float | dict[int, float] | list[PlanEntry],
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
        **options,  # type: ignore[arg-type]
    )


def _priority_plan(targets: list[Pointing], hours: int) -> Plan:
    config = targets[0].config
    assert config is not None
    return PriorityPlanner(
        config, targets, BEGIN, BEGIN + timedelta(hours=hours)
    ).schedule()


class TestStartingPoint:
    def test_without_search_the_plan_is_the_priority_first_plan(self) -> None:
        targets = _targets(4)
        greedy = _priority_plan(targets, 4)

        planner = _planner(targets, 4, iterations=0)
        plan = planner.schedule()

        assert planner.start_score == planner.initial_score
        assert _science(plan) == _science(greedy)
        assert plan.metadata is not None
        assert plan.metadata["planner"] == "local_search"

    def test_snapshots_the_priority_first_planner_moved_start_where_it_left_them(
        self,
    ) -> None:
        """Moved, and sometimes cut short, they still decode to the same plan.

        Where the priority-first planner fitted a snapshot before another, the
        slew to that other one starts as late as it can, and may arrive before
        its collection starts. The decoder starts that collection on arrival,
        so it can start a little earlier, for the same length.
        """
        hours = 12
        config = _config(hours)
        greedy = PriorityPlanner(
            config, _random_targets(config), BEGIN, BEGIN + timedelta(hours=hours)
        )
        greedy.schedule()
        assert any("shorter" in e.description for e in greedy.log.events)

        planner = _planner(_random_targets(config), hours, iterations=0)
        plan = planner.schedule()

        def snapshots(plan: Plan) -> list[tuple[int, float, float]]:
            return [
                (int(e.obsid), float(e.begin), _collection_seconds(e))
                for e in plan
                if e.obstype != ObsType.GSP
            ]

        assert planner.start_score == planner.initial_score
        assert snapshots(plan) == snapshots(greedy.plan)
        for decoded, placed in zip(plan, greedy.plan):
            assert decoded.collection_begin is not None
            assert placed.collection_begin is not None
            assert decoded.collection_begin <= placed.collection_begin

    def test_with_slews_before_targets_rise_the_start_is_the_priority_plan(
        self,
    ) -> None:
        config = _config(4)
        config.spacecraft_bus.attitude_control.slew_before_visible = True
        greedy = _priority_plan(_targets(4, config), 4)

        planner = _planner(_targets(4, config), 4, iterations=0)
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
        executed = sum(
            hk.collection_seconds or 0.0 for hk in ditl.telemetry.housekeeping
        )
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

    def test_with_slews_before_targets_rise_the_plan_executes_exactly(self) -> None:
        config = _config(4)
        config.spacecraft_bus.attitude_control.slew_before_visible = True
        targets = _targets(4, config)
        planner = _planner(targets, 4, iterations=300)

        plan = planner.schedule()

        assert planner.score >= planner.initial_score
        ditl = _execute(targets, plan, 4)
        assert ditl.validate_plan_matches_execution() == []


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


class TestDynamicMerit:
    """The search keeps the cadence and program-share rules of the
    priority-first planner."""

    HOURS = 4

    def _monitor_config(self) -> MissionConfig:
        return _config(
            self.HOURS,
            weights=TargetConfig(cadence_weight=50),
            categories=[MONITOR],
        )

    def test_cadence_visits_stay_apart(self) -> None:
        config = self._monitor_config()
        targets = _monitored(config)

        plan = _planner(targets, self.HOURS, iterations=300).schedule()

        visits = _visits(plan, 1)
        assert len(visits) >= 2
        for (_, end), (begin, _) in zip(visits, visits[1:]):
            assert begin >= end + HOUR
        ditl = _execute(targets, plan, self.HOURS)
        assert ditl.validate_plan_matches_execution() == []

    def test_committed_visits_delay_the_next(self) -> None:
        config = self._monitor_config()
        targets = _monitored(config)

        plan = _planner(
            targets,
            self.HOURS,
            iterations=300,
            reserved_visits={1: T0 + 20 * MIN},
        ).schedule()

        first, _ = _visits(plan, 1)[0]
        assert first >= T0 + 80 * MIN

    def test_cadence_visits_stay_clear_of_a_locked_visit(self) -> None:
        config = self._monitor_config()
        lock_start = T0 + 90 * MIN
        locked = PlanEntry(
            config=config,
            name="locked",
            ra=PATCH[0][0],
            dec=PATCH[0][1],
            roll=0.0,
            obsid=1,
            obstype=ObsType.AT,
            begin=lock_start,
            end=lock_start + 10 * MIN,
            collection_begin=lock_start,
            collection_end=lock_start + 10 * MIN,
            ss_min=60,
        )

        plan = _planner(
            _monitored(config), self.HOURS, iterations=300, locked=[locked]
        ).schedule()

        planned = [
            v for v in _visits(plan, 1) if v != (lock_start, lock_start + 10 * MIN)
        ]
        assert planned
        for begin, end in planned:
            assert end + HOUR <= lock_start or begin >= lock_start + 10 * MIN + HOUR

    def test_entries_carry_their_value_where_they_are_placed(self) -> None:
        """Each entry's merit counts the program shares the plan delivers before
        it, so a Target of Opportunity must beat that value to interrupt it."""
        config = _config(
            self.HOURS,
            weights=TargetConfig(completion_deficit_weight=100),
            categories=[
                ObservationCategory(
                    name="A", obsid_min=100, obsid_max=200, time_share=0.5
                ),
                ObservationCategory(
                    name="B", obsid_min=200, obsid_max=300, time_share=0.5
                ),
            ],
        )
        a = [
            _target(config, 100 + i, ra, dec, merit=100, minutes=120, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH[:3])
        ]
        b = [
            _target(config, 200 + i, ra, dec, merit=60, minutes=120, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH[2:])
        ]
        planner = _planner([*a, *b], self.HOURS, iterations=300)

        plan = planner.schedule()

        programs: dict[str, float] = {}
        merits = []
        for entry in sorted(
            (e for e in plan if e.obstype != ObsType.GSP), key=lambda e: e.begin
        ):
            request = planner._by_obsid[int(entry.obsid)]
            expected = planner._value(request, program_shares(programs))
            merits.append(float(entry.merit))
            assert float(entry.merit) == pytest.approx(expected)
            program = request.category.program_name
            programs[program] = programs.get(
                program, 0.0
            ) + entry.collection_seconds_between(entry.begin, entry.end)
        # Not all at the value they had at the start of the horizon.
        assert len(set(merits)) > 2


class TestArguments:
    @pytest.mark.parametrize(
        "options",
        [
            {"time_limit": -1.0},
            {"max_iterations": -1},
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
