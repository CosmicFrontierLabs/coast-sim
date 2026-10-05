"""PriorityPlanner: plans must execute in DITL exactly as planned."""

from datetime import timedelta
from unittest.mock import patch

import pytest

from conops import DITL
from conops.common import ObsType
from conops.config import MissionConfig, TargetConfig
from conops.config.observation_categories import ObservationCategory
from conops.schedulers import PriorityPlanner
from conops.targets import Plan, PlanEntry

from .planning_scenario import (
    BEGIN,
    CONSTRAINED,
    HOUR,
    MIN,
    PATCH,
    T0,
)
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target


def _execute(config: MissionConfig, plan: Plan, hours: int, step: int) -> DITL:
    """Execute a serialized copy of the plan in DITL."""
    replay = Plan.model_validate_json(plan.model_dump_json())
    ditl = DITL(
        config=config,
        ephem=config.constraint.ephem,
        plan=replay,
        begin=BEGIN,
        end=BEGIN + timedelta(hours=hours),
    )
    ditl.step_size = step
    assert ditl.calc()
    return ditl


def _science(plan: Plan) -> list[PlanEntry]:
    return [entry for entry in plan if entry.obstype != ObsType.GSP]


def _planned_collection(plan: Plan) -> float:
    return sum(
        float(e.collection_end) - float(e.collection_begin)
        for e in _science(plan)
        if e.collection_begin is not None and e.collection_end is not None
    )


class TestExecution:
    def test_plan_executes_exactly_as_planned(self) -> None:
        hours = 4
        config = _config(hours)
        targets = [
            _target(config, 100 + i, ra, dec, merit=90 - i, minutes=40, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH + CONSTRAINED)
        ]
        planner = PriorityPlanner(
            config, targets, BEGIN, BEGIN + timedelta(hours=hours)
        )

        plan = planner.schedule()
        ditl = _execute(config, plan, hours, planner.ctx.step_size)

        assert len(_science(plan)) >= 5
        assert ditl.validate_plan_matches_execution() == []
        executed = sum(hk.collection_seconds for hk in ditl.telemetry.housekeeping)
        assert executed == pytest.approx(_planned_collection(plan))

    def test_plan_with_ground_passes_executes_exactly_as_planned(self) -> None:
        hours = 12
        config = _config(hours, stations=True)
        targets = [
            _target(config, 100 + i, ra, dec, merit=90 - i, minutes=60, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH + CONSTRAINED)
        ]
        planner = PriorityPlanner(
            config, targets, BEGIN, BEGIN + timedelta(hours=hours)
        )

        plan = planner.schedule()
        ditl = _execute(config, plan, hours, planner.ctx.step_size)

        assert any(entry.obstype == ObsType.GSP for entry in plan)
        assert ditl.validate_plan_matches_execution() == []
        executed = sum(hk.collection_seconds for hk in ditl.telemetry.housekeeping)
        assert executed == pytest.approx(_planned_collection(plan))

    def test_plan_entries_start_on_simulation_steps(self) -> None:
        config = _config(2)
        targets = [
            _target(config, 100 + i, ra, dec) for i, (ra, dec) in enumerate(PATCH)
        ]
        planner = PriorityPlanner(config, targets, BEGIN, BEGIN + timedelta(hours=2))

        plan = planner.schedule()

        assert all((entry.begin - T0) % planner.ctx.step_size == 0 for entry in plan)


class TestOrdering:
    def test_higher_merit_is_placed_first(self) -> None:
        config = _config(2)
        low = _target(config, 1, 105.0, 10.0, merit=10)
        high = _target(config, 2, 110.0, 10.0, merit=90)

        plan = PriorityPlanner(
            config, [low, high], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        assert [e.obsid for e in _science(plan)] == [2, 1]
        assert [e.merit for e in _science(plan)] == [90.0, 10.0]

    def test_tier_outranks_merit(self) -> None:
        categories = [ObservationCategory(name="Cal", obsid_min=1, obsid_max=2, tier=1)]
        config = _config(2, categories=categories)
        calibration = _target(config, 1, 105.0, 10.0, merit=1)
        science = _target(config, 2, 110.0, 10.0, merit=90)

        plan = PriorityPlanner(
            config, [science, calibration], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        assert [e.obsid for e in _science(plan)] == [1, 2]


class TestSnapshots:
    def test_exposure_is_split_into_snapshots(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, minutes=40, snapshot=20)

        plan = PriorityPlanner(
            config, [target], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        snapshots = _science(plan)
        assert len(snapshots) == 2
        assert all(entry.exposure == 20 * MIN for entry in snapshots)

    def test_remainder_below_minimum_snapshot_is_dropped(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, minutes=25, snapshot=20, ss_min=10)

        planner = PriorityPlanner(config, [target], BEGIN, BEGIN + timedelta(hours=2))
        plan = planner.schedule()

        assert [entry.exposure for entry in _science(plan)] == [20 * MIN]
        assert planner.unplaced == []

    def test_inputs_are_not_modified(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, minutes=40, snapshot=20)

        PriorityPlanner(config, [target], BEGIN, BEGIN + timedelta(hours=2)).schedule()

        assert target.exptime == 40 * MIN
        assert not target.done


class TestDeadlinesAndConstraints:
    def test_collection_starts_by_the_deadline(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, deadline=T0 + 30 * MIN)

        plan = PriorityPlanner(
            config, [target], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        (entry,) = _science(plan)
        assert entry.collection_begin is not None
        assert entry.collection_begin <= T0 + 30 * MIN

    def test_priority_order_can_cost_a_short_window_request(self) -> None:
        """Placed first, a flexible high-merit request takes the slot B needed."""
        config = _config(3)
        flexible = _target(config, 1, 105.0, 10.0, merit=100, minutes=60)
        urgent = _target(
            config, 2, 110.0, 10.0, merit=70, minutes=30, deadline=T0 + 30 * MIN
        )

        planner = PriorityPlanner(
            config, [flexible, urgent], BEGIN, BEGIN + timedelta(hours=3)
        )
        plan = planner.schedule()

        assert [e.obsid for e in _science(plan)] == [1]
        assert planner.unplaced == [urgent]

    def test_urgency_orders_the_short_window_request_first(self) -> None:
        config = _config(3, weights=TargetConfig(urgency_weight=50))
        flexible = _target(config, 1, 105.0, 10.0, merit=100, minutes=60)
        urgent = _target(
            config, 2, 110.0, 10.0, merit=70, minutes=30, deadline=T0 + 30 * MIN
        )

        planner = PriorityPlanner(
            config, [flexible, urgent], BEGIN, BEGIN + timedelta(hours=3)
        )
        plan = planner.schedule()

        assert [e.obsid for e in _science(plan)] == [2, 1]
        assert planner.unplaced == []

    def test_target_never_clear_of_the_sun_is_unplaced(self) -> None:
        config = _config(2)
        near_sun = _target(config, 1, 216.0, -14.5)

        planner = PriorityPlanner(config, [near_sun], BEGIN, BEGIN + timedelta(hours=2))
        plan = planner.schedule()

        assert _science(plan) == []
        assert planner.unplaced == [near_sun]


class TestLockedEntries:
    def test_locked_collection_window_is_kept_and_executes(self) -> None:
        hours = 3
        config = _config(hours)
        locked = PlanEntry(
            config=config,
            name="Locked",
            ra=115.0,
            dec=10.0,
            roll=0.0,
            obsid=900,
            obstype=ObsType.AT,
            begin=T0 + HOUR,
            end=T0 + HOUR + 20 * MIN,
            collection_begin=T0 + HOUR,
            collection_end=T0 + HOUR + 20 * MIN,
            ss_min=60,
        )
        others = [
            _target(config, 100 + i, ra, dec, minutes=40)
            for i, (ra, dec) in enumerate(PATCH)
        ]

        planner = PriorityPlanner(
            config, others, BEGIN, BEGIN + timedelta(hours=hours), locked=[locked]
        )
        plan = planner.schedule()

        kept = next(entry for entry in plan if entry.obsid == 900)
        assert (kept.collection_begin, kept.collection_end) == (
            T0 + HOUR,
            T0 + HOUR + 20 * MIN,
        )
        ditl = _execute(config, plan, hours, planner.ctx.step_size)
        assert ditl.validate_plan_matches_execution() == []

    def test_locked_entry_needs_a_collection_window(self) -> None:
        config = _config(1)
        locked = PlanEntry(config=config, obsid=900, begin=T0, end=T0 + 600)

        with pytest.raises(ValueError, match="collection window"):
            PriorityPlanner(
                config, [], BEGIN, BEGIN + timedelta(hours=1), locked=[locked]
            ).schedule()


class TestSuccessorRetries:
    """A limit on later starts tried in a gap whose successor is unreachable."""

    HOURS = 12

    def _plan(self, retries: int | None) -> tuple[MissionConfig, PriorityPlanner, int]:
        config = _config(self.HOURS, stations=True)
        targets = [
            _target(config, 100 + i, ra, dec, merit=90 - 3 * i, minutes=60, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH + CONSTRAINED)
        ]
        planner = PriorityPlanner(
            config,
            targets,
            BEGIN,
            BEGIN + timedelta(hours=self.HOURS),
            successor_retries=retries,
        )
        with patch.object(planner, "_connect", wraps=planner._connect) as connect:
            planner.schedule()
        return config, planner, connect.call_count

    def test_a_limit_searches_less_and_still_executes_exactly(self) -> None:
        _, exhaustive, exhaustive_calls = self._plan(None)
        config, limited, limited_calls = self._plan(0)

        assert limited_calls < exhaustive_calls
        assert len(limited.plan) > 0
        ditl = _execute(config, limited.plan, self.HOURS, 60)
        assert ditl.validate_plan_matches_execution() == []

    def test_rejects_a_negative_limit(self) -> None:
        config = _config(1)

        with pytest.raises(ValueError, match="successor_retries"):
            PriorityPlanner(
                config, [], BEGIN, BEGIN + timedelta(hours=1), successor_retries=-1
            )
