"""Observation timing per category: setup, cleanup and handoff budgets that
differ between categories, used by the planners, dispatch and execution."""

from datetime import timedelta
from unittest.mock import Mock

import pytest

from conops import DITL, create_ditl
from conops.config import MissionConfig, ObservationTiming, Payload
from conops.config.observation_categories import (
    ObservationCategories,
    ObservationCategory,
)
from conops.config.observation_timing import timing_for
from conops.schedulers import PriorityScheduler, SequentialScheduler
from conops.targets import Plan, Pointing

from .planning_scenario import BEGIN, PATCH
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 6
STEP = 60
PAYLOAD = ObservationTiming(setup_seconds=60, cleanup_seconds=30, handoff_seconds=0)
SLOW = ObservationTiming(setup_seconds=600, cleanup_seconds=120, handoff_seconds=60)
FAST = ObservationTiming(setup_seconds=0, cleanup_seconds=0, handoff_seconds=0)


def _timed_config(hours: int = HOURS) -> MissionConfig:
    """Obsids 100 to 101 take ten minutes of setup, 102 none, the rest the payload's minute."""
    config = _config(hours)
    config.payload.observation_timing = PAYLOAD
    config.observation_categories = ObservationCategories(
        categories=[
            ObservationCategory(
                name="Slow", obsid_min=100, obsid_max=102, observation_timing=SLOW
            ),
            ObservationCategory(
                name="Fast", obsid_min=102, obsid_max=103, observation_timing=FAST
            ),
            ObservationCategory(name="Plain", obsid_min=103, obsid_max=200),
        ]
    )
    return config


def _targets(config: MissionConfig) -> list[Pointing]:
    return [
        _target(config, 100 + i, ra, dec, merit=90 - i, minutes=40, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH)
    ]


def _expected(obsid: int) -> ObservationTiming:
    if obsid in (100, 101):
        return SLOW
    if obsid == 102:
        return FAST
    return PAYLOAD


class TestLookup:
    def test_the_category_overrides_the_payload(self) -> None:
        config = _timed_config()
        assert config.observation_timing_for(100) is SLOW
        assert config.observation_timing_for(102) is FAST
        assert config.observation_timing_for(103) is PAYLOAD
        assert config.observation_timing_for(999_999) is PAYLOAD
        assert config.observation_timings() == [PAYLOAD, SLOW, FAST]
        assert timing_for(config, 101) is SLOW
        target = _target(config, 100, 105.0, 10.0)
        assert target.observation_timing is SLOW

    def test_the_bound_is_the_largest_of_each_budget(self) -> None:
        bound = ObservationTiming.bound([PAYLOAD, SLOW, FAST])
        assert (bound.setup_seconds, bound.cleanup_seconds, bound.handoff_seconds) == (
            600,
            120,
            60,
        )
        assert ObservationTiming.bound([]) == ObservationTiming()

    def test_a_test_double_gets_the_payloads_or_the_default(self) -> None:
        config = Mock()
        config.payload.observation_timing = PAYLOAD
        assert timing_for(config, 5) is PAYLOAD
        assert timing_for(object(), 5) == ObservationTiming()
        assert (
            timing_for(MissionConfig(payload=Payload(observation_timing=SLOW)), 1)
            is SLOW
        )

    def test_the_sequential_scheduler_rejects_category_budgets(self) -> None:
        config = _config(2)
        config.observation_categories = ObservationCategories(
            categories=[
                ObservationCategory(
                    name="Slow", obsid_min=100, obsid_max=200, observation_timing=SLOW
                )
            ]
        )
        with pytest.raises(ValueError, match="timing budgets"):
            SequentialScheduler(config, [], BEGIN, BEGIN + timedelta(hours=2))

    def test_round_trips_through_json(self) -> None:
        config = _timed_config()
        loaded = MissionConfig.model_validate_json(config.model_dump_json())
        assert loaded.observation_timing_for(100) == SLOW
        assert loaded.observation_timing_for(102) == FAST


def _check_entries(plan: Plan | list) -> None:
    """Every science entry's collection window reserves its own category's budgets."""
    checked = 0
    for entry in plan:
        if entry.collection_begin is None or entry.obsid < 100 or entry.obsid > 200:
            continue
        timing = _expected(int(entry.obsid))
        assert float(entry.collection_begin) == pytest.approx(
            float(entry.begin) + float(entry.slewtime) + timing.setup_seconds
        )
        assert float(entry.end) - float(entry.collection_end) == pytest.approx(
            timing.post_collection_seconds
        )
        checked += 1
    assert checked >= 4


class TestPlanning:
    def test_the_priority_plan_reserves_each_categorys_budgets_and_executes(
        self,
    ) -> None:
        config = _timed_config()
        scheduler = PriorityScheduler(
            config,
            _targets(config),
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            step_size=STEP,
        )
        plan = scheduler.schedule()
        obsids = {int(e.obsid) for e in plan}
        assert {100, 101, 102, 103} <= obsids
        _check_entries(plan)
        # The context's bound is the slowest category's.
        assert scheduler.ctx.setup_seconds == 600
        assert scheduler.ctx.setup_for(102) == 0
        replay = Plan.model_validate_json(plan.model_dump_json())
        ditl = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=replay,
            begin=BEGIN,
            end=BEGIN + timedelta(hours=HOURS),
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        # Execution collects within each entry's own window.
        for entry in ditl.plan:
            if entry.collection_begin is None or int(entry.obsid) not in (100, 102):
                continue
            seconds = sum(
                hk.collection_seconds
                for hk in ditl.telemetry.housekeeping
                if hk.obsid == entry.obsid
                and float(entry.begin) <= hk.timestamp.timestamp() < float(entry.end)
            )
            assert seconds == pytest.approx(
                float(entry.collection_end) - float(entry.collection_begin), abs=STEP
            )

    def test_dispatch_reserves_each_categorys_budgets(self) -> None:
        config = _timed_config()
        ditl = create_ditl(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        ditl.step_size = STEP
        assert ditl.calc()
        observed = {hk.obsid for hk in ditl.telemetry.housekeeping if hk.obsid}
        assert {100, 101, 102, 103} <= observed
        _check_entries(ditl.plan)
        slow = [e for e in ditl.plan if int(e.obsid) == 100 and e.collection_begin]
        fast = [e for e in ditl.plan if int(e.obsid) == 102 and e.collection_begin]
        assert slow and fast
        assert (
            float(slow[0].collection_begin) - float(slow[0].begin) - slow[0].slewtime
            == 600
        )
        assert (
            float(fast[0].collection_begin) - float(fast[0].begin) - fast[0].slewtime
            == 0
        )

    def test_without_category_budgets_nothing_changes(self) -> None:
        config = _config(HOURS)
        config.payload.observation_timing = PAYLOAD
        targets = _targets(config)
        plan = PriorityScheduler(
            config, targets, BEGIN, BEGIN + timedelta(hours=HOURS), step_size=STEP
        ).schedule()
        for entry in plan:
            if entry.collection_begin is not None:
                assert float(entry.collection_begin) == pytest.approx(
                    float(entry.begin) + float(entry.slewtime) + 60
                )
        assert any(e.collection_begin is not None for e in plan)
