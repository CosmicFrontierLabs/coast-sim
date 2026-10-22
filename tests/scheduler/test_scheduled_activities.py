"""Scheduled activities: fixed-time blocks with no science, reserved by the
planners and executed by both simulations."""

from datetime import timedelta

import pytest

from conops import DITL, create_ditl
from conops.common import ObsType
from conops.common.enums import ACSMode
from conops.config import MissionConfig
from conops.ditl import RollingHorizonDITL
from conops.schedulers import PriorityScheduler
from conops.targets import (
    ACTIVITY_OBSID,
    Plan,
    Pointing,
    ScheduledActivity,
    assign_obsids,
)

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 8
STEP = 60
END = BEGIN + timedelta(hours=HOURS)
CALIBRATION = ScheduledActivity(
    name="Lamp calibration", begin=T0 + 2 * HOUR, duration_seconds=30 * MIN
)
MAINTENANCE = ScheduledActivity(
    name="Maintenance", begin=T0 + 5 * HOUR + 10 * MIN, duration_seconds=15 * MIN
)
ACTIVITIES = [MAINTENANCE, CALIBRATION]


def _targets(config: MissionConfig) -> list[Pointing]:
    return [
        _target(config, 100 + i, ra, dec, merit=90 - i, minutes=200, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH)
    ]


def _science(plan: Plan | list) -> list:
    return [e for e in plan if e.obstype not in (ObsType.GSP, ObsType.ACTIVITY)]


def _activities(plan: Plan | list) -> list:
    return [e for e in plan if e.obstype == ObsType.ACTIVITY]


def _overlaps(entry, activity: ScheduledActivity) -> bool:
    return float(entry.begin) < activity.end and float(entry.end) > activity.begin


def _check_execution(ditl) -> None:
    """No science, contact or charge mode while an activity is under way, and
    the activities are in the plan with their times."""
    entries = _activities(ditl.plan)
    assert [e.name for e in entries] == ["Lamp calibration", "Maintenance"]
    for entry, activity in zip(entries, sorted(ACTIVITIES, key=lambda a: a.begin)):
        assert float(entry.end) == activity.end
        assert float(entry.begin) + entry.slewtime <= activity.begin
        assert entry.obsid == activity.obsid or entry.obsid >= ACTIVITY_OBSID
        for hk in ditl.telemetry.housekeeping:
            t = hk.timestamp.timestamp()
            if activity.begin <= t < activity.end:
                assert hk.acs_mode not in (
                    ACSMode.SCIENCE,
                    ACSMode.PASS,
                    ACSMode.CHARGING,
                ), (activity.name, t, hk.acs_mode)
    for entry in _science(ditl.plan):
        for activity in ACTIVITIES:
            assert not _overlaps(entry, activity), (entry.obsid, activity.name)
    events = [e for e in ditl.log.events if e.event_type == "ACTIVITY"]
    assert [e.description for e in events] == [
        "Scheduled activity Lamp calibration begins",
        "Scheduled activity Lamp calibration ends",
        "Scheduled activity Maintenance begins",
        "Scheduled activity Maintenance ends",
    ]


class TestModel:
    def test_obsids_are_assigned_in_time_order(self) -> None:
        assigned = assign_obsids(ACTIVITIES)
        assert [a.name for a in assigned] == ["Lamp calibration", "Maintenance"]
        assert [a.obsid for a in assigned] == [ACTIVITY_OBSID, ACTIVITY_OBSID + 1]
        given = ScheduledActivity(name="X", begin=T0, duration_seconds=60, obsid=7)
        assert assign_obsids([given, CALIBRATION])[0].obsid == 7
        assert CALIBRATION.obsid is None

    def test_begin_accepts_a_datetime(self) -> None:
        activity = ScheduledActivity(
            name="X", begin=BEGIN + timedelta(hours=1), duration_seconds=60
        )
        assert activity.begin == T0 + HOUR
        assert activity.end == T0 + HOUR + 60

    def test_a_duration_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ScheduledActivity(name="X", begin=T0, duration_seconds=0)

    def test_the_plan_entry(self) -> None:
        activity = assign_obsids([CALIBRATION])[0]
        entry = activity.plan_entry(
            None, (10.0, 20.0, 30.0), begin=activity.begin - 120
        )
        assert entry.obstype == ObsType.ACTIVITY
        assert (entry.ra, entry.dec, entry.roll) == (10.0, 20.0, 30.0)
        assert (float(entry.begin), float(entry.end)) == (
            activity.begin - 120,
            activity.end,
        )
        assert entry.name == "Lamp calibration" and entry.obsid == ACTIVITY_OBSID
        assert entry.collection_begin is None


class TestPlanning:
    def test_the_plan_reserves_the_activities_and_executes(self) -> None:
        config = _config(HOURS)
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=STEP, activities=ACTIVITIES
        )
        plan = scheduler.schedule()
        entries = _activities(plan)
        assert [e.name for e in entries] == ["Lamp calibration", "Maintenance"]
        for entry, activity in zip(entries, sorted(ACTIVITIES, key=lambda a: a.begin)):
            assert float(entry.end) == activity.end
            assert float(entry.begin) + entry.slewtime <= activity.begin
        science = _science(plan)
        assert len(science) >= 6
        for entry in science:
            for activity in ACTIVITIES:
                assert not _overlaps(entry, activity)
        # Science on both sides of each activity.
        assert any(float(e.end) <= CALIBRATION.begin for e in science)
        assert any(
            CALIBRATION.end <= float(e.begin) < MAINTENANCE.begin for e in science
        )
        # The activity holds in IDLE, which the battery ledger counts as a hold.
        assert scheduler._battery.addition(entries[0]) is not None
        replay = Plan.model_validate_json(plan.model_dump_json())
        ditl = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=replay,
            begin=BEGIN,
            end=END,
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        _check_execution(ditl)

    def test_an_activity_outside_the_horizon_is_ignored(self) -> None:
        config = _config(HOURS)
        later = ScheduledActivity(
            name="Later", begin=T0 + 20 * HOUR, duration_seconds=600
        )
        plan = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=STEP, activities=[later]
        ).schedule()
        assert _activities(plan) == []

    def test_without_activities_nothing_changes(self) -> None:
        config = _config(HOURS)
        plain = PriorityScheduler(config, _targets(config), BEGIN, END, step_size=STEP)
        plan = plain.schedule()
        assert _activities(plan) == [] and plain.activities == []


class TestDispatch:
    def test_dispatch_holds_through_the_activities(self) -> None:
        config = _config(HOURS)
        ditl = create_ditl(config, _targets(config), BEGIN, END, activities=ACTIVITIES)
        ditl.step_size = STEP
        assert ditl.calc()
        _check_execution(ditl)
        # Dispatch holds the attitude it has: the activity entries start on time.
        for entry, activity in zip(
            _activities(ditl.plan), sorted(ACTIVITIES, key=lambda a: a.begin)
        ):
            assert float(entry.begin) == activity.begin
            assert entry.slewtime == 0
        # No observation was under way: the next activity is a deadline, so
        # none starts that would run into it.
        science = _science(ditl.plan)
        assert any(float(e.begin) >= CALIBRATION.end for e in science)
        # No observation runs into an activity: the next activity is a deadline.
        for entry in science:
            assert (
                float(entry.end) <= CALIBRATION.begin
                or float(entry.begin) >= CALIBRATION.end
            )


class TestRolling:
    def test_each_plan_reserves_the_activities(self) -> None:
        config = _config(HOURS)
        ditl = RollingHorizonDITL(
            config,
            _targets(config),
            begin=BEGIN,
            end=END,
            horizon=timedelta(hours=3),
            replan_interval=timedelta(hours=2),
            activities=ACTIVITIES,
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        _check_execution(ditl)
