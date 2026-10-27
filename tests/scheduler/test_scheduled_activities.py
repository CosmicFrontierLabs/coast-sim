"""Scheduled activities: fixed-time blocks with no science, reserved by the
planners and executed by both simulations."""

from datetime import timedelta
from unittest.mock import patch

import pytest

from conops import DITL, QueueDITL, create_ditl
from conops.common import ObsType
from conops.common.enums import ACSMode
from conops.config import MissionConfig
from conops.ditl import RollingHorizonDITL
from conops.schedulers import LocalSearchScheduler, PriorityScheduler, SchedulingContext
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


def _check_following(plan: Plan | list, config: MissionConfig) -> None:
    """Each activity entry holds the attitude of the entry before it, with
    no slew of its own, and IDLE can hold that attitude through it."""
    entries = sorted(plan, key=lambda e: float(e.begin))
    ctx = SchedulingContext(config, BEGIN, END, step_size=STEP)
    for index, entry in enumerate(entries):
        if entry.obstype != ObsType.ACTIVITY:
            continue
        assert entry.slewtime == 0 and entry.slewdist == 0
        attitude = (float(entry.ra), float(entry.dec), float(entry.roll))
        if index > 0:
            before = entries[index - 1]
            assert before.obstype != ObsType.ACTIVITY
            assert attitude == pytest.approx(
                (float(before.ra), float(before.dec), float(before.roll))
            )
        assert (
            ctx.first_hold_violation(
                attitude, float(entry.begin), float(entry.end), ACSMode.IDLE
            )
            is None
        )


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

    def test_an_activity_takes_the_attitude_of_what_precedes_it(self) -> None:
        """The science before each activity ends at the block with no slew
        back to a reserved attitude: the activity holds the attitude that
        science leaves, and the science after it slews from there."""
        config = _config(HOURS)
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=STEP, activities=ACTIVITIES
        )
        plan = scheduler.schedule()
        _check_following(plan, config)
        entries = sorted(plan, key=lambda e: float(e.begin))
        for activity in ACTIVITIES:
            entry = next(e for e in entries if e.name == activity.name)
            assert float(entry.begin) == activity.begin
            before = entries[entries.index(entry) - 1]
            # Science right up to the block, less a snapshot that did not fit.
            assert float(before.end) <= activity.begin
            assert (before.ra, before.dec) != pytest.approx(
                scheduler.ctx.initial_attitude(scheduler.ctx.ustart)[:2]
            )
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

    def test_local_search_keeps_the_activities_following(self) -> None:
        config = _config(HOURS)
        scheduler = LocalSearchScheduler(
            config,
            _targets(config),
            BEGIN,
            END,
            step_size=STEP,
            activities=ACTIVITIES,
            time_limit=5.0,
        )
        plan = scheduler.schedule()
        assert [e.name for e in _activities(plan)] == [
            "Lamp calibration",
            "Maintenance",
        ]
        _check_following(plan, config)
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

    def test_an_activity_outside_the_horizon_is_ignored(self) -> None:
        config = _config(HOURS)
        later = ScheduledActivity(
            name="Later", begin=T0 + 20 * HOUR, duration_seconds=600
        )
        plan = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=STEP, activities=[later]
        ).schedule()
        assert _activities(plan) == []

    def test_a_sliding_activity_due_at_the_horizon_end_is_left_out(self) -> None:
        """A sliding block is reserved from its latest start; at or past the
        horizon's end that reservation would have no length, and the plan
        leaves the block out rather than holding a block of no length."""
        config = _config(HOURS)
        at_the_end = ScheduledActivity(
            name="Late calibration",
            begin=T0 + (HOURS - 2) * HOUR,
            duration_seconds=1800,
            slide_seconds=3 * HOUR,
        )
        scheduler = PriorityScheduler(
            config,
            _targets(config),
            BEGIN,
            END,
            step_size=STEP,
            activities=[at_the_end],
        )
        plan = scheduler.schedule()
        assert _activities(plan) == []
        assert any(
            "at or past the end of the horizon" in e.description
            for e in scheduler.log.events
        )
        # Within the horizon, the same block is reserved and settles early.
        inside = at_the_end.model_copy(update={"slide_seconds": HOUR})
        plan = PriorityScheduler(
            config,
            _targets(config),
            BEGIN,
            END,
            step_size=STEP,
            activities=[inside],
        ).schedule()
        [entry] = _activities(plan)
        assert float(entry.end) - float(entry.begin) == 1800

    def test_cpsat_leaves_the_block_at_the_horizon_end_out(self) -> None:
        pytest.importorskip("ortools")
        from conops.schedulers import CpSatScheduler

        config = _config(HOURS)
        at_the_end = ScheduledActivity(
            name="Late calibration",
            begin=T0 + (HOURS - 2) * HOUR,
            duration_seconds=1800,
            slide_seconds=3 * HOUR,
        )
        scheduler = CpSatScheduler(
            config,
            _targets(config),
            BEGIN,
            END,
            step_size=STEP,
            activities=[at_the_end],
            time_limit=5.0,
        )
        plan = scheduler.schedule()
        assert _activities(plan) == []
        assert _science(plan)

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


def _slow_slew_run(
    activities: list[ScheduledActivity],
) -> tuple[QueueDITL, list[Pointing]]:
    """Two targets with slews of some minutes between them, dispatched."""
    config = _config(6)
    acs = config.spacecraft_bus.attitude_control
    acs.max_slew_rate = 0.02
    acs.slew_acceleration = 0.01
    targets = [
        _target(config, 1, *PATCH[0], merit=90.0, minutes=60, snapshot=60),
        _target(config, 2, *PATCH[4], merit=80.0, minutes=60, snapshot=60),
    ]
    ditl = create_ditl(
        config, targets, BEGIN, BEGIN + timedelta(hours=6), activities=activities
    )
    assert isinstance(ditl, QueueDITL)
    ditl.step_size = STEP
    assert ditl.calc()
    return ditl, targets


def _lamp_after_first(**fields: bool) -> ScheduledActivity:
    """A half-hour activity due two minutes after the first target's
    observation ends, inside the slew to the second target."""
    ditl, _ = _slow_slew_run([])
    first = next(e for e in ditl.plan if e.obsid == 1)
    return ScheduledActivity(
        name="lamp",
        begin=float(first.end) + 2 * MIN,
        duration_seconds=30 * MIN,
        **fields,
    )


class TestSlewsIntoActivities:
    def test_no_slew_is_admitted_that_crosses_the_activity_start(self) -> None:
        """The second target's slew would end inside the block, so it is
        passed over: the block begins on time at the first target's
        attitude, with no slewing inside, and the target follows it."""
        lamp = _lamp_after_first()
        ditl, _ = _slow_slew_run([lamp])
        first = next(e for e in ditl.plan if e.obsid == 1)
        activity = next(e for e in ditl.plan if e.obstype == ObsType.ACTIVITY)
        second = next(e for e in ditl.plan if e.obsid == 2)
        # The activity begins on the first step from its start, as always.
        assert lamp.begin <= float(activity.begin) < lamp.begin + STEP
        assert float(activity.end) == lamp.end
        assert (activity.ra, activity.dec) == pytest.approx((first.ra, first.dec))
        assert float(second.begin) >= lamp.end
        modes = {
            int(hk.acs_mode)
            for hk in ditl.telemetry.housekeeping
            if lamp.begin <= hk.timestamp.timestamp() < lamp.end
        }
        assert modes == {int(ACSMode.IDLE)}

    def test_a_slew_under_way_delays_the_activity_which_keeps_its_length(
        self,
    ) -> None:
        """With the activity deadline disabled, the second target's slew is
        admitted and is under way when the activity is due. The activity
        waits for the slew's end, runs its full length from there, and holds
        the attitude the slew reached."""
        lamp = _lamp_after_first()
        with patch.object(QueueDITL, "_next_activity_begin", return_value=None):
            ditl, _ = _slow_slew_run([lamp])
        second = next(e for e in ditl.plan if e.obsid == 2)
        activity = next(e for e in ditl.plan if e.obstype == ObsType.ACTIVITY)
        slew_end = float(second.begin) + second.slewtime
        assert float(second.begin) < lamp.begin < slew_end
        assert slew_end <= float(activity.begin) < slew_end + STEP
        assert float(activity.end) - float(activity.begin) == pytest.approx(
            lamp.duration_seconds
        )
        assert (activity.ra, activity.dec) == pytest.approx((second.ra, second.dec))
        descriptions = [
            e.description for e in ditl.log.events if e.event_type == "ACTIVITY"
        ]
        assert descriptions[0] == "Scheduled activity lamp waits for the slew under way"
        assert descriptions[1].startswith("Scheduled activity lamp begins, ")
        assert descriptions[1].endswith("s after its start")
        modes = {
            int(hk.acs_mode)
            for hk in ditl.telemetry.housekeeping
            if float(activity.begin) <= hk.timestamp.timestamp() < float(activity.end)
        }
        assert modes == {int(ACSMode.IDLE)}

    def test_a_fixed_activity_begins_at_its_start_regardless(self) -> None:
        lamp = _lamp_after_first(fixed=True)
        with patch.object(QueueDITL, "_next_activity_begin", return_value=None):
            ditl, _ = _slow_slew_run([lamp])
        activity = next(e for e in ditl.plan if e.obstype == ObsType.ACTIVITY)
        assert float(activity.begin) < lamp.begin + STEP
        assert float(activity.end) == lamp.end
        assert not any("waits for the slew" in e.description for e in ditl.log.events)


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
        _check_following(ditl.plan, config)
