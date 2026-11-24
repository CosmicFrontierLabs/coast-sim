"""A dump that waits for its fixed firing attitude, and a fixed-attitude
activity that safe mode interrupts."""

from datetime import datetime, timedelta

import pytest
from rust_ephem import EarthLimbConstraint

from conops import DITL, create_ditl
from conops.config import (
    ActivityAttitude,
    MissionConfig,
    StoredMomentumConfig,
    ThrusterDumpConfig,
)
from conops.schedulers import PriorityScheduler
from conops.targets import Plan, PlanEntry, ScheduledActivity

from .planning_scenario import BEGIN, MIN, make_config, make_target

FIRING = ActivityAttitude(kind="fixed", ra_deg=80, dec_deg=-20, roll_deg=0)
INHERITED = ActivityAttitude()
HOLD_POSE = ActivityAttitude(kind="fixed", ra_deg=150, dec_deg=20, roll_deg=0)
# The Earth's limb blocks the firing attitude when a run starts here: it is
# allowed again from about half an hour in.
BLOCKED = BEGIN + timedelta(minutes=70)


def _config(attitude: ActivityAttitude) -> MissionConfig:
    """Wheels over their planning level from the start, so a dump is called
    for at once, with the Earth's limb as the only attitude rule."""
    config = make_config(8)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.constraint.safety_constraint = EarthLimbConstraint(min_angle=10)
    bus = config.spacecraft_bus
    bus.inertia_tensor_body_kg_m2 = ((1000, 0, 0), (0, 1200, 0), (0, 0, 800))
    bus.attitude_control.stored_momentum = StoredMomentumConfig(
        gravity_gradient_enabled=True,
        capacity_n_m_s=10,
        planning_level=0.6,
        initial_momentum_body_n_m_s=(7, 0, 0),
        thruster_dump=ThrusterDumpConfig(duration_seconds=600, attitude=attitude),
    )
    return config


def _dumps(plan: list[PlanEntry], begin: datetime) -> list[int]:
    """The minute, from ``begin``, at which each dump in ``plan`` fires."""
    return [
        round((float(e.activity_hold_begin or 0.0) - begin.timestamp()) / MIN)
        for e in plan
        if e.desaturation
    ]


def _dispatch(
    attitude: ActivityAttitude,
    minutes: float,
    *,
    snapshot: int = 5,
    activities: list[ScheduledActivity] | None = None,
) -> tuple[list[int], float]:
    """Run dispatch from ``BLOCKED``; the dumps and the minutes of science."""
    config = _config(attitude)
    target = make_target(config, 1, 100, 0, minutes=600, snapshot=snapshot)
    run = create_ditl(
        config,
        [target],
        BLOCKED,
        BLOCKED + timedelta(minutes=minutes),
        activities=activities or [],
    )
    run.step_size = 60
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
    science = sum(
        float(e.collection_end - e.collection_begin)
        for e in run.plan
        if e.collection_begin is not None
        and e.collection_end is not None
        and int(e.obsid) == 1
    )
    return _dumps(list(run.plan), BLOCKED), science / MIN


def _planned(
    attitude: ActivityAttitude,
    begin: datetime,
    minutes: float,
    *,
    snapshot: int = 5,
    activities: list[ScheduledActivity] | None = None,
) -> tuple[MissionConfig, Plan]:
    config = _config(attitude)
    target = make_target(config, 1, 100, 0, minutes=600, snapshot=snapshot)
    plan = PriorityScheduler(
        config,
        [target],
        begin,
        begin + timedelta(minutes=minutes),
        step_size=60,
        activities=activities or [],
    ).schedule()
    return config, plan


@pytest.mark.parametrize("snapshot", [5, 20])
def test_one_dump_is_called_for_while_it_waits(snapshot: int) -> None:
    """While a dump waits for its firing attitude the wheels stay over the
    planning level. Before, each observation then considered called for
    another dump, eight or more fired in three hours, and science stopped."""
    dumps, science = _dispatch(FIRING, 180, snapshot=snapshot)
    assert len(dumps) == 1
    assert dumps[0] >= 25  # it waited for the attitude
    _, unhindered = _dispatch(INHERITED, 180, snapshot=snapshot)
    # Science goes on while the dump waits.
    assert science >= 0.8 * unhindered


@pytest.mark.parametrize("fault_minute", [18, 20, 21, 30, 50])
@pytest.mark.parametrize("fixed_start", [False, True])
def test_safe_mode_during_a_fixed_attitude_activity(
    fault_minute: int, fixed_start: bool
) -> None:
    """Safe mode that begins during the turn to a fixed attitude, or during
    the hold there, takes the spacecraft to its own attitude. The activity is
    no longer recorded as holding the attitude it asked for. Before, the run
    failed its plan and execution check."""
    config = make_config(2)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.battery.watthour = 1000.0
    start = BEGIN.timestamp()
    activity = ScheduledActivity(
        name="Hold",
        begin=start + 20 * MIN,
        duration_seconds=40 * MIN,
        attitude=HOLD_POSE,
        fixed=fixed_start,
    )
    target = make_target(config, 100, 105.0, 10.0, minutes=60, snapshot=20)
    run = create_ditl(
        config, [target], BEGIN, BEGIN + timedelta(hours=2), activities=[activity]
    )
    run.step_size = 60
    manage = run._manage_activities

    def with_fault(utime: float, ra: float, dec: float, roll: float) -> bool:
        if utime >= start + fault_minute * MIN:
            # Below the battery's red limit from here on.
            run.battery.charge_level = 0.3 * run.battery.watthour
        return manage(utime, ra, dec, roll)

    run._manage_activities = with_fault  # type: ignore[method-assign]
    assert run.calc()
    assert run.acs.in_safe_mode
    assert run.validate_plan_matches_execution() == []
    assert any(
        "Safe mode began during scheduled activity Hold" in e.description
        for e in run.log.events
    )


@pytest.mark.parametrize("minutes", [20, 30])
def test_dispatch_leaves_a_dump_still_blocked_at_the_end_to_the_next_plan(
    minutes: int,
) -> None:
    """The firing attitude stays blocked to the end of a short run, with the
    wheels under their capacity. Before, the run stopped with an
    ``ActivitySchedulingError``."""
    dumps, _ = _dispatch(FIRING, minutes)
    # No firing completes inside the run.
    assert all(minute + 10 > minutes for minute in dumps)


@pytest.mark.parametrize("minutes", [20, 30, 38])
def test_the_planner_leaves_a_dump_still_blocked_at_the_end_to_the_next_plan(
    minutes: int,
) -> None:
    _, plan = _planned(FIRING, BLOCKED, minutes)
    assert _dumps(list(plan), BLOCKED) == []
    # The inherited dump is planned, so the level is reached in the horizon.
    _, inherited = _planned(INHERITED, BLOCKED, minutes)
    assert _dumps(list(inherited), BLOCKED)


@pytest.mark.parametrize("start", [25, 26, 27])
def test_a_dump_fires_on_arrival_where_the_firing_is_allowed(start: int) -> None:
    """The firing attitude is allowed for the firing but not for a longest
    slew more. With science after it to turn to, the dump fires on arrival.
    Before, it was postponed by most of an orbit, behind an added hold."""
    begin = BEGIN + timedelta(minutes=start)
    config, plan = _planned(FIRING, begin, 180, snapshot=20)
    assert _dumps(list(plan), begin) == [25]
    assert not [e for e in plan if e.name == "Hold before thruster dump"]
    run = DITL(config=config, plan=plan, begin=begin, end=begin + timedelta(hours=3))
    run.step_size = 60
    assert run.calc()
    assert run.validate_plan_matches_execution() == []


@pytest.mark.parametrize("hold_minute", [26, 29, 32])
def test_the_planner_moves_a_waiting_dump_past_a_mission_activity(
    hold_minute: int,
) -> None:
    """A mission activity lies across the first firing the attitude allows,
    or the approach to it. The dump is reserved after the activity. Before,
    only that one start was tried and planning raised."""
    hold = ScheduledActivity(
        name="Mission hold",
        begin=BLOCKED.timestamp() + hold_minute * MIN,
        duration_seconds=5 * MIN,
        fixed=True,
    )
    _, plan = _planned(FIRING, BLOCKED, 180, activities=[hold])
    [dump] = _dumps(list(plan), BLOCKED)
    assert dump >= hold_minute + 5
    [entry] = [e for e in plan if e.name == "Mission hold"]
    assert entry.activity_hold_begin == hold.begin


@pytest.mark.parametrize("hold_minute", [30, 36])
def test_dispatch_moves_a_waiting_dump_past_a_fixed_attitude_activity(
    hold_minute: int,
) -> None:
    """Before, the dump was moved onto a fixed-start, fixed-attitude activity,
    which then could not reach its attitude in time and stopped the run."""
    hold = ScheduledActivity(
        name="Mission hold",
        begin=BLOCKED.timestamp() + hold_minute * MIN,
        duration_seconds=5 * MIN,
        fixed=True,
        attitude=HOLD_POSE,
    )
    dumps, _ = _dispatch(FIRING, 180, activities=[hold])
    [dump] = dumps
    assert dump >= hold_minute + 5


def _run_with_activities(
    begin: datetime,
    momentum: float,
    activities: list[ScheduledActivity],
    minutes: float = 180,
) -> tuple[list[int], list[str]]:
    """Dispatch with a fixed firing attitude from ``begin``: the dumps, and
    the faults raised about the stored momentum."""
    config = _config(FIRING)
    stored = config.spacecraft_bus.attitude_control.stored_momentum
    stored.initial_momentum_body_n_m_s = (momentum, 0, 0)
    target = make_target(config, 1, 100, 0, minutes=600, snapshot=20)
    run = create_ditl(
        config,
        [target],
        begin,
        begin + timedelta(minutes=minutes),
        activities=activities,
    )
    run.step_size = 60
    assert run.calc()
    assert not run.acs.in_safe_mode
    assert run.validate_plan_matches_execution() == []
    faults = [str(e) for e in run.fault_management.events if "momentum" in str(e)]
    return _dumps(list(run.plan), begin), faults


def _recurring(begin: datetime, every: int) -> list[ScheduledActivity]:
    """Five-minute activities that hold the attitude they find, every
    ``every`` minutes from ten minutes in."""
    return [
        ScheduledActivity(
            name=f"Recurring {k}",
            begin=begin.timestamp() + (10 + every * k) * MIN,
            duration_seconds=5 * MIN,
            fixed=True,
        )
        for k in range(170 // every)
    ]


@pytest.mark.parametrize("momentum", [7.0, 8.5, 9.0])
def test_a_dump_fires_between_recurring_activities(momentum: float) -> None:
    """Activities every 20 minutes leave 15 minutes between them, enough for
    the turn and the firing. Before, the dump asked for a longest slew clear
    on both sides, never fitted, and the wheels reached their capacity."""
    dumps, faults = _run_with_activities(BEGIN, momentum, _recurring(BEGIN, 20))
    assert dumps == [18]  # after the first activity, and done before the second
    assert faults == []


def test_a_dump_stops_giving_way_before_the_wheels_fill() -> None:
    """Activities every 12 minutes leave no room for the firing. Near its
    capacity the dump fires all the same and the activity after it begins
    late. Further from it, the dump goes on giving way."""
    full, faults = _run_with_activities(BEGIN, 9.8, _recurring(BEGIN, 12))
    assert len(full) == 1
    assert faults == []
    waiting, faults = _run_with_activities(BEGIN, 9.0, _recurring(BEGIN, 12))
    assert waiting == []
    assert faults == []


def test_a_dump_is_not_fired_where_the_next_activity_would_keep_its_attitude() -> None:
    """An activity that begins as the firing ends holds the firing attitude.
    Where that attitude is not allowed for as long, the dump waits. Before,
    the spacecraft was kept there into the Earth's limb and entered safe
    mode."""
    dumps, _ = _run_with_activities(BLOCKED, 9.0, _recurring(BLOCKED, 16))
    assert len(dumps) == 1


@pytest.mark.parametrize(("begin", "slide"), [(10, 60), (10, 120), (20, 150)])
def test_an_activity_that_may_slide_does_not_hold_the_dump_back(
    begin: int, slide: int
) -> None:
    """An activity with a long slide window is late only past its latest
    start. Before, its whole window counted as taken: dispatch dumped at 79
    minutes and entered safe mode, and the planner reserved the dump after
    two hours or not at all."""
    activity = ScheduledActivity(
        name="Slides",
        begin=BLOCKED.timestamp() + begin * MIN,
        duration_seconds=5 * MIN,
        slide_seconds=slide * MIN,
    )
    dumps, science = _dispatch(FIRING, 180, snapshot=20, activities=[activity])
    assert dumps == [31]
    _, unhindered = _dispatch(INHERITED, 180, snapshot=20, activities=[activity])
    assert science >= 0.9 * unhindered
    config, plan = _planned(FIRING, BLOCKED, 180, snapshot=20, activities=[activity])
    assert _dumps(list(plan), BLOCKED) == [33]
    run = DITL(
        config=config, plan=plan, begin=BLOCKED, end=BLOCKED + timedelta(hours=3)
    )
    run.step_size = 60
    assert run.calc()
    assert run.validate_plan_matches_execution() == []


@pytest.mark.parametrize("hold_minute", [13, 15, 17, 20, 22])
def test_the_planner_fits_a_dump_around_a_hold_soon_after_it(hold_minute: int) -> None:
    """A mission hold follows within a few minutes of where the dump would
    fire on arrival, with no room for the turn between them. The dump is
    reserved after the hold. Before, planning raised."""
    hold = ScheduledActivity(
        name="Mission hold",
        begin=BEGIN.timestamp() + hold_minute * MIN,
        duration_seconds=5 * MIN,
        fixed=True,
    )
    config, plan = _planned(FIRING, BEGIN, 180, snapshot=20, activities=[hold])
    (dump,) = _dumps(list(plan), BEGIN)
    assert dump >= hold_minute + 5
    run = DITL(config=config, plan=plan, begin=BEGIN, end=BEGIN + timedelta(hours=3))
    run.step_size = 60
    assert run.calc()
    assert run.validate_plan_matches_execution() == []


def _faulted_firing(
    fault_minute: int, minutes: int, attitude: ActivityAttitude, *, dump: bool
) -> tuple[DITL, list[PlanEntry]]:
    """A ten-minute activity from 20 minutes in, with the battery below its
    red limit from ``fault_minute``; the run and the activity's entries."""
    config = make_config(2)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.battery.watthour = 1000.0
    bus = config.spacecraft_bus
    bus.inertia_tensor_body_kg_m2 = ((1000, 0, 0), (0, 1200, 0), (0, 0, 800))
    bus.attitude_control.stored_momentum = StoredMomentumConfig(
        gravity_gradient_enabled=True,
        capacity_n_m_s=10,
        planning_level=0.9,
        initial_momentum_body_n_m_s=(3, 0, 0),
        thruster_dump=ThrusterDumpConfig(duration_seconds=600),
    )
    start = BEGIN.timestamp()
    activity = ScheduledActivity(
        name="Firing",
        begin=start + 20 * MIN,
        duration_seconds=(10 if dump else 40) * MIN,
        attitude=attitude,
        desaturation=dump,
    )
    target = make_target(config, 100, 105.0, 10.0, minutes=60, snapshot=20)
    run = create_ditl(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(minutes=minutes),
        activities=[activity],
    )
    run.step_size = 60
    manage = run._manage_activities

    def with_fault(utime: float, ra: float, dec: float, roll: float) -> bool:
        if utime >= start + fault_minute * MIN:
            run.battery.charge_level = 0.3 * run.battery.watthour
        return manage(utime, ra, dec, roll)

    run._manage_activities = with_fault  # type: ignore[method-assign]
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
    return run, [e for e in run.plan if e.name == "Firing"]


@pytest.mark.parametrize("fault_minute", [18, 19, 20, 21, 25])
def test_a_dump_that_safe_mode_stops_unloads_nothing(fault_minute: int) -> None:
    """The thrusters do not fire in safe mode. The dump's entry stops where
    safe mode began and is recorded at the attitude safe mode reaches.
    Before, the dump was credited with emptying the wheels, and its entry
    recorded the requested attitude or one in the middle of a slew."""
    run, (entry,) = _faulted_firing(fault_minute, 120, HOLD_POSE, dump=True)
    assert not entry.activity_finishes
    assert not [e for e in run.log.events if "unloads" in e.description]
    tracker = run._stored_momentum_tracker
    assert tracker is not None
    assert tracker.stored_momentum_norm_n_m_s > 2.5
    safe = (float(run.acs.ra), float(run.acs.dec))
    assert (entry.ra, entry.dec) == pytest.approx(safe, abs=0.5)


def test_a_dump_due_in_safe_mode_is_not_begun() -> None:
    run, entries = _faulted_firing(19, 120, INHERITED, dump=True)
    assert entries == []
    assert not [e for e in run.log.events if "unloads" in e.description]


@pytest.mark.parametrize("end_minute", [22, 24, 26, 30])
def test_safe_mode_on_the_last_step_of_a_fixed_attitude_activity(
    end_minute: int,
) -> None:
    """The fault comes a minute before the run ends, after the step's
    activities were managed. Before, the run failed its plan and execution
    check."""
    _, (entry,) = _faulted_firing(end_minute - 1, end_minute, HOLD_POSE, dump=False)
    assert entry.end == BEGIN.timestamp() + end_minute * MIN
