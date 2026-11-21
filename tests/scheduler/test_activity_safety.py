"""Mandatory activities use a reachable attitude safe for the entire hold."""

from contextlib import nullcontext
from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
from rust_ephem import EarthLimbConstraint

from conops import DITL
from conops.common import ACSMode, ObsType
from conops.ditl import RollingHorizonDITL
from conops.schedulers import (
    ActivitySchedulingError,
    PriorityScheduler,
    SchedulingContext,
)
from conops.schedulers.priority_scheduler import StartState
from conops.targets import MovingPointing, Plan, ScheduledActivity, track_from_table

from .planning_scenario import (
    BEGIN,
    HOUR,
    MIN,
    PATCH,
    T0,
    activity_entries,
    make_config,
    make_target,
    replay,
)
from .planning_scenario import planner as _planner


def _config():
    config = make_config(4, open_sky=True)
    config.constraint.safety_constraint = EarthLimbConstraint(min_angle=0)
    return config


@pytest.mark.parametrize(
    "kind, slide, rolling",
    [
        ("priority", 0, False),
        ("local_search", 20 * MIN, False),
        ("cp_sat", 0, False),
        ("priority", 20 * MIN, True),
    ],
)
def test_an_unsafe_inherited_hold_is_reserved_and_executed(
    kind, slide, rolling, tmp_path
):
    config = _config()
    end = BEGIN + timedelta(hours=4)
    activity = ScheduledActivity(
        name="Mandatory calibration",
        begin=T0 + HOUR,
        duration_seconds=HOUR,
        fixed=not slide,
        slide_seconds=slide,
        data_gb=1.5,
    )
    targets = [make_target(config, 100, *PATCH[0], minutes=40, snapshot=10)]
    scheduler, options = _planner(kind)
    builder = scheduler(
        config,
        targets,
        BEGIN,
        end,
        activities=[activity],
        include_passes=False,
        **options,
    )
    plan = builder.schedule()
    # The old placement gate dropped this block before trying another attitude.
    initial = builder.ctx.initial_attitude(T0)
    assert (
        builder.ctx.first_hold_violation(
            initial,
            activity.begin,
            activity.end,
            ACSMode.IDLE,
        )
        is not None
    )
    assert not any("leaving it out" in e.description for e in builder.log)

    if rolling:
        ditl = RollingHorizonDITL(
            config,
            targets,
            begin=BEGIN,
            end=end,
            activities=[activity],
            scheduler=scheduler,
            scheduler_options=options,
            include_passes=False,
            horizon=timedelta(hours=3),
            replan_interval=timedelta(minutes=30),
            commit_lead_time=timedelta(minutes=10),
        )
    else:
        loaded = Plan.load(plan.save(tmp_path / "plan.json"))
        ditl = DITL(config=config, plan=loaded, begin=BEGIN, end=end)
    assert ditl.calc()
    assert ditl.validate_plan_matches_execution() == []
    [entry] = activity_entries(ditl.plan)
    assert activity.begin <= entry.activity_hold_begin <= activity.latest_start
    assert entry.activity_hold_end - entry.activity_hold_begin == HOUR
    assert float(entry.end) == entry.activity_hold_end
    assert float(entry.begin) + entry.slewtime <= entry.activity_hold_begin
    attitude = (entry.ra, entry.dec, entry.roll)
    assert (
        builder.ctx.first_hold_violation(
            attitude,
            entry.activity_hold_begin,
            entry.activity_hold_end,
            ACSMode.IDLE,
        )
        is None
    )
    assert all(
        float(e.end) <= float(entry.begin) or float(e.begin) >= float(entry.end)
        for e in ditl.plan
        if e.obstype != ObsType.ACTIVITY
    )
    modes = [
        int(hk.acs_mode)
        for hk in ditl.telemetry.housekeeping
        if entry.activity_hold_begin
        <= hk.timestamp.timestamp()
        < entry.activity_hold_end
    ]
    assert len(modes) == HOUR // config.constraint.ephem.step_size
    assert set(modes) <= {int(ACSMode.IDLE), int(ACSMode.SAA)}


@pytest.mark.parametrize(
    "slide, fragment",
    [(0, False), (20 * MIN, False), (0, True)],
    ids=["fixed", "sliding", "horizon-fragment"],
)
def test_an_attitude_safe_at_start_but_unsafe_midway_gets_an_ingress(slide, fragment):
    config = _config()
    begin = BEGIN + timedelta(minutes=54)
    end = BEGIN + timedelta(hours=3)
    # A fragment's horizon ends before the violation, but its pose must survive the whole hold.
    horizon_end = BEGIN + timedelta(minutes=65) if fragment else end
    activity = ScheduledActivity(
        name="Long calibration",
        begin=T0 + HOUR,
        duration_seconds=HOUR,
        fixed=not slide,
        slide_seconds=slide,
    )
    ephem = config.constraint.ephem
    index = ephem.index(BEGIN + timedelta(hours=1))
    inherited = (
        (float(ephem.earth_ra_deg[index]) + 180) % 360,
        -float(ephem.earth_dec_deg[index]),
        0.0,
    )
    builder = PriorityScheduler(
        config,
        [],
        begin,
        horizon_end,
        simulation_end=end,
        activities=[activity],
        include_passes=False,
        start_state=StartState(time=begin.timestamp(), attitude=inherited),
    )
    plan = builder.schedule()
    violation = builder.ctx.first_hold_violation(
        inherited,
        activity.begin,
        activity.end,
        ACSMode.IDLE,
    )
    assert activity.begin < violation < activity.end
    if fragment:
        assert horizon_end.timestamp() < violation
    [entry] = plan.entries
    assert entry.slewtime > 0
    assert float(entry.begin) < entry.activity_hold_begin
    assert activity.begin <= entry.activity_hold_begin <= activity.latest_start
    assert float(entry.begin) + entry.slewtime <= entry.activity_hold_begin
    assert entry.activity_hold_end - entry.activity_hold_begin == HOUR
    assert float(entry.end) == min(entry.activity_hold_end, horizon_end.timestamp())
    assert (
        builder.ctx.first_hold_violation(
            (entry.ra, entry.dec, entry.roll),
            entry.activity_hold_begin,
            entry.activity_hold_end,
            ACSMode.IDLE,
        )
        is None
    )
    if not fragment:
        replay(config, plan, begin, end, pose=inherited)


@pytest.mark.parametrize(
    "overdue, reason",
    [(False, "full duration"), (True, "start window closed")],
    ids=["no-safe-attitude", "window-closed"],
)
def test_an_unplaceable_mandatory_activity_is_reported(overdue, reason):
    config = _config()
    activity = ScheduledActivity(
        name="Mandatory calibration",
        begin=T0 + HOUR,
        duration_seconds=(2 if overdue else 1) * HOUR,
        fixed=not overdue,
        slide_seconds=30 * MIN if overdue else 0,
    )
    builder = PriorityScheduler(
        config,
        [],
        BEGIN + timedelta(hours=2 if overdue else 0),
        BEGIN + timedelta(hours=4),
        simulation_start=BEGIN,
        activities=[activity],
        include_passes=False,
    )
    # Without the overdue start, make every candidate attitude unsafe.
    unsafe = patch.multiple(
        SchedulingContext,
        first_hold_violation=Mock(return_value=T0 + HOUR),
        idle_hold_candidates=Mock(return_value=iter([(0.0, 0.0, 0.0)])),
    )
    with nullcontext() if overdue else unsafe:
        with pytest.raises(ActivitySchedulingError, match=activity.name) as error:
            builder.schedule()
    assert error.value.activity.obsid is not None
    assert reason in error.value.reason


@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
def test_a_sliding_activity_remains_due_after_its_nominal_start(kind):
    """A committed science visit ends inside an uncommitted block's window.

    At the one-hour replan, the long visit is committed through about 02:00,
    but the activity's nominal 01:00 start is past. Its still-open window must
    keep the full calibration in the new plan, without calling it resumed.
    """
    config = make_config(4, open_sky=True)
    scheduler, options = _planner(kind)
    activity = ScheduledActivity(
        name="Sliding calibration",
        begin=T0 + HOUR,
        duration_seconds=30 * MIN,
        slide_seconds=2 * HOUR,
        data_gb=1.5,
    )
    target = make_target(config, 100, *PATCH[0], minutes=120, snapshot=120)
    ditl = RollingHorizonDITL(
        config,
        [target],
        begin=BEGIN,
        end=BEGIN + timedelta(hours=4),
        scheduler=scheduler,
        scheduler_options=options,
        include_passes=False,
        activities=[activity],
        horizon=timedelta(hours=4),
        replan_interval=timedelta(hours=1),
        commit_lead_time=timedelta(minutes=40),
    )
    assert ditl.calc()
    [entry] = activity_entries(ditl.plan)
    assert T0 + 2 * HOUR <= entry.activity_hold_begin <= activity.latest_start
    assert float(entry.end) - entry.activity_hold_begin == activity.duration_seconds
    assert entry.fixed_data_gb == activity.data_gb
    assert not entry.activity_partial
    assert ditl.validate_plan_matches_execution() == []


@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
@pytest.mark.parametrize("rolling", [False, True])
def test_a_calibration_after_tracking_holds_the_final_attitude(kind, rolling):
    """Do not slew back to the moving body's arrival position during a hold."""
    config = make_config(4, open_sky=True)
    config.spacecraft_bus.attitude_control.settle_time = 600
    target = MovingPointing(
        config=config,
        obsid=700,
        name="Slow mover",
        exptime=40 * MIN,
        ss_min=40 * MIN,
        ss_max=40 * MIN,
        track=track_from_table(
            [T0 + k * HOUR for k in range(5)],
            [105 + k * 0.01 for k in range(5)],
            [10.0] * 5,
        ),
    )
    activity = ScheduledActivity(
        name="After tracking",
        begin=T0 + HOUR,
        duration_seconds=HOUR,
        slide_seconds=HOUR,
    )
    scheduler, options = _planner(kind)
    end = BEGIN + timedelta(hours=4)
    if rolling:
        ditl = RollingHorizonDITL(
            config,
            [target],
            begin=BEGIN,
            end=end,
            scheduler=scheduler,
            scheduler_options=options,
            activities=[activity],
            include_passes=False,
            horizon=timedelta(hours=3),
            replan_interval=timedelta(minutes=30),
        )
    else:
        plan = scheduler(
            config,
            [target],
            BEGIN,
            end,
            activities=[activity],
            include_passes=False,
            **options,
        ).schedule()
        ditl = DITL(config=config, plan=plan, begin=BEGIN, end=end)
    assert ditl.calc()
    [entry] = activity_entries(ditl.plan)
    modes = [
        int(hk.acs_mode)
        for hk in ditl.telemetry.housekeeping
        if entry.activity_hold_begin
        <= hk.timestamp.timestamp()
        < entry.activity_hold_end
    ]
    assert len(modes) == 60
    assert set(modes) == {int(ACSMode.IDLE)}
    assert ditl.validate_plan_matches_execution() == []
    assert entry.activity_requires_stationary
    index = next(
        i
        for i, t in enumerate(ditl.utime)
        if entry.activity_hold_begin <= t < entry.activity_hold_end
    )
    original = ditl.mode[index]
    ditl.mode[index] = int(ACSMode.SLEWING)
    assert ditl.validate_plan_matches_execution()
    entry.activity_requires_stationary = False
    assert ditl.validate_plan_matches_execution() == []
    ditl.mode[index] = original


@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
def test_a_sliding_hold_finishes_inside_the_physical_run(kind):
    config = _config()
    scheduler, options = _planner(kind)
    activity = ScheduledActivity(
        name="Calibration before shutdown",
        begin=T0 + 3 * HOUR,
        duration_seconds=30 * MIN,
        slide_seconds=2 * HOUR,
        data_gb=1.5,
    )
    end = BEGIN + timedelta(hours=4)
    planner = scheduler(
        config, [], BEGIN, end, activities=[activity], include_passes=False, **options
    )
    [entry] = planner.schedule().entries
    assert T0 + 3 * HOUR <= entry.activity_hold_begin <= T0 + 3.5 * HOUR
    assert entry.activity_hold_end - entry.activity_hold_begin == 30 * MIN
    assert float(entry.end) == entry.activity_hold_end <= end.timestamp()
    assert entry.fixed_data_gb == activity.data_gb
    first_violation = planner.ctx.first_hold_violation(
        planner.ctx.initial_attitude(T0), T0, entry.activity_hold_begin, ACSMode.IDLE
    )
    assert first_violation is not None
    assert float(entry.begin) < first_violation
    run = replay(config, Plan(entries=[entry]), BEGIN, end)
    assert run.fault_management.events == []
