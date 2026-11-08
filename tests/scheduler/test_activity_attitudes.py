"""Explicit activity poses reserve ingress and a full stationary hold."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from conops import DITL, QueueDITL
from conops.common import ACSMode, ObsType
from conops.common.vector import quaternion_attitude_delta
from conops.config import (
    ActivityAttitude,
    MissionConfig,
    ObservationTiming,
    SpacecraftFrame,
    StoredMomentumConfig,
    ThrusterDumpConfig,
)
from conops.ditl import RollingHorizonDITL
from conops.schedulers import (
    ActivitySchedulingError,
    PriorityScheduler,
    SchedulingContext,
)
from conops.targets import (
    ObservationSequence,
    Plan,
    Pointing,
    Program,
    ScheduledActivity,
    SequenceStep,
)

from .planning_scenario import BEGIN, HOUR, MIN, T0, make_config, make_target
from .test_activity_safety import _planner

POSE = ActivityAttitude(kind="fixed", ra_deg=105, dec_deg=10, roll_deg=25)


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "fixed"},
        {"kind": "inherited", "ra_deg": 1},
        {"kind": "fixed", "ra_deg": float("nan"), "dec_deg": 0, "roll_deg": 0},
        {"kind": "fixed", "ra_deg": 10, "dec_deg": 91, "roll_deg": 0},
    ],
)
def test_invalid_attitude_requests_are_rejected(fields):
    with pytest.raises(ValidationError):
        ActivityAttitude(**fields)


def test_attitude_roundtrips_in_program_and_dump_configuration(tmp_path):
    activity = ScheduledActivity(
        name="Pointed hold", begin=T0 + HOUR, duration_seconds=20 * MIN, attitude=POSE
    )
    program = Program(begin=T0, end=T0 + 4 * HOUR, activities=[activity])
    assert Program.load(program.save(tmp_path / "program.json")) == program
    dump = ThrusterDumpConfig(attitude=POSE)
    assert ThrusterDumpConfig.model_validate_json(dump.model_dump_json()) == dump
    config = make_config(4)
    config.spacecraft_bus.attitude_control.stored_momentum = StoredMomentumConfig(
        capacity_n_m_s=10,
        thruster_dump=dump,
    )
    config.to_yaml_file(str(tmp_path / "mission.yaml"))
    restored = MissionConfig.from_yaml_file(str(tmp_path / "mission.yaml"))
    assert (
        restored.spacecraft_bus.attitude_control.stored_momentum.thruster_dump.attitude
        == POSE
    )
    assert ActivityAttitude().fixed_pointing is None


@pytest.mark.parametrize(
    "kind,rolling",
    [
        ("dispatch", False),
        ("priority", False),
        ("local_search", False),
        ("cp_sat", False),
        ("priority", True),
        ("local_search", True),
        ("cp_sat", True),
    ],
)
@pytest.mark.parametrize("fixed", [False, True])
def test_explicit_hold_attitude_is_flown_for_the_full_duration(
    kind, rolling, fixed, tmp_path
):
    config = make_config(4)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    end = BEGIN + timedelta(hours=4)
    activity = ScheduledActivity(
        name="Pointed calibration",
        begin=T0 + HOUR,
        duration_seconds=20 * MIN,
        fixed=fixed,
        slide_seconds=0 if fixed else 20 * MIN,
        attitude=POSE,
        data_gb=1.5,
    )
    if kind == "dispatch":
        ditl = QueueDITL(config=config, begin=BEGIN, end=end, activities=[activity])
    else:
        scheduler, options = _planner(kind)
        if rolling:
            ditl = RollingHorizonDITL(
                config,
                [],
                begin=BEGIN,
                end=end,
                activities=[activity],
                scheduler=scheduler,
                scheduler_options=options,
                include_passes=False,
                horizon=timedelta(hours=3),
                replan_interval=timedelta(minutes=30),
            )
        else:
            plan = scheduler(
                config,
                [],
                BEGIN,
                end,
                activities=[activity],
                include_passes=False,
                **options,
            ).schedule()
            ditl = DITL(
                config=config,
                plan=Plan.load(plan.save(tmp_path / "plan.json")),
                begin=BEGIN,
                end=end,
            )
    ditl._add_activity_data = Mock(wraps=ditl._add_activity_data)
    assert ditl.calc()
    [entry] = [e for e in ditl.plan if e.obstype == ObsType.ACTIVITY]
    assert (entry.ra, entry.dec, entry.roll) == pytest.approx(POSE.fixed_pointing)
    assert entry.activity_requires_stationary
    assert activity.begin <= entry.activity_hold_begin <= activity.latest_start
    assert float(entry.begin) < entry.activity_hold_begin
    assert float(entry.begin) + entry.slewtime <= entry.activity_hold_begin
    assert float(entry.end) - entry.activity_hold_begin == activity.duration_seconds
    samples = [
        hk
        for hk in ditl.telemetry.housekeeping
        if entry.activity_hold_begin <= hk.timestamp.timestamp() < float(entry.end)
    ]
    assert len(samples) == 20
    assert {int(hk.acs_mode) for hk in samples} == {int(ACSMode.IDLE)}
    assert all(
        quaternion_attitude_delta(hk.ra, hk.dec, hk.roll, *POSE.fixed_pointing)[0]
        < 1e-6
        for hk in samples
    )
    assert ditl.validate_plan_matches_execution() == []
    ditl._add_activity_data.assert_called_once()


def test_a_forbidden_fixed_attitude_is_not_replaced_with_another_pose():
    from rust_ephem import SunConstraint

    config = make_config(4)
    config.constraint.safety_constraint = SunConstraint(min_angle=180)
    activity = ScheduledActivity(
        name="Illegal pose",
        begin=T0 + HOUR,
        duration_seconds=MIN,
        fixed=True,
        attitude=POSE,
    )
    with pytest.raises(ActivitySchedulingError, match="Illegal pose"):
        PriorityScheduler(
            config,
            [],
            BEGIN,
            BEGIN + timedelta(hours=4),
            activities=[activity],
            include_passes=False,
        ).schedule()


def test_a_fixed_pose_searches_its_slide_window_instead_of_substituting_a_pose():
    config = make_config(4)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    activity = ScheduledActivity(
        name="Windowed pose",
        begin=T0 + HOUR,
        duration_seconds=20 * MIN,
        slide_seconds=HOUR,
        attitude=POSE,
    )
    boundary = T0 + 90 * MIN

    def violation(ctx, attitude, begin, end, mode):
        return max(begin, boundary) if end > boundary else None

    with patch.object(SchedulingContext, "first_hold_violation", violation):
        plan = PriorityScheduler(
            config,
            [],
            BEGIN,
            BEGIN + timedelta(hours=4),
            activities=[activity],
            include_passes=False,
        ).schedule()
    [entry] = plan.entries
    assert (entry.ra, entry.dec, entry.roll) == POSE.fixed_pointing
    assert activity.begin <= entry.activity_hold_begin < activity.latest_start
    assert float(entry.end) <= boundary
    assert float(entry.end) - entry.activity_hold_begin == activity.duration_seconds


@pytest.mark.parametrize(
    "kind,rolling",
    [
        ("dispatch", False),
        ("priority", False),
        ("local_search", False),
        ("cp_sat", False),
        ("priority", True),
        ("local_search", True),
        ("cp_sat", True),
    ],
)
@pytest.mark.parametrize("payload_frame", [False, True])
def test_an_automatic_dump_reserves_its_firing_attitude_and_unloads_at_the_end(
    kind, rolling, payload_frame
):
    config = make_config(4)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    if payload_frame:
        config.spacecraft_bus.frame = SpacecraftFrame(
            boresight=(0, 0, 1), sun_side=(1, 0, 0)
        )
    config.spacecraft_bus.inertia_tensor_body_kg_m2 = (
        (1000, 0, 0),
        (0, 1200, 0),
        (0, 0, 800),
    )
    config.spacecraft_bus.attitude_control.stored_momentum = StoredMomentumConfig(
        gravity_gradient_enabled=True,
        capacity_n_m_s=10,
        planning_level=0.6,
        initial_momentum_body_n_m_s=(7, 0, 0),
        thruster_dump=ThrusterDumpConfig(duration_seconds=120, attitude=POSE),
    )
    end = BEGIN + timedelta(minutes=30)
    if kind == "dispatch":
        ditl = QueueDITL(config=config, begin=BEGIN, end=end)
    else:
        scheduler, options = _planner(kind)
        if rolling:
            ditl = RollingHorizonDITL(
                config,
                [],
                begin=BEGIN,
                end=end,
                scheduler=scheduler,
                scheduler_options=options,
                include_passes=False,
                horizon=timedelta(minutes=30),
                replan_interval=timedelta(minutes=3),
            )
        else:
            builder = scheduler(config, [], BEGIN, end, include_passes=False, **options)
            plan = builder.schedule()
            [reserved] = builder.dumps
            assert reserved.attitude == POSE
            ditl = DITL(config=config, plan=plan, begin=BEGIN, end=end)
    ditl._unload_stored_momentum = Mock(wraps=ditl._unload_stored_momentum)
    assert ditl.calc()
    [entry] = [e for e in ditl.plan if e.desaturation]
    assert (entry.ra, entry.dec, entry.roll) == pytest.approx(POSE.fixed_pointing)
    assert float(entry.begin) < entry.activity_hold_begin
    assert float(entry.begin) + entry.slewtime <= entry.activity_hold_begin
    assert float(entry.end) - entry.activity_hold_begin == 120
    samples = [
        hk
        for hk in ditl.telemetry.housekeeping
        if entry.activity_hold_begin <= hk.timestamp.timestamp() < float(entry.end)
    ]
    assert len(samples) == 2
    assert {int(hk.acs_mode) for hk in samples} == {int(ACSMode.DESAT)}
    assert all(
        quaternion_attitude_delta(hk.ra, hk.dec, hk.roll, *POSE.fixed_pointing)[0]
        < 1e-6
        for hk in samples
    )
    ditl._unload_stored_momentum.assert_called_once_with(float(entry.end), entry.obsid)
    assert ditl.validate_plan_matches_execution() == []


@pytest.mark.parametrize("kind", ["dispatch", "priority", "local_search", "cp_sat"])
def test_a_truncated_explicit_dump_does_not_unload(kind):
    config = make_config(2)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    activity = ScheduledActivity(
        name="Cut firing",
        begin=T0 + HOUR,
        duration_seconds=2 * MIN,
        fixed=True,
        desaturation=True,
        attitude=POSE,
    )
    end = BEGIN + timedelta(minutes=61)
    if kind == "dispatch":
        ditl = QueueDITL(config=config, begin=BEGIN, end=end, activities=[activity])
    else:
        scheduler, options = _planner(kind)
        plan = scheduler(
            config,
            [],
            BEGIN,
            end,
            activities=[activity],
            include_passes=False,
            **options,
        ).schedule()
        ditl = DITL(config=config, plan=plan, begin=BEGIN, end=end)
    ditl._unload_stored_momentum = Mock()
    assert ditl.calc()
    [entry] = [e for e in ditl.plan if e.desaturation]
    assert entry.activity_partial and not entry.activity_finishes
    assert entry.activity_hold_end == activity.end
    ditl._unload_stored_momentum.assert_not_called()
    assert ditl.validate_plan_matches_execution() == []


def test_dispatch_rejects_an_unsafe_required_pose():
    from rust_ephem import SunConstraint

    config = make_config(4)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.constraint.safety_constraint = SunConstraint(min_angle=0, max_angle=80)
    activity = ScheduledActivity(
        name="Illegal pose",
        begin=T0 + HOUR,
        duration_seconds=MIN,
        fixed=True,
        attitude=POSE,
    )
    ditl = QueueDITL(
        config=config,
        begin=BEGIN,
        end=BEGIN + timedelta(hours=4),
        activities=[activity],
    )
    with pytest.raises(ActivitySchedulingError, match="Illegal pose"):
        ditl.calc()


@pytest.mark.parametrize("scheduled", [False, True])
def test_a_configured_dump_does_not_split_a_sequence_visit(scheduled):
    config = make_config(2)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.inertia_tensor_body_kg_m2 = (
        (1000, 0, 0),
        (0, 1200, 0),
        (0, 0, 800),
    )
    config.spacecraft_bus.attitude_control.stored_momentum = StoredMomentumConfig(
        gravity_gradient_enabled=True,
        capacity_n_m_s=10,
        planning_level=0.6,
        initial_momentum_body_n_m_s=(0 if scheduled else 7, 0, 0),
        thruster_dump=ThrusterDumpConfig(duration_seconds=120, attitude=POSE),
    )
    visit = Pointing(
        config=config,
        obsid=700,
        name="Protected visit",
        ra=105,
        dec=10,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(name="A", duration_seconds=20 * MIN),
                SequenceStep(name="B", duration_seconds=20 * MIN, roll_offset_deg=15),
            ]
        ),
        visits=1,
    )
    activities = (
        [
            ScheduledActivity(
                name="Dump during visit",
                begin=T0 + 20 * MIN,
                duration_seconds=2 * MIN,
                desaturation=True,
                attitude=POSE,
            )
        ]
        if scheduled
        else []
    )
    ditl = QueueDITL(
        config=config,
        begin=BEGIN,
        end=BEGIN + timedelta(hours=2),
        activities=activities,
    )
    ditl.queue.add_request(visit)
    assert ditl.calc()
    parts = [e for e in ditl.plan if e.obsid == visit.obsid]
    assert len(parts) == 2
    dumps = [e for e in ditl.plan if e.desaturation]
    assert dumps
    assert all(
        float(e.end) <= float(parts[0].begin) or float(e.begin) >= float(parts[-1].end)
        for e in dumps
    )
    assert ditl.validate_plan_matches_execution() == []


def test_a_delayed_dump_does_not_hide_another_unstarted_activity():
    config = make_config(4)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    activities = [
        ScheduledActivity(
            name="Window",
            begin=T0 + HOUR,
            duration_seconds=HOUR,
            fixed=True,
            attitude=POSE,
        ),
        ScheduledActivity(
            name="Waiting dump",
            begin=T0 + HOUR + 10 * MIN,
            duration_seconds=20 * MIN,
            desaturation=True,
            attitude=POSE,
        ),
        ScheduledActivity(
            name="Waiting calibration",
            begin=T0 + HOUR + 20 * MIN,
            duration_seconds=10 * MIN,
            attitude=POSE,
        ),
    ]
    ditl = QueueDITL(
        config=config,
        begin=BEGIN,
        end=BEGIN + timedelta(hours=4),
        activities=activities,
    )
    assert ditl.calc()
    entries = [e for e in ditl.plan if e.obstype == ObsType.ACTIVITY]
    assert [e.name for e in entries] == [a.name for a in activities]
    assert float(entries[1].end) == T0 + 2 * HOUR + 20 * MIN
    assert float(entries[2].end) == T0 + 2 * HOUR + 30 * MIN
    assert ditl.validate_plan_matches_execution() == []


def test_dispatch_cancels_pending_collection_before_fixed_pose_ingress():
    config = make_config(2)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.payload.observation_timing = ObservationTiming(cleanup_seconds=10 * MIN)
    hold = ScheduledActivity(
        name="Fixed hold before an exposure can finish teardown",
        begin=T0 + 5 * MIN,
        duration_seconds=20 * MIN,
        fixed=True,
        attitude=ActivityAttitude(kind="fixed", ra_deg=105, dec_deg=10, roll_deg=0),
    )
    run = QueueDITL(
        config=config, begin=BEGIN, end=BEGIN + timedelta(hours=2), activities=[hold]
    )
    run.ustart, run.uend, run.step_size = T0, T0 + 2 * HOUR, MIN
    target = make_target(config, 100, 105, 10, minutes=30)
    target.collection_begin = T0 + MIN
    target.collection_end = T0 + 30 * MIN
    target.end = T0 + 40 * MIN
    run.ppt = target
    assert run._manage_activities(T0, run.acs.ra, run.acs.dec, run.acs.roll)
    assert run._active_activity is not None
    assert run._active_activity.name == hold.name
    assert run.ppt is None


@pytest.mark.parametrize("fixed", [True, False])
def test_dispatch_budgets_teardown_before_fixed_pose_ingress(fixed):
    config = make_config(4)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.payload.observation_timing = ObservationTiming(
        cleanup_seconds=300, handoff_seconds=120
    )
    pose = ActivityAttitude(kind="fixed", ra_deg=170, dec_deg=35, roll_deg=20)
    hold = ScheduledActivity(
        name="Required safe hold during a long exposure",
        begin=T0 + HOUR,
        duration_seconds=HOUR,
        fixed=fixed,
        slide_seconds=0 if fixed else 15 * MIN,
        attitude=pose,
    )
    run = QueueDITL(
        config=config, begin=BEGIN, end=BEGIN + timedelta(hours=4), activities=[hold]
    )
    run.queue.add_request(
        make_target(config, 100, 105, 10, minutes=150, snapshot=150, ss_min=5)
    )
    assert run.calc()
    assert not run.validate_plan_matches_execution()
    block = next(e for e in run.plan if e.obstype == ObsType.ACTIVITY)
    assert block.activity_finishes
    assert block.activity_hold_end - block.activity_hold_begin == HOUR
    assert hold.begin <= block.activity_hold_begin <= hold.latest_start
    before = [
        e
        for e in run.plan
        if e.obstype in (ObsType.PPT, ObsType.AT) and e.end <= block.begin
    ]
    assert before
    assert before[-1].collection_end + 420 <= block.begin
    collected = sum(
        float(h.collection_seconds or 0) for h in run.telemetry.housekeeping
    )
    planned = sum(
        float(e.collection_end - e.collection_begin)
        for e in run.plan
        if e.collection_begin is not None and e.collection_end is not None
    )
    assert collected == pytest.approx(planned)
