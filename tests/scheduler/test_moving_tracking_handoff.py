"""A cut during the first tracking interval still needs a physical brake."""

from datetime import timedelta

import numpy as np
import pytest

from conops import DITL, create_ditl
from conops.common import ACSMode, ObsType
from conops.config import ObservationTiming, TargetConfig
from conops.ditl import QueueDITL, RollingHorizonDITL
from conops.schedulers import PriorityScheduler
from conops.simulation.attitude import AttitudeTrajectory
from conops.targets import (
    MovingPointing,
    ObservationSequence,
    Plan,
    PlanEntry,
    Pointing,
    SequenceStep,
    TOORequest,
    track_from_table,
)

from .planning_scenario import BEGIN, MIN, T0, collected_seconds, make_config


def tracked_entry(initial_pose=(105.0, 10.0, 0.0)):
    config = make_config(2, open_sky=True)
    config.payload.observation_timing = ObservationTiming()
    config.spacecraft_bus.attitude_control.settle_time = 0
    entry = PlanEntry(
        config=config,
        obsid=700,
        ra=105,
        dec=10,
        roll=0,
        begin=T0,
        end=T0 + 3600,
        slewtime=0,
        collection_begin=T0,
        collection_end=T0 + 3600,
        track=track_from_table([T0, T0 + 7200], [105, 107], [10, 10]),
    )
    run = DITL(
        config=config,
        ephem=config.constraint.ephem,
        plan=Plan(entries=[entry]),
        begin=BEGIN,
        end=BEGIN + timedelta(hours=2),
    )
    run.step_size = 60
    pose = initial_pose
    run.acs.ra, run.acs.dec, run.acs.roll = pose
    run.acs._hold_idle_attitude(*pose, T0)
    run.acs.pointing(T0)
    run._command_entry(entry, T0)
    run.acs.pointing(T0)
    run.acs.pointing(T0 + 20)
    assert run.acs._tracked_slew is not None
    entry.end = entry.collection_end = float(run.acs._tracked_slew.slewend) + 39
    return run, entry


def test_rolling_handoff_uses_the_installed_first_tracking_interval():
    run, entry = tracked_entry()
    rolling = object.__new__(RollingHorizonDITL)
    rolling.acs, rolling.config = run.acs, run.config
    rolling._entry_passes = {}
    before = run.acs._executor.state
    braking = AttitudeTrajectory.braking(
        run.acs._executor.predict(float(entry.end)),
        run.config.spacecraft_bus.attitude_control,
    )
    assert braking is not None
    rest, attitude = rolling._rest_after(entry)
    assert rest > float(entry.end)
    assert rest == braking.end
    assert attitude == pytest.approx(braking.state(braking.end).attitude)
    assert run.acs._executor.state == before


def test_expired_entry_stops_tracking_at_its_continuous_end():
    run, entry = tracked_entry()
    braking = AttitudeTrajectory.braking(
        run.acs._executor.predict(float(entry.end)),
        run.config.spacecraft_bus.attitude_control,
    )
    assert braking is not None and braking.end < T0 + 60
    run._finish_active_entry(T0 + 60)
    run.acs.pointing(T0 + 60)
    assert run.acs._tracked_slew is None
    assert np.linalg.norm(run.acs.angular_velocity_body) == pytest.approx(0)
    assert run.acs._executor.state.attitude == pytest.approx(
        braking.state(braking.end).attitude
    )


def test_future_handoff_is_predicted_while_ingress_is_still_under_way():
    run, entry = tracked_entry((0.0, 0.0, 0.0))
    assert run.acs.get_mode(T0 + 20) == ACSMode.SLEWING
    before = run.acs._executor.state
    state = run.acs.science_handoff_state(entry, float(entry.end))
    assert state is not None and state.utime > float(entry.end)
    assert np.linalg.norm(state.angular_velocity_body) == pytest.approx(0)
    assert run.acs._executor.state == before


def test_dispatch_predicts_the_brake_it_commands_during_moving_ingress():
    run, _ = tracked_entry((0.0, 0.0, 0.0))
    now = T0 + 20
    queue = object.__new__(QueueDITL)
    queue.acs = run.acs
    before = run.acs._executor.state
    brake = AttitudeTrajectory.braking(
        run.acs._executor.predict(now), run.config.spacecraft_bus.attitude_control
    )
    assert brake is not None
    assert queue._ppt_slew_execution_time(now) == brake.end
    pose = queue._expected_slew_start_attitude(now, brake.end + 60)
    assert pose == pytest.approx(brake.state(brake.end).attitude)
    assert run.acs._executor.state == before
    run.acs.end_science_observation(now)
    run.acs.pointing(brake.end + 60)
    assert run.acs._executor.state.attitude == pytest.approx(pose)


def test_dispatch_too_during_moving_ingress_collects_both_sequence_steps():
    config = make_config(
        2,
        weights=TargetConfig(urgency_weight=0, completion_deficit_weight=0),
        open_sky=True,
    )
    config.payload.observation_timing = ObservationTiming()
    config.battery.planning_level = None
    limits = config.spacecraft_bus.attitude_control
    limits.max_slew_rate, limits.slew_acceleration, limits.settle_time = 0.1, 0.01, 0
    mover = MovingPointing(
        config=config,
        obsid=700,
        merit=100,
        exptime=3600,
        ss_min=MIN,
        ss_max=3600,
        roll_deg=0,
        track=track_from_table([T0, T0 + 7200], [105, 107], [10, 10]),
    )
    run = create_ditl(config, [mover], BEGIN, BEGIN + timedelta(hours=2))
    assert isinstance(run, QueueDITL)
    run.step_size = 60
    run.submit_too(
        TOORequest(
            request=Pointing(
                config=config,
                obsid=900,
                ra=110,
                dec=10,
                merit=1000,
                sequence=ObservationSequence(
                    steps=[
                        SequenceStep(name="first", duration_seconds=9 * MIN),
                        SequenceStep(name="second", duration_seconds=MIN),
                    ]
                ),
            ),
            submit_time=T0 + MIN,
            deadline=T0 + 40 * MIN,
        )
    )
    assert run.calc()
    entries = [e for e in run.plan if e.obsid == 900]
    assert [e.step for e in entries] == ["first", "second"]
    assert [e.exposure for e in entries] == [9 * MIN, MIN]
    assert collected_seconds(run, 900) == 10 * MIN
    assert run.validate_plan_matches_execution() == []


def test_rapid_replan_does_not_lose_a_step_after_a_short_tracking_cut():
    config = make_config(
        2,
        weights=TargetConfig(urgency_weight=0, completion_deficit_weight=0),
        open_sky=True,
    )
    config.payload.observation_timing = ObservationTiming()
    config.battery.planning_level = None
    limits = config.spacecraft_bus.attitude_control
    limits.max_slew_rate, limits.slew_acceleration, limits.settle_time = 0.1, 0.01, 0
    mover = MovingPointing(
        config=config,
        obsid=700,
        merit=100,
        exptime=3600,
        ss_min=MIN,
        ss_max=3600,
        roll_deg=0,
        track=track_from_table([T0, T0 + 7200], [105, 107], [10, 10]),
    )
    run = RollingHorizonDITL(
        config=config,
        targets=[mover],
        begin=BEGIN,
        end=BEGIN + timedelta(hours=2),
        horizon=timedelta(hours=2),
        replan_interval=timedelta(hours=2),
        include_passes=False,
    )
    run.step_size = 60
    run.submit_too(
        obsid=900,
        ra=110,
        dec=10,
        merit=1000,
        exptime=10 * MIN,
        submit_time=T0 + MIN,
        deadline=T0 + 40 * MIN,
    )
    assert run.calc()
    assert any(r.interrupted_obsid == mover.obsid for r in run.replans)
    collected = collected_seconds(run, 900, science=True)
    assert collected == 10 * MIN, f"collected {collected} of {10 * MIN} seconds"
    assert run.validate_plan_matches_execution() == []


def test_planner_reserves_the_terminal_tracking_brake():
    from .test_moving_pointing import END, HOURS, STEP, _config, _fillers, _mover

    config = _config(HOURS)
    mover = _mover(config)
    scheduler = PriorityScheduler(
        config, [mover, *_fillers(config)], BEGIN, END, step_size=STEP
    )
    scheduler.schedule()
    [block] = [b for b in scheduler.timeline if b.entry.obsid == 700]
    assert block.slew is not None
    limits = config.spacecraft_bus.attitude_control
    physical_slew = block.slew.model_copy(
        update={"at": block.entry, "obstype": ObsType.PPT}
    )
    ingress = AttitudeTrajectory.from_slew(physical_slew, limits)
    run = DITL(config=config, plan=Plan(entries=[block.entry]), begin=BEGIN, end=END)
    motion = run.acs._tracking_after(physical_slew, ingress, limits)
    assert motion is not None and motion.end > float(block.entry.end)
    assert block.end == motion.end
    assert block.attitude_out == pytest.approx(motion.state(motion.end).attitude)
    following = [b for b in scheduler.timeline if b.entry.begin > block.entry.begin]
    assert all(float(b.entry.begin) >= motion.end for b in following)
