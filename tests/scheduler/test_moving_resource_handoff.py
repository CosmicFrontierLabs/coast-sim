"""Moving resource reservations include the actual physical tracking stop."""

from datetime import timedelta

import numpy as np
import pytest

from conops import DITL
from conops.common import ACSMode, ObsType
from conops.config import ObservationTiming
from conops.ditl import RollingHorizonDITL
from conops.schedulers import PriorityScheduler
from conops.simulation.attitude import AttitudeTrajectory
from conops.targets import Plan, PlanEntry, track_from_table

from .planning_scenario import BEGIN, T0, make_config


def tracked_entry(initial_pose=(105.0, 10.0, 0.0)):
    config = make_config(2)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
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


def test_future_handoff_is_predicted_while_ingress_is_still_under_way():
    run, entry = tracked_entry((0.0, 0.0, 0.0))
    assert run.acs.get_mode(T0 + 20) == ACSMode.SLEWING
    before = run.acs._executor.state
    state = run.acs.science_handoff_state(entry, float(entry.end))
    assert state is not None and state.utime > float(entry.end)
    assert np.linalg.norm(state.angular_velocity_body) == pytest.approx(0)
    assert run.acs._executor.state == before


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
