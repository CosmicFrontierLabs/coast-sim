"""Contact forecasts use the physical tracking profile installed by ACS."""

from datetime import timedelta

import numpy as np
import pytest

from conops.common import ACSCommandType, ObsType
from conops.ditl import RollingHorizonDITL
from conops.schedulers.context import SchedulingContext
from conops.schedulers.momentum import MomentumForecast
from conops.schedulers.priority_scheduler import _Block
from conops.simulation.acs_command import ACSCommand
from conops.simulation.attitude import AttitudeExecutor
from conops.simulation.passes import Pass
from conops.targets import PlanEntry

from .test_momentum_ledger import BEGIN, T0, _config

INITIAL = (0.1, -0.2, 0.3)


def _scenario(late=0, wait=0):
    config = _config()
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    gspass = Pass(
        config=config,
        station="Test station",
        obsid=900,
        begin=T0,
        length=600,
        utime=[T0 + k * 60 for k in range(11)],
        ra=[80 + k * 0.8 for k in range(11)],
        dec=[-20] * 11,
        roll=[10] * 11,
    )
    acquire = T0 + late
    begin = acquire - wait
    profile = list(zip(gspass.ra, gspass.dec, gspass.roll, strict=True))
    origin = gspass.attitude_for_profile_at(profile, acquire)
    assert origin is not None
    entry = PlanEntry(
        config=config,
        obstype=ObsType.GSP,
        obsid=gspass.obsid,
        begin=begin,
        end=gspass.end,
        contact_begin=acquire,
        contact_end=gspass.end,
        slewtime=0,
        ra=origin[0],
        dec=origin[1],
        roll=origin[2],
    )
    run = RollingHorizonDITL(
        config, [], begin=BEGIN, end=BEGIN + timedelta(seconds=720)
    )
    run.step_size = 10
    run._entry_passes = {id(entry): gspass}
    run.acs.ra, run.acs.dec, run.acs.roll = origin
    run.acs._executor = AttitudeExecutor(begin, origin)
    run.acs.passrequests.passes = [gspass]
    run.acs._start_pass(
        ACSCommand(command_type=ACSCommandType.START_PASS, execution_time=acquire),
        acquire,
    )
    motion = run.acs._executor._trajectory
    assert motion is not None and motion.end > entry.end
    context = SchedulingContext(
        config, BEGIN + timedelta(seconds=begin - T0), BEGIN + timedelta(seconds=720)
    )
    block = _Block(
        entry=entry,
        attitude_in=origin,
        attitude_out=motion.state(motion.end).attitude,
        ready=acquire,
        end=motion.end,
        gspass=gspass,
    )
    return config, context, block, motion, run


def _physical(run, motion, origin, begin, end, initial):
    # The ACS motion's start and physical stop are integration boundaries,
    # including a wait before acquisition and an off-grid terminal brake.
    cursor = begin
    for stop in sorted({motion.start, motion.end, end}):
        stop = min(stop, end)
        if stop <= cursor:
            continue
        initial = run._forecast_momentum(
            cursor,
            stop,
            lambda t: origin if t < motion.start else motion.state(t).attitude,
            start_eci_n_m_s=initial,
        ).end_eci_n_m_s
        cursor = stop
    return initial


@pytest.mark.parametrize("model", ["committed", "planner"])
@pytest.mark.parametrize("late,wait", [(0, 0), (30, 0), (60, 60)])
def test_contact_forecast_tracks_acquisition_and_terminal_braking(model, late, wait):
    config, context, block, motion, run = _scenario(late, wait)
    expected = _physical(
        run, motion, block.attitude_in, context.ustart, context.uend, INITIAL
    )
    run._stored_momentum_tracker.start_from_eci(INITIAL)
    before = run._stored_momentum_tracker.momentum_eci_n_m_s.copy()
    if model == "committed":
        # This variant predicts a contact that has not yet been installed.
        run.acs.current_pass = None
        actual = run._momentum_start(context.ustart, context.uend, [block.entry])
    else:
        forecast = MomentumForecast(config, context, start_momentum_eci_n_m_s=INITIAL)
        forecast.profile([block])
        actual = forecast.tracker.momentum_eci_n_m_s
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(
        run._stored_momentum_tracker.momentum_eci_n_m_s, before
    )


@pytest.mark.parametrize("late", [0, 30])
@pytest.mark.parametrize("at", [300, 600.02])
def test_contact_forecast_during_tracking_or_braking(late, at):
    _, context, block, motion, run = _scenario(late)
    begin = T0 + at
    initial = _physical(run, motion, block.attitude_in, context.ustart, begin, INITIAL)
    expected = _physical(run, motion, block.attitude_in, begin, context.uend, initial)
    run._stored_momentum_tracker.start_from_eci(initial)
    before = run._stored_momentum_tracker.momentum_eci_n_m_s.copy()
    if begin > block.entry.end:
        # END_PASS has run, but its physical brake has not finished yet.
        run.acs.current_pass = None
    actual = run._momentum_start(begin, context.uend, [block.entry])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(
        run._stored_momentum_tracker.momentum_eci_n_m_s, before
    )


def test_contact_wheel_demand_includes_tracking_rate():
    config, context, block, motion, _ = _scenario()
    momentum = config.spacecraft_bus.attitude_control.stored_momentum
    momentum.gravity_gradient_enabled = False
    momentum.capacity_n_m_s = 1
    momentum.slew_momentum_enabled = True
    forecast = MomentumForecast(config, context, start_momentum_eci_n_m_s=(0, 0, 0))
    times, demand = forecast.demand([block])
    inertia = np.asarray(config.spacecraft_bus.inertia_tensor_body_kg_m2)
    expected = [
        np.linalg.norm(inertia @ np.deg2rad(motion.state(t).angular_velocity_body))
        for t in times
    ]
    assert max(expected) > 0.1
    np.testing.assert_allclose(demand, expected, rtol=0, atol=1e-10)
    assert demand[-1] == pytest.approx(0)


def test_contact_cache_distinguishes_selected_profiles_with_identical_endpoints():
    config, context, block, _, _ = _scenario()
    profile = block.gspass.ra.copy()
    profile[5] += 1
    different = block.model_copy(
        update={"gspass": block.gspass.model_copy(update={"ra": profile})}
    )
    forecast = MomentumForecast(config, context, start_momentum_eci_n_m_s=INITIAL)
    forecast.profile([block])
    first = forecast.tracker.momentum_eci_n_m_s.copy()
    forecast.profile([different])
    changed = forecast.tracker.momentum_eci_n_m_s.copy()
    fresh = MomentumForecast(config, context, start_momentum_eci_n_m_s=INITIAL)
    fresh.profile([different])
    np.testing.assert_array_equal(changed, fresh.tracker.momentum_eci_n_m_s)
    assert np.linalg.norm(first - changed) > 1e-4


@pytest.mark.parametrize("model", ["committed", "planner"])
def test_contact_forecast_can_end_during_the_acquisition_wait(model):
    config, _, block, motion, run = _scenario(late=60, wait=60)
    stop = T0 + 30
    expected = _physical(run, motion, block.attitude_in, T0, stop, INITIAL)
    if model == "committed":
        run._stored_momentum_tracker.start_from_eci(INITIAL)
        actual = run._momentum_start(T0, stop, [block.entry])
    else:
        context = SchedulingContext(config, BEGIN, BEGIN + timedelta(seconds=30))
        forecast = MomentumForecast(config, context, start_momentum_eci_n_m_s=INITIAL)
        times, _ = forecast.profile([block])
        assert times[-1] == stop
        actual = forecast.tracker.momentum_eci_n_m_s
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_installed_contact_profile_survives_a_new_selected_profile():
    _, context, block, motion, run = _scenario()
    profile = block.gspass.ra.copy()
    profile[5] += 1
    # The plan's new pass object names the same contact, but selecting its
    # future profile cannot rewrite the motion already installed in ACS.
    replacement = block.gspass.model_copy(update={"ra": profile})
    run._entry_passes[id(block.entry)] = replacement
    run._stored_momentum_tracker.start_from_eci(INITIAL)
    before = run.acs._executor.state
    expected = _physical(
        run, motion, block.attitude_in, context.ustart, context.uend, INITIAL
    )
    actual = run._momentum_start(context.ustart, context.uend, [block.entry])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    assert run.acs._executor.state == before
