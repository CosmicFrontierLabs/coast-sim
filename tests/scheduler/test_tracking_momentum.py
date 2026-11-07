"""Planner momentum follows the continuous science motion used by ACS."""

from datetime import timedelta

import numpy as np
import pytest

from conops.common import ObsType
from conops.common.vector import quaternion_to_rotation_matrix
from conops.ditl import RollingHorizonDITL
from conops.schedulers.context import SchedulingContext
from conops.schedulers.momentum import MomentumForecast
from conops.schedulers.priority_scheduler import _Block
from conops.simulation.attitude import AttitudeTrajectory
from conops.simulation.slew import Slew
from conops.targets import PlanEntry, track_from_table

from .test_momentum_ledger import BEGIN, T0, _config


def _scenario(idle_seconds=120, middle=84):
    config = _config()
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    origin = (80.0, -20.0, 10.0)
    entry = PlanEntry(
        config=config,
        obsid=702,
        obstype=ObsType.PPT,
        ra=80,
        dec=-20,
        roll=10,
        begin=T0,
        end=T0 + 1200,
        slewtime=0,
        track=track_from_table(
            [T0, T0 + 600, T0 + 1200, T0 + 3600],
            [80, middle, 88, 100],
            [-20] * 4,
        ),
    )
    limits = config.spacecraft_bus.attitude_control
    end = BEGIN + timedelta(seconds=1200 + idle_seconds)
    run = RollingHorizonDITL(config, [], begin=BEGIN, end=end)
    run.step_size = 10
    slew = Slew.idle_hold(config, *origin, T0)
    slew.obstype, slew.at, slew.obsid = ObsType.PPT, entry, entry.obsid
    ingress = AttitudeTrajectory.from_slew(slew, limits)
    motion = run.acs._tracking_after(slew, ingress, limits)
    assert motion is not None
    context = SchedulingContext(config, BEGIN, end)
    block = _Block(
        entry=entry,
        attitude_in=origin,
        attitude_out=motion.state(motion.end).attitude,
        ready=T0,
        end=motion.end,
    )
    return config, context, block, motion, run


@pytest.mark.parametrize("idle_seconds", [0, 120])
def test_planner_integrates_tracking_and_terminal_stop(idle_seconds):
    config, context, block, motion, run = _scenario(idle_seconds)
    initial = (0.1, -0.2, 0.3)
    stop = context.uend
    rest = min(stop, motion.end)
    expected = run._forecast_momentum(
        T0, rest, lambda t: motion.state(t).attitude, start_eci_n_m_s=initial
    ).end_eci_n_m_s
    if stop > rest:
        expected = run._forecast_momentum(
            rest, stop, lambda t: motion.state(t).attitude, start_eci_n_m_s=expected
        ).end_eci_n_m_s
    forecast = MomentumForecast(config, context, start_momentum_eci_n_m_s=initial)
    times, _ = forecast.profile([block])
    assert times[-1] == stop
    np.testing.assert_allclose(
        forecast.tracker.momentum_eci_n_m_s, expected, rtol=0, atol=1e-10
    )


def test_tracking_demand_includes_body_angular_velocity():
    config, context, block, motion, _ = _scenario()
    momentum = config.spacecraft_bus.attitude_control.stored_momentum
    momentum.gravity_gradient_enabled = False
    momentum.capacity_n_m_s = 1
    momentum.slew_momentum_enabled = True
    forecast = MomentumForecast(config, context, start_momentum_eci_n_m_s=(0, 0, 0))
    times, demand = forecast.demand([block])
    inertia = np.asarray(config.spacecraft_bus.inertia_tensor_body_kg_m2)
    expected = []
    for time in times:
        state = motion.state(time)
        # No external torque and zero initial H: the wheels counter the
        # body's I*omega during tracking, then unload to zero as it stops.
        body = quaternion_to_rotation_matrix(np.asarray(state.quaternion)) @ np.zeros(3)
        expected.append(
            np.linalg.norm(body - inertia @ np.deg2rad(state.angular_velocity_body))
        )
    assert max(expected) > 0.05
    np.testing.assert_allclose(demand, expected, rtol=0, atol=1e-10)
    assert demand[-1] == pytest.approx(0)
    assert forecast.loading(block) >= max(expected) - 1e-10


def test_tracking_cache_distinguishes_paths_with_the_same_endpoints(monkeypatch):
    config, context, first, _, _ = _scenario(middle=83)
    _, _, second, _, _ = _scenario(middle=86)
    second.entry.config = config
    forecast = MomentumForecast(
        config, context, start_momentum_eci_n_m_s=(0.1, -0.2, 0.3)
    )
    forecast.profile([first])
    first_h = forecast.tracker.momentum_eci_n_m_s.copy()
    with monkeypatch.context() as patch:

        def uncached_update(**kwargs):
            pytest.fail("An equivalent rebuilt track should reuse its integration")

        patch.setattr(forecast.tracker, "update", uncached_update)
        forecast.profile([first.model_copy(update={"entry": first.entry.model_copy()})])
    forecast.profile([second])
    second_h = forecast.tracker.momentum_eci_n_m_s.copy()
    fresh = MomentumForecast(config, context, start_momentum_eci_n_m_s=(0.1, -0.2, 0.3))
    fresh.profile([second])
    np.testing.assert_array_equal(second_h, fresh.tracker.momentum_eci_n_m_s)
    assert np.linalg.norm(first_h - second_h) > 1e-4


def test_loading_covers_the_whole_block_beyond_a_short_context():
    config, context, block, _, _ = _scenario()
    full = MomentumForecast(config, context).loading(block)
    short = SchedulingContext(config, BEGIN, BEGIN + timedelta(seconds=300))
    assert full > 0.5
    assert MomentumForecast(config, short).loading(block) == pytest.approx(
        full, rel=0, abs=1e-10
    )
