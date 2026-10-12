from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from conops import (
    AttitudeControlSystem,
    MissionConfig,
    SpacecraftBus,
    StoredMomentumConfig,
)
from conops.common.vector import attitude_to_quat
from conops.simulation.attitude import AttitudeExecutor, AttitudeTrajectory
from conops.simulation.momentum import StoredMomentumTracker
from conops.simulation.momentum_budget import MomentumBudget, MomentumLimitError


def setup_budget(
    *,
    capacity=20.0,
    gravity=False,
    inertia=(1000.0, 1000.0, 1000.0),
    attitude=(0.0, 0.0, 0.0),
    step=2,
):
    epoch = datetime(2027, 1, 1, tzinfo=timezone.utc)
    ephem = SimpleNamespace(
        timestamp=[epoch + timedelta(seconds=i) for i in range(0, 3601, step)],
        gcrs_pv=SimpleNamespace(
            position=np.tile([7000.0, 0.0, 0.0], (3600 // step + 1, 1))
        ),
    )
    config = MissionConfig(
        spacecraft_bus=SpacecraftBus(
            inertia_tensor_body_kg_m2=np.diag(inertia),
            attitude_control=AttitudeControlSystem(
                max_slew_rate=2.0,
                slew_acceleration=1.0,
                settle_time=90,
                stored_momentum=StoredMomentumConfig(
                    capacity_n_m_s=capacity, gravity_gradient_enabled=gravity
                ),
            ),
        )
    )
    executor = AttitudeExecutor(epoch.timestamp(), attitude)
    executor.momentum = MomentumBudget(config, ephem, executor.state)
    return config, ephem, executor


def test_short_slew_peak_cannot_hide_between_scheduler_ticks():
    config, _, executor = setup_budget(capacity=10.0)
    start = executor.state.utime
    trajectory = AttitudeTrajectory.turn(
        start,
        executor.state.attitude,
        (0, 0, 1),
        config.spacecraft_bus.attitude_control,
    )
    assert trajectory.end - start < 60
    assert (
        trajectory.state(start).angular_velocity_body
        == trajectory.state(trajectory.end).angular_velocity_body
        == (0, 0, 0)
    )
    state, momentum = executor.state, executor.momentum.current
    with pytest.raises(MomentumLimitError, match="Actuator momentum exceeds"):
        executor.install(trajectory)
    assert executor.state is state and executor.momentum.current is momentum


def test_speculative_forecasts_do_not_consume_momentum_or_time():
    config, _, executor = setup_budget(
        capacity=50.0, gravity=True, inertia=(1000, 800, 600), attitude=(0, 45, 0)
    )
    start = executor.state.utime
    before = executor.momentum.current
    a = executor.forecast_momentum(start + 100)
    b = executor.forecast_momentum(start + 100)
    assert a.sample == b.sample and a.peak_n_m_s == b.peak_n_m_s
    assert a.sample.stored_momentum_norm_n_m_s > 0.01
    assert executor.momentum.current is before and executor.state.utime == start
    executor.advance(start + 100)
    assert executor.momentum.current.sample == a.sample


def test_actuator_demand_is_vector_difference_not_sum_of_norms():
    tracker = StoredMomentumTracker(
        np.eye(3) * 1000, (0, 0, 5), gravity_gradient_enabled=False
    )
    q = attitude_to_quat(0, 0, 0)
    for time, rate in [(0, (0, 0, 0)), (1, (0, 0, 0.01))]:
        sample = tracker.update(
            utime=time,
            position_eci_km=[7000, 0, 0],
            attitude_quaternion_eci_to_body=q,
            angular_velocity_body_rad_s=rate,
        )
    assert sample.stored_momentum_norm_n_m_s == pytest.approx(5)
    assert sample.actuator_momentum_norm_n_m_s == pytest.approx(5)
    assert sample.actuator_momentum_body_n_m_s == pytest.approx((0, 0, -5))


def test_initial_moving_body_is_included_in_total_momentum():
    tracker = StoredMomentumTracker(
        np.eye(3) * 1000, (1, 2, 3), gravity_gradient_enabled=False
    )
    sample = tracker.update(
        utime=0,
        position_eci_km=[7000, 0, 0],
        attitude_quaternion_eci_to_body=attitude_to_quat(20, 30, 40),
        angular_velocity_body_rad_s=[0.01, 0.02, 0.03],
    )
    assert sample.actuator_momentum_body_n_m_s == pytest.approx((1, 2, 3))


def test_idle_overflow_is_not_silently_counted_as_valid_execution():
    _, _, executor = setup_budget(
        capacity=0.01, gravity=True, inertia=(1000, 800, 600), attitude=(0, 45, 0)
    )
    state, before = executor.state, executor.momentum.current
    with pytest.raises(MomentumLimitError):
        executor.advance(state.utime + 100)
    assert executor.state is state and executor.momentum.current is before


def test_momentum_is_carried_across_installed_trajectories():
    config, _, executor = setup_budget(
        capacity=50, gravity=True, inertia=(1000, 800, 600), attitude=(0, 45, 0)
    )
    start = executor.state.utime
    executor.advance(start + 100)
    accumulated = executor.momentum.current.sample.stored_momentum_norm_n_m_s
    trajectory = AttitudeTrajectory.turn_from_state(
        executor.state, (10, 50, 20), config.spacecraft_bus.attitude_control
    )
    forecast = executor.check_momentum(trajectory)
    executor.install(trajectory)
    assert executor.momentum.current.sample.stored_momentum_norm_n_m_s == pytest.approx(
        accumulated
    )
    executor.advance(trajectory.end)
    assert executor.momentum.current.sample == forecast.sample


def test_forecast_checks_ephemeris_coverage():
    _, _, executor = setup_budget()
    with pytest.raises(ValueError, match="ephemeris coverage"):
        executor.forecast_momentum(executor.state.utime + 3601)


def test_fine_motion_does_not_disguise_coarse_orbit():
    with pytest.raises(ValueError, match="finer ephemeris"):
        setup_budget(gravity=True, step=60)


def test_contact_tracking_uses_the_same_momentum_guard():
    config, _, executor = setup_budget(capacity=10.0)
    start = executor.state.utime
    tracking = AttitudeTrajectory.tracking(
        [(start, (0, 0, 0)), (start + 10, (10, 0, 0)), (start + 20, (10, 0, 0))],
        config.spacecraft_bus.attitude_control,
        initial_rate=(0, 0, 0),
    )
    with pytest.raises(MomentumLimitError):
        executor.install(tracking)
    assert executor._trajectory is None


def test_braking_cannot_hide_momentum_transfer_from_the_body():
    config, ephem, prior = setup_budget(capacity=10.0)
    executor = AttitudeExecutor(prior.state.utime, (0, 0, 0), (0, 0, 1))
    executor.momentum = MomentumBudget(config, ephem, executor.state)
    state, before = executor.state, executor.momentum.current
    with pytest.raises(MomentumLimitError):
        executor.request_stop(state.utime, config.spacecraft_bus.attitude_control)
    assert executor.state is state and executor.momentum.current is before


@pytest.mark.parametrize("initial", [(11, 0, 0), (7, 7, 7)])
def test_initial_actuator_momentum_must_fit_capacity(initial):
    config, ephem, executor = setup_budget(capacity=10.0)
    config.spacecraft_bus.attitude_control.stored_momentum.initial_momentum_body_n_m_s = initial
    with pytest.raises(MomentumLimitError):
        MomentumBudget(config, ephem, executor.state)


@pytest.mark.parametrize("capacity", [0, -1, float("nan"), float("inf")])
def test_invalid_capacity_rejected(capacity):
    with pytest.raises(ValueError):
        StoredMomentumConfig(capacity_n_m_s=capacity)


def test_capacity_requires_inertia_even_without_external_torque():
    with pytest.raises(ValueError, match="inertia_tensor"):
        SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                stored_momentum=StoredMomentumConfig(capacity_n_m_s=5)
            )
        )


@pytest.mark.parametrize(
    "fractions", [(), (0.5,), (1, 0.5, 0.75), (1, 0), (1, float("nan"))]
)
def test_invalid_speed_search_rejected(fractions):
    with pytest.raises(ValueError):
        StoredMomentumConfig(slew_rate_fractions=fractions)
