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
from conops.common.quaternion_curve import QuaternionHermite
from conops.common.vector import attitude_to_quat, quaternion_to_rotation_matrix
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
                    capacity_n_m_s=capacity,
                    gravity_gradient_enabled=gravity,
                    enforce_capacity=True,
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
    with pytest.raises(MomentumLimitError, match="Actuator momentum exceeds"):
        executor.install(trajectory, momentum_end=start)
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


def short_tracking_curve(config, executor, tail=100.0):
    start = executor.state.utime
    return AttitudeTrajectory.tracking(
        [
            (start, (0, 0, 0)),
            (start + 0.2, (0.005, 0, 0)),
            (start + 0.4, (0, 0, 0)),
            (start + tail, (0, 0, 0)),
        ],
        config.spacecraft_bus.attitude_control,
        initial_rate=(0, 0, 0),
    )


@pytest.mark.parametrize("tail", [60.0, 100.0, 101.0])
def test_tracking_interior_peak_cannot_hide_between_integration_samples(tail):
    config, _, executor = setup_budget(capacity=0.5)
    trajectory = short_tracking_curve(config, executor, tail)
    start = trajectory.start
    peak_rate = trajectory.state(start + 0.1).angular_velocity_body
    assert np.linalg.norm(np.deg2rad(peak_rate)) * 1000 == pytest.approx(
        0.654498, rel=1e-5
    )
    before = executor.momentum.current
    with pytest.raises(
        MomentumLimitError, match="Actuator momentum could not be certified"
    ):
        executor.check_momentum(trajectory)
    with pytest.raises(
        MomentumLimitError, match="Actuator momentum could not be certified"
    ):
        executor.install(trajectory)
    assert executor.momentum.current is before
    assert executor.state.utime == start
    assert executor._trajectory is None


def test_execution_checks_tracking_interior_peak_without_committing_failure():
    config, _, executor = setup_budget(capacity=0.7)
    trajectory = short_tracking_curve(config, executor)
    executor.install(trajectory)
    # Tighten the guard after installation to exercise the execution check,
    # independently of the admission check that normally catches this curve.
    executor.momentum.capacity = 0.5
    state, before = executor.state, executor.momentum.current
    with pytest.raises(
        MomentumLimitError, match="Actuator momentum could not be certified"
    ):
        executor.advance(trajectory.start + 60)
    assert executor.state is state
    assert executor.momentum.current is before


def test_feasible_short_tracking_curve_is_not_rejected_by_the_bound():
    config, _, executor = setup_budget(capacity=0.7)
    trajectory = short_tracking_curve(config, executor)
    executor.install(trajectory)
    executor.advance(trajectory.start + 0.05)
    executor.advance(trajectory.start + 0.15)
    executor.advance(trajectory.start + 60)
    executor.advance(trajectory.end)
    assert executor.momentum.current.utime == trajectory.end


@pytest.mark.parametrize("first,last", [(0.0, 1.0), (0.2, 0.7), (0.47, 0.52)])
def test_tracking_momentum_bound_matches_dense_vector_demand(first, last):
    curve = QuaternionHermite(
        tuple(attitude_to_quat(20, -10, 35)),
        tuple(attitude_to_quat(23, -8, 40)),
        (0.1, -0.2, 0.15),
        (-0.15, 0.1, 0.2),
        30.0,
    )
    inertia = np.array(
        [[1000.0, 120.0, 90.0], [120.0, 850.0, -70.0], [90.0, -70.0, 600.0]]
    )
    # Nonzero initial momentum plus a changing torque, in local interval time.
    momentum = np.array([[-0.4, 0.7, -0.2], [0.03, -0.08, 0.1], [0.06, 0.03, -0.07]])
    demands = []
    for u in np.linspace(0, 1, 2001):
        q, rate, _ = curve.evaluate(first + (last - first) * u)
        total = momentum[0] + u * momentum[1] + u**2 * momentum[2]
        demands.append(
            np.linalg.norm(
                quaternion_to_rotation_matrix(q) @ total - inertia @ np.deg2rad(rate)
            )
        )
    peak = max(demands)
    assert curve.within_momentum_limit(first, last, inertia, momentum, peak * 1.01)
    assert not curve.within_momentum_limit(first, last, inertia, momentum, peak * 0.99)


def test_tracking_momentum_bound_retains_vector_cancellation():
    curve = QuaternionHermite(
        tuple(attitude_to_quat(0, 0, 0)),
        tuple(attitude_to_quat(0.005, 0, 0)),
        (0, 0, 0),
        (0, 0, 0),
        0.2,
    )
    inertia = np.eye(3) * 1000
    momentum = np.array([[0.0, 0.0, 0.35], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # I omega alone reaches 0.6545, but H - I omega stays within +/-0.35.
    assert curve.within_momentum_limit(0, 1, inertia, momentum, 0.4)
    assert not curve.within_momentum_limit(0, 1, inertia, -momentum, 0.4)


def test_tracking_with_gravity_gradient_carries_the_integrated_momentum():
    config, _, executor = setup_budget(
        capacity=1.0, gravity=True, inertia=(1000, 800, 600)
    )
    trajectory = short_tracking_curve(config, executor)
    before = executor.momentum.current
    forecast = executor.check_momentum(trajectory)
    assert forecast.uncertified_interval is None
    assert forecast.sample.stored_momentum_norm_n_m_s > 0
    assert executor.momentum.current is before
    executor.install(trajectory)
    executor.advance(trajectory.end)
    assert executor.momentum.current.sample == forecast.sample


def test_uncertified_tracking_interval_remains_rejected_when_forecast_is_extended():
    config, _, executor = setup_budget(capacity=0.5)
    trajectory = short_tracking_curve(config, executor)
    forecast = executor.momentum.forecast(
        trajectory.state, trajectory.start + 0.2, trajectory=trajectory
    )
    assert forecast.uncertified_interval is not None
    extended = executor.momentum.forecast(
        trajectory.state, trajectory.end, initial=forecast, trajectory=trajectory
    )
    assert extended.uncertified_interval == forecast.uncertified_interval
    with pytest.raises(
        MomentumLimitError, match="Actuator momentum could not be certified"
    ):
        executor.momentum.commit(extended)


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
                stored_momentum=StoredMomentumConfig(
                    capacity_n_m_s=5, enforce_capacity=True
                )
            )
        )


@pytest.mark.parametrize(
    "fractions", [(), (0.5,), (1, 0.5, 0.75), (1, 0), (1, float("nan"))]
)
def test_invalid_speed_search_rejected(fractions):
    with pytest.raises(ValueError):
        StoredMomentumConfig(slew_rate_fractions=fractions)


def test_slew_interior_peak_is_certified_between_grid_samples():
    config, ephem, executor = setup_budget(capacity=1000)
    config.spacecraft_bus.inertia_tensor_body_kg_m2 = [
        [1000, 100, 150],
        [100, 900, 100],
        [150, 100, 800],
    ]
    config.spacecraft_bus.attitude_control.stored_momentum.initial_momentum_body_n_m_s = (
        10,
        7,
        3,
    )
    executor.momentum = MomentumBudget(config, ephem, executor.state)
    trajectory = AttitudeTrajectory.turn_from_state(
        executor.state, (0, 0, 150), config.spacecraft_bus.attitude_control
    )
    forecast = executor.check_momentum(trajectory)
    # The constant-rate interior peak lies between the five-degree samples.
    assert (
        max(node.sample.actuator_momentum_norm_n_m_s for node in forecast.path.nodes)
        < 28.5268
    )
    executor.momentum.capacity = 28.5268
    before = executor.momentum.current
    with pytest.raises(MomentumLimitError, match="could not be certified"):
        executor.install(trajectory)
    assert executor.momentum.current is before
    assert executor._trajectory is None


def test_chunked_execution_reuses_admission_grid_and_torque_integral():
    config, _, executor = setup_budget(
        capacity=50, gravity=True, inertia=(1000, 800, 600), attitude=(0, 45, 0)
    )
    start = executor.state.utime
    trajectory = AttitudeTrajectory.turn_from_state(
        executor.state, (90, 30, 70), config.spacecraft_bus.attitude_control
    )
    end = trajectory.end + 127.31
    expected = executor.check_momentum(trajectory, end)
    executor.install(trajectory, momentum_end=end)
    path = executor.momentum.current.path
    nodes = tuple(path.times)
    for time in np.arange(start + 0.731, end, 3.173):
        executor.advance(float(time))
        assert executor.momentum.current.path is path
    executor.advance(end)
    assert tuple(path.times) == nodes
    assert executor.momentum.current.sample == expected.sample
    assert executor.momentum.current.peak_n_m_s == expected.peak_n_m_s


def test_install_certifies_pending_motion_and_full_hold_once(monkeypatch):
    config, _, executor = setup_budget(capacity=50)
    start = executor.state.utime
    trajectory = AttitudeTrajectory.turn_from_state(
        executor.state, (40, 15, 0), config.spacecraft_bus.attitude_control
    )
    budget = executor.momentum
    original = budget.forecast
    forecasts = []

    def record(state_at, end, **kwargs):
        forecasts.append(end)
        return original(state_at, end, **kwargs)

    monkeypatch.setattr(budget, "forecast", record)
    executor.install(trajectory, momentum_end=trajectory.end + 100)
    assert forecasts == [start, trajectory.end + 100]


@pytest.mark.parametrize(
    "coefficients",
    [
        ((1, 2, 3), (150, -100, 80), (120, 80, 40)),  # growing H, far beyond midpoint
        ((1, 2, 3), (400, 120, -100), (-400, -100, 100)),  # interior peak
        ((100, -50, 80), (-200, 100, -160), (0, 0, 0)),  # zero midpoint H
    ],
)
def test_fixed_axis_bound_covers_changing_quadratic_momentum(coefficients):
    from conops.simulation.attitude import AttitudeState

    _, _, executor = setup_budget(capacity=1000, inertia=(1000, 800, 600))
    budget = executor.momentum
    initial = AttitudeState(0, tuple(attitude_to_quat(15, 25, 10)), (0, 0, 20))
    duration = 3.0
    coefficients = np.asarray(coefficients, dtype=float)
    # Independent dense evaluation of the physical demand, without the
    # midpoint variation bound or its recursive subdivision.
    norms = []
    for time in np.linspace(0, duration, 10001):
        state = initial.coast(float(time))
        fraction = time / duration
        total = (
            coefficients[0] + fraction * coefficients[1] + fraction**2 * coefficients[2]
        )
        demand = quaternion_to_rotation_matrix(
            state.quaternion
        ) @ total - budget.inertia @ np.deg2rad(state.angular_velocity_body)
        norms.append(np.linalg.norm(demand))
    peak = max(norms)
    budget.capacity = peak * 1.001
    assert budget._fixed_axis_within_limit(initial.coast, 0, duration, coefficients)
    budget.capacity = peak * 0.999
    assert not budget._fixed_axis_within_limit(initial.coast, 0, duration, coefficients)
