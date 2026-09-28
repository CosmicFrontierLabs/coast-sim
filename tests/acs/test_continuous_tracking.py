"""Continuous-rate tracking, interval bounds, and interruption regressions."""

from functools import cached_property

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from conops import AttitudeControlSystem
from conops.common.quaternion_curve import QuaternionHermite
from conops.common.vector import (
    attitude_to_quat,
    quat_to_attitude,
    quaternion_attitude_delta,
)
from conops.simulation.attitude import (
    AttitudeExecutionError,
    AttitudeExecutor,
    AttitudeTrajectory,
)


@pytest.fixture
def limits():
    return AttitudeControlSystem(
        max_slew_rate=0.2, slew_acceleration=0.05, settle_time=0
    )


@pytest.mark.parametrize("cadence", [2, 10, 60])
def test_established_constant_rate_track_is_cadence_independent(limits, cadence):
    samples = [(float(time), (0.1 * time, 0.0, 0.0)) for time in range(0, 121, cadence)]
    trajectory = AttitudeTrajectory.tracking(samples, limits)
    executor = AttitudeExecutor(0, (0, 0, 0), (0, 0, 0.1))
    executor.install(trajectory)
    for time in np.linspace(0, 120, 97):
        state = executor.predict(time)
        assert state.attitude == pytest.approx((0.1 * time, 0, 0), abs=1e-8)
        assert state.angular_velocity_body == pytest.approx((0, 0, 0.1), abs=1e-10)
    # Profile exhaustion brakes instead of instantaneously stopping.
    assert trajectory.end == pytest.approx(122)
    assert trajectory.state(121).angular_velocity_body == pytest.approx((0, 0, 0.05))
    assert trajectory.state(trajectory.end).angular_velocity_body == (0, 0, 0)
    assert trajectory.state(122).attitude == pytest.approx((12.1, 0, 0), abs=1e-8)


def test_braking_takes_four_seconds_not_the_remaining_404(limits):
    executor = AttitudeExecutor(0, (0, 0, 0))
    trajectory = AttitudeTrajectory.turn(0, (0, 0, 0), (100, 0, 0), limits)
    executor.install(trajectory)
    executor.advance(100)
    before = executor.state
    assert trajectory.next_rest_time(100) == pytest.approx(504)
    assert executor.request_stop(100, limits) == pytest.approx(104)
    assert executor.state == before
    assert executor.predict(102).angular_velocity_body == pytest.approx((0, 0, 0.1))
    assert executor.predict(104).angular_velocity_body == (0, 0, 0)
    assert executor.predict(104).attitude == pytest.approx((20, 0, 0), abs=1e-8)
    assert executor.predict(160).attitude == pytest.approx((20, 0, 0), abs=1e-8)


def test_moving_handoff_matches_angular_velocity(limits):
    executor = AttitudeExecutor(0, (0, 0, 0), (0, 0, 0.1))
    executor.install(
        AttitudeTrajectory.tracking([(0, (0, 0, 0)), (60, (6, 0, 0))], limits)
    )
    executor.advance(20)
    before = executor.state
    replacement = AttitudeTrajectory.tracking(
        [(20, before.attitude), (80, (4, 2, 1)), (140, (6, 3, 2))],
        limits,
        initial_rate=before.angular_velocity_body,
    )
    executor.install(replacement)
    assert executor.state == before
    np.testing.assert_allclose(
        executor.predict(20 + 1e-5).angular_velocity_body,
        before.angular_velocity_body,
        atol=1e-7,
    )


def test_matching_orientation_with_wrong_rate_is_rejected(limits):
    executor = AttitudeExecutor(0, (0, 0, 0), (0, 0, 0.1))
    wrong_rate = AttitudeTrajectory.tracking([(0, (0, 0, 0)), (60, (3, 0, 0))], limits)
    with pytest.raises(AttitudeExecutionError, match="angular velocity"):
        executor.install(wrong_rate)


def test_acquisition_from_rest_must_fit_the_first_interval(limits):
    with pytest.raises(AttitudeExecutionError):
        AttitudeTrajectory.tracking(
            [(0, (0, 0, 0)), (2, (0.2, 0, 0)), (4, (0.4, 0, 0))],
            limits,
            initial_rate=(0, 0, 0),
        )
    curve = AttitudeTrajectory.tracking(
        [(0, (0, 0, 0)), (60, (6, 0, 0)), (120, (12, 0, 0))],
        limits,
        initial_rate=(0, 0, 0),
    )
    assert curve.state(0).angular_velocity_body == (0, 0, 0)
    assert curve.state(60).angular_velocity_body == pytest.approx((0, 0, 0.1))


def test_interval_check_catches_rate_peak_between_endpoints():
    curve = QuaternionHermite(
        tuple(attitude_to_quat(0, 0, 0)),
        tuple(attitude_to_quat(6, 0, 0)),
        (0, 0, 0),
        (0, 0, 0),
        60,
    )
    assert np.linalg.norm(curve.evaluate(0)[1]) == 0
    assert np.linalg.norm(curve.evaluate(1)[1]) < 1e-12
    assert np.linalg.norm(curve.evaluate(0.5)[1]) > 0.14
    assert not curve.within_limits((0.12,) * 3, (0.05,) * 3)
    assert curve.within_limits((0.2,) * 3, (0.05,) * 3)
    assert not curve.within_limits((0.2,) * 3, (0.001,) * 3)


def test_quaternion_curve_rates_and_acceleration_match_independent_differences():
    curve = QuaternionHermite(
        tuple(attitude_to_quat(10, 20, 30)),
        tuple(attitude_to_quat(12, 22, 31)),
        (0.1, 0.02, 0.03),
        (0.02, 0.01, 0.04),
        60,
    )
    assert curve.within_limits((0.2, 0.1, 0.1), (0.05, 0.03, 0.02))
    for fraction in np.linspace(0, 1, 101):
        _, rate, acceleration = curve.evaluate(fraction)
        q0, w0, _ = curve.evaluate(fraction - 1e-5)
        q1, w1, _ = curve.evaluate(fraction + 1e-5)
        angle, axis = quaternion_attitude_delta(
            *quat_to_attitude(q0), *quat_to_attitude(q1)
        )
        np.testing.assert_allclose(angle * np.asarray(axis) / 0.0012, rate, atol=1e-8)
        np.testing.assert_allclose((w1 - w0) / 0.0012, acceleration, atol=1e-8)
        assert np.linalg.norm(rate / (0.2, 0.1, 0.1)) <= 1
        assert np.linalg.norm(acceleration / (0.05, 0.03, 0.02)) <= 1


def test_mixed_axis_braking_respects_coupled_acceleration():
    limits = AttitudeControlSystem(
        max_slew_rate_body=(0.2, 0.1, 0.3), slew_acceleration_body=(0.05, 0.03, 0.02)
    )
    executor = AttitudeExecutor(1000, (43, 67, 220), (0.04, -0.06, 0.08))
    initial = executor.state
    end = executor.request_stop(1000, limits)
    times = np.linspace(1000, end, 101)
    states = [executor.predict(time) for time in times]
    rates = np.asarray([state.angular_velocity_body for state in states])
    accelerations = np.diff(rates, axis=0) / np.diff(times)[:, None]
    assert (
        np.max(np.linalg.norm(accelerations / limits.slew_acceleration_body, axis=1))
        <= 1 + 1e-9
    )
    assert states[0] == initial
    assert states[-1].angular_velocity_body == (0, 0, 0)
    for first, last in zip(states, states[1:]):
        angle, axis = quaternion_attitude_delta(*first.attitude, *last.attitude)
        np.testing.assert_allclose(
            angle * np.asarray(axis) / (last.utime - first.utime),
            (np.asarray(first.angular_velocity_body) + last.angular_velocity_body) / 2,
            atol=1e-7,
        )


@pytest.mark.parametrize(
    "attitudes",
    [
        [(359, 0, 0), (1, 0, 0), (4, 0, 0)],
        [(0, 89, 40), (180, 89, 40), (179, 88, 41)],
        [(0, 0, 0), (180, 0, 0), (180, 0, 0)],
        [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(0, 0, 0), (2, 0, 0), (0, 0, 0)],
    ],
)
def test_nonuniform_tracking_wrap_pole_half_turn_and_reversal(attitudes):
    limits = AttitudeControlSystem(max_slew_rate=1, slew_acceleration=0.1)
    trajectory = AttitudeTrajectory.tracking(
        list(zip([0, 600, 1500], attitudes)), limits, initial_rate=(0, 0, 0)
    )
    for time, expected in zip([0, 600, 1500], attitudes):
        assert (
            quaternion_attitude_delta(*trajectory.state(time).attitude, *expected)[0]
            < 1e-7
        )
    before = trajectory.state(600 - 1e-5).angular_velocity_body
    after = trajectory.state(600 + 1e-5).angular_velocity_body
    np.testing.assert_allclose(before, after, atol=1e-7)


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan")])
def test_curve_rejects_invalid_duration(duration):
    with pytest.raises(ValueError, match="duration"):
        QuaternionHermite((1, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0), duration)


class _PolynomialReference(QuaternionHermite):
    """Independent, general-purpose algebra for the optimized coefficient builder."""

    @cached_property
    def _polynomials(self):
        p = [Polynomial(component) for component in self._coefficients.T]
        dp = [component.deriv() for component in p]
        norm = sum(component * component for component in p)
        rate = [
            -2
            * np.rad2deg(1.0)
            / self.duration
            * (dp[i] * p[0] - dp[0] * p[i] - dp[j] * p[k] + dp[k] * p[j])
            for i, j, k in ((1, 2, 3), (2, 3, 1), (3, 1, 2))
        ]
        accel = [(w.deriv() * norm - w * norm.deriv()) / self.duration for w in rate]

        def columns(polynomials):
            size = max(len(polynomial.coef) for polynomial in polynomials)
            return np.column_stack(
                [
                    np.pad(polynomial.coef, (0, size - len(polynomial.coef)))
                    for polynomial in polynomials
                ]
            )

        return norm.coef, columns(rate), columns(accel)


@pytest.mark.parametrize("seed", range(30))
def test_fixed_size_polynomials_match_generic_reference(seed):
    rng = np.random.default_rng(seed)
    endpoints = rng.normal(size=(2, 4))
    endpoints /= np.linalg.norm(endpoints, axis=1)[:, None]
    duration = 10 ** rng.uniform(0, 3)
    rates = rng.normal(size=(2, 3)) / duration
    if seed == 0:  # Trailing-zero coefficients: stationary identity quaternion.
        endpoints[:] = (1, 0, 0, 0)
        rates[:] = 0
    elif seed == 1:  # Single-axis, rest-to-rest curve.
        endpoints = np.array([attitude_to_quat(0, 0, 0), attitude_to_quat(6, 0, 0)])
        rates[:] = 0
    args = (*map(tuple, endpoints), *map(tuple, rates), duration)
    actual, reference = QuaternionHermite(*args), _PolynomialReference(*args)
    fractions = np.linspace(0, 1, 51)
    expected = np.array([reference.evaluate(fraction)[1:] for fraction in fractions])
    for fraction in fractions:
        for value, oracle in zip(
            actual.evaluate(fraction), reference.evaluate(fraction)
        ):
            np.testing.assert_allclose(value, oracle, rtol=1e-10, atol=1e-8)
    axes = rng.uniform(0.5, 2.0, (2, 3))
    peaks = np.max(np.linalg.norm(expected / axes, axis=2), axis=0)
    for margin in (0.99, 1.1):
        rate_axes, accel_axes = axes * np.maximum(peaks, 1e-6)[:, None] * margin
        assert actual.within_limits(
            tuple(rate_axes), tuple(accel_axes)
        ) == reference.within_limits(tuple(rate_axes), tuple(accel_axes))
