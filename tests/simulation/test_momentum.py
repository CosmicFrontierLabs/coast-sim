import numpy as np
import pytest

from conops.common import attitude_to_quat
from conops.config import AttitudeControlSystem
from conops.simulation import Slew
from conops.simulation.momentum import (
    EARTH_GRAVITATIONAL_PARAMETER_KM3_S2,
    StoredMomentumTracker,
    gravity_gradient_torque_body,
)


def _diagonal_inertia(*values: float) -> np.ndarray:
    return np.diag(values)


def test_gravity_gradient_torque_is_zero_for_spherical_inertia() -> None:
    torque = gravity_gradient_torque_body(
        (6878.0, 100.0, -50.0),
        attitude_to_quat(23.0, -14.0, 71.0),
        _diagonal_inertia(10.0, 10.0, 10.0),
    )

    assert torque == pytest.approx((0.0, 0.0, 0.0), abs=1e-15)


def test_gravity_gradient_torque_matches_analytic_case() -> None:
    radius_km = 6878.0
    radial_component = radius_km / np.sqrt(2.0)
    torque = gravity_gradient_torque_body(
        (radial_component, radial_component, 0.0),
        attitude_to_quat(0.0, 0.0, 0.0),
        _diagonal_inertia(10.0, 5.0, 1.0),
    )
    coefficient = 3.0 * EARTH_GRAVITATIONAL_PARAMETER_KM3_S2 / radius_km**3

    assert torque == pytest.approx((0.0, 0.0, -2.5 * coefficient), abs=1e-15)


def test_gravity_gradient_torque_uses_roll() -> None:
    radius_km = 6878.0
    torque_without_roll = gravity_gradient_torque_body(
        (0.0, radius_km, 0.0),
        attitude_to_quat(0.0, 0.0, 0.0),
        _diagonal_inertia(10.0, 5.0, 1.0),
    )
    torque_with_roll = gravity_gradient_torque_body(
        (0.0, radius_km, 0.0),
        attitude_to_quat(0.0, 0.0, 45.0),
        _diagonal_inertia(10.0, 5.0, 1.0),
    )
    coefficient = 3.0 * EARTH_GRAVITATIONAL_PARAMETER_KM3_S2 / radius_km**3

    assert torque_without_roll == pytest.approx((0.0, 0.0, 0.0), abs=1e-15)
    assert torque_with_roll == pytest.approx((2.0 * coefficient, 0.0, 0.0))


def test_gravity_gradient_torque_scales_with_inverse_radius_cubed() -> None:
    inertia = _diagonal_inertia(10.0, 5.0, 1.0)
    quaternion = attitude_to_quat(0.0, 0.0, 0.0)
    direction = np.array((1.0, 1.0, 0.0)) / np.sqrt(2.0)
    low = gravity_gradient_torque_body(direction * 6678.0, quaternion, inertia)
    high = gravity_gradient_torque_body(direction * 7178.0, quaternion, inertia)

    assert np.linalg.norm(low) / np.linalg.norm(high) == pytest.approx(
        (7178.0 / 6678.0) ** 3
    )


@pytest.mark.parametrize(
    "position",
    [(0.0, 0.0, 0.0), (np.nan, 0.0, 1.0), (1.0, 2.0)],
)
def test_gravity_gradient_torque_rejects_invalid_positions(position) -> None:
    with pytest.raises(ValueError, match="position_eci_km"):
        gravity_gradient_torque_body(
            position,
            attitude_to_quat(0.0, 0.0, 0.0),
            np.eye(3),
        )


def test_tracker_integrates_constant_torque() -> None:
    radius_km = 6878.0
    radial_component = radius_km / np.sqrt(2.0)
    position = (radial_component, radial_component, 0.0)
    quaternion = attitude_to_quat(0.0, 0.0, 0.0)
    tracker = StoredMomentumTracker(_diagonal_inertia(10.0, 5.0, 1.0))

    first = tracker.update(
        utime=100.0,
        position_eci_km=position,
        attitude_quaternion_eci_to_body=quaternion,
    )
    second = tracker.update(
        utime=110.0,
        position_eci_km=position,
        attitude_quaternion_eci_to_body=quaternion,
    )

    assert first.stored_momentum_norm_n_m_s == pytest.approx(0.0)
    assert second.stored_momentum_body_n_m_s == pytest.approx(
        np.asarray(second.gravity_gradient_torque_body_n_m) * 10.0
    )


def test_tracker_reexpresses_initial_momentum_after_attitude_change() -> None:
    tracker = StoredMomentumTracker(
        np.eye(3), initial_momentum_body_n_m_s=(1.0, 0.0, 0.0)
    )
    tracker.update(
        utime=100.0,
        position_eci_km=(6878.0, 0.0, 0.0),
        attitude_quaternion_eci_to_body=attitude_to_quat(0.0, 0.0, 0.0),
    )
    rotated = tracker.update(
        utime=110.0,
        position_eci_km=(6878.0, 0.0, 0.0),
        attitude_quaternion_eci_to_body=attitude_to_quat(90.0, 0.0, 0.0),
    )

    assert rotated.stored_momentum_body_n_m_s == pytest.approx((0.0, -1.0, 0.0))
    assert rotated.stored_momentum_norm_n_m_s == pytest.approx(1.0)


def test_tracker_rejects_decreasing_timestamps() -> None:
    tracker = StoredMomentumTracker(np.eye(3))
    quaternion = attitude_to_quat(0.0, 0.0, 0.0)
    tracker.update(
        utime=100.0,
        position_eci_km=(6878.0, 0.0, 0.0),
        attitude_quaternion_eci_to_body=quaternion,
    )

    with pytest.raises(ValueError, match="nondecreasing"):
        tracker.update(
            utime=99.0,
            position_eci_km=(6878.0, 0.0, 0.0),
            attitude_quaternion_eci_to_body=quaternion,
        )


@pytest.mark.parametrize("interval", [0.0, -1.0, np.nan, np.inf, 10.1])
def test_tracker_rejects_invalid_sampling_limit(interval) -> None:
    with pytest.raises(ValueError, match="max_sample_interval_s"):
        StoredMomentumTracker(np.eye(3), max_sample_interval_s=interval)


def test_rejected_gap_does_not_advance_tracker() -> None:
    tracker = StoredMomentumTracker(np.diag([10.0, 8.0, 6.0]))
    arguments = dict(
        position_eci_km=(5000.0, 5000.0, 0.0),
        attitude_quaternion_eci_to_body=attitude_to_quat(0.0, 0.0, 0.0),
    )
    tracker.update(utime=0.0, **arguments)
    with pytest.raises(ValueError, match="sample interval 60 s exceeds 10 s"):
        tracker.update(utime=60.0, **arguments)
    sample = tracker.update(utime=10.0, **arguments)
    assert sample.stored_momentum_body_n_m_s == pytest.approx(
        np.asarray(sample.gravity_gradient_torque_body_n_m) * 10.0
    )


def test_fractional_cadence_allows_unix_timestamp_roundoff() -> None:
    tracker = StoredMomentumTracker(np.eye(3), max_sample_interval_s=0.1)
    arguments = dict(
        position_eci_km=(6878.0, 0.0, 0.0),
        attitude_quaternion_eci_to_body=(1.0, 0.0, 0.0, 0.0),
    )
    for offset in range(11):
        tracker.update(utime=1_700_000_000.0 + offset / 10.0, **arguments)
    with pytest.raises(ValueError, match="sample interval"):
        tracker.update(utime=1_700_000_001.101, **arguments)


def test_fast_slew_rejects_coarse_samples_and_converges_at_fine_cadence() -> None:
    radius = 6878.0
    orbit_rate = np.sqrt(EARTH_GRAVITATIONAL_PARAMETER_KM3_S2 / radius**3)
    slew = Slew(
        acs_config=AttitudeControlSystem(
            max_slew_rate=2.0, slew_acceleration=1.0, settle_time=0.0
        ),
        endra=90.0 + np.rad2deg(orbit_rate * 60.0),
    )
    slew.calc_slewtime()
    assert slew.slewtime == 49.0

    def integrate(step: float):
        tracker = StoredMomentumTracker(
            np.diag([1000.0, 800.0, 600.0]), max_sample_interval_s=2.5
        )
        for time in np.arange(0.0, 60.0 + step / 2.0, step):
            sample = tracker.update(
                utime=time,
                position_eci_km=radius
                * np.array([np.cos(orbit_rate * time), np.sin(orbit_rate * time), 0.0]),
                attitude_quaternion_eci_to_body=attitude_to_quat(*slew.attitude(time)),
            )
        return sample.stored_momentum_norm_n_m_s

    with pytest.raises(ValueError, match="sample interval 60 s exceeds 2.5 s"):
        integrate(60.0)
    reference = integrate(0.1)
    assert reference == pytest.approx(0.01081989139, rel=1e-8)
    for step in (2.5, 1.0):
        assert integrate(step) == pytest.approx(reference, rel=1e-3)
