import math
from dataclasses import dataclass

import numpy as np
import pytest

from conops.simulation.reaction_wheel import ReactionWheel


@dataclass
class DummyPass:
    ra_deg: float
    dec_deg: float

    def ra_dec(self, utime: float) -> tuple[float, float]:
        return self.ra_deg, self.dec_deg


def _make_orthogonal_wheels(max_torque: float = 10.0, max_momentum: float = 10.0):
    return [
        ReactionWheel(
            max_torque=max_torque,
            max_momentum=max_momentum,
            orientation=(1.0, 0.0, 0.0),
            current_momentum=0.0,
            name="rw_x",
        ),
        ReactionWheel(
            max_torque=max_torque,
            max_momentum=max_momentum,
            orientation=(0.0, 1.0, 0.0),
            current_momentum=0.0,
            name="rw_y",
        ),
        ReactionWheel(
            max_torque=max_torque,
            max_momentum=max_momentum,
            orientation=(0.0, 0.0, 1.0),
            current_momentum=0.0,
            name="rw_z",
        ),
    ]


def test_pass_tracking_torque_matches_inertia(acs):
    acs.ra = 0.0
    acs.dec = 0.0
    acs.reaction_wheels = _make_orthogonal_wheels()
    acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
    acs._compute_disturbance_torque = lambda _ut: np.zeros(3)
    acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)
    acs.current_pass = DummyPass(1.0, 0.0)

    acs._apply_pass_wheel_update(dt=1.0, utime=0.0)

    expected_torque = (math.pi / 180.0) * 10.0
    assert math.isclose(acs._last_pass_rate_deg_s, 1.0, rel_tol=1e-6)
    assert math.isclose(acs._last_pass_torque_target_mag, expected_torque, rel_tol=1e-6)


def test_pass_tracking_updates_wheel_momentum_axis(acs):
    """Verify that pass tracking applies torque along the correct body-frame axis.

    When pointing at (RA=0, Dec=0) and tracking to (RA=1, Dec=0):
    - The inertial-frame rotation axis is approximately the celestial pole (0, 0, 1)
    - In body frame (where body-Z points at RA=0, Dec=0), this axis transforms
      to body-Y (perpendicular to body-Z in the RA direction)
    - The Y-wheel should absorb momentum, not the Z-wheel
    """
    acs.ra = 0.0
    acs.dec = 0.0
    acs.reaction_wheels = _make_orthogonal_wheels()
    acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
    acs._compute_disturbance_torque = lambda _ut: np.zeros(3)
    acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)
    acs.current_pass = DummyPass(1.0, 0.0)

    acs._apply_pass_wheel_update(dt=1.0, utime=0.0)

    expected_torque = (math.pi / 180.0) * 10.0

    # Verify wheel momentum goes to the correct wheel based on frame transformation.
    # Pointing at (RA=0°, Dec=0°). With pole-reference rotation matrix:
    #   z_b = (1, 0, 0), y_b = (0, 0, 1), x_b = (0, 1, 0)
    # Celestial pole (0,0,1) transforms to body frame as:
    #   R @ (0,0,1) = (0, 1, 0) = body Y
    # So Y-wheel should absorb momentum for RA tracking.
    # Sign is negative: to rotate body in +Y direction, wheel spins in -Y direction.
    assert math.isclose(
        acs.reaction_wheels[1].current_momentum, -expected_torque, rel_tol=2e-2
    )
    assert math.isclose(acs.reaction_wheels[0].current_momentum, 0.0, abs_tol=1e-9)
    assert math.isclose(acs.reaction_wheels[2].current_momentum, 0.0, abs_tol=1e-9)


def test_pass_tracking_no_change_when_no_motion(acs):
    acs.ra = 0.0
    acs.dec = 0.0
    acs.reaction_wheels = _make_orthogonal_wheels()
    acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
    acs._compute_disturbance_torque = lambda _ut: np.zeros(3)
    acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)
    acs.current_pass = DummyPass(0.0, 0.0)

    acs._apply_pass_wheel_update(dt=1.0, utime=0.0)

    assert acs._last_pass_torque_target_mag == 0.0
    assert acs._last_pass_torque_actual_mag == 0.0
    for wheel in acs.reaction_wheels:
        assert wheel.current_momentum == 0.0


def test_pass_tracking_momentum_delta_matches_torque(acs):
    acs.ra = 0.0
    acs.dec = 0.0
    acs.reaction_wheels = _make_orthogonal_wheels()
    acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
    acs._compute_disturbance_torque = lambda _ut: np.zeros(3)
    acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)
    acs.current_pass = DummyPass(1.0, 0.0)

    before = {w.name: w.current_momentum for w in acs.reaction_wheels}
    dt = 0.5
    acs._apply_pass_wheel_update(dt=dt, utime=0.0)

    after = {w.name: w.current_momentum for w in acs.reaction_wheels}
    snapshot = acs.wheel_snapshot()
    assert snapshot.wheels
    for reading in snapshot.wheels:
        delta = after[reading.name] - before[reading.name]
        assert delta == pytest.approx(reading.torque_applied * dt, rel=1e-6, abs=1e-9)


def test_pass_tracking_torque_uses_rate_delta(acs):
    """Verify pass tracking uses rate delta, not absolute rate, for torque calculation.

    The pass update now uses _last_pointing_rate_rad_s for rate tracking
    (shared with the consistency check), not a separate _last_pass_rate_deg_s.
    """
    acs.ra = 0.0
    acs.dec = 0.0
    acs.reaction_wheels = _make_orthogonal_wheels()
    acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
    acs._compute_disturbance_torque = lambda _ut: np.zeros(3)
    acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)
    acs.current_pass = DummyPass(1.0, 0.0)
    # Set previous pointing rate (now uses rad/s shared tracker)
    acs._last_pointing_rate_rad_s = 0.2 * (math.pi / 180.0)

    acs._apply_pass_wheel_update(dt=1.0, utime=0.0)

    accel_req = 0.8  # (1.0 - 0.2) deg/s^2
    expected_torque = (math.pi / 180.0) * 10.0 * accel_req
    assert acs._last_pass_rate_deg_s == pytest.approx(1.0, rel=1e-6)
    assert acs._last_pass_torque_target_mag == pytest.approx(expected_torque, rel=1e-6)


def test_pass_tracking_axis_transform_at_nonzero_dec(acs):
    """Verify pass tracking transforms rotation axis to body frame at non-zero Dec.

    At (RA=0, Dec=45°):
    - Body-Z points at celestial coordinates (RA=0, Dec=45°)
    - The rotation axis for RA tracking transforms differently than at Dec=0

    This test verifies that the body-frame transformation is applied correctly
    and the correct wheel receives momentum for RA-tracking at non-zero Dec.

    Key physics: At Dec=45°, RA tracking still rotates around celestial pole
    (inertial Z), but that axis transforms to a different body-frame direction
    than at Dec=0.
    """
    # Position spacecraft at (RA=0, Dec=45°)
    acs.ra = 0.0
    acs.dec = 45.0
    acs.reaction_wheels = _make_orthogonal_wheels()
    acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
    acs._compute_disturbance_torque = lambda _ut: np.zeros(3)
    acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)

    # Set up previous pointing state (needed for axis calculation)
    acs._last_pointing_ra = 0.0
    acs._last_pointing_dec = 45.0
    acs._last_pointing_utime = -1.0
    acs._last_pointing_rate_rad_s = 0.0

    # Target: track to RA=1° at same Dec (pure RA tracking)
    acs.current_pass = DummyPass(1.0, 45.0)

    acs._apply_pass_wheel_update(dt=1.0, utime=0.0)

    # At Dec=45°:
    # - Body-Z = (cos(45)cos(0), cos(45)sin(0), sin(45)) = (0.707, 0, 0.707)
    # - Celestial pole (inertial Z) = (0, 0, 1)
    # - With pole-reference Gram-Schmidt:
    #   y_b = pole - (pole·z_b)*z_b = (0,0,1) - 0.707*(0.707, 0, 0.707)
    #       = (0,0,1) - (0.5, 0, 0.5) = (-0.5, 0, 0.5) normalized = (-0.707, 0, 0.707)
    # - This means body-Y has significant X component in inertial frame

    # The celestial pole maps to a mix of body axes, but Y should dominate
    # because the Gram-Schmidt process aligns Y with the pole projection
    y_wheel_momentum = acs.reaction_wheels[1].current_momentum

    # At Dec=45°, the Y-wheel should still receive momentum for RA tracking
    # (the pole-reference Gram-Schmidt keeps Y aligned toward the pole)
    expected_torque = (math.pi / 180.0) * 10.0  # 1 deg/s² × 10 kg·m² = 0.1745 Nm

    # Y-wheel should have significant momentum (sign depends on geometry)
    assert abs(y_wheel_momentum) > 0.1 * expected_torque, (
        f"Y-wheel should receive significant momentum for RA tracking at Dec=45°: "
        f"Y={y_wheel_momentum:.4f}, expected magnitude ~{expected_torque:.4f}"
    )

    # X and Z wheels should have less momentum than Y
    x_wheel_momentum = abs(acs.reaction_wheels[0].current_momentum)
    z_wheel_momentum = abs(acs.reaction_wheels[2].current_momentum)

    # Y should be dominant for RA tracking
    assert abs(y_wheel_momentum) > x_wheel_momentum, (
        f"Y-wheel should dominate for RA tracking: Y={abs(y_wheel_momentum):.4f}, X={x_wheel_momentum:.4f}"
    )
    assert abs(y_wheel_momentum) > z_wheel_momentum, (
        f"Y-wheel should dominate for RA tracking: Y={abs(y_wheel_momentum):.4f}, Z={z_wheel_momentum:.4f}"
    )
