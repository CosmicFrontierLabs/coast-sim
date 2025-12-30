"""Tests for momentum conservation in the ACS module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from conops.simulation.acs import ACS


class MockEphemeris:
    """Minimal ephemeris mock for ACS initialization."""

    def __init__(self):
        self.earth = [MagicMock(ra=MagicMock(deg=0.0), dec=MagicMock(deg=0.0))]
        self.gcrs_pv = MagicMock()
        self.gcrs_pv.position = [[7000e3, 0, 0]]
        self.gcrs_pv.velocity = [[0, 7500, 0]]

    def index(self, dt):
        return 0


class MockConstraint:
    """Mock constraint with ephemeris."""

    def __init__(self):
        self.ephem = MockEphemeris()


def make_acs_with_wheels():
    """Create ACS instance with 3-axis wheel configuration."""
    config = MagicMock()
    config.constraint = MockConstraint()
    config.spacecraft_bus = MagicMock()
    config.spacecraft_bus.attitude_control = MagicMock()
    config.spacecraft_bus.attitude_control.slew_acceleration = 0.5
    config.spacecraft_bus.attitude_control.max_slew_rate = 0.25
    config.spacecraft_bus.attitude_control.settle_time = 60.0
    config.spacecraft_bus.attitude_control.wheel_enabled = False
    config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)
    config.spacecraft_bus.attitude_control.strict_wheel_validation = False
    # 3-axis wheel configuration
    config.spacecraft_bus.attitude_control.wheels = [
        {
            "orientation": [1, 0, 0],
            "max_torque": 0.1,
            "max_momentum": 10.0,
            "name": "X",
        },
        {
            "orientation": [0, 1, 0],
            "max_torque": 0.1,
            "max_momentum": 10.0,
            "name": "Y",
        },
        {
            "orientation": [0, 0, 1],
            "max_torque": 0.1,
            "max_momentum": 10.0,
            "name": "Z",
        },
    ]
    config.spacecraft_bus.attitude_control.magnetorquers = []
    config.solar_panel = None

    acs = ACS(config=config, log=None)
    return acs


class TestMomentumConservation:
    """Tests for momentum conservation bookkeeping."""

    def test_body_momentum_initialized_to_zero(self):
        """Body momentum starts at zero."""
        acs = make_acs_with_wheels()
        assert np.allclose(acs.spacecraft_body_momentum, [0, 0, 0])

    def test_total_system_momentum_includes_wheels_and_body(self):
        """Total system momentum = wheel momentum + body momentum."""
        acs = make_acs_with_wheels()

        # Set some wheel momentum
        acs.reaction_wheels[0].current_momentum = 1.0  # X wheel
        acs.reaction_wheels[1].current_momentum = 2.0  # Y wheel
        acs.reaction_wheels[2].current_momentum = 3.0  # Z wheel

        # Set body momentum
        acs.spacecraft_body_momentum = np.array([0.5, 0.5, 0.5])

        h_total = acs._get_total_system_momentum()
        expected = np.array([1.5, 2.5, 3.5])  # wheels + body
        assert np.allclose(h_total, expected)

    def test_wheel_torque_conserves_momentum(self):
        """Wheel torque application conserves total momentum."""
        acs = make_acs_with_wheels()

        # Record initial total momentum
        h_initial = acs._get_total_system_momentum().copy()

        # Apply wheel torques
        taus = np.array([0.05, 0.03, 0.02])  # N·m per wheel
        dt = 1.0  # 1 second

        acs._apply_wheel_torques_conserving(taus, dt)

        # Total momentum should be unchanged (internal exchange)
        h_final = acs._get_total_system_momentum()
        assert np.allclose(h_initial, h_final, atol=1e-10)

    def test_wheel_torque_updates_body_momentum_oppositely(self):
        """Body momentum changes opposite to wheel momentum change."""
        acs = make_acs_with_wheels()

        # Record initial states
        h_wheels_before = acs._get_total_wheel_momentum().copy()
        h_body_before = acs.spacecraft_body_momentum.copy()

        # Apply X-axis wheel torque
        taus = np.array([0.1, 0.0, 0.0])
        dt = 2.0

        acs._apply_wheel_torques_conserving(taus, dt)

        # Wheel momentum change
        h_wheels_after = acs._get_total_wheel_momentum()
        delta_h_wheels = h_wheels_after - h_wheels_before

        # Body momentum change should be opposite
        delta_h_body = acs.spacecraft_body_momentum - h_body_before

        assert np.allclose(delta_h_wheels, -delta_h_body, atol=1e-10)

    def test_external_torque_changes_total_momentum(self):
        """External torque changes total system momentum."""
        acs = make_acs_with_wheels()

        # Initialize conservation tracking
        acs._initial_total_momentum = acs._get_total_system_momentum().copy()

        h_initial = acs._get_total_system_momentum().copy()

        # Apply external torque
        external_torque = np.array([0.01, 0.02, 0.03])
        dt = 5.0
        acs._apply_external_torque(external_torque, dt, source="test")

        # Total momentum should change by external impulse
        h_final = acs._get_total_system_momentum()
        expected_change = external_torque * dt

        assert np.allclose(h_final - h_initial, expected_change, atol=1e-10)

    def test_external_impulse_tracking(self):
        """Cumulative external impulse is tracked correctly."""
        acs = make_acs_with_wheels()

        # Apply several external torques
        acs._apply_external_torque(np.array([0.01, 0, 0]), 1.0, source="test1")
        acs._apply_external_torque(np.array([0, 0.02, 0]), 2.0, source="test2")
        acs._apply_external_torque(np.array([0, 0, 0.03]), 3.0, source="test3")

        expected = np.array([0.01, 0.04, 0.09])
        assert np.allclose(acs._cumulative_external_impulse, expected, atol=1e-10)

    def test_conservation_check_passes_when_valid(self):
        """Conservation check passes when momentum is conserved."""
        acs = make_acs_with_wheels()

        # Initialize
        utime = 1000.0
        acs._check_momentum_conservation(utime)  # Sets initial

        # Apply wheel torques (internal, conserves momentum)
        acs._apply_wheel_torques_conserving(np.array([0.05, 0.05, 0.05]), 1.0)

        # Apply external torque
        external = np.array([0.01, 0.02, 0.01])
        acs._apply_external_torque(external, 2.0, source="test")

        # Check should pass
        assert acs._check_momentum_conservation(utime + 60)

    def test_body_momentum_returns_to_zero_after_slew(self):
        """Body momentum should return near zero after complete slew cycle.

        During a slew:
        - Accel phase: wheels speed up, body gets negative momentum (reaction)
        - Decel phase: wheels slow down, body momentum returns toward zero
        - After settle: body momentum ≈ 0
        """
        acs = make_acs_with_wheels()

        # Simulate symmetric slew: accel then decel
        dt = 1.0
        accel_torque = np.array([0.05, 0.0, 0.0])

        # Accel phase (10 steps)
        for _ in range(10):
            acs._apply_wheel_torques_conserving(accel_torque, dt)

        # Body momentum should be non-zero during slew
        h_body_mid = acs.spacecraft_body_momentum.copy()
        assert np.linalg.norm(h_body_mid) > 0.1

        # Decel phase (10 steps, opposite torque)
        for _ in range(10):
            acs._apply_wheel_torques_conserving(-accel_torque, dt)

        # Body momentum should return near zero
        h_body_final = acs.spacecraft_body_momentum
        assert np.linalg.norm(h_body_final) < 1e-9

    def test_get_body_momentum_magnitude(self):
        """get_body_momentum_magnitude returns correct value."""
        acs = make_acs_with_wheels()
        acs.spacecraft_body_momentum = np.array([3.0, 4.0, 0.0])
        assert acs.get_body_momentum_magnitude() == pytest.approx(5.0)


class TestMomentumBookkeepingIntegration:
    """Integration tests for momentum bookkeeping with full ACS flow."""

    def test_hold_torque_tracks_disturbance_as_external(self):
        """Hold torque properly tracks disturbance as external impulse."""
        acs = make_acs_with_wheels()
        acs._last_pointing_time = 0.0

        # Apply a "disturbance" torque via hold
        disturbance = np.array([0.001, 0.002, 0.001])
        dt = 60.0
        utime = 60.0

        initial_ext = acs._cumulative_external_impulse.copy()
        acs._apply_hold_wheel_torque(disturbance, dt, utime)

        # External impulse should have increased by disturbance * dt
        delta_ext = acs._cumulative_external_impulse - initial_ext
        assert np.allclose(delta_ext, disturbance * dt, atol=1e-10)

    def test_wheel_headroom_along_axis(self):
        """Wheel headroom calculation considers current momentum."""
        acs = make_acs_with_wheels()

        # Full headroom when empty: max * margin = 10.0 * 0.9 = 9.0
        axis = np.array([1.0, 0.0, 0.0])
        full_headroom = acs._get_wheel_headroom_along_axis(axis)
        assert full_headroom == pytest.approx(9.0)  # 10.0 * 0.9 margin

        # Reduce headroom by loading X wheel
        acs.reaction_wheels[0].current_momentum = 5.0

        partial_headroom = acs._get_wheel_headroom_along_axis(axis)

        # Should have less headroom: 9.0 - 5.0 = 4.0
        assert partial_headroom < full_headroom
        assert partial_headroom == pytest.approx(4.0)
