"""Tests for ACS momentum bookkeeping and pre-slew budget checks."""

from unittest.mock import Mock

import numpy as np
import pytest

from conops.simulation.reaction_wheel import ReactionWheel
from conops.simulation.slew import Slew


class TestMomentumBookkeeping:
    """Test momentum vector computation and tracking."""

    def test_get_total_wheel_momentum_empty(self, acs):
        """No wheels returns zero vector."""
        acs.reaction_wheels = []
        acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
        h_total = acs._get_total_wheel_momentum()
        assert np.allclose(h_total, np.zeros(3))

    def test_get_total_wheel_momentum_single_wheel(self, acs):
        """Single wheel returns correct momentum vector."""
        wheel = ReactionWheel(
            max_torque=0.1,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
        )
        wheel.current_momentum = 0.5
        acs.reaction_wheels = [wheel]
        acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync

        h_total = acs._get_total_wheel_momentum()
        assert np.allclose(h_total, np.array([0.5, 0.0, 0.0]))

    def test_get_total_wheel_momentum_multiple_wheels(self, acs):
        """Multiple wheels sum correctly."""
        wheel_x = ReactionWheel(
            max_torque=0.1,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
        )
        wheel_x.current_momentum = 0.3
        wheel_y = ReactionWheel(
            max_torque=0.1,
            max_momentum=1.0,
            orientation=(0.0, 1.0, 0.0),
        )
        wheel_y.current_momentum = -0.2
        wheel_z = ReactionWheel(
            max_torque=0.1,
            max_momentum=1.0,
            orientation=(0.0, 0.0, 1.0),
        )
        wheel_z.current_momentum = 0.4
        acs.reaction_wheels = [wheel_x, wheel_y, wheel_z]
        acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync

        h_total = acs._get_total_wheel_momentum()
        assert np.allclose(h_total, np.array([0.3, -0.2, 0.4]))

    def test_get_wheel_headroom_along_axis(self, acs):
        """Headroom calculation accounts for current momentum."""
        wheel = ReactionWheel(
            max_torque=0.1,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
        )
        wheel.current_momentum = 0.3
        acs.reaction_wheels = [wheel]
        acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
        acs.wheel_dynamics._momentum_margin = 1.0  # No margin for this test

        # Headroom along X axis
        headroom = acs._get_wheel_headroom_along_axis(np.array([1.0, 0.0, 0.0]))
        assert pytest.approx(headroom, abs=0.01) == 0.7

    def test_get_wheel_headroom_orthogonal_axis(self, acs):
        """Headroom is zero for orthogonal axis."""
        wheel = ReactionWheel(
            max_torque=0.1,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
        )
        wheel.current_momentum = 0.0
        acs.reaction_wheels = [wheel]
        acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync

        # Y axis is orthogonal to X-oriented wheel
        headroom = acs._get_wheel_headroom_along_axis(np.array([0.0, 1.0, 0.0]))
        assert headroom == 0.0


class TestMomentumBudget:
    """Test pre-slew momentum budget calculations."""

    def test_compute_slew_peak_momentum(self, acs, mock_config):
        """Peak momentum computed from slew geometry and MOI."""
        slew = Slew(config=mock_config)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 0.0
        slew.slewdist = 10.0
        slew.rotation_axis = (0.0, 0.0, 1.0)
        slew._accel_override = 0.5  # deg/s^2

        h_peak, axis = acs._compute_slew_peak_momentum(slew)

        # Peak should be positive
        assert h_peak > 0
        # Axis should be normalized
        assert np.allclose(np.linalg.norm(axis), 1.0)

    def test_check_slew_momentum_budget_sufficient(self, acs, mock_config):
        """Budget check passes with sufficient headroom."""
        # Set up 3 orthogonal wheels with plenty of headroom for 3-axis control.
        # This is required since the rotation axis is transformed from inertial
        # to body frame, and may not align with a single wheel.
        wheels = [
            ReactionWheel(
                max_torque=0.1, max_momentum=10.0, orientation=(1.0, 0.0, 0.0)
            ),
            ReactionWheel(
                max_torque=0.1, max_momentum=10.0, orientation=(0.0, 1.0, 0.0)
            ),
            ReactionWheel(
                max_torque=0.1, max_momentum=10.0, orientation=(0.0, 0.0, 1.0)
            ),
        ]
        for w in wheels:
            w.current_momentum = 0.0
        acs.reaction_wheels = wheels
        acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync

        # Small slew
        slew = Slew(config=mock_config)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 5.0
        slew.enddec = 0.0
        slew.slewdist = 5.0
        slew.slewtime = 30.0
        slew.rotation_axis = (0.0, 0.0, 1.0)
        slew._accel_override = 0.5

        # Mock disturbance model
        acs.disturbance_model = Mock()
        acs.disturbance_model.compute = Mock(return_value=(np.zeros(3), {}))

        feasible, msg = acs._check_slew_momentum_budget(slew, 1000.0)
        assert feasible
        assert "Budget OK" in msg

    def test_check_slew_momentum_budget_insufficient(self, acs, mock_config):
        """Budget check fails with insufficient headroom."""
        # Set up wheel near saturation
        wheel = ReactionWheel(
            max_torque=0.1,
            max_momentum=0.5,  # Small capacity
            orientation=(0.0, 0.0, 1.0),
        )
        wheel.current_momentum = 0.45  # 90% full
        acs.reaction_wheels = [wheel]
        acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
        acs.wheel_dynamics._momentum_margin = 0.9

        # Large slew requiring lots of momentum
        slew = Slew(config=mock_config)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 90.0  # Large slew
        slew.enddec = 0.0
        slew.slewdist = 90.0
        slew.slewtime = 60.0
        slew.rotation_axis = (0.0, 0.0, 1.0)
        slew._accel_override = 1.0

        # Mock disturbance model
        acs.disturbance_model = Mock()
        acs.disturbance_model.compute = Mock(return_value=(np.zeros(3), {}))

        feasible, msg = acs._check_slew_momentum_budget(slew, 1000.0)
        assert not feasible
        assert "Insufficient" in msg

    def test_budget_check_no_wheels(self, acs, mock_config):
        """Budget check always passes with no wheels."""
        acs.reaction_wheels = []

        slew = Slew(config=mock_config)
        slew.slewdist = 90.0
        slew.rotation_axis = (0.0, 0.0, 1.0)

        feasible, msg = acs._check_slew_momentum_budget(slew, 1000.0)
        assert feasible
        assert "No wheels" in msg


class TestMomentumWarnings:
    """Test warning accumulation and retrieval."""

    def test_get_momentum_warnings_initially_empty(self, acs):
        """Warnings list is empty initially."""
        assert acs.get_momentum_warnings() == []

    def test_get_momentum_warnings_returns_copy(self, acs):
        """get_momentum_warnings returns a copy."""
        acs._momentum_warnings.append("test warning")
        warnings = acs.get_momentum_warnings()
        warnings.append("should not affect original")
        assert len(acs._momentum_warnings) == 1

    def test_clear_momentum_warnings(self, acs):
        """clear_momentum_warnings empties the list."""
        acs._momentum_warnings.append("test warning")
        acs.clear_momentum_warnings()
        assert acs.get_momentum_warnings() == []

    def test_get_momentum_summary(self, acs):
        """Momentum summary includes expected fields."""
        wheel = ReactionWheel(
            max_torque=0.1,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
        )
        wheel.current_momentum = 0.5
        acs.reaction_wheels = [wheel]
        acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync

        summary = acs.get_momentum_summary()

        # WheelDynamics uses wheel_momentum instead of total_momentum_vector
        assert "wheel_momentum" in summary
        assert "total_momentum" in summary
        assert "num_warnings" in summary
        assert summary["wheel_momentum_mag"] > 0


class TestSlewMomentumRecording:
    """Test momentum state recording at slew boundaries."""

    def test_record_slew_start_momentum(self, acs, mock_config):
        """Slew start records current wheel momentum."""
        wheel = ReactionWheel(
            max_torque=0.1,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
        )
        wheel.current_momentum = 0.3
        acs.reaction_wheels = [wheel]
        acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync

        slew = Slew(config=mock_config)
        slew.slewdist = 10.0
        slew.rotation_axis = (1.0, 0.0, 0.0)
        slew._accel_override = 0.5

        acs._record_slew_start_momentum(slew)

        # Access via WheelDynamics
        assert acs.wheel_dynamics._slew_momentum_at_start is not None
        assert np.allclose(
            acs.wheel_dynamics._slew_momentum_at_start, np.array([0.3, 0.0, 0.0])
        )

    def test_verify_slew_end_momentum_clears_state(self, acs, mock_config):
        """Verification clears recorded state."""
        wheel = ReactionWheel(
            max_torque=0.1,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
        )
        wheel.current_momentum = 0.3
        acs.reaction_wheels = [wheel]
        acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
        # Set via WheelDynamics
        acs.wheel_dynamics._slew_momentum_at_start = np.array([0.3, 0.0, 0.0])

        slew = Slew(config=mock_config)
        slew.slewtime = 30.0
        slew.slewdist = 10.0
        slew.rotation_axis = (1.0, 0.0, 0.0)
        slew._accel_override = 0.5

        # Mock disturbance
        acs.disturbance_model = Mock()
        acs.disturbance_model.compute = Mock(return_value=(np.zeros(3), {}))

        acs._verify_slew_end_momentum(slew, 1000.0)

        # Access via WheelDynamics
        assert acs.wheel_dynamics._slew_momentum_at_start is None
