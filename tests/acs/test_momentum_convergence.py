"""Tests for momentum convergence and physical consistency."""

import numpy as np
import pytest

from conops.config import MissionConfig
from conops.simulation.acs import ACS
from conops.simulation.reaction_wheel import ReactionWheel
from conops.simulation.slew import Slew


class DummyAngle:
    def __init__(self, deg):
        self.deg = deg


class DummyBody:
    def __init__(self, ra, dec):
        self.ra = DummyAngle(ra)
        self.dec = DummyAngle(dec)


class DummyEphem:
    def __init__(self):
        self.earth = [DummyBody(0.0, 0.0)]
        self.sun = [DummyBody(0.0, 0.0)]

    def index(self, dt):
        return 0


def make_acs_with_wheels(wheels_config: list[dict]) -> ACS:
    """Create ACS with configured wheels."""
    cfg = MissionConfig()
    cfg.constraint.ephem = DummyEphem()
    acs_cfg = cfg.spacecraft_bus.attitude_control
    acs_cfg.wheels = wheels_config
    return ACS(config=cfg, log=None)


def make_three_axis_wheels() -> list[dict]:
    """Standard 3-axis orthogonal wheel configuration."""
    return [
        {
            "orientation": [1.0, 0.0, 0.0],
            "max_torque": 0.1,
            "max_momentum": 1.0,
            "name": "X",
        },
        {
            "orientation": [0.0, 1.0, 0.0],
            "max_torque": 0.1,
            "max_momentum": 1.0,
            "name": "Y",
        },
        {
            "orientation": [0.0, 0.0, 1.0],
            "max_torque": 0.1,
            "max_momentum": 1.0,
            "name": "Z",
        },
    ]


class TestWheelMomentumLimits:
    """Tests that wheel momentum stays within hardware limits."""

    def test_wheel_momentum_clamped_at_max(self):
        """Wheel momentum is clamped at max_momentum."""
        wheel = ReactionWheel(max_torque=0.1, max_momentum=1.0)
        wheel.current_momentum = 0.9

        # Apply torque that would exceed max
        wheel.apply_torque(0.1, 20.0)  # Would add 2.0 N*m*s

        assert wheel.current_momentum == 1.0

    def test_wheel_momentum_clamped_at_negative_max(self):
        """Wheel momentum is clamped at -max_momentum."""
        wheel = ReactionWheel(max_torque=0.1, max_momentum=1.0)
        wheel.current_momentum = -0.9

        # Apply negative torque that would exceed negative max
        wheel.apply_torque(-0.1, 20.0)

        assert wheel.current_momentum == -1.0

    def test_wheel_momentum_updates_correctly(self):
        """Normal torque application updates momentum correctly."""
        wheel = ReactionWheel(max_torque=0.1, max_momentum=1.0)
        wheel.current_momentum = 0.0

        wheel.apply_torque(0.05, 10.0)

        assert wheel.current_momentum == pytest.approx(0.5)


class TestMomentumBookkeepingIntegration:
    """Tests for momentum bookkeeping system."""

    def test_get_total_wheel_momentum_zero_initial(self):
        """Total wheel momentum is zero when wheels start at zero."""
        acs = make_acs_with_wheels(make_three_axis_wheels())

        h_total = acs._get_total_wheel_momentum()

        assert np.allclose(h_total, np.zeros(3))

    def test_get_total_wheel_momentum_single_axis(self):
        """Total momentum reflects single wheel with non-zero momentum."""
        acs = make_acs_with_wheels(make_three_axis_wheels())
        acs.reaction_wheels[0].current_momentum = 0.5  # X-axis wheel

        h_total = acs._get_total_wheel_momentum()

        assert h_total[0] == pytest.approx(0.5)
        assert h_total[1] == pytest.approx(0.0)
        assert h_total[2] == pytest.approx(0.0)

    def test_get_total_wheel_momentum_multi_axis(self):
        """Total momentum correctly sums multiple wheels."""
        acs = make_acs_with_wheels(make_three_axis_wheels())
        acs.reaction_wheels[0].current_momentum = 0.3
        acs.reaction_wheels[1].current_momentum = -0.2
        acs.reaction_wheels[2].current_momentum = 0.5

        h_total = acs._get_total_wheel_momentum()

        assert h_total[0] == pytest.approx(0.3)
        assert h_total[1] == pytest.approx(-0.2)
        assert h_total[2] == pytest.approx(0.5)

    def test_get_momentum_summary_structure(self):
        """Momentum summary contains expected fields."""
        acs = make_acs_with_wheels(make_three_axis_wheels())
        acs.reaction_wheels[0].current_momentum = 0.5

        summary = acs.get_momentum_summary()

        assert "total_momentum_vector" in summary
        assert "total_momentum_magnitude" in summary
        assert "wheels" in summary
        assert "num_warnings" in summary
        assert len(summary["wheels"]) == 3


class TestHeadroomCalculation:
    """Tests for wheel headroom calculation."""

    def test_headroom_along_aligned_axis(self):
        """Headroom is correct when query axis aligns with wheel."""
        acs = make_acs_with_wheels(make_three_axis_wheels())
        acs.reaction_wheels[0].current_momentum = 0.3  # X-axis at 0.3 of 1.0

        headroom = acs._get_wheel_headroom_along_axis(np.array([1.0, 0.0, 0.0]))

        # Available headroom = max*margin - current = 1.0*0.9 - 0.3 = 0.6
        assert headroom == pytest.approx(0.6, abs=0.01)

    def test_headroom_along_orthogonal_axis(self):
        """Headroom is zero along axis orthogonal to all wheels."""
        acs = make_acs_with_wheels(
            [
                {
                    "orientation": [1.0, 0.0, 0.0],
                    "max_torque": 0.1,
                    "max_momentum": 1.0,
                },
            ]
        )

        # Y-axis is orthogonal to X-oriented wheel
        headroom = acs._get_wheel_headroom_along_axis(np.array([0.0, 1.0, 0.0]))

        assert headroom == pytest.approx(0.0)

    def test_headroom_partial_projection(self):
        """Headroom accounts for partial axis projection."""
        # 45-degree wheel between X and Y
        acs = make_acs_with_wheels(
            [
                {
                    "orientation": [0.707, 0.707, 0.0],
                    "max_torque": 0.1,
                    "max_momentum": 1.0,
                },
            ]
        )
        acs.reaction_wheels[0].current_momentum = 0.0

        # Query along X - wheel projects ~0.707 onto X
        headroom_x = acs._get_wheel_headroom_along_axis(np.array([1.0, 0.0, 0.0]))

        # Full headroom (1.0*0.9 margin) * projection (0.707) ≈ 0.636
        assert headroom_x == pytest.approx(0.636, abs=0.02)


class TestSlewBudgetCheck:
    """Tests for pre-slew momentum budget validation."""

    def test_budget_check_passes_with_headroom(self):
        """Budget check passes when wheels have sufficient headroom."""
        cfg = MissionConfig()
        cfg.constraint.ephem = DummyEphem()
        cfg.spacecraft_bus.attitude_control.wheels = [
            {"orientation": [0.0, 0.0, 1.0], "max_torque": 0.1, "max_momentum": 10.0},
        ]
        acs = ACS(config=cfg, log=None)

        # Small slew around Z axis
        slew = Slew(config=cfg)
        slew.slewdist = 5.0
        slew.rotation_axis = (0.0, 0.0, 1.0)
        slew._accel_override = 0.1
        slew.slewtime = 30.0

        feasible, msg = acs._check_slew_momentum_budget(slew, 1000.0)

        assert feasible
        assert "OK" in msg

    def test_budget_check_fails_near_saturation(self):
        """Budget check fails when wheels are near saturation."""
        cfg = MissionConfig()
        cfg.constraint.ephem = DummyEphem()
        cfg.spacecraft_bus.attitude_control.wheels = [
            {"orientation": [0.0, 0.0, 1.0], "max_torque": 0.1, "max_momentum": 0.5},
        ]
        acs = ACS(config=cfg, log=None)
        # Pre-saturate the wheel
        acs.reaction_wheels[0].current_momentum = 0.48

        # Large slew
        slew = Slew(config=cfg)
        slew.slewdist = 90.0
        slew.rotation_axis = (0.0, 0.0, 1.0)
        slew._accel_override = 1.0
        slew.slewtime = 60.0

        feasible, msg = acs._check_slew_momentum_budget(slew, 1000.0)

        assert not feasible
        assert "Insufficient" in msg

    def test_budget_check_passes_with_no_wheels(self):
        """Budget check passes when no wheels are configured."""
        cfg = MissionConfig()
        cfg.constraint.ephem = DummyEphem()
        acs = ACS(config=cfg, log=None)

        slew = Slew(config=cfg)
        slew.slewdist = 90.0
        slew.rotation_axis = (0.0, 0.0, 1.0)

        feasible, msg = acs._check_slew_momentum_budget(slew, 1000.0)

        assert feasible
        assert "No wheels" in msg


class TestMomentumWarnings:
    """Tests for momentum warning system."""

    def test_warnings_initially_empty(self):
        """Momentum warnings list starts empty."""
        acs = make_acs_with_wheels(make_three_axis_wheels())

        assert acs.get_momentum_warnings() == []

    def test_warnings_returns_copy(self):
        """get_momentum_warnings returns a copy, not the original list."""
        acs = make_acs_with_wheels(make_three_axis_wheels())
        acs._momentum_warnings.append("test warning")

        warnings = acs.get_momentum_warnings()
        warnings.append("new warning")

        assert len(acs._momentum_warnings) == 1

    def test_clear_warnings(self):
        """clear_momentum_warnings empties the warning list."""
        acs = make_acs_with_wheels(make_three_axis_wheels())
        acs._momentum_warnings.append("test warning")

        acs.clear_momentum_warnings()

        assert acs.get_momentum_warnings() == []


class TestReserveImpulse:
    """Tests for wheel impulse reservation."""

    def test_reserve_impulse_allows_full_torque(self):
        """Full torque allowed when headroom is sufficient."""
        wheel = ReactionWheel(max_torque=0.1, max_momentum=1.0)
        wheel.current_momentum = 0.0

        reserved = wheel.reserve_impulse(0.05, 10.0)  # Needs 0.5 N*m*s

        assert reserved == pytest.approx(0.05)

    def test_reserve_impulse_reduces_torque(self):
        """Torque is reduced when it would exceed headroom."""
        wheel = ReactionWheel(max_torque=0.1, max_momentum=1.0)
        wheel.current_momentum = 0.9  # Only 0.1 headroom

        reserved = wheel.reserve_impulse(0.1, 10.0)  # Would need 1.0 N*m*s

        # Should scale down to 0.01 (0.1 headroom / 10s)
        assert reserved == pytest.approx(0.01)

    def test_reserve_impulse_zero_at_saturation(self):
        """Zero torque returned when wheel is saturated."""
        wheel = ReactionWheel(max_torque=0.1, max_momentum=1.0)
        wheel.current_momentum = 1.0  # Fully saturated

        reserved = wheel.reserve_impulse(0.1, 10.0)

        assert reserved == pytest.approx(0.0)


class TestPeakMomentumCalculation:
    """Tests for slew peak momentum estimation."""

    def test_compute_peak_momentum_scales_with_angle(self):
        """Larger slew angles produce larger peak momentum."""
        cfg = MissionConfig()
        cfg.constraint.ephem = DummyEphem()
        cfg.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)
        acs = ACS(config=cfg, log=None)

        slew_small = Slew(config=cfg)
        slew_small.slewdist = 10.0
        slew_small.rotation_axis = (0.0, 0.0, 1.0)
        slew_small._accel_override = 0.5

        slew_large = Slew(config=cfg)
        slew_large.slewdist = 90.0
        slew_large.rotation_axis = (0.0, 0.0, 1.0)
        slew_large._accel_override = 0.5

        h_small, _ = acs._compute_slew_peak_momentum(slew_small)
        h_large, _ = acs._compute_slew_peak_momentum(slew_large)

        assert h_large > h_small

    def test_compute_peak_momentum_scales_with_inertia(self):
        """Larger inertia produces larger peak momentum."""
        cfg1 = MissionConfig()
        cfg1.constraint.ephem = DummyEphem()
        cfg1.spacecraft_bus.attitude_control.spacecraft_moi = (5.0, 5.0, 5.0)
        acs1 = ACS(config=cfg1, log=None)

        cfg2 = MissionConfig()
        cfg2.constraint.ephem = DummyEphem()
        cfg2.spacecraft_bus.attitude_control.spacecraft_moi = (20.0, 20.0, 20.0)
        acs2 = ACS(config=cfg2, log=None)

        slew = Slew(config=cfg1)
        slew.slewdist = 45.0
        slew.rotation_axis = (0.0, 0.0, 1.0)
        slew._accel_override = 0.5

        h1, _ = acs1._compute_slew_peak_momentum(slew)
        h2, _ = acs2._compute_slew_peak_momentum(slew)

        assert h2 > h1


class TestTimestepConsistency:
    """Tests that results are consistent across different timesteps."""

    def test_wheel_momentum_accumulation_independent_of_substeps(self):
        """Total momentum change is same regardless of substep count."""
        # Apply 1.0 N*m*s impulse in one step vs many
        wheel1 = ReactionWheel(max_torque=0.1, max_momentum=10.0)
        wheel2 = ReactionWheel(max_torque=0.1, max_momentum=10.0)

        # Single large step
        wheel1.apply_torque(0.1, 10.0)  # 1.0 N*m*s

        # Many small steps
        for _ in range(100):
            wheel2.apply_torque(0.1, 0.1)  # 0.01 N*m*s each, 1.0 total

        assert wheel1.current_momentum == pytest.approx(
            wheel2.current_momentum, rel=1e-10
        )

    def test_total_momentum_preserved_without_saturation(self):
        """Total impulse equals final momentum when not saturating."""
        wheel = ReactionWheel(max_torque=0.1, max_momentum=10.0)
        wheel.current_momentum = 0.0

        total_impulse = 0.0
        for i in range(50):
            torque = 0.05 * (1 if i % 2 == 0 else -1)  # Alternating
            dt = 1.0
            wheel.apply_torque(torque, dt)
            total_impulse += torque * dt

        # Net impulse should equal final momentum
        assert wheel.current_momentum == pytest.approx(total_impulse, abs=1e-10)
