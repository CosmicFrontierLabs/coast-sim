"""System-level momentum conservation tests.

These tests verify that total angular momentum is conserved according to:
    H_total = H_initial + ∫τ_external dt

where H_total = H_wheels + H_body, and τ_external includes disturbances and MTQ.

These tests run realistic scenarios that would catch bugs like:
- Wheel saturation not updating body momentum correctly
- MTQ bleeding not tracked as external impulse
- Slew momentum not returning to baseline
- Conservation tracking initialized at wrong time
"""

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from conops import MissionConfig, Queue, QueueDITL
from conops.simulation.acs import ACS

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
TLE_PATH = EXAMPLES_DIR / "example.tle"


def make_realistic_config(ephem, mtq_dipole: float = 32.0) -> MissionConfig:
    """Create config with realistic disturbances and MTQ enabled."""
    example_config = EXAMPLES_DIR / "example_config.json"
    if example_config.exists():
        cfg = MissionConfig.from_json_file(str(example_config))
    else:
        cfg = MissionConfig()

    acs = cfg.spacecraft_bus.attitude_control
    acs.spacecraft_moi = (45.0, 45.0, 45.0)
    acs.wheel_enabled = False

    # Magnetorquers - active momentum bleeding
    acs.magnetorquers = [
        {
            "name": "mtq_x",
            "orientation": (1, 0, 0),
            "dipole_strength": mtq_dipole,
            "power_draw": 5.0,
        },
        {
            "name": "mtq_y",
            "orientation": (0, 1, 0),
            "dipole_strength": mtq_dipole,
            "power_draw": 5.0,
        },
        {
            "name": "mtq_z",
            "orientation": (0, 0, 1),
            "dipole_strength": mtq_dipole,
            "power_draw": 5.0,
        },
    ]
    acs.magnetorquer_bfield_T = 3e-5

    # Disturbances - realistic external torques
    acs.cp_offset_body = (0.25, 0.0, 0.0)
    acs.residual_magnetic_moment = (0.05, 0, 0)
    acs.drag_area_m2 = 1.3
    acs.drag_coeff = 2.2
    acs.solar_area_m2 = 1.3
    acs.solar_reflectivity = 1.3
    acs.use_msis_density = True

    # Pyramid wheel array - can saturate during long slews
    s = math.sqrt(2 / 3)
    c = math.sqrt(1 / 3)
    acs.wheels = [
        {
            "name": "rw1",
            "orientation": (+s, 0.0, +c),
            "max_torque": 0.06,
            "max_momentum": 1.0,
        },
        {
            "name": "rw2",
            "orientation": (-s, 0.0, +c),
            "max_torque": 0.06,
            "max_momentum": 1.0,
        },
        {
            "name": "rw3",
            "orientation": (0.0, +s, -c),
            "max_torque": 0.06,
            "max_momentum": 1.0,
        },
        {
            "name": "rw4",
            "orientation": (0.0, -s, -c),
            "max_torque": 0.06,
            "max_momentum": 1.0,
        },
    ]

    cfg.constraint.ephem = ephem
    return cfg


class TestMomentumConservation:
    """Test that total system momentum is conserved."""

    @pytest.fixture
    def realistic_ditl(self):
        """Create a realistic DITL that exercises MTQ, saturation, and slews."""
        if not TLE_PATH.exists():
            pytest.skip(f"TLE file not found: {TLE_PATH}")

        from rust_ephem import TLEEphemeris

        # 2-hour simulation - long enough to accumulate errors, short enough for CI
        begin = datetime(2025, 1, 1)
        end = begin + timedelta(hours=2)
        ephem = TLEEphemeris(tle=str(TLE_PATH), begin=begin, end=end, step_size=10)

        cfg = make_realistic_config(ephem)
        cfg.constraint.ephem = ephem

        # Queue with targets that will cause slews
        np.random.seed(42)
        n_targets = 50
        queue = Queue(ephem=ephem, config=cfg)
        queue.slew_distance_weight = 0.0
        for i in range(n_targets):
            queue.add(
                merit=40,
                ra=np.random.uniform(0, 360),
                dec=np.random.uniform(-90, 90),
                obsid=10000 + i,
                name=f"target_{i}",
                exptime=60,
                ss_min=20,
            )

        ditl = QueueDITL(config=cfg, ephem=ephem, begin=begin, end=end, queue=queue)
        ditl.acs._wheel_mom_margin = 1.0  # Allow full capacity for saturation testing
        ditl.step_size = 10
        ditl.calc()

        return ditl

    def test_total_momentum_conservation(self, realistic_ditl):
        """Verify H_total = H_initial + cumulative_external_impulse throughout simulation."""
        wd = realistic_ditl.acs.wheel_dynamics

        h_current = wd.get_total_system_momentum()
        h_initial = wd._initial_total_momentum
        ext_impulse = wd._cumulative_external_impulse

        assert h_initial is not None, "Initial momentum should be recorded"

        h_expected = h_initial + ext_impulse
        error = h_current - h_expected
        error_mag = float(np.linalg.norm(error))

        # Tolerance: 10% of current momentum or 0.01 N·m·s minimum
        h_mag = float(np.linalg.norm(h_current))
        tolerance = max(0.1 * h_mag, 0.01)

        assert error_mag < tolerance, (
            f"Momentum conservation violated!\n"
            f"  H_current: {h_current}\n"
            f"  H_expected: {h_expected}\n"
            f"  Error: {error} (mag={error_mag:.4f})\n"
            f"  Tolerance: {tolerance:.4f}"
        )

    def test_no_conservation_warnings_logged(self, realistic_ditl):
        """Verify no conservation violations were logged during simulation."""
        warnings = realistic_ditl.acs.get_momentum_warnings()
        conservation_warnings = [w for w in warnings if "conservation" in w.lower()]

        assert len(conservation_warnings) == 0, (
            "Conservation violations logged:\n"
            + "\n".join(f"  {w}" for w in conservation_warnings[:5])
        )

    def test_wheel_saturation_preserves_conservation(self, realistic_ditl):
        """Verify conservation holds even when wheels saturate."""
        wd = realistic_ditl.acs.wheel_dynamics

        # Check if any wheel reached saturation
        wm_raw = np.array(getattr(realistic_ditl, "wheel_momentum_fraction_raw", []))
        if wm_raw.size == 0:
            pytest.skip("No wheel momentum telemetry")

        saturated_steps = (wm_raw >= 0.95).sum()

        # Even with saturation, conservation should hold
        h_current = wd.get_total_system_momentum()
        h_expected = wd._initial_total_momentum + wd._cumulative_external_impulse
        error_mag = float(np.linalg.norm(h_current - h_expected))

        h_mag = float(np.linalg.norm(h_current))
        tolerance = max(0.1 * h_mag, 0.01)

        assert error_mag < tolerance, (
            f"Conservation violated with {saturated_steps} saturation steps\n"
            f"  Error magnitude: {error_mag:.4f} N·m·s"
        )

    def test_mtq_bleeding_tracked_correctly(self, realistic_ditl):
        """Verify MTQ momentum bleeding is properly tracked as external impulse."""
        wd = realistic_ditl.acs.wheel_dynamics

        # MTQ should have been active (check power telemetry)
        mtq_power = np.array(getattr(realistic_ditl, "mtq_power", []))
        if mtq_power.size == 0:
            pytest.skip("No MTQ power telemetry")

        mtq_active_steps = (mtq_power > 0).sum()
        assert mtq_active_steps > 0, "MTQ should be active during slews"

        # External impulse should be non-zero (MTQ bleeds momentum)
        ext_impulse_mag = float(np.linalg.norm(wd._cumulative_external_impulse))
        assert ext_impulse_mag > 0.01, (
            f"External impulse too small ({ext_impulse_mag:.4f} N·m·s) "
            f"with {mtq_active_steps} MTQ-active steps"
        )

        # Conservation should still hold
        h_current = wd.get_total_system_momentum()
        h_expected = wd._initial_total_momentum + wd._cumulative_external_impulse
        error_mag = float(np.linalg.norm(h_current - h_expected))

        assert error_mag < 0.1, (
            f"Conservation violated with MTQ active\n"
            f"  External impulse: {ext_impulse_mag:.4f} N·m·s\n"
            f"  Error: {error_mag:.4f} N·m·s"
        )


class TestSlewMomentumSymmetry:
    """Test that symmetric slews return momentum to baseline."""

    def test_single_slew_conserves_momentum(self):
        """A single slew should conserve total system momentum."""
        from unittest.mock import MagicMock

        cfg = MissionConfig()
        acs_cfg = cfg.spacecraft_bus.attitude_control
        acs_cfg.spacecraft_moi = (10.0, 10.0, 10.0)
        acs_cfg.slew_acceleration = 0.5
        acs_cfg.max_slew_rate = 10.0
        acs_cfg.settle_time = 0.0

        # Simple 3-axis wheels
        acs_cfg.wheels = [
            {
                "name": "X",
                "orientation": (1, 0, 0),
                "max_torque": 1.0,
                "max_momentum": 20.0,
            },
            {
                "name": "Y",
                "orientation": (0, 1, 0),
                "max_torque": 1.0,
                "max_momentum": 20.0,
            },
            {
                "name": "Z",
                "orientation": (0, 0, 1),
                "max_torque": 1.0,
                "max_momentum": 20.0,
            },
        ]

        # Disable disturbances for clean test
        acs_cfg.cp_offset_body = (0.0, 0.0, 0.0)
        acs_cfg.residual_magnetic_moment = (0.0, 0.0, 0.0)
        acs_cfg.drag_area_m2 = 0.0
        acs_cfg.solar_area_m2 = 0.0
        acs_cfg.magnetorquers = []

        ephem = MagicMock()
        ephem.step_size = 1.0
        ephem.index = lambda dt: 0
        cfg.constraint.ephem = ephem

        acs = ACS(config=cfg, log=None)
        wd = acs.wheel_dynamics

        # Record initial state
        h_initial = wd.get_total_system_momentum().copy()

        # Create slew
        angle_deg = 10.0
        accel = 0.5
        t_peak = math.sqrt(angle_deg / accel)
        motion_time = 2 * t_peak

        slew = MagicMock()
        slew.slewdist = angle_deg
        slew.slewstart = 1000.0
        slew.slewtime = motion_time
        slew.slewend = 1000.0 + motion_time
        slew.rotation_axis = (0.0, 0.0, 1.0)
        slew.accel = accel  # Use actual accel value
        slew.vmax = 10.0  # High enough for triangular profile
        slew.is_slewing = lambda t: 1000.0 <= t < slew.slewend

        # Execute slew
        dt = 0.1
        acs._last_pointing_time = 1000.0
        acs.current_slew = slew
        acs.last_slew = slew
        acs._was_slewing = False

        for t in np.arange(1000.0, slew.slewend + dt, dt):
            acs._update_wheel_momentum(t + dt)
            acs._last_pointing_time = t + dt

        # Final state
        h_final = wd.get_total_system_momentum()
        ext_impulse = wd._cumulative_external_impulse

        # With no external torques, total momentum should be unchanged
        delta = np.linalg.norm(h_final - h_initial - ext_impulse)
        assert delta < 0.01, (
            f"Slew violated conservation!\n"
            f"  H_initial: {h_initial}\n"
            f"  H_final: {h_final}\n"
            f"  External impulse: {ext_impulse}\n"
            f"  Delta: {delta:.6f} N·m·s"
        )

    def test_slew_wheel_momentum_returns_to_baseline(self):
        """Wheel momentum should return to baseline after symmetric slew."""
        from unittest.mock import MagicMock

        cfg = MissionConfig()
        acs_cfg = cfg.spacecraft_bus.attitude_control
        acs_cfg.spacecraft_moi = (10.0, 10.0, 10.0)
        acs_cfg.slew_acceleration = 0.5
        acs_cfg.max_slew_rate = 10.0
        acs_cfg.settle_time = 0.0

        acs_cfg.wheels = [
            {
                "name": "Z",
                "orientation": (0, 0, 1),
                "max_torque": 1.0,
                "max_momentum": 20.0,
            },
        ]
        acs_cfg.cp_offset_body = (0.0, 0.0, 0.0)
        acs_cfg.residual_magnetic_moment = (0.0, 0.0, 0.0)
        acs_cfg.drag_area_m2 = 0.0
        acs_cfg.solar_area_m2 = 0.0
        acs_cfg.magnetorquers = []

        ephem = MagicMock()
        ephem.step_size = 1.0
        ephem.index = lambda dt: 0
        cfg.constraint.ephem = ephem

        acs = ACS(config=cfg, log=None)

        h_initial = acs._get_total_wheel_momentum().copy()

        # Triangular profile slew
        angle_deg = 10.0
        accel = 0.5
        t_peak = math.sqrt(angle_deg / accel)
        motion_time = 2 * t_peak

        slew = MagicMock()
        slew.slewdist = angle_deg
        slew.slewstart = 1000.0
        slew.slewtime = motion_time
        slew.slewend = 1000.0 + motion_time
        slew.rotation_axis = (0.0, 0.0, 1.0)
        slew.accel = accel  # Use actual accel value
        slew.vmax = 10.0  # High enough for triangular profile
        slew.is_slewing = lambda t: 1000.0 <= t < slew.slewend

        dt = 0.1
        acs._last_pointing_time = 1000.0
        acs.current_slew = slew
        acs.last_slew = slew
        acs._was_slewing = False

        for t in np.arange(1000.0, slew.slewend + dt, dt):
            acs._update_wheel_momentum(t + dt)
            acs._last_pointing_time = t + dt

        h_final = acs._get_total_wheel_momentum()
        delta = np.linalg.norm(h_final - h_initial)

        assert delta < 0.01, (
            f"Wheel momentum not returning to baseline!\n"
            f"  H_initial: {h_initial}\n"
            f"  H_final: {h_final}\n"
            f"  Delta: {delta:.6f} N·m·s"
        )


class TestWheelSaturationConservation:
    """Test that conservation holds when wheels saturate."""

    def test_saturation_clamp_preserves_conservation(self):
        """When wheel saturates, body momentum must compensate."""
        from conops.simulation.reaction_wheel import ReactionWheel
        from conops.simulation.wheel_dynamics import WheelDynamics

        # Single wheel near saturation
        wheel = ReactionWheel(
            max_torque=1.0,
            max_momentum=1.0,
            orientation=(0, 0, 1),
            current_momentum=0.9,  # Near limit
        )

        inertia = np.diag([10.0, 10.0, 10.0])
        wd = WheelDynamics(wheels=[wheel], inertia_matrix=inertia)

        # Record initial state
        h_initial = wd.get_total_system_momentum().copy()

        # Apply torque that would exceed capacity
        taus = np.array([0.5])  # Would add 0.5 N·m·s in 1s, exceeding 1.0 limit
        dt = 1.0
        wd.apply_wheel_torques(taus, dt)

        h_final = wd.get_total_system_momentum()

        # Total momentum should be conserved (no external torque)
        delta = np.linalg.norm(h_final - h_initial)
        assert delta < 1e-9, (
            f"Saturation broke conservation!\n"
            f"  Wheel momentum: {wheel.current_momentum}\n"
            f"  Body momentum: {wd.body_momentum}\n"
            f"  H_initial: {h_initial}\n"
            f"  H_final: {h_final}\n"
            f"  Delta: {delta}"
        )

        # Wheel should be at max
        assert wheel.current_momentum == pytest.approx(1.0, abs=1e-9)

        # Body should have absorbed the reaction
        assert wd.body_momentum[2] == pytest.approx(
            -0.1, abs=1e-9
        )  # -0.1 = -(1.0 - 0.9)
