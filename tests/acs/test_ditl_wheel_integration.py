"""Integration tests for DITL simulation with reaction wheel modeling.

These tests validate that:
1. QueueDITL runs successfully with wheels enabled
2. Wheel telemetry is recorded properly
3. No NaN/Inf values appear in wheel data
4. Momentum accumulates and stays within bounds
"""

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from rust_ephem import TLEEphemeris

from conops import MissionConfig, Pointing, QueueDITL
from conops.targets import Queue

# Path to example TLE file
TLE_PATH = Path(__file__).parent.parent.parent / "examples" / "example.tle"


@pytest.fixture
def wheel_enabled_config():
    """Create a mission config with reaction wheels enabled."""
    cfg = MissionConfig()

    # Configure 4-wheel pyramid (redundant, well-conditioned)
    acs_cfg = cfg.spacecraft_bus.attitude_control
    theta = math.radians(54.74)
    acs_cfg.wheels = [
        {
            "name": "rw1",
            "orientation": [math.sin(theta), 0.0, math.cos(theta)],
            "max_torque": 0.1,
            "max_momentum": 2.0,
            "idle_power_w": 5.0,
            "torque_power_coeff": 50.0,
        },
        {
            "name": "rw2",
            "orientation": [0.0, math.sin(theta), math.cos(theta)],
            "max_torque": 0.1,
            "max_momentum": 2.0,
            "idle_power_w": 5.0,
            "torque_power_coeff": 50.0,
        },
        {
            "name": "rw3",
            "orientation": [-math.sin(theta), 0.0, math.cos(theta)],
            "max_torque": 0.1,
            "max_momentum": 2.0,
            "idle_power_w": 5.0,
            "torque_power_coeff": 50.0,
        },
        {
            "name": "rw4",
            "orientation": [0.0, -math.sin(theta), math.cos(theta)],
            "max_torque": 0.1,
            "max_momentum": 2.0,
            "idle_power_w": 5.0,
            "torque_power_coeff": 50.0,
        },
    ]
    acs_cfg.spacecraft_moi = (10.0, 10.0, 8.0)  # kg*m^2

    # Configure disturbance model (small but non-zero)
    acs_cfg.cp_offset_body = (0.01, 0.0, 0.0)  # 1cm offset
    acs_cfg.drag_area_m2 = 2.0
    acs_cfg.solar_area_m2 = 4.0

    return cfg


@pytest.fixture
def short_ephem():
    """Create a short ephemeris for quick testing (~30 minutes)."""
    begin = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    end = begin + timedelta(minutes=30)
    return TLEEphemeris(tle=str(TLE_PATH), begin=begin, end=end, step_size=10)


@pytest.fixture
def orbit_ephem():
    """Create an ephemeris for one full orbit (~93 minutes)."""
    begin = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    end = begin + timedelta(minutes=93)
    return TLEEphemeris(tle=str(TLE_PATH), begin=begin, end=end, step_size=10)


def create_target(config, ra, dec, exptime=120, obsid=1):
    """Helper to create a Pointing object."""
    return Pointing(
        config=config,
        ra=ra,
        dec=dec,
        exptime=exptime,
        obsid=obsid,
        name=f"Target_{obsid}",
        merit=100.0 - obsid,  # Higher priority for lower obsid
    )


class TestDITLWheelIntegration:
    """Integration tests for DITL with reaction wheels."""

    def test_queue_ditl_runs_with_wheels(self, wheel_enabled_config, short_ephem):
        """Verify QueueDITL completes without errors when wheels are enabled."""
        cfg = wheel_enabled_config
        cfg.constraint.ephem = short_ephem

        ditl = QueueDITL(config=cfg)
        ditl.ephem = short_ephem
        ditl.begin = short_ephem.timestamp[0]
        ditl.end = short_ephem.timestamp[-1]

        # Create a queue with a target
        queue = Queue(config=cfg)
        queue.ephem = short_ephem
        queue.append(create_target(cfg, ra=45.0, dec=30.0, exptime=120, obsid=1))
        ditl.queue = queue

        # Run simulation
        result = ditl.calc()

        assert result is True
        assert len(ditl.utime) > 0

    def test_wheel_telemetry_recorded(self, wheel_enabled_config, short_ephem):
        """Verify wheel telemetry arrays are populated during simulation."""
        cfg = wheel_enabled_config
        cfg.constraint.ephem = short_ephem

        ditl = QueueDITL(config=cfg)
        ditl.ephem = short_ephem
        ditl.begin = short_ephem.timestamp[0]
        ditl.end = short_ephem.timestamp[-1]

        queue = Queue(config=cfg)
        queue.ephem = short_ephem
        queue.append(create_target(cfg, ra=45.0, dec=30.0, exptime=120, obsid=1))
        ditl.queue = queue

        ditl.calc()

        # Check telemetry arrays exist and have correct length
        simlen = len(ditl.utime)
        assert len(ditl.wheel_momentum_fraction) == simlen
        assert len(ditl.wheel_torque_fraction) == simlen
        assert len(ditl.wheel_saturation) == simlen
        assert len(ditl.wheel_power) == simlen

    def test_wheel_momentum_no_nan_inf(self, wheel_enabled_config, short_ephem):
        """Verify no NaN or Inf values in wheel telemetry."""
        cfg = wheel_enabled_config
        cfg.constraint.ephem = short_ephem

        ditl = QueueDITL(config=cfg)
        ditl.ephem = short_ephem
        ditl.begin = short_ephem.timestamp[0]
        ditl.end = short_ephem.timestamp[-1]

        queue = Queue(config=cfg)
        queue.ephem = short_ephem
        queue.append(create_target(cfg, ra=45.0, dec=30.0, exptime=120, obsid=1))
        ditl.queue = queue

        ditl.calc()

        # Check for NaN/Inf
        assert all(np.isfinite(ditl.wheel_momentum_fraction))
        assert all(np.isfinite(ditl.wheel_torque_fraction))
        assert all(np.isfinite(ditl.wheel_power))

        # Check per-wheel histories if available
        for name, hist in ditl.wheel_momentum_history.items():
            assert all(np.isfinite(hist)), f"NaN/Inf in momentum history for {name}"

    def test_wheel_momentum_within_bounds(self, wheel_enabled_config, orbit_ephem):
        """Verify wheel momentum stays within configured limits over an orbit."""
        cfg = wheel_enabled_config
        cfg.constraint.ephem = orbit_ephem

        ditl = QueueDITL(config=cfg)
        ditl.ephem = orbit_ephem
        ditl.begin = orbit_ephem.timestamp[0]
        ditl.end = orbit_ephem.timestamp[-1]

        # Add multiple targets to exercise slewing
        queue = Queue(config=cfg)
        queue.ephem = orbit_ephem
        queue.append(create_target(cfg, ra=0.0, dec=0.0, exptime=300, obsid=1))
        queue.append(create_target(cfg, ra=90.0, dec=45.0, exptime=300, obsid=2))
        queue.append(create_target(cfg, ra=180.0, dec=-30.0, exptime=300, obsid=3))
        ditl.queue = queue

        ditl.calc()

        # Check momentum fractions stay in [0, 1] range
        assert all(0.0 <= f <= 1.0 for f in ditl.wheel_momentum_fraction)
        assert all(0.0 <= f <= 1.0 for f in ditl.wheel_torque_fraction)

        # Check raw per-wheel maxima
        for name, max_frac in ditl.wheel_per_wheel_max_raw.items():
            assert 0.0 <= max_frac <= 1.0, f"Wheel {name} exceeded bounds: {max_frac}"

    def test_disturbance_torques_recorded(self, wheel_enabled_config, short_ephem):
        """Verify disturbance torque telemetry is recorded."""
        cfg = wheel_enabled_config
        cfg.constraint.ephem = short_ephem

        ditl = QueueDITL(config=cfg)
        ditl.ephem = short_ephem
        ditl.begin = short_ephem.timestamp[0]
        ditl.end = short_ephem.timestamp[-1]

        queue = Queue(config=cfg)
        queue.ephem = short_ephem
        queue.append(create_target(cfg, ra=45.0, dec=30.0, exptime=120, obsid=1))
        ditl.queue = queue

        ditl.calc()

        simlen = len(ditl.utime)
        assert len(ditl.disturbance_total) == simlen
        assert len(ditl.disturbance_gg) == simlen
        assert len(ditl.disturbance_drag) == simlen
        assert len(ditl.disturbance_srp) == simlen

    def test_wheel_config_validation_runs(self, wheel_enabled_config, short_ephem):
        """Verify wheel configuration validation runs during ACS init."""
        cfg = wheel_enabled_config
        cfg.constraint.ephem = short_ephem

        ditl = QueueDITL(config=cfg)

        # Check validation attributes were set
        assert hasattr(ditl.acs, "_wheel_config_rank")
        assert hasattr(ditl.acs, "_wheel_config_n_wheels")
        assert ditl.acs._wheel_config_rank == 3  # Full 3-axis control
        assert ditl.acs._wheel_config_n_wheels == 4  # 4 wheels configured

    def test_wheel_power_recorded(self, wheel_enabled_config, short_ephem):
        """Verify wheel power consumption is tracked."""
        cfg = wheel_enabled_config
        cfg.constraint.ephem = short_ephem

        ditl = QueueDITL(config=cfg)
        ditl.ephem = short_ephem
        ditl.begin = short_ephem.timestamp[0]
        ditl.end = short_ephem.timestamp[-1]

        queue = Queue(config=cfg)
        queue.ephem = short_ephem
        queue.append(create_target(cfg, ra=45.0, dec=30.0, exptime=120, obsid=1))
        ditl.queue = queue

        ditl.calc()

        # With 4 wheels at 5W idle each, minimum power should be 20W
        assert all(p >= 20.0 for p in ditl.wheel_power)
        # Power should be finite
        assert all(np.isfinite(ditl.wheel_power))


class TestWheelMomentumAccumulation:
    """Tests for momentum accumulation behavior."""

    def test_momentum_changes_during_simulation(
        self, wheel_enabled_config, orbit_ephem
    ):
        """Verify momentum changes over time due to disturbances and maneuvers."""
        cfg = wheel_enabled_config
        # Increase disturbance to make effect more visible
        cfg.spacecraft_bus.attitude_control.cp_offset_body = (0.1, 0.0, 0.0)
        cfg.constraint.ephem = orbit_ephem

        ditl = QueueDITL(config=cfg)
        ditl.ephem = orbit_ephem
        ditl.begin = orbit_ephem.timestamp[0]
        ditl.end = orbit_ephem.timestamp[-1]

        # Multiple targets to force slews
        queue = Queue(config=cfg)
        queue.ephem = orbit_ephem
        queue.append(create_target(cfg, ra=0.0, dec=0.0, exptime=600, obsid=1))
        queue.append(create_target(cfg, ra=180.0, dec=0.0, exptime=600, obsid=2))
        ditl.queue = queue

        ditl.calc()

        # Check that momentum fraction changes over time (not constant)
        if len(ditl.wheel_momentum_fraction) > 10:
            # At least some variation should exist
            assert max(ditl.wheel_momentum_fraction) > min(ditl.wheel_momentum_fraction)

    def test_slew_increases_wheel_activity(self, wheel_enabled_config, short_ephem):
        """Verify that slewing causes wheel torque activity."""
        cfg = wheel_enabled_config
        cfg.constraint.ephem = short_ephem

        ditl = QueueDITL(config=cfg)
        ditl.ephem = short_ephem
        ditl.begin = short_ephem.timestamp[0]
        ditl.end = short_ephem.timestamp[-1]

        # Two widely separated targets to force a large slew
        queue = Queue(config=cfg)
        queue.ephem = short_ephem
        queue.append(create_target(cfg, ra=0.0, dec=0.0, exptime=60, obsid=1))
        queue.append(create_target(cfg, ra=90.0, dec=45.0, exptime=60, obsid=2))
        ditl.queue = queue

        ditl.calc()

        # During slews, torque fraction should be non-zero at some point
        assert max(ditl.wheel_torque_fraction) > 0.0
