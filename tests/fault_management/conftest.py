"""Test fixtures for fault management subsystem tests."""

from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np
import pytest
import rust_ephem

from conops import (
    ACS,
    Battery,
    Constraint,
    FaultManagement,
    GroundStationRegistry,
    MissionConfig,
    Payload,
    SolarPanelSet,
    SpacecraftBus,
)
from conops.config.fault_management import FaultConstraint
from conops.ditl.telemetry import Housekeeping


@pytest.fixture
def acs_stub() -> Mock:
    acs = Mock()
    acs.in_safe_mode = False
    acs.acsmode = None
    acs.ephem = Mock(step_size=1.0)
    return acs


@pytest.fixture
def fm_with_yellow_state(base_config: MissionConfig) -> tuple[FaultManagement, ACS]:
    """Fixture providing fault management after checking yellow state."""
    fm = base_config.fault_management
    acs = ACS(config=base_config)
    battery_threshold = next(t for t in fm.thresholds if t.name == "battery_level")
    base_config.battery.charge_level = base_config.battery.watthour * (
        battery_threshold.yellow - 0.01
    )
    hk = Housekeeping(
        timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
        battery_level=base_config.battery.battery_level,
        recorder_fill_fraction=0.0,
        star_tracker_functional_count=0,
    )
    fm.check(hk, acs=acs)
    return fm, acs


@pytest.fixture
def fm_with_red_state(base_config: MissionConfig) -> tuple[FaultManagement, ACS]:
    """Fixture providing fault management after checking red state."""
    fm = base_config.fault_management
    acs = ACS(config=base_config)
    battery_threshold = next(t for t in fm.thresholds if t.name == "battery_level")
    base_config.battery.charge_level = base_config.battery.watthour * (
        battery_threshold.red - 0.01
    )
    hk = Housekeeping(
        timestamp=datetime.fromtimestamp(2000.0, tz=timezone.utc),
        battery_level=base_config.battery.battery_level,
        recorder_fill_fraction=0.0,
        star_tracker_functional_count=0,
    )
    fm.check(hk, acs=acs)
    return fm, acs


@pytest.fixture
def fm_with_multiple_cycles(base_config: MissionConfig) -> tuple[FaultManagement, ACS]:
    """Fixture providing fault management after multiple yellow cycles."""
    fm = base_config.fault_management
    acs = ACS(config=base_config)
    battery_threshold = next(t for t in fm.thresholds if t.name == "battery_level")
    yellow_limit = battery_threshold.yellow

    # Cycle 1: nominal (no accumulation)
    base_config.battery.charge_level = base_config.battery.watthour * (
        yellow_limit + 0.05
    )
    hk = Housekeeping(
        timestamp=datetime.fromtimestamp(3000.0, tz=timezone.utc),
        battery_level=base_config.battery.battery_level,
        recorder_fill_fraction=0.0,
        star_tracker_functional_count=0,
    )
    fm.check(hk, acs=acs)

    # Cycle 2: yellow
    base_config.battery.charge_level = base_config.battery.watthour * (
        yellow_limit - 0.01
    )
    hk = Housekeeping(
        timestamp=datetime.fromtimestamp(3060.0, tz=timezone.utc),
        battery_level=base_config.battery.battery_level,
        recorder_fill_fraction=0.0,
        star_tracker_functional_count=0,
    )
    fm.check(hk, acs=acs)

    # Cycle 3: yellow again
    hk = Housekeeping(
        timestamp=datetime.fromtimestamp(3120.0, tz=timezone.utc),
        battery_level=base_config.battery.battery_level,
        recorder_fill_fraction=0.0,
        star_tracker_functional_count=0,
    )
    fm.check(hk, acs=acs)
    return fm, acs


@pytest.fixture
def fm_with_above_threshold(acs_stub) -> FaultManagement:
    """Fixture providing fault management with 'above' direction threshold after multiple checks."""
    fm = FaultManagement()
    fm.add_threshold("battery_level", yellow=50.0, red=60.0, direction="above")

    # Test nominal
    hk = Housekeeping(
        timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
        battery_level=40.0,
    )
    fm.check(hk, acs=acs_stub)

    # Test yellow
    hk = Housekeeping(
        timestamp=datetime.fromtimestamp(1001.0, tz=timezone.utc),
        battery_level=55.0,
    )
    fm.check(hk, acs=acs_stub)

    # Test red
    hk = Housekeeping(
        timestamp=datetime.fromtimestamp(1002.0, tz=timezone.utc),
        battery_level=65.0,
    )
    fm.check(hk, acs=acs_stub)
    return fm


class DummyBattery:
    """Simple battery mock for testing."""

    def __init__(self):
        self.charge_level = 800.0
        self.watthour = 1000
        self.capacity = 1000
        self.max_depth_of_discharge = 0.6

    @property
    def battery_level(self):
        return self.charge_level / self.watthour


class DummyEphemeris:
    """Minimal mock ephemeris for testing."""

    def __init__(self):
        self.step_size = 60.0
        self.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
        self.sun = [Mock(ra=Mock(deg=45.0), dec=Mock(deg=23.5))]
        # New direct array access (rust-ephem 0.3.0+)
        self.earth_ra_deg = [0.0]
        self.earth_dec_deg = [0.0]
        self.sun_ra_deg = [45.0]
        self.sun_dec_deg = [23.5]
        # Mock position/velocity data for roll calculation
        self.sun_pv = Mock(position=np.array([[1.5e8, 0.0, 0.0]]))
        self.gcrs_pv = Mock(position=np.array([[0.0, 0.0, 6378.0]]))

    def index(self, time):
        return 0


@pytest.fixture
def base_config() -> MissionConfig:
    # Minimal mocks for required subsystems
    spacecraft_bus = Mock(spec=SpacecraftBus)
    spacecraft_bus.attitude_control = Mock()
    spacecraft_bus.attitude_control.predict_slew = Mock(return_value=(45.0, []))
    spacecraft_bus.attitude_control.slew_time = Mock(return_value=100.0)

    solar_panel = Mock(spec=SolarPanelSet)
    solar_panel.optimal_charging_pointing = Mock(return_value=(45.0, 23.5))

    payload = Mock(spec=Payload)

    # Use real Battery object
    battery = Battery(watthour=1000, max_depth_of_discharge=0.6)
    battery.charge_level = 800.0

    constraint = Mock(spec=Constraint)
    constraint.ephem = DummyEphemeris()  # Use DummyEphemeris instead of Mock
    constraint.in_eclipse = Mock(return_value=False)
    ground_stations = Mock(spec=GroundStationRegistry)
    fm = FaultManagement()
    cfg = MissionConfig(
        spacecraft_bus=spacecraft_bus,
        solar_panel=solar_panel,
        payload=payload,
        battery=battery,
        constraint=constraint,
        ground_stations=ground_stations,
        fault_management=fm,
    )
    cfg.init_fault_management_defaults()
    return cfg


# Fixtures for common data used across tests
@pytest.fixture
def ephem() -> rust_ephem.TLEEphemeris:
    return rust_ephem.TLEEphemeris(
        tle="examples/example.tle",
        begin=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 1, 2, tzinfo=timezone.utc),
        step_size=60,
    )


@pytest.fixture
def fm() -> FaultManagement:
    return FaultManagement()


@pytest.fixture
def fm_safe() -> FaultManagement:
    return FaultManagement(safe_mode_on_red=True)


@pytest.fixture
def constraint_sun_30() -> rust_ephem.SunConstraint:
    return rust_ephem.SunConstraint(min_angle=30.0)


@pytest.fixture
def constraint_sun_90() -> rust_ephem.SunConstraint:
    return rust_ephem.SunConstraint(min_angle=90.0)


@pytest.fixture
def constraint_earth_10() -> rust_ephem.EarthLimbConstraint:
    return rust_ephem.EarthLimbConstraint(min_angle=10.0)


@pytest.fixture
def constraint_moon_5() -> rust_ephem.MoonConstraint:
    return rust_ephem.MoonConstraint(min_angle=5.0)


@pytest.fixture
def fault_constraint() -> FaultConstraint:
    return FaultConstraint(
        name="test_sun_limit",
        constraint=rust_ephem.SunConstraint(min_angle=30.0),
        time_threshold_seconds=300.0,
        description="Test sun constraint",
    )


@pytest.fixture
def fault_monitor_constraint() -> FaultConstraint:
    return FaultConstraint(
        name="test_monitor",
        constraint=rust_ephem.MoonConstraint(min_angle=5.0),
        time_threshold_seconds=None,
        description="Monitoring only",
    )
