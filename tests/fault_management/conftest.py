"""Test fixtures for fault management subsystem tests."""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
import rust_ephem

from conops import (
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
        self.step_size = 1.0
        self.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
        self.sun = [Mock(ra=Mock(deg=45.0), dec=Mock(deg=23.5))]

    def index(self, time):
        return 0


@pytest.fixture
def base_config():
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
def ephem():
    return rust_ephem.TLEEphemeris(
        tle="examples/example.tle",
        begin=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 1, 2, tzinfo=timezone.utc),
        step_size=60,
    )


@pytest.fixture
def fm():
    return FaultManagement()


@pytest.fixture
def fm_safe():
    return FaultManagement(safe_mode_on_red=True)


@pytest.fixture
def constraint_sun_30():
    return rust_ephem.SunConstraint(min_angle=30.0)


@pytest.fixture
def constraint_sun_90():
    return rust_ephem.SunConstraint(min_angle=90.0)


@pytest.fixture
def constraint_earth_10():
    return rust_ephem.EarthLimbConstraint(min_angle=10.0)


@pytest.fixture
def constraint_moon_5():
    return rust_ephem.MoonConstraint(min_angle=5.0)


@pytest.fixture
def fault_constraint():
    return FaultConstraint(
        name="test_sun_limit",
        constraint=rust_ephem.SunConstraint(min_angle=30.0),
        time_threshold_seconds=300.0,
        description="Test sun constraint",
    )


@pytest.fixture
def fault_monitor_constraint():
    return FaultConstraint(
        name="test_monitor",
        constraint=rust_ephem.MoonConstraint(min_angle=5.0),
        time_threshold_seconds=None,
        description="Monitoring only",
    )
