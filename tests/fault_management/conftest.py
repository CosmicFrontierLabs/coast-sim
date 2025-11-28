"""Test fixtures for fault management subsystem tests."""

from unittest.mock import Mock

import pytest

from conops import (
    Battery,
    Config,
    Constraint,
    FaultManagement,
    GroundStationRegistry,
    Payload,
    SolarPanelSet,
    SpacecraftBus,
)


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
    cfg = Config(
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
