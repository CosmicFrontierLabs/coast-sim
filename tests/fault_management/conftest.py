"""Test fixtures for fault management subsystem tests."""

from unittest.mock import Mock

import pytest

from conops.battery import Battery
from conops.config import Config
from conops.constraint import Constraint
from conops.fault_management import FaultManagement
from conops.groundstation import GroundStationRegistry
from conops.instrument import Payload
from conops.solar_panel import SolarPanelSet
from conops.spacecraft_bus import SpacecraftBus


@pytest.fixture
def base_config():
    # Minimal mocks for required subsystems
    spacecraft_bus = SpacecraftBus()
    solar_panel = SolarPanelSet(panels=[])
    payload = Payload(instruments=[])
    battery = Battery(watthour=1000, max_depth_of_discharge=0.6)
    constraint = Mock(spec=Constraint)
    constraint.ephem = Mock()
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
