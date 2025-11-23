"""Test fixtures for config subsystem tests."""

from unittest.mock import Mock

import pytest

from conops.battery import Battery
from conops.config import Config
from conops.constraint import Constraint
from conops.groundstation import GroundStationRegistry
from conops.instrument import Payload
from conops.solar_panel import SolarPanelSet
from conops.spacecraft_bus import SpacecraftBus


@pytest.fixture
def minimal_config():
    name = "Test Config"
    spacecraft_bus = Mock(spec=SpacecraftBus)
    solar_panel = Mock(spec=SolarPanelSet)
    payload = Mock(spec=Payload)
    battery = Mock(spec=Battery)
    constraint = Mock(spec=Constraint)
    ground_stations = Mock(spec=GroundStationRegistry)

    config = Config(
        name=name,
        spacecraft_bus=spacecraft_bus,
        solar_panel=solar_panel,
        payload=payload,
        battery=battery,
        constraint=constraint,
        ground_stations=ground_stations,
    )

    return {
        "config": config,
        "spacecraft_bus": spacecraft_bus,
        "solar_panel": solar_panel,
        "payload": payload,
        "battery": battery,
        "constraint": constraint,
        "ground_stations": ground_stations,
    }
