"""Test fixtures for config subsystem tests."""

from unittest.mock import Mock

import pytest

from conops import (
    Battery,
    Constraint,
    GroundStationRegistry,
    MissionConfig,
    Payload,
    SolarPanelSet,
    SpacecraftBus,
)


@pytest.fixture
def minimal_config():
    name = "Test Config"
    spacecraft_bus = Mock(spec=SpacecraftBus)
    solar_panel = Mock(spec=SolarPanelSet)
    payload = Mock(spec=Payload)
    battery = Mock(spec=Battery)
    constraint = Mock(spec=Constraint)
    ground_stations = Mock(spec=GroundStationRegistry)
    fault_management = None

    config = MissionConfig(
        name=name,
        spacecraft_bus=spacecraft_bus,
        solar_panel=solar_panel,
        payload=payload,
        battery=battery,
        constraint=constraint,
        ground_stations=ground_stations,
        fault_management=fault_management,
    )

    return {
        "config": config,
        "spacecraft_bus": spacecraft_bus,
        "solar_panel": solar_panel,
        "payload": payload,
        "battery": battery,
        "constraint": constraint,
        "ground_stations": ground_stations,
        "fault_management": fault_management,
    }
