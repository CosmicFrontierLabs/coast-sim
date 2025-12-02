"""Shared pytest fixtures for test suite."""

from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_ephem():
    """Create mock ephemeris."""
    ephem = Mock()
    ephem.step_size = 60.0
    ephem.timestamp = Mock()
    ephem.timestamp.unix = [1514764800.0 + i * 60.0 for i in range(1440)]
    ephem.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0)) for _ in range(1440)]
    ephem.sun = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
    ephem.index.return_value = 0
    return ephem


@pytest.fixture
def base_constraint():
    """Create a basic constraint fixture."""
    from conops import Constraint

    return Constraint()


@pytest.fixture
def payload_constraint():
    """Create a payload constraint fixture."""
    from conops import Constraint

    return Constraint()


@pytest.fixture
def config_with_payload_constraint(base_constraint, payload_constraint):
    """Create a config with payload constraint."""
    from conops import (
        Battery,
        FaultManagement,
        GroundStationRegistry,
        MissionConfig,
        Payload,
        SolarPanelSet,
        SpacecraftBus,
    )

    config = MissionConfig(
        name="Test Config",
        spacecraft_bus=SpacecraftBus(),
        solar_panel=SolarPanelSet(),
        payload=Payload(),
        battery=Battery(),
        constraint=base_constraint,
        ground_stations=GroundStationRegistry(),
        fault_management=FaultManagement(),
    )
    config.payload_constraint = payload_constraint
    return config
