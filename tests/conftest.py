"""Shared pytest fixtures for test suite."""

from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_ephem():
    """Create mock ephemeris with both legacy and new rust-ephem 0.3.0+ attributes."""

    import numpy as np

    ephem = Mock()
    ephem.index = Mock(return_value=0)

    # Legacy SkyCoord-style access
    sun_mock = Mock()
    sun_mock.ra = Mock()
    sun_mock.ra.deg = 90.0
    sun_mock.dec = Mock()
    sun_mock.dec.deg = 30.0
    sun_mock.separation = Mock(return_value=Mock(deg=45.0))
    ephem.sun = np.array([sun_mock])

    # New direct array access (rust-ephem 0.3.0+)
    ephem.sun_ra_deg = np.array([90.0])
    ephem.sun_dec_deg = np.array([30.0])

    # Earth position
    earth_mock = Mock()
    earth_mock.separation = Mock(return_value=Mock(deg=0.5))
    ephem.earth = np.array([earth_mock])

    # Earth radius angle
    ephem.earth_radius_angle = np.array([Mock(deg=0.3)])

    ephem._tle_ephem = Mock()

    return ephem


@pytest.fixture
def mock_spacecraft_bus() -> Mock:
    """Create a mock SpacecraftBus with the subsystem stubs ACS/passes production
    code reads directly (attitude_control, radiators, star_trackers).

    Centralizes a block that used to be copy-pasted across multiple test files'
    mock_config fixtures; add new subsystem stubs here as production code grows
    to depend on them, rather than patching each fixture individually.
    """
    spacecraft_bus = Mock()
    spacecraft_bus.attitude_control = Mock()
    spacecraft_bus.attitude_control.predict_slew = Mock(return_value=(45.0, []))
    spacecraft_bus.attitude_control.slew_time = Mock(return_value=100.0)
    spacecraft_bus.radiators = Mock()
    spacecraft_bus.radiators.num_radiators = Mock(return_value=0)
    spacecraft_bus.star_trackers = Mock()
    spacecraft_bus.star_trackers.num_trackers = Mock(return_value=0)
    return spacecraft_bus


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
