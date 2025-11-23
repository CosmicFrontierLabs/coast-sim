"""Test fixtures for groundstation subsystem tests."""

import pytest

from conops import GroundStation, GroundStationRegistry


@pytest.fixture
def sample_groundstation():
    return GroundStation(
        code="GHA", name="Ghana", latitude_deg=5.74, longitude_deg=0.30
    )


@pytest.fixture
def groundstation_registry():
    return GroundStationRegistry()


@pytest.fixture
def default_registry():
    return GroundStationRegistry.default()
