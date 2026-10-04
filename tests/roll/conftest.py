"""Test fixtures for roll subsystem tests."""

from unittest.mock import Mock

import numpy as np
import pytest

from conops import SolarPanel, SolarPanelSet


@pytest.fixture
def mock_ephem():
    ephem = Mock()
    ephem.index = Mock(return_value=0)
    # New direct array access (rust-ephem 0.3.0+)
    sun_pv = Mock()
    sun_pv.position = [np.array([1000, 500, 800])]
    ephem.sun_pv = sun_pv
    gcrs_pv = Mock()
    gcrs_pv.position = [np.array([0.0, 0.0, 0.0])]
    ephem.gcrs_pv = gcrs_pv
    return ephem


@pytest.fixture
def mock_sun_coord():
    sun_coord = Mock()
    sun_coord.cartesian.xyz.to_value = Mock(return_value=np.array([1000, 500, 800]))
    return sun_coord


def _solar_panel_set(
    *panels: tuple[tuple[float, float, float], float],
) -> SolarPanelSet:
    return SolarPanelSet(
        panels=[
            SolarPanel(
                normal=normal,
                conversion_efficiency=0.3,
                max_power=max_power,
            )
            for normal, max_power in panels
        ],
        conversion_efficiency=0.3,
    )


@pytest.fixture
def mock_solar_panel_single():
    return _solar_panel_set(((0.0, 1.0, 0.0), 800.0))


@pytest.fixture
def mock_solar_panel_multiple():
    return _solar_panel_set(
        ((0.0, 1.0, 0.0), 800.0),
        ((0.0, 0.0, -1.0), 600.0),
    )


@pytest.fixture
def mock_solar_panel_canted():
    return _solar_panel_set(((0.1, 0.866, -0.5), 800.0))


@pytest.fixture
def mock_ephem_sidemount():
    ephem = Mock()
    ephem.index = Mock(return_value=0)
    # Mock the sun attribute to be subscriptable (legacy)
    sun_mock = Mock()
    sun_mock.cartesian.xyz.to_value = Mock(return_value=np.array([1000, 500, 800]))
    ephem.sun = Mock()
    ephem.sun.__getitem__ = Mock(return_value=sun_mock)
    # New direct array access (rust-ephem 0.3.0+)
    sun_pv = Mock()
    sun_pv.position = [np.array([1000, 500, 800])]
    ephem.sun_pv = sun_pv
    gcrs_pv = Mock()
    gcrs_pv.position = [np.array([0.0, 0.0, 0.0])]
    ephem.gcrs_pv = gcrs_pv
    return ephem
