"""Test fixtures for roll subsystem tests."""

from unittest.mock import Mock

import numpy as np
import pytest


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


@pytest.fixture
def mock_solar_panel_single():
    solar_panel = Mock()
    mock_panel = Mock()
    mock_panel.normal = (0.0, 1.0, 0.0)  # Side-mounted
    mock_panel.conversion_efficiency = 0.3
    mock_panel.max_power = 800.0
    solar_panel.panels = [mock_panel]
    solar_panel.conversion_efficiency = 0.3
    return solar_panel


@pytest.fixture
def mock_solar_panel_multiple():
    solar_panel = Mock()
    mock_panel1 = Mock()
    mock_panel1.normal = (0.0, 1.0, 0.0)  # Side-mounted
    mock_panel1.conversion_efficiency = 0.3
    mock_panel1.max_power = 800.0
    mock_panel2 = Mock()
    mock_panel2.normal = (0.0, 0.0, -1.0)  # Body-mounted
    mock_panel2.conversion_efficiency = 0.3
    mock_panel2.max_power = 600.0
    solar_panel.panels = [mock_panel1, mock_panel2]
    solar_panel.conversion_efficiency = 0.3
    return solar_panel


@pytest.fixture
def mock_solar_panel_canted():
    solar_panel = Mock()
    mock_panel = Mock()
    mock_panel.normal = (0.1, 0.866, -0.5)  # Canted normal vector
    mock_panel.conversion_efficiency = 0.3
    mock_panel.max_power = 800.0
    solar_panel.panels = [mock_panel]
    solar_panel.conversion_efficiency = 0.3
    return solar_panel


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
