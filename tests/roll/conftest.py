"""Test fixtures for roll subsystem tests."""

from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def mock_ephem():
    ephem = Mock()
    ephem.index = Mock(return_value=0)
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
    mock_panel.sidemount = True
    mock_panel.cant_x = None
    mock_panel.cant_y = None
    mock_panel.conversion_efficiency = 0.3
    mock_panel.max_power = 800.0
    mock_panel.azimuth_deg = 0.0
    solar_panel._effective_panels = Mock(return_value=[mock_panel])
    solar_panel.conversion_efficiency = 0.3
    return solar_panel


@pytest.fixture
def mock_solar_panel_multiple():
    solar_panel = Mock()
    mock_panel1 = Mock()
    mock_panel1.sidemount = True
    mock_panel1.cant_x = 0.0
    mock_panel1.cant_y = 0.0
    mock_panel1.conversion_efficiency = 0.3
    mock_panel1.max_power = 800.0
    mock_panel1.azimuth_deg = 0.0
    mock_panel2 = Mock()
    mock_panel2.sidemount = False
    mock_panel2.cant_x = 0.0
    mock_panel2.cant_y = 0.0
    mock_panel2.conversion_efficiency = 0.3
    mock_panel2.max_power = 600.0
    mock_panel2.azimuth_deg = 90.0
    solar_panel._effective_panels = Mock(return_value=[mock_panel1, mock_panel2])
    solar_panel.conversion_efficiency = 0.3
    return solar_panel


@pytest.fixture
def mock_solar_panel_canted():
    solar_panel = Mock()
    mock_panel = Mock()
    mock_panel.sidemount = True
    mock_panel.cant_x = 10.0
    mock_panel.cant_y = 5.0
    mock_panel.conversion_efficiency = 0.3
    mock_panel.max_power = 800.0
    mock_panel.azimuth_deg = 45.0
    solar_panel._effective_panels = Mock(return_value=[mock_panel])
    solar_panel.conversion_efficiency = 0.3
    return solar_panel


@pytest.fixture
def mock_ephem_sidemount():
    ephem = Mock()
    ephem.index = Mock(return_value=0)
    ephem.sunvec = [np.array([1000, 500, 800])]
    return ephem
