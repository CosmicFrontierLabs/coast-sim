"""Test fixtures for slew subsystem tests."""

from unittest.mock import Mock

import numpy as np
import pytest

from conops import Slew


@pytest.fixture
def ephem():
    return Mock()


@pytest.fixture
def constraint(ephem):
    constraint = Mock()
    constraint.ephem = ephem
    return constraint


@pytest.fixture
def acs_config():
    acs = Mock()
    # Slew.__init__ now calls get_accel_cap() and max_slew_rate
    acs.get_accel_cap = Mock(return_value=0.5)
    acs.max_slew_rate = 0.25
    return acs


@pytest.fixture
def mock_config(constraint, acs_config):
    config = Mock()
    config.constraint = constraint
    config.spacecraft_bus = Mock()
    config.spacecraft_bus.attitude_control = acs_config
    return config


@pytest.fixture
def slew(mock_config):
    return Slew(config=mock_config)


@pytest.fixture
def slew_with_positions(slew):
    slew.startra = 45.0
    slew.startdec = 30.0
    slew.endra = 90.0
    slew.enddec = 60.0
    slew.slewstart = 1700000000.0
    return slew


@pytest.fixture
def slew_slewing(slew):
    slew.slewstart = 1700000000.0
    slew.slewend = 1700000100.0
    return slew


@pytest.fixture
def slew_ra_dec(slew):
    slew.startra = 45.0
    slew.startdec = 30.0
    slew.endra = 90.0
    slew.enddec = 60.0
    slew.slewstart = 1700000000.0
    slew.slewend = 1700000100.0
    slew.slewpath = (np.array([45.0, 90.0]), np.array([30.0, 60.0]))
    slew.slewsecs = np.array([0.0, 100.0])
    return slew


@pytest.fixture
def slew_setup(slew, acs_config):
    acs_config.s_of_t = Mock(return_value=0.0)
    acs_config.motion_time = Mock(return_value=100.0)
    slew.startra = 45.0
    slew.startdec = 30.0
    slew.endra = 90.0
    slew.enddec = 60.0
    slew.slewstart = 1700000000.0
    slew.slewend = 1700000100.0
    slew.slewpath = (np.array([45.0, 90.0]), np.array([30.0, 60.0]))
    slew.slewdist = 50.0
    return slew


@pytest.fixture
def slew_interpolation(slew, acs_config):
    acs_config.s_of_t = Mock(return_value=25.0)
    acs_config.motion_time = Mock(return_value=100.0)
    slew.startra = 45.0
    slew.startdec = 30.0
    slew.endra = 90.0
    slew.enddec = 60.0
    slew.slewstart = 1700000000.0
    slew.slewend = 1700000100.0
    slew.slewpath = (np.array([45.0, 90.0]), np.array([30.0, 60.0]))
    slew.slewdist = 50.0
    return slew


@pytest.fixture
def slew_modern_path(slew, acs_config):
    acs_config.s_of_t = Mock(return_value=0.0)
    acs_config.motion_time = Mock(return_value=100.0)
    slew.startra = 45.0
    slew.startdec = 30.0
    slew.endra = 90.0
    slew.enddec = 60.0
    slew.slewstart = 1700000000.0
    slew.slewend = 1700000100.0
    slew.slewpath = (np.array([67.5]), np.array([45.0]))
    slew.slewdist = 50.0
    return slew


@pytest.fixture
def slew_acs(slew, acs_config):
    acs_config.motion_time = Mock(return_value=100.0)
    acs_config.s_of_t = Mock(return_value=50.0)
    slew.startra = 0.0
    slew.startdec = 0.0
    slew.endra = 10.0
    slew.enddec = 10.0
    slew.slewstart = 1700000000.0
    slew.slewend = 1700000100.0
    slew.slewdist = 14.142
    slew.slewpath = (np.linspace(0.0, 10.0, 20), np.linspace(0.0, 10.0, 20))
    return slew


@pytest.fixture
def slew_interp_start(slew):
    slew.startra = 0.0
    slew.startdec = 0.0
    slew.slewstart = 1700000000.0
    slew.slewpath = (np.array([0.0, 10.0]), np.array([0.0, 10.0]))
    slew.slewsecs = np.array([0.0, 100.0])
    slew.slewdist = 0
    return slew


@pytest.fixture
def slew_calc_setup(slew, acs_config):
    acs_config.predict_slew = Mock(
        return_value=(10.0, (np.array([0.0, 10.0]), np.array([0.0, 10.0])))
    )
    acs_config.slew_time = Mock(return_value=50.0)
    slew.startra = 0.0
    slew.startdec = 0.0
    slew.endra = 10.0
    slew.enddec = 10.0
    slew.slewstart = 1700000000.0
    return slew


@pytest.fixture
def slew_calc_setup_alt(slew, acs_config):
    acs_config.predict_slew = Mock(
        return_value=(5.0, (np.array([0.0, 5.0]), np.array([0.0, 5.0])))
    )
    acs_config.slew_time = Mock(return_value=30.0)
    slew.startra = 0.0
    slew.startdec = 0.0
    slew.endra = 5.0
    slew.enddec = 5.0
    slew.slewstart = 1700000000.0
    return slew


@pytest.fixture
def slew_predict_setup(slew, acs_config):
    ra_path = np.linspace(45.0, 90.0, 100)
    dec_path = np.linspace(30.0, 60.0, 100)
    acs_config.predict_slew = Mock(return_value=(14.142, (ra_path, dec_path)))
    slew.startra = 45.0
    slew.startdec = 30.0
    slew.endra = 90.0
    slew.enddec = 60.0
    return slew, ra_path, dec_path
