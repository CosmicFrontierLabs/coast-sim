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
    return Mock()


@pytest.fixture
def slew(constraint, acs_config):
    return Slew(constraint=constraint, acs_config=acs_config)


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
