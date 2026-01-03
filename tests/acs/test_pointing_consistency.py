"""Pointing consistency tests for ACS."""

import pytest

from conops import Slew


def test_science_holds_pointing_without_active_slew(acs, mock_config):
    """SCIENCE should hold RA/Dec when no slew is active."""
    acs.ra = 10.0
    acs.dec = 20.0
    acs.in_safe_mode = False
    acs.current_pass = None
    acs.last_slew = None

    acs._calculate_pointing(utime=10.0)

    assert acs.ra == pytest.approx(10.0)
    assert acs.dec == pytest.approx(20.0)


def test_pointing_updates_during_active_slew(acs, mock_config):
    """RA/Dec should update when a slew is actively in progress."""
    acs.ra = 10.0
    acs.dec = 20.0
    acs.in_safe_mode = False
    acs.current_pass = None

    slew = Slew(config=mock_config)
    slew.startra = 10.0
    slew.startdec = 20.0
    slew.endra = 30.0
    slew.enddec = 40.0
    slew.slewstart = 0.0
    slew.slewtime = 100.0
    slew.slewend = 100.0
    slew.slewdist = 10.0
    slew.slewpath = ([10.0, 30.0], [20.0, 40.0])

    acs.last_slew = slew

    acs._calculate_pointing(utime=50.0)

    assert acs.ra != pytest.approx(10.0) or acs.dec != pytest.approx(20.0)
