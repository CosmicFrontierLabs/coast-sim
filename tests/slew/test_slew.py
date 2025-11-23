"""Tests for conops.slew module."""

from unittest.mock import Mock

import numpy as np
import pytest

from conops.slew import Slew


class TestSlewInit:
    """Test Slew initialization."""

    def test_slew_init_with_constraint_and_acs(self):
        """Test Slew initialization with constraint and ACS config."""
        # Mock ephemeris
        ephem = Mock()

        # Mock constraint
        constraint = Mock()
        constraint.ephem = ephem

        # Mock ACS config
        acs_config = Mock()

        slew = Slew(constraint=constraint, acs_config=acs_config)

        assert slew.constraint == constraint
        assert slew.ephem == ephem
        assert slew.acs_config == acs_config
        assert slew.slewtime == 0
        assert slew.slewdist == 0
        assert slew.slewstart == 0
        assert slew.slewend == 0

    def test_slew_init_missing_constraint(self):
        """Test Slew initialization without constraint raises error."""
        acs_config = Mock()
        with pytest.raises(AssertionError, match="Constraint must be set"):
            Slew(constraint=None, acs_config=acs_config)

    def test_slew_init_missing_ephemeris(self):
        """Test Slew initialization when constraint has no ephemeris."""
        constraint = Mock()
        constraint.ephem = None
        acs_config = Mock()

        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            Slew(constraint=constraint, acs_config=acs_config)

    def test_slew_init_missing_acs_config(self):
        """Test Slew initialization without ACS config raises error."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem

        with pytest.raises(AssertionError, match="ACS config must be set"):
            Slew(constraint=constraint, acs_config=None)

    def test_slew_init_default_values(self):
        """Test Slew initialization sets correct default values."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()

        slew = Slew(constraint=constraint, acs_config=acs_config)

        assert slew.slewrequest == 0
        assert slew.startra == 0
        assert slew.startdec == 0
        assert slew.endra == 0
        assert slew.enddec == 0
        assert slew.obstype == "PPT"
        assert slew.obsid == 0
        assert slew.mode == 0
        assert slew.at is False


class TestSlewStr:
    """Test Slew string representation."""

    def test_slew_str_representation(self):
        """Test Slew __str__ method."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.startra = 45.0
        slew.startdec = 30.0
        slew.endra = 90.0
        slew.enddec = 60.0
        slew.slewstart = 1700000000.0

        str_repr = str(slew)

        assert "Slew from" in str_repr
        assert "45.000" in str_repr
        assert "30.000" in str_repr
        assert "90.0" in str_repr


class TestIsSlewing:
    """Test is_slewing method."""

    def test_is_slewing_during_slew(self):
        """Test is_slewing returns True during slew."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.slewstart = 1700000000.0
        slew.slewend = 1700000100.0

        # Time during slew
        assert slew.is_slewing(1700000050.0) is True

    def test_is_slewing_before_slew(self):
        """Test is_slewing returns False before slew."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.slewstart = 1700000000.0
        slew.slewend = 1700000100.0

        assert slew.is_slewing(1699999999.0) is False

    def test_is_slewing_after_slew(self):
        """Test is_slewing returns False after slew."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.slewstart = 1700000000.0
        slew.slewend = 1700000100.0

        assert slew.is_slewing(1700000101.0) is False

    def test_is_slewing_at_start_boundary(self):
        """Test is_slewing at exact start time."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.slewstart = 1700000000.0
        slew.slewend = 1700000100.0

        assert slew.is_slewing(1700000000.0) is True

    def test_is_slewing_at_end_boundary(self):
        """Test is_slewing at exact end time."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.slewstart = 1700000000.0
        slew.slewend = 1700000100.0

        assert slew.is_slewing(1700000100.0) is False


class TestRaDec:
    """Test ra_dec method."""

    def test_ra_dec_calls_slew_ra_dec(self):
        """Test ra_dec delegates to slew_ra_dec."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.startra = 45.0
        slew.startdec = 30.0
        slew.endra = 90.0
        slew.enddec = 60.0
        slew.slewstart = 1700000000.0
        slew.slewend = 1700000100.0
        slew.slewpath = (np.array([45.0, 90.0]), np.array([30.0, 60.0]))
        slew.slewsecs = np.array([0.0, 100.0])

        ra, dec = slew.ra_dec(1700000000.0)

        assert isinstance(ra, (float, np.floating))
        assert isinstance(dec, (float, np.floating))


class TestSlewRaDec:
    """Test slew_ra_dec method."""

    def test_slew_ra_dec_legacy_with_path(self):
        """Test slew_ra_dec with legacy linear interpolation."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()
        acs_config.s_of_t = Mock(return_value=0.0)
        acs_config.motion_time = Mock(return_value=100.0)

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.startra = 45.0
        slew.startdec = 30.0
        slew.endra = 90.0
        slew.enddec = 60.0
        slew.slewstart = 1700000000.0
        slew.slewend = 1700000100.0
        slew.slewpath = (np.array([45.0, 90.0]), np.array([30.0, 60.0]))
        slew.slewsecs = np.array([0.0, 100.0])
        slew.slewdist = 0

        ra, dec = slew.slew_ra_dec(1700000000.0)

        assert isinstance(ra, (float, np.floating))
        assert isinstance(dec, (float, np.floating))
        assert 0 <= ra < 360
        assert -90 <= dec <= 90

    def test_slew_ra_dec_no_path_returns_start(self):
        """Test slew_ra_dec returns start position when no path exists."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.startra = 45.0
        slew.startdec = 30.0
        slew.slewstart = 1700000000.0

        ra, dec = slew.slew_ra_dec(1700000000.0)

        assert ra == 45.0
        assert dec == 30.0

    def test_slew_ra_dec_acs_bang_bang_profile(self):
        """Test slew_ra_dec with ACS bang-bang control profile."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()
        acs_config.motion_time = Mock(return_value=100.0)
        acs_config.s_of_t = Mock(return_value=50.0)  # Halfway through slew

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 10.0
        slew.slewstart = 1700000000.0
        slew.slewend = 1700000100.0
        slew.slewdist = 14.142  # ~sqrt(100+100) degrees
        slew.slewpath = (
            np.linspace(0.0, 10.0, 20),
            np.linspace(0.0, 10.0, 20),
        )

        ra, dec = slew.slew_ra_dec(1700000050.0)

        assert isinstance(ra, (float, np.floating))
        assert isinstance(dec, (float, np.floating))
        assert 0 <= ra < 360
        assert -90 <= dec <= 90

    def test_slew_ra_dec_interpolates_correctly(self):
        """Test slew_ra_dec interpolates between start and end."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.slewstart = 1700000000.0
        slew.slewpath = (np.array([0.0, 10.0]), np.array([0.0, 10.0]))
        slew.slewsecs = np.array([0.0, 100.0])
        slew.slewdist = 0

        # At start
        ra, dec = slew.slew_ra_dec(1700000000.0)
        assert np.isclose(ra, 0.0)
        assert np.isclose(dec, 0.0)


class TestCalcSlewtime:
    """Test calc_slewtime method."""

    def test_calc_slewtime_basic(self):
        """Test calc_slewtime calculates slew time."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()
        acs_config.predict_slew = Mock(
            return_value=(10.0, (np.array([0.0, 10.0]), np.array([0.0, 10.0])))
        )
        acs_config.slew_time = Mock(return_value=50.0)

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 10.0
        slew.slewstart = 1700000000.0

        slewtime = slew.calc_slewtime()

        assert slewtime == 50.0
        assert slew.slewend == 1700000050.0

    def test_calc_slewtime_sets_slewend(self):
        """Test calc_slewtime correctly sets slewend."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()
        acs_config.predict_slew = Mock(
            return_value=(5.0, (np.array([0.0, 5.0]), np.array([0.0, 5.0])))
        )
        acs_config.slew_time = Mock(return_value=30.0)

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 5.0
        slew.enddec = 5.0
        slew.slewstart = 1700000000.0

        slew.calc_slewtime()

        assert slew.slewend == 1700000030.0

    def test_calc_slewtime_handles_nan_distance(self):
        """Test calc_slewtime handles NaN distance."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()
        acs_config.predict_slew = Mock(
            return_value=(np.nan, (np.array([0.0]), np.array([0.0])))
        )
        acs_config.slew_time = Mock(return_value=0.0)

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 10.0
        slew.slewstart = 1700000000.0

        slewtime = slew.calc_slewtime()

        assert slewtime == 0.0
        assert slew.slewdist == 0.0

    def test_calc_slewtime_handles_negative_distance(self):
        """Test calc_slewtime handles negative distance."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()
        acs_config.predict_slew = Mock(
            return_value=(-5.0, (np.array([0.0]), np.array([0.0])))
        )
        acs_config.slew_time = Mock(return_value=0.0)

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 10.0
        slew.slewstart = 1700000000.0

        slewtime = slew.calc_slewtime()

        assert slewtime == 0.0
        assert slew.slewdist == 0.0


class TestPredictSlew:
    """Test predict_slew method."""

    def test_predict_slew_calls_acs_predict_slew(self):
        """Test predict_slew calls ACS predict_slew method."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()
        ra_path = np.linspace(0.0, 10.0, 20)
        dec_path = np.linspace(0.0, 10.0, 20)
        acs_config.predict_slew = Mock(return_value=(10.0, (ra_path, dec_path)))

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 10.0

        slew.predict_slew()

        acs_config.predict_slew.assert_called_once_with(0.0, 0.0, 10.0, 10.0, steps=20)
        assert slew.slewdist == 10.0
        assert len(slew.slewpath[0]) == 20
        assert len(slew.slewpath[1]) == 20

    def test_predict_slew_sets_slewdist_and_slewpath(self):
        """Test predict_slew sets slewdist and slewpath."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()
        ra_path = np.linspace(45.0, 90.0, 20)
        dec_path = np.linspace(30.0, 60.0, 20)
        acs_config.predict_slew = Mock(return_value=(14.142, (ra_path, dec_path)))

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.startra = 45.0
        slew.startdec = 30.0
        slew.endra = 90.0
        slew.enddec = 60.0

        slew.predict_slew()

        assert np.isclose(slew.slewdist, 14.142)
        assert np.allclose(slew.slewpath[0], ra_path)
        assert np.allclose(slew.slewpath[1], dec_path)

    def test_predict_slew_uses_20_steps(self):
        """Test predict_slew uses 20 steps for path."""
        ephem = Mock()
        constraint = Mock()
        constraint.ephem = ephem
        acs_config = Mock()
        acs_config.predict_slew = Mock(
            return_value=(5.0, (np.linspace(0, 5, 20), np.linspace(0, 5, 20)))
        )

        slew = Slew(constraint=constraint, acs_config=acs_config)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 5.0
        slew.enddec = 5.0

        slew.predict_slew()

        # Verify predict_slew was called with steps=20
        call_args = acs_config.predict_slew.call_args
        assert call_args[1]["steps"] == 20
