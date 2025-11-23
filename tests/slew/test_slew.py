"""Tests for conops.slew module."""

from unittest.mock import Mock

import numpy as np
import pytest

from conops.slew import Slew


class TestSlewInit:
    """Test Slew initialization."""

    def test_slew_init_constraint_set(self, slew, constraint):
        assert slew.constraint == constraint

    def test_slew_init_ephem_set(self, slew, ephem):
        assert slew.ephem == ephem

    def test_slew_init_acs_config_set(self, slew, acs_config):
        assert slew.acs_config == acs_config

    def test_slew_init_slewtime_zero(self, slew):
        assert slew.slewtime == 0

    def test_slew_init_slewdist_zero(self, slew):
        assert slew.slewdist == 0

    def test_slew_init_slewstart_zero(self, slew):
        assert slew.slewstart == 0

    def test_slew_init_slewend_zero(self, slew):
        assert slew.slewend == 0

    def test_slew_init_slewrequest_zero(self, slew):
        assert slew.slewrequest == 0

    def test_slew_init_startra_zero(self, slew):
        assert slew.startra == 0

    def test_slew_init_startdec_zero(self, slew):
        assert slew.startdec == 0

    def test_slew_init_endra_zero(self, slew):
        assert slew.endra == 0

    def test_slew_init_enddec_zero(self, slew):
        assert slew.enddec == 0

    def test_slew_init_obsid_zero(self, slew):
        assert slew.obsid == 0

    def test_slew_init_mode_zero(self, slew):
        assert slew.mode == 0

    def test_slew_init_at_false(self, slew):
        assert slew.at is False

    def test_slew_init_obstype_ppt(self, slew):
        assert slew.obstype == "PPT"

    def test_slew_init_missing_constraint(self, acs_config):
        with pytest.raises(AssertionError, match="Constraint must be set"):
            Slew(constraint=None, acs_config=acs_config)

    def test_slew_init_missing_ephemeris(self, constraint, acs_config):
        constraint.ephem = None
        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            Slew(constraint=constraint, acs_config=acs_config)

    def test_slew_init_missing_acs_config(self, constraint):
        with pytest.raises(AssertionError, match="ACS config must be set"):
            Slew(constraint=constraint, acs_config=None)


class TestSlewStr:
    """Test Slew string representation."""

    def test_slew_str_contains_slew_from(self, slew_with_positions):
        str_repr = str(slew_with_positions)
        assert "Slew from" in str_repr

    def test_slew_str_contains_45_000(self, slew_with_positions):
        str_repr = str(slew_with_positions)
        assert "45.000" in str_repr

    def test_slew_str_contains_30_000(self, slew_with_positions):
        str_repr = str(slew_with_positions)
        assert "30.000" in str_repr

    def test_slew_str_contains_90_0(self, slew_with_positions):
        str_repr = str(slew_with_positions)
        assert "90.0" in str_repr


class TestIsSlewing:
    """Test is_slewing method."""

    def test_is_slewing_at_1700000050_true(self, slew_slewing):
        assert slew_slewing.is_slewing(1700000050.0) is True

    def test_is_slewing_at_1699999999_false(self, slew_slewing):
        assert slew_slewing.is_slewing(1699999999.0) is False

    def test_is_slewing_at_1700000101_false(self, slew_slewing):
        assert slew_slewing.is_slewing(1700000101.0) is False

    def test_is_slewing_at_1700000000_true(self, slew_slewing):
        assert slew_slewing.is_slewing(1700000000.0) is True

    def test_is_slewing_at_1700000100_false(self, slew_slewing):
        assert slew_slewing.is_slewing(1700000100.0) is False


class TestRaDec:
    """Test ra_dec method."""

    def test_ra_dec_returns_ra_float(self, slew_ra_dec):
        ra, dec = slew_ra_dec.ra_dec(1700000000.0)
        assert isinstance(ra, (float, np.floating))

    def test_ra_dec_returns_dec_float(self, slew_ra_dec):
        ra, dec = slew_ra_dec.ra_dec(1700000000.0)
        assert isinstance(dec, (float, np.floating))


class TestSlewRaDec:
    """Test slew_ra_dec method."""

    def test_slew_ra_dec_legacy_returns_ra_float(self, slew, acs_config):
        acs_config.s_of_t = Mock(return_value=0.0)
        acs_config.motion_time = Mock(return_value=100.0)
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

    def test_slew_ra_dec_legacy_returns_dec_float(self, slew, acs_config):
        acs_config.s_of_t = Mock(return_value=0.0)
        acs_config.motion_time = Mock(return_value=100.0)
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
        assert isinstance(dec, (float, np.floating))

    def test_slew_ra_dec_legacy_ra_in_range(self, slew, acs_config):
        acs_config.s_of_t = Mock(return_value=0.0)
        acs_config.motion_time = Mock(return_value=100.0)
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
        assert 0 <= ra < 360

    def test_slew_ra_dec_legacy_dec_in_range(self, slew, acs_config):
        acs_config.s_of_t = Mock(return_value=0.0)
        acs_config.motion_time = Mock(return_value=100.0)
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
        assert -90 <= dec <= 90

    def test_slew_ra_dec_no_path_returns_ra_start(self, slew):
        slew.startra = 45.0
        slew.startdec = 30.0
        slew.slewstart = 1700000000.0
        ra, dec = slew.slew_ra_dec(1700000000.0)
        assert ra == 45.0

    def test_slew_ra_dec_no_path_returns_dec_start(self, slew):
        slew.startra = 45.0
        slew.startdec = 30.0
        slew.slewstart = 1700000000.0
        ra, dec = slew.slew_ra_dec(1700000000.0)
        assert dec == 30.0

    def test_slew_ra_dec_acs_returns_ra_float(self, slew, acs_config):
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
        ra, dec = slew.slew_ra_dec(1700000050.0)
        assert isinstance(ra, (float, np.floating))

    def test_slew_ra_dec_acs_returns_dec_float(self, slew, acs_config):
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
        ra, dec = slew.slew_ra_dec(1700000050.0)
        assert isinstance(dec, (float, np.floating))

    def test_slew_ra_dec_acs_ra_in_range(self, slew, acs_config):
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
        ra, dec = slew.slew_ra_dec(1700000050.0)
        assert 0 <= ra < 360

    def test_slew_ra_dec_acs_dec_in_range(self, slew, acs_config):
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
        ra, dec = slew.slew_ra_dec(1700000050.0)
        assert -90 <= dec <= 90

    def test_slew_ra_dec_interpolates_ra_at_start(self, slew):
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.slewstart = 1700000000.0
        slew.slewpath = (np.array([0.0, 10.0]), np.array([0.0, 10.0]))
        slew.slewsecs = np.array([0.0, 100.0])
        slew.slewdist = 0
        ra, dec = slew.slew_ra_dec(1700000000.0)
        assert np.isclose(ra, 0.0)

    def test_slew_ra_dec_interpolates_dec_at_start(self, slew):
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.slewstart = 1700000000.0
        slew.slewpath = (np.array([0.0, 10.0]), np.array([0.0, 10.0]))
        slew.slewsecs = np.array([0.0, 100.0])
        slew.slewdist = 0
        ra, dec = slew.slew_ra_dec(1700000000.0)
        assert np.isclose(dec, 0.0)


class TestCalcSlewtime:
    """Test calc_slewtime method."""

    def test_calc_slewtime_returns_slewtime(self, slew, acs_config):
        acs_config.predict_slew = Mock(
            return_value=(10.0, (np.array([0.0, 10.0]), np.array([0.0, 10.0])))
        )
        acs_config.slew_time = Mock(return_value=50.0)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 10.0
        slew.slewstart = 1700000000.0
        slewtime = slew.calc_slewtime()
        assert slewtime == 50.0

    def test_calc_slewtime_sets_slewend(self, slew, acs_config):
        acs_config.predict_slew = Mock(
            return_value=(10.0, (np.array([0.0, 10.0]), np.array([0.0, 10.0])))
        )
        acs_config.slew_time = Mock(return_value=50.0)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 10.0
        slew.slewstart = 1700000000.0
        slew.calc_slewtime()
        assert slew.slewend == 1700000050.0

    def test_calc_slewtime_sets_slewend_correctly(self, slew, acs_config):
        acs_config.predict_slew = Mock(
            return_value=(5.0, (np.array([0.0, 5.0]), np.array([0.0, 5.0])))
        )
        acs_config.slew_time = Mock(return_value=30.0)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 5.0
        slew.enddec = 5.0
        slew.slewstart = 1700000000.0
        slew.calc_slewtime()
        assert slew.slewend == 1700000030.0

    def test_calc_slewtime_handles_nan_distance_slewtime(self, slew, acs_config):
        acs_config.predict_slew = Mock(
            return_value=(np.nan, (np.array([0.0]), np.array([0.0])))
        )
        acs_config.slew_time = Mock(return_value=0.0)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 10.0
        slew.slewstart = 1700000000.0
        slewtime = slew.calc_slewtime()
        assert slewtime == 0.0

    def test_calc_slewtime_handles_nan_distance_slewdist(self, slew, acs_config):
        acs_config.predict_slew = Mock(
            return_value=(np.nan, (np.array([0.0]), np.array([0.0])))
        )
        acs_config.slew_time = Mock(return_value=0.0)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 10.0
        slew.slewstart = 1700000000.0
        _ = slew.calc_slewtime()
        assert slew.slewdist == 0.0

    def test_calc_slewtime_handles_negative_distance_slewtime(self, slew, acs_config):
        acs_config.predict_slew = Mock(
            return_value=(-5.0, (np.array([0.0]), np.array([0.0])))
        )
        acs_config.slew_time = Mock(return_value=0.0)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 10.0
        slew.slewstart = 1700000000.0
        slewtime = slew.calc_slewtime()
        assert slewtime == 0.0

    def test_calc_slewtime_handles_negative_distance_slewdist(self, slew, acs_config):
        acs_config.predict_slew = Mock(
            return_value=(-5.0, (np.array([0.0]), np.array([0.0])))
        )
        acs_config.slew_time = Mock(return_value=0.0)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 10.0
        slew.slewstart = 1700000000.0
        _ = slew.calc_slewtime()
        assert slew.slewdist == 0.0


class TestPredictSlew:
    """Test predict_slew method."""

    def test_predict_slew_calls_acs_predict_slew(self, slew, acs_config):
        ra_path = np.linspace(0.0, 10.0, 20)
        dec_path = np.linspace(0.0, 10.0, 20)
        acs_config.predict_slew = Mock(return_value=(10.0, (ra_path, dec_path)))
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 10.0
        slew.predict_slew()
        acs_config.predict_slew.assert_called_once_with(0.0, 0.0, 10.0, 10.0, steps=20)

    def test_predict_slew_sets_slewdist(self, slew, acs_config):
        ra_path = np.linspace(45.0, 90.0, 20)
        dec_path = np.linspace(30.0, 60.0, 20)
        acs_config.predict_slew = Mock(return_value=(14.142, (ra_path, dec_path)))
        slew.startra = 45.0
        slew.startdec = 30.0
        slew.endra = 90.0
        slew.enddec = 60.0
        slew.predict_slew()
        assert slew.slewdist == 14.142

    def test_predict_slew_sets_path_ra_length(self, slew, acs_config):
        ra_path = np.linspace(45.0, 90.0, 20)
        dec_path = np.linspace(30.0, 60.0, 20)
        acs_config.predict_slew = Mock(return_value=(14.142, (ra_path, dec_path)))
        slew.startra = 45.0
        slew.startdec = 30.0
        slew.endra = 90.0
        slew.enddec = 60.0
        slew.predict_slew()
        assert len(slew.slewpath[0]) == 20

    def test_predict_slew_sets_path_dec_length(self, slew, acs_config):
        ra_path = np.linspace(45.0, 90.0, 20)
        dec_path = np.linspace(30.0, 60.0, 20)
        acs_config.predict_slew = Mock(return_value=(14.142, (ra_path, dec_path)))
        slew.startra = 45.0
        slew.startdec = 30.0
        slew.endra = 90.0
        slew.enddec = 60.0
        slew.predict_slew()
        assert len(slew.slewpath[1]) == 20

    def test_predict_slew_sets_path_ra_values(self, slew, acs_config):
        ra_path = np.linspace(45.0, 90.0, 20)
        dec_path = np.linspace(30.0, 60.0, 20)
        acs_config.predict_slew = Mock(return_value=(14.142, (ra_path, dec_path)))
        slew.startra = 45.0
        slew.startdec = 30.0
        slew.endra = 90.0
        slew.enddec = 60.0
        slew.predict_slew()
        assert np.allclose(slew.slewpath[0], ra_path)

    def test_predict_slew_sets_path_dec_values(self, slew, acs_config):
        ra_path = np.linspace(45.0, 90.0, 20)
        dec_path = np.linspace(30.0, 60.0, 20)
        acs_config.predict_slew = Mock(return_value=(14.142, (ra_path, dec_path)))
        slew.startra = 45.0
        slew.startdec = 30.0
        slew.endra = 90.0
        slew.enddec = 60.0
        slew.predict_slew()
        assert np.allclose(slew.slewpath[1], dec_path)
