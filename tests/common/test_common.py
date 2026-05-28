"""Tests for conops.common module."""

import numpy as np
import pytest

from conops import (
    ACSMode,
    givename,
    ics_date_conv,
    unixtime2date,
    unixtime2yearday,
)
from conops.common import (
    attitude_for_body_vector_tracking,
    attitude_from_body_axes,
    body_vector_to_eci,
    scbodyvector,
)


class TestACSMode:
    """Test ACS Mode enum."""

    def test_acs_mode_science(self):
        """Test SCIENCE mode value."""
        assert ACSMode.SCIENCE == 0

    def test_acs_mode_slewing(self):
        """Test SLEWING mode value."""
        assert ACSMode.SLEWING == 1

    def test_acs_mode_saa(self):
        """Test SAA mode value."""
        assert ACSMode.SAA == 2

    def test_acs_mode_pass(self):
        """Test PASS mode value."""
        assert ACSMode.PASS == 3

    def test_acs_mode_charging(self):
        """Test CHARGING mode value."""
        assert ACSMode.CHARGING == 4

    def test_acs_mode_is_int(self):
        """Test that ACS mode values are integers."""
        assert isinstance(ACSMode.SCIENCE, int)


class TestGivename:
    """Test givename coordinate naming function."""

    def test_givename_positive_dec(self):
        """Test givename with positive declination."""
        # RA = 45 degrees = 3 hours, Dec = 30 degrees
        name = givename(45.0, 30.0)
        assert "J" in name
        assert "03" in name  # 45/15 = 3 hours
        assert "30" in name  # Dec degrees

    def test_givename_negative_dec(self):
        """Test givename with negative declination."""
        # RA = 60 degrees = 4 hours, Dec = -45 degrees
        name = givename(60.0, -45.0)
        assert "J" in name
        assert "-" in name  # Negative indicator

    def test_givename_with_stem(self):
        """Test givename with stem prefix."""
        name = givename(45.0, 30.0, stem="TEST")
        assert "TEST" in name

    def test_givename_zero_ra_dec(self):
        """Test givename with zero RA and Dec."""
        name = givename(0.0, 0.0)
        assert isinstance(name, str)
        assert len(name) > 0


class TestUnixtime2date:
    """Test Unix timestamp to date conversion."""

    def test_unixtime2date_known_time(self):
        """Test conversion of known Unix timestamp."""
        # Unix timestamp for 2023-01-01 00:00:00 UTC
        utime = 1672531200.0
        date_str = unixtime2date(utime)
        assert isinstance(date_str, str)
        assert "2023" in date_str

    def test_unixtime2date_format(self):
        """Test that date format is correct."""
        utime = 1700000000.0
        date_str = unixtime2date(utime)
        # Format should be YYYY-DDD-HH:MM:SS
        parts = date_str.split("-")
        assert len(parts) >= 2
        assert len(parts[0]) == 4  # Year is 4 digits


class TestIcsDateConv:
    """Test ICS date conversion function."""

    def test_ics_date_conv_valid_date(self):
        """Test ICS date conversion with valid date."""
        # Format: "YYYY/DDD-HH:MM:SS"
        ics_date = "2023/001-00:00:00"
        unix_time = ics_date_conv(ics_date)
        assert isinstance(unix_time, (int, float))

    def test_ics_date_conv_another_date(self):
        """Test ICS date conversion with another date."""
        ics_date = "2023/100-12:30:45"
        unix_time = ics_date_conv(ics_date)
        assert isinstance(unix_time, (int, float))
        assert unix_time > 0


class TestUnixtimeToYearday:
    """Test Unix timestamp to year and day of year conversion."""

    def test_unixtime2yearday_known_time(self):
        """Test conversion of known Unix timestamp."""
        # Unix timestamp for 2023-01-01 00:00:00 UTC
        utime = 1672531200.0
        year, day = unixtime2yearday(utime)
        assert year == 2023
        assert day == 1

    def test_unixtime2yearday_mid_year(self):
        """Test conversion for mid-year date."""
        # Unix timestamp for 2023-07-02 (approximately day 183)
        utime = 1688169600.0
        year, day = unixtime2yearday(utime)
        assert year == 2023
        assert day > 100  # Mid-year


class TestBodyVectorTrackingAttitude:
    """Test attitude solutions for tracking arbitrary body vectors."""

    @pytest.mark.parametrize(
        "body_vector",
        [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ],
    )
    def test_body_vector_tracks_target(self, body_vector):
        target = np.array([0.3, -0.4, 0.8660254])
        target = target / np.linalg.norm(target)

        attitude = attitude_for_body_vector_tracking(body_vector, target)
        assert attitude is not None
        ra, dec, roll = attitude
        tracked = body_vector_to_eci(ra, dec, roll, body_vector)
        assert tracked is not None
        target_in_body = scbodyvector(
            np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(roll), target
        )

        assert np.dot(tracked, target) == pytest.approx(1.0, abs=1e-9)
        assert target_in_body == pytest.approx(body_vector, abs=1e-9)

    @pytest.mark.parametrize(
        "attitude",
        [
            (0.0, 0.0, 30.0),
            (90.0, 45.0, 30.0),
            (120.0, -20.0, 275.0),
        ],
    )
    @pytest.mark.parametrize(
        "body_vector",
        [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        ],
    )
    def test_body_vector_to_eci_matches_spacecraft_body_transform(
        self, attitude, body_vector
    ):
        ra, dec, roll = attitude

        inertial_vector = body_vector_to_eci(ra, dec, roll, body_vector)
        assert inertial_vector is not None
        recovered_body_vector = scbodyvector(
            np.deg2rad(ra),
            np.deg2rad(dec),
            np.deg2rad(roll),
            inertial_vector,
        )

        assert recovered_body_vector == pytest.approx(body_vector, abs=1e-9)

    def test_legacy_minus_x_tracking_keeps_zero_roll(self):
        target = np.array([0.3, -0.4, 0.8660254])
        target = target / np.linalg.norm(target)

        attitude = attitude_for_body_vector_tracking((-1.0, 0.0, 0.0), target)
        assert attitude is not None
        ra, dec, roll = attitude
        minus_x = body_vector_to_eci(ra, dec, roll, (-1.0, 0.0, 0.0))
        plus_x = body_vector_to_eci(ra, dec, roll, (1.0, 0.0, 0.0))
        assert minus_x is not None
        assert plus_x is not None

        assert np.dot(minus_x, target) == pytest.approx(1.0, abs=1e-9)
        assert np.dot(plus_x, -target) == pytest.approx(1.0, abs=1e-9)
        assert roll == pytest.approx(0.0, abs=1e-9)

    def test_tracking_skips_zero_reference_vectors(self):
        target = np.array([0.3, -0.4, 0.8660254])
        target = target / np.linalg.norm(target)

        attitude = attitude_for_body_vector_tracking(
            (0.0, 1.0, 0.0),
            target,
            reference_body=(0.0, 0.0, 0.0),
            reference_eci=(0.0, 0.0, 0.0),
        )

        assert attitude is not None
        ra, dec, roll = attitude
        tracked = body_vector_to_eci(ra, dec, roll, (0.0, 1.0, 0.0))
        assert tracked is not None
        assert np.dot(tracked, target) == pytest.approx(1.0, abs=1e-9)

    def test_attitude_from_body_axes_handles_parallel_z_reference(self):
        attitude = attitude_from_body_axes((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))

        assert attitude is not None
        ra, dec, roll = attitude
        body_x = body_vector_to_eci(ra, dec, roll, (1.0, 0.0, 0.0))
        assert body_x is not None
        assert body_x == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)

    def test_degenerate_tracking_vectors_return_none(self):
        target = np.array([0.3, -0.4, 0.8660254])
        target = target / np.linalg.norm(target)

        assert attitude_for_body_vector_tracking((0.0, 0.0, 0.0), target) is None
        assert (
            attitude_for_body_vector_tracking((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)) is None
        )
        assert body_vector_to_eci(0.0, 0.0, 0.0, (0.0, 0.0, 0.0)) is None
