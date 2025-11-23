"""Tests for conops.common module."""

from conops import (
    ACSMode,
    givename,
    ics_date_conv,
    unixtime2date,
    unixtime2yearday,
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
