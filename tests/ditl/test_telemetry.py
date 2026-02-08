"""Unit tests for telemetry.py."""

import pytest

from conops.common.enums import ACSMode
from conops.ditl.telemetry import (
    Housekeeping,
    HousekeepingList,
    PayloadData,
    Telemetry,
)


class TestHousekeeping:
    """Test Housekeeping class."""

    def test_housekeeping_creation_with_none_values(self) -> None:
        """Test creating Housekeeping with None values."""
        hk = Housekeeping()
        assert hk.timestamp is None
        assert hk.ra is None
        assert hk.dec is None
        assert hk.roll == 0.0  # Has default
        assert hk.acs_mode is None

    def test_housekeeping_creation_with_values(self) -> None:
        """Test creating Housekeeping with specific values."""
        from datetime import datetime, timezone

        expected_dt = datetime.fromtimestamp(1234567890.0, tz=timezone.utc)
        hk = Housekeeping(
            timestamp=expected_dt,
            ra=45.0,
            dec=30.0,
            roll=10.0,
            acs_mode=ACSMode.SCIENCE,
            panel_illumination=0.8,
            power_usage=100.0,
            battery_level=0.9,
            obsid=42,
        )
        assert hk.timestamp == expected_dt
        assert hk.ra == 45.0
        assert hk.dec == 30.0
        assert hk.roll == 10.0
        assert hk.acs_mode == ACSMode.SCIENCE
        assert hk.panel_illumination == 0.8
        assert hk.power_usage == 100.0
        assert hk.battery_level == 0.9
        assert hk.obsid == 42

    def test_extract_field_empty_list(self) -> None:
        """Test extract_field with empty list."""
        result = Housekeeping.extract_field([], "ra")
        assert result == []

    def test_extract_field_single_record(self) -> None:
        """Test extract_field with single record."""
        hk = Housekeeping(ra=45.0, dec=30.0)
        result = Housekeeping.extract_field([hk], "ra")
        assert result == [45.0]

    def test_extract_field_multiple_records(self) -> None:
        """Test extract_field with multiple records."""
        hk1 = Housekeeping(ra=45.0, dec=30.0)
        hk2 = Housekeeping(ra=90.0, dec=60.0)
        hk3 = Housekeeping(ra=None, dec=45.0)
        result = Housekeeping.extract_field([hk1, hk2, hk3], "ra")
        assert result == [45.0, 90.0, None]

    def test_extract_field_invalid_attribute(self) -> None:
        """Test extract_field with invalid attribute."""
        hk = Housekeeping(ra=45.0)
        with pytest.raises(AttributeError):
            Housekeeping.extract_field([hk], "invalid_field")

    def test_extract_fields_empty_list(self) -> None:
        """Test extract_fields with empty list."""
        result = Housekeeping.extract_fields([], ["ra", "dec"])
        assert result == {"ra": [], "dec": []}

    def test_extract_fields_multiple_records(self) -> None:
        """Test extract_fields with multiple records."""
        hk1 = Housekeeping(ra=45.0, dec=30.0, roll=10.0)
        hk2 = Housekeeping(ra=90.0, dec=60.0, roll=20.0)
        hk3 = Housekeeping(ra=None, dec=45.0, roll=None)
        result = Housekeeping.extract_fields([hk1, hk2, hk3], ["ra", "dec", "roll"])
        expected = {
            "ra": [45.0, 90.0, None],
            "dec": [30.0, 60.0, 45.0],
            "roll": [10.0, 20.0, None],
        }
        assert result == expected

    def test_extract_fields_invalid_attribute(self) -> None:
        """Test extract_fields with invalid attribute."""
        hk = Housekeeping(ra=45.0)
        with pytest.raises(AttributeError):
            Housekeeping.extract_fields([hk], ["ra", "invalid_field"])


class TestPayloadData:
    """Test PayloadData class."""

    def test_payload_data_creation(self) -> None:
        """Test creating PayloadData."""
        from datetime import datetime, timezone

        expected_dt = datetime.fromtimestamp(1234567890.0, tz=timezone.utc)
        pd = PayloadData(timestamp=expected_dt, data_size_gb=1.5)
        assert pd.timestamp == expected_dt
        assert pd.data_size_gb == 1.5


class TestTelemetry:
    """Test Telemetry class."""

    def test_init_empty(self) -> None:
        """Test initialization with no data."""
        tm = Telemetry()
        assert isinstance(tm.housekeeping, HousekeepingList)
        assert isinstance(tm.data, list)
        assert len(tm.housekeeping) == 0
        assert len(tm.data) == 0

    def test_init_with_data(self) -> None:
        """Test initialization with data."""
        hk = Housekeeping(ra=45.0)
        pd = PayloadData(timestamp=1234567890.0, data_size_gb=1.5)
        tm = Telemetry(housekeeping=HousekeepingList([hk]), data=[pd])
        assert len(tm.housekeeping) == 1
        assert len(tm.data) == 1
        assert tm.housekeeping[0].ra == 45.0
        assert tm.data[0].data_size_gb == 1.5

    def test_list_assignment(self) -> None:
        """Test that lists can be assigned."""
        tm = Telemetry()
        # Initially should be empty
        assert isinstance(tm.housekeeping, HousekeepingList)
        assert isinstance(tm.data, list)

        # Setting new lists
        tm.housekeeping = HousekeepingList([Housekeeping(ra=45.0)])
        tm.data = [PayloadData(timestamp=1234567890.0, data_size_gb=1.5)]
        assert isinstance(tm.housekeeping, HousekeepingList)
        assert isinstance(tm.data, list)
        assert tm.housekeeping.ra == [45.0]
        # data is a regular list, so no attribute access
        assert tm.data[0].data_size_gb == 1.5

    def test_attribute_access_on_housekeeping(self) -> None:
        """Test attribute access on housekeeping list."""
        hk1 = Housekeeping(
            ra=45.0,
            dec=30.0,
            acs_mode=ACSMode.SCIENCE,
            sun_angle_deg=10.0,
            in_eclipse=False,
        )
        hk2 = Housekeeping(
            ra=90.0,
            dec=60.0,
            acs_mode=ACSMode.SAFE,
            sun_angle_deg=20.0,
            in_eclipse=True,
        )
        tm = Telemetry(housekeeping=HousekeepingList([hk1, hk2]))

        assert tm.housekeeping.ra == [45.0, 90.0]
        assert tm.housekeeping.dec == [30.0, 60.0]
        assert tm.housekeeping.acs_mode == [ACSMode.SCIENCE, ACSMode.SAFE]
        assert tm.housekeeping.sun_angle_deg == [10.0, 20.0]
        assert tm.housekeeping.in_eclipse == [False, True]

    def test_attribute_access_on_data(self) -> None:
        """Test attribute access on data list."""
        from datetime import datetime, timezone

        dt1 = datetime.fromtimestamp(1234567890.0, tz=timezone.utc)
        dt2 = datetime.fromtimestamp(1234567891.0, tz=timezone.utc)
        pd1 = PayloadData(timestamp=dt1, data_size_gb=1.5)
        pd2 = PayloadData(timestamp=dt2, data_size_gb=2.0)
        tm = Telemetry(data=[pd1, pd2])

        # data is a regular list, so access attributes directly
        assert [pd.timestamp for pd in tm.data] == [dt1, dt2]
        assert [pd.data_size_gb for pd in tm.data] == [1.5, 2.0]

    def test_model_config(self) -> None:
        """Test that model config is properly set."""
        tm = Telemetry()
        assert tm.model_config["arbitrary_types_allowed"] is True
