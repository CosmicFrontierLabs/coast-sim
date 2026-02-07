"""Unit tests for telemetry.py."""

import pytest

from conops.common.enums import ACSMode
from conops.ditl.telemetry import (
    Housekeeping,
    HousekeepingList,
    PayloadData,
    Telemetry,
    _CacheInvalidatingList,
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
        assert hk.mode is None

    def test_housekeeping_creation_with_values(self) -> None:
        """Test creating Housekeeping with specific values."""
        from datetime import datetime, timezone

        expected_dt = datetime.fromtimestamp(1234567890.0, tz=timezone.utc)
        hk = Housekeeping(
            timestamp=expected_dt,
            ra=45.0,
            dec=30.0,
            roll=10.0,
            mode=ACSMode.SCIENCE,
            panel_illumination=0.8,
            power_usage=100.0,
            battery_level=0.9,
            obsid=42,
        )
        assert hk.timestamp == expected_dt
        assert hk.ra == 45.0
        assert hk.dec == 30.0
        assert hk.roll == 10.0
        assert hk.mode == ACSMode.SCIENCE
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


class TestCacheInvalidatingList:
    """Test _CacheInvalidatingList class."""

    def test_init_without_telemetry(self) -> None:
        """Test initialization without telemetry reference."""
        lst = _CacheInvalidatingList([1, 2, 3])
        assert list(lst) == [1, 2, 3]
        assert lst._telemetry is None

    def test_init_with_telemetry(self) -> None:
        """Test initialization with telemetry reference."""
        telemetry = Telemetry()
        lst = _CacheInvalidatingList([1, 2, 3], telemetry=telemetry)
        assert list(lst) == [1, 2, 3]
        assert lst._telemetry is telemetry

    def test_append_without_telemetry(self) -> None:
        """Test append without telemetry (no cache clearing)."""
        lst = _CacheInvalidatingList([1, 2])
        lst.append(3)
        assert list(lst) == [1, 2, 3]

    def test_append_with_telemetry(self) -> None:
        """Test append with telemetry (cache clearing)."""
        telemetry = Telemetry()
        telemetry._field_cache["test"] = [1, 2, 3]
        lst = _CacheInvalidatingList([1, 2], telemetry=telemetry)
        lst.append(3)
        assert telemetry._field_cache == {}  # Cache should be cleared

    def test_extend(self) -> None:
        """Test extend method."""
        telemetry = Telemetry()
        telemetry._field_cache["test"] = [1, 2, 3]
        lst = _CacheInvalidatingList([1, 2], telemetry=telemetry)
        lst.extend([3, 4])
        assert list(lst) == [1, 2, 3, 4]
        assert telemetry._field_cache == {}  # Cache should be cleared

    def test_setitem_single_index(self) -> None:
        """Test __setitem__ with single index."""
        telemetry = Telemetry()
        telemetry._field_cache["test"] = [1, 2, 3]
        lst = _CacheInvalidatingList([1, 2, 3], telemetry=telemetry)
        lst[1] = 99
        assert list(lst) == [1, 99, 3]
        assert telemetry._field_cache == {}  # Cache should be cleared

    def test_setitem_slice(self) -> None:
        """Test __setitem__ with slice."""
        telemetry = Telemetry()
        telemetry._field_cache["test"] = [1, 2, 3]
        lst = _CacheInvalidatingList([1, 2, 3, 4], telemetry=telemetry)
        lst[1:3] = [99, 100]
        assert list(lst) == [1, 99, 100, 4]
        assert telemetry._field_cache == {}  # Cache should be cleared

    def test_delitem_single_index(self) -> None:
        """Test __delitem__ with single index."""
        telemetry = Telemetry()
        telemetry._field_cache["test"] = [1, 2, 3]
        lst = _CacheInvalidatingList([1, 2, 3], telemetry=telemetry)
        del lst[1]
        assert list(lst) == [1, 3]
        assert telemetry._field_cache == {}  # Cache should be cleared

    def test_delitem_slice(self) -> None:
        """Test __delitem__ with slice."""
        telemetry = Telemetry()
        telemetry._field_cache["test"] = [1, 2, 3]
        lst = _CacheInvalidatingList([1, 2, 3, 4], telemetry=telemetry)
        del lst[1:3]
        assert list(lst) == [1, 4]
        assert telemetry._field_cache == {}  # Cache should be cleared

    def test_insert(self) -> None:
        """Test insert method."""
        telemetry = Telemetry()
        telemetry._field_cache["test"] = [1, 2, 3]
        lst = _CacheInvalidatingList([1, 2, 4], telemetry=telemetry)
        lst.insert(2, 3)
        assert list(lst) == [1, 2, 3, 4]
        assert telemetry._field_cache == {}  # Cache should be cleared

    def test_remove(self) -> None:
        """Test remove method."""
        telemetry = Telemetry()
        telemetry._field_cache["test"] = [1, 2, 3]
        lst = _CacheInvalidatingList([1, 2, 3], telemetry=telemetry)
        lst.remove(2)
        assert list(lst) == [1, 3]
        assert telemetry._field_cache == {}  # Cache should be cleared

    def test_pop(self) -> None:
        """Test pop method."""
        telemetry = Telemetry()
        telemetry._field_cache["test"] = [1, 2, 3]
        lst = _CacheInvalidatingList([1, 2, 3], telemetry=telemetry)
        result = lst.pop(1)
        assert result == 2
        assert list(lst) == [1, 3]
        assert telemetry._field_cache == {}  # Cache should be cleared

    def test_clear(self) -> None:
        """Test clear method."""
        telemetry = Telemetry()
        telemetry._field_cache["test"] = [1, 2, 3]
        lst = _CacheInvalidatingList([1, 2, 3], telemetry=telemetry)
        lst.clear()
        assert list(lst) == []
        assert telemetry._field_cache == {}  # Cache should be cleared

    def test_getattr_valid_attribute(self) -> None:
        """Test __getattr__ with valid attribute."""
        hk1 = Housekeeping(ra=45.0, dec=30.0)
        hk2 = Housekeeping(ra=90.0, dec=60.0)
        lst = _CacheInvalidatingList([hk1, hk2])
        assert lst.ra == [45.0, 90.0]
        assert lst.dec == [30.0, 60.0]

    def test_getattr_with_none_values(self) -> None:
        """Test __getattr__ with None values."""
        hk1 = Housekeeping(ra=45.0, dec=None)
        hk2 = Housekeeping(ra=None, dec=60.0)
        lst = _CacheInvalidatingList([hk1, hk2])
        assert lst.ra == [45.0, None]
        assert lst.dec == [None, 60.0]

    def test_getattr_empty_list(self) -> None:
        """Test __getattr__ with empty list."""
        lst = _CacheInvalidatingList()
        with pytest.raises(AttributeError, match="object has no attribute 'ra'"):
            _ = lst.ra

    def test_getattr_invalid_attribute(self) -> None:
        """Test __getattr__ with invalid attribute."""
        hk = Housekeeping(ra=45.0)
        lst = _CacheInvalidatingList([hk])
        with pytest.raises(AttributeError, match="object has no attribute 'invalid'"):
            _ = lst.invalid


class TestTelemetry:
    """Test Telemetry class."""

    def test_init_empty(self) -> None:
        """Test initialization with no data."""
        tm = Telemetry()
        assert isinstance(tm.housekeeping, HousekeepingList)
        assert isinstance(tm.data, _CacheInvalidatingList)
        assert len(tm.housekeeping) == 0
        assert len(tm.data) == 0
        assert tm._field_cache == {}
        assert tm._fields_cache == {}

    def test_init_with_data(self) -> None:
        """Test initialization with data."""
        hk = Housekeeping(ra=45.0)
        pd = PayloadData(timestamp=1234567890.0, data_size_gb=1.5)
        tm = Telemetry(housekeeping=[hk], data=[pd])
        assert len(tm.housekeeping) == 1
        assert len(tm.data) == 1
        assert tm.housekeeping[0].ra == 45.0
        assert tm.data[0].data_size_gb == 1.5

    def test_list_wrapping(self) -> None:
        """Test that lists are properly wrapped."""
        tm = Telemetry()
        # Initially should be wrapped
        assert isinstance(tm.housekeeping, _CacheInvalidatingList)
        assert isinstance(tm.data, _CacheInvalidatingList)

        # Setting new lists should wrap them
        tm.housekeeping = [Housekeeping(ra=45.0)]
        tm.data = [PayloadData(timestamp=1234567890.0, data_size_gb=1.5)]
        assert isinstance(tm.housekeeping, _CacheInvalidatingList)
        assert isinstance(tm.data, _CacheInvalidatingList)
        assert tm.housekeeping.ra == [45.0]
        assert tm.data.data_size_gb == [1.5]

    def test_get_housekeeping_field_caching(self) -> None:
        """Test get_housekeeping_field with caching."""
        hk1 = Housekeeping(ra=45.0, dec=30.0)
        hk2 = Housekeeping(ra=90.0, dec=60.0)
        tm = Telemetry(housekeeping=[hk1, hk2])

        # First call should cache
        result1 = tm.get_housekeeping_field("ra")
        assert result1 == [45.0, 90.0]
        assert "ra" in tm._field_cache

        # Second call should use cache
        result2 = tm.get_housekeeping_field("ra")
        assert result2 == [45.0, 90.0]
        assert result1 is result2  # Same object from cache

    def test_get_housekeeping_field_cache_invalidation(self) -> None:
        """Test that cache is invalidated when list is modified."""
        hk = Housekeeping(ra=45.0)
        tm = Telemetry(housekeeping=[hk])

        # Cache the field
        result1 = tm.get_housekeeping_field("ra")
        assert result1 == [45.0]

        # Modify the list (should clear cache)
        tm.housekeeping.append(Housekeeping(ra=90.0))

        # Cache should be cleared, new call should recompute
        result2 = tm.get_housekeeping_field("ra")
        assert result2 == [45.0, 90.0]
        assert result1 != result2  # Different objects

    def test_get_housekeeping_fields_caching(self) -> None:
        """Test get_housekeeping_fields with caching."""
        hk1 = Housekeeping(ra=45.0, dec=30.0)
        hk2 = Housekeeping(ra=90.0, dec=60.0)
        tm = Telemetry(housekeeping=[hk1, hk2])

        # First call should cache
        result1 = tm.get_housekeeping_fields(["ra", "dec"])
        expected = {"ra": [45.0, 90.0], "dec": [30.0, 60.0]}
        assert result1 == expected
        assert ("dec", "ra") in tm._fields_cache  # Sorted tuple

        # Second call should use cache
        result2 = tm.get_housekeeping_fields(["ra", "dec"])
        assert result2 == expected
        assert result1 is result2  # Same object from cache

    def test_get_housekeeping_fields_different_order(self) -> None:
        """Test get_housekeeping_fields with different field order."""
        hk1 = Housekeeping(ra=45.0, dec=30.0)
        tm = Telemetry(housekeeping=[hk1])

        result1 = tm.get_housekeeping_fields(["ra", "dec"])
        result2 = tm.get_housekeeping_fields(["dec", "ra"])
        assert result1 == result2  # Same result regardless of order

    def test_clear_cache(self) -> None:
        """Test clear_cache method."""
        hk = Housekeeping(ra=45.0)
        tm = Telemetry(housekeeping=[hk])

        # Populate caches
        tm.get_housekeeping_field("ra")
        tm.get_housekeeping_fields(["ra", "dec"])

        assert tm._field_cache
        assert tm._fields_cache

        # Clear cache
        tm.clear_cache()

        assert tm._field_cache == {}
        assert tm._fields_cache == {}

    def test_attribute_access_on_housekeeping(self) -> None:
        """Test attribute access on housekeeping list."""
        hk1 = Housekeeping(ra=45.0, dec=30.0, mode=ACSMode.SCIENCE)
        hk2 = Housekeeping(ra=90.0, dec=60.0, mode=ACSMode.SAFE)
        tm = Telemetry(housekeeping=[hk1, hk2])

        assert tm.housekeeping.ra == [45.0, 90.0]
        assert tm.housekeeping.dec == [30.0, 60.0]
        assert tm.housekeeping.mode == [ACSMode.SCIENCE, ACSMode.SAFE]

    def test_attribute_access_on_data(self) -> None:
        """Test attribute access on data list."""
        from datetime import datetime, timezone

        dt1 = datetime.fromtimestamp(1234567890.0, tz=timezone.utc)
        dt2 = datetime.fromtimestamp(1234567891.0, tz=timezone.utc)
        pd1 = PayloadData(timestamp=dt1, data_size_gb=1.5)
        pd2 = PayloadData(timestamp=dt2, data_size_gb=2.0)
        tm = Telemetry(data=[pd1, pd2])

        assert tm.data.timestamp == [dt1, dt2]
        assert tm.data.data_size_gb == [1.5, 2.0]

    def test_model_config(self) -> None:
        """Test that model config is properly set."""
        tm = Telemetry()
        assert tm.model_config["arbitrary_types_allowed"] is True
