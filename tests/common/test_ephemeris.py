"""Position snapshots avoid full ephemeris copies without changing values."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import rust_ephem

from conops.common.ephemeris import _position_snapshot, position_vectors


@pytest.fixture
def ephem():
    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return rust_ephem.GroundEphemeris(
        latitude=34.0,
        longitude=-118.0,
        height=100.0,
        begin=begin,
        end=begin + timedelta(minutes=2),
        step_size=60,
    )


@pytest.mark.parametrize("body", ["gcrs", "sun"])
def test_snapshot_reuses_read_only_positions(ephem, body):
    _position_snapshot.cache_clear()
    expected = getattr(ephem, f"{body}_pv").position
    snapshot = position_vectors(ephem, body)
    np.testing.assert_array_equal(snapshot, expected)
    assert position_vectors(ephem, body) is snapshot
    assert _position_snapshot.cache_info().misses == 1
    assert _position_snapshot.cache_info().hits == 1
    with pytest.raises(ValueError, match="read-only"):
        snapshot[0, 0] = 0
    # Public getters still yield independent, writable copies.
    expected[0, 0] = float("nan")
    assert np.isfinite(snapshot).all()
    assert np.isfinite(getattr(ephem, f"{body}_pv").position).all()


def test_snapshot_cache_is_bounded_and_separates_ephemerides(ephem):
    _position_snapshot.cache_clear()
    first = position_vectors(ephem, "gcrs")
    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for latitude in range(10):
        other = rust_ephem.GroundEphemeris(
            latitude=latitude,
            longitude=0.0,
            height=0.0,
            begin=begin,
            end=begin + timedelta(minutes=1),
            step_size=60,
        )
        assert position_vectors(other, "gcrs") is not first
    assert _position_snapshot.cache_info().currsize == 8


@pytest.mark.parametrize("mock", [False, True])
def test_mutable_adapters_are_not_cached(mock):
    ephem = Mock(spec=rust_ephem.Ephemeris) if mock else SimpleNamespace()
    ephem.gcrs_pv = SimpleNamespace(position=np.zeros((1, 3)))
    assert position_vectors(ephem, "gcrs") is ephem.gcrs_pv.position
    ephem.gcrs_pv.position = np.ones((1, 3))
    np.testing.assert_array_equal(position_vectors(ephem, "gcrs"), np.ones((1, 3)))
    assert ephem.gcrs_pv.position.flags.writeable
