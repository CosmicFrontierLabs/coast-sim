"""Position snapshots avoid full ephemeris copies without changing values."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import rust_ephem

from conops.common.ephemeris import _position_snapshot, orbit_at, position_vectors
from conops.simulation.momentum import OrbitTables


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


@pytest.mark.parametrize("numpy_dates", [False, True])
def test_orbit_tables_accept_adapter_epochs_without_optional_vectors(
    numpy_dates: bool,
) -> None:
    begin = datetime(2027, 1, 1, tzinfo=timezone.utc)
    dates = [begin, begin + timedelta(seconds=1, microseconds=125000)]
    timestamps = (
        np.array([date.replace(tzinfo=None) for date in dates], dtype="datetime64[us]")
        if numpy_dates
        else dates
    )
    positions = np.array([[7000.0, 0.0, 0.0], [7000.0, 7.0, 0.0]])
    ephem = SimpleNamespace(
        timestamp=timestamps, gcrs_pv=SimpleNamespace(position=positions)
    )

    orbit = OrbitTables.from_ephem(ephem)

    np.testing.assert_array_equal(orbit.times, [date.timestamp() for date in dates])
    np.testing.assert_array_equal(orbit.positions_eci_km, positions)
    assert orbit.velocities_eci_km_s is None
    assert orbit.positions_ecef_km is None
    assert orbit.sun_positions_eci_km is None


ORBIT_BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)


def _orbit(step_size: int) -> rust_ephem.TLEEphemeris:
    return rust_ephem.TLEEphemeris(
        tle="examples/example.tle",
        begin=ORBIT_BEGIN,
        end=ORBIT_BEGIN + timedelta(minutes=30),
        step_size=step_size,
    )


class TestOrbitAt:
    def test_a_sample_time_gives_the_stored_orbit(self):
        ephem = _orbit(60)
        orbit = orbit_at(ephem, ephem.timestamp[7].timestamp())
        np.testing.assert_array_equal(orbit.position_eci_km, ephem.gcrs_pv.position[7])
        np.testing.assert_array_equal(
            orbit.velocity_eci_km_s, ephem.gcrs_pv.velocity[7]
        )
        np.testing.assert_array_equal(
            orbit.sun_position_eci_km, ephem.sun_pv.position[7]
        )
        np.testing.assert_array_equal(orbit.position_ecef_km, ephem.itrs_pv.position[7])

    def test_between_samples_it_follows_the_orbit(self):
        """Midway between 60 s samples, against the orbit propagated there:
        within a metre, where the chord between the samples cuts kilometres
        inside it."""
        coarse, fine = _orbit(60), _orbit(30)
        truth = fine.gcrs_pv.position[11]
        orbit = orbit_at(coarse, fine.timestamp[11].timestamp())
        assert np.linalg.norm(orbit.position_eci_km - truth) < 1e-3
        velocity_error = orbit.velocity_eci_km_s - fine.gcrs_pv.velocity[11]
        assert np.linalg.norm(velocity_error) < 1e-4
        assert np.linalg.norm(orbit.position_ecef_km - fine.itrs_pv.position[11]) < 1e-3
        chord = (coarse.gcrs_pv.position[5] + coarse.gcrs_pv.position[6]) / 2.0
        assert np.linalg.norm(chord) < np.linalg.norm(truth) - 1.0

    def test_outside_the_ephemeris_it_raises(self):
        ephem = _orbit(60)
        with pytest.raises(ValueError, match="outside ephemeris range"):
            orbit_at(ephem, ephem.timestamp[-1].timestamp() + 5.0)
