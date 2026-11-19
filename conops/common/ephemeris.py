"""Bounded snapshots of immutable Rust ephemeris position arrays.

The Rust Python getters return full-array copies. Keep those copies across
sample lookups instead of copying an entire day to read one position. This
does not change the ephemeris or the public getter's independent-copy contract.
"""

from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import rust_ephem

_IMMUTABLE_EPHEMERIDES = (
    rust_ephem.TLEEphemeris,
    rust_ephem.FileEphemeris,
    rust_ephem.GroundEphemeris,
    rust_ephem.OEMEphemeris,
    rust_ephem.ParquetEphemeris,
    rust_ephem.SPICEEphemeris,
)


@lru_cache(maxsize=8)
def _position_snapshot(
    ephem: rust_ephem.Ephemeris, body: Literal["gcrs", "sun"]
) -> npt.NDArray[np.float64]:
    pv = ephem.gcrs_pv if body == "gcrs" else ephem.sun_pv
    positions = np.asarray(pv.position, dtype=np.float64)
    positions.setflags(write=False)
    return positions


@lru_cache(maxsize=8)
def _velocity_snapshot(
    ephem: rust_ephem.Ephemeris, body: Literal["gcrs"]
) -> npt.NDArray[np.float64]:
    velocities = np.asarray(getattr(ephem, f"{body}_pv").velocity, dtype=np.float64)
    velocities.setflags(write=False)
    return velocities


def velocity_vectors(
    ephem: rust_ephem.Ephemeris, body: Literal["gcrs"] = "gcrs"
) -> npt.NDArray[np.float64] | None:
    """The velocity at each sample in km/s, as :func:`position_vectors`
    gives positions; None for an ephemeris without velocities, such as a
    test double."""
    pv = getattr(ephem, f"{body}_pv", None)
    velocity = getattr(pv, "velocity", None) if pv is not None else None
    if velocity is None:
        return None
    if type(ephem) in _IMMUTABLE_EPHEMERIDES:
        return _velocity_snapshot(ephem, body)
    try:
        velocities = np.asarray(velocity, dtype=np.float64)
    except (TypeError, ValueError):
        # A test double's ephemeris: no velocities to speak of.
        return None
    if velocities.ndim != 2 or velocities.shape[1] != 3:
        return None
    return velocities


def position_vectors(
    ephem: rust_ephem.Ephemeris, body: Literal["gcrs", "sun"]
) -> npt.NDArray[np.float64]:
    """Return read-only snapshots for Rust ephemerides; adapters remain live."""
    if type(ephem) in _IMMUTABLE_EPHEMERIDES:
        return _position_snapshot(ephem, body)
    # Test/custom adapters may be mutable; do not silently cache their values.
    return ephem.gcrs_pv.position if body == "gcrs" else ephem.sun_pv.position


def sample_times(ephem: rust_ephem.Ephemeris) -> npt.NDArray[np.float64]:
    """The ephemeris sample times in Unix seconds; a test double may give
    them as ``numpy.datetime64``."""
    return np.array(
        [
            float((t - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s"))
            if isinstance(t, np.datetime64)
            else t.timestamp()
            for t in ephem.timestamp
        ],
        dtype=np.float64,
    )


class OrbitState(NamedTuple):
    """The orbit at one time. All but the position may be missing from a
    test double's ephemeris."""

    position_eci_km: npt.NDArray[np.float64]
    velocity_eci_km_s: npt.NDArray[np.float64] | None
    sun_position_eci_km: npt.NDArray[np.float64] | None
    """The Sun from the Earth's centre, not from the spacecraft."""
    position_ecef_km: npt.NDArray[np.float64] | None


class OrbitSampler:
    """The orbit at any time within an ephemeris, held at its first or last
    sample outside it.

    A Rust ephemeris interpolates between its samples itself: cubic Hermite
    in position and velocity, as its constraints see the orbit at such a
    time, and exact at a sample. Any other ephemeris, such as a test double,
    is interpolated linearly, which cuts the corner of the orbit between
    samples: by some kilometres in low Earth orbit at a 60 s step.
    """

    def __init__(self, ephem: rust_ephem.Ephemeris) -> None:
        self.ephem = ephem
        self.hermite = type(ephem) in _IMMUTABLE_EPHEMERIDES
        """Whether the ephemeris interpolates itself."""
        self._tables: (
            tuple[
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64] | None,
                npt.NDArray[np.float64] | None,
                npt.NDArray[np.float64] | None,
            ]
            | None
        ) = None
        if self.hermite:
            stamps = ephem.timestamp
            self._first = float(stamps[0].timestamp())
            self._last = float(stamps[-1].timestamp())

    def at(self, utime: float) -> OrbitState:
        """The orbit at ``utime``, in Unix seconds."""
        if not self.hermite:
            return self._linear(utime)
        when = datetime.fromtimestamp(
            min(max(utime, self._first), self._last), tz=timezone.utc
        )
        ephem = self.ephem
        orbit = ephem.gcrs_pv_at(when)
        return OrbitState(
            np.asarray(orbit.position[0], dtype=np.float64),
            np.asarray(orbit.velocity[0], dtype=np.float64),
            np.asarray(ephem.sun_pv_at(when).position[0], dtype=np.float64),
            np.asarray(ephem.itrs_pv_at(when).position[0], dtype=np.float64),
        )

    def _linear(self, utime: float) -> OrbitState:
        if self._tables is None:
            ephem = self.ephem
            sun = getattr(getattr(ephem, "sun_pv", None), "position", None)
            ecef = getattr(getattr(ephem, "itrs_pv", None), "position", None)
            self._tables = (
                sample_times(ephem),
                np.asarray(position_vectors(ephem, "gcrs"), dtype=np.float64),
                velocity_vectors(ephem),
                np.asarray(sun, dtype=np.float64) if sun is not None else None,
                np.asarray(ecef, dtype=np.float64) if ecef is not None else None,
            )
        times, positions, velocities, sun, ecef = self._tables

        def interpolate(
            table: npt.NDArray[np.float64] | None,
        ) -> npt.NDArray[np.float64] | None:
            if table is None:
                return None
            return np.array(
                [np.interp(utime, times, table[:, k]) for k in range(3)],
                dtype=np.float64,
            )

        position = interpolate(positions)
        assert position is not None
        return OrbitState(
            position, interpolate(velocities), interpolate(sun), interpolate(ecef)
        )
