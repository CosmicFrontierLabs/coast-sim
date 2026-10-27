"""Bounded snapshots of immutable Rust ephemeris position arrays.

The Rust Python getters return full-array copies. Keep those copies across
sample lookups instead of copying an entire day to read one position. This
does not change the ephemeris or the public getter's independent-copy contract.
"""

from functools import lru_cache
from typing import Literal, cast

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
    positions = np.asarray(getattr(ephem, f"{body}_pv").position, dtype=np.float64)
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
    return cast(npt.NDArray[np.float64], velocities)


def position_vectors(
    ephem: rust_ephem.Ephemeris, body: Literal["gcrs", "sun"]
) -> npt.NDArray[np.float64]:
    """Return read-only snapshots for Rust ephemerides; adapters remain live."""
    if type(ephem) in _IMMUTABLE_EPHEMERIDES:
        return _position_snapshot(ephem, body)
    # Test/custom adapters may be mutable; do not silently cache their values.
    return cast(npt.NDArray[np.float64], getattr(ephem, f"{body}_pv").position)
