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

from .common import dtutcfromtimestamp
from .numerics import VECTOR_COMPONENTS
from .vector import scbodyvector

_IMMUTABLE_EPHEMERIDES = (
    rust_ephem.TLEEphemeris,
    rust_ephem.FileEphemeris,
    rust_ephem.GroundEphemeris,
    rust_ephem.OEMEphemeris,
    rust_ephem.ParquetEphemeris,
    rust_ephem.SPICEEphemeris,
)


_ARRAY_DIMENSIONS = 2
"""Velocities come as one row of three components per sample."""


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
    if velocities.ndim != _ARRAY_DIMENSIONS or velocities.shape[1] != VECTOR_COMPONENTS:
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


def sun_direction(ephem: rust_ephem.Ephemeris, utime: float) -> npt.NDArray[np.float64]:
    """The Sun's position from the spacecraft in the GCRS frame at ``utime``."""
    index = ephem.index(dtutcfromtimestamp(float(utime)))
    return cast(
        npt.NDArray[np.float64],
        position_vectors(ephem, "sun")[index] - position_vectors(ephem, "gcrs")[index],
    )


def sun_vector_change_between(
    sun: npt.NDArray[np.float64],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float:
    """The angle, in degrees, between the directions of ``sun`` (see
    :func:`sun_direction`) in the body frame at two attitudes (RA, Dec, roll
    in degrees): what a slew between them moves the Sun by, and what the
    thermal settling table is written in."""
    start_ra, start_dec, start_roll = (float(np.radians(v)) for v in start)
    end_ra, end_dec, end_roll = (float(np.radians(v)) for v in end)
    before = scbodyvector(start_ra, start_dec, start_roll, sun)
    after = scbodyvector(end_ra, end_dec, end_roll, sun)
    cosine = float(np.dot(before, after)) / float(
        np.linalg.norm(before) * np.linalg.norm(after)
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def sun_vector_change_deg(
    ephem: rust_ephem.Ephemeris,
    utime: float,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float:
    """The angle, in degrees, between the Sun's directions in the body frame
    at two attitudes (RA, Dec, roll in degrees), with the Sun where it is at
    ``utime`` (see :func:`sun_vector_change_between`)."""
    return sun_vector_change_between(sun_direction(ephem, utime), start, end)
