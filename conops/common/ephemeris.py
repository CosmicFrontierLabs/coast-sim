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


def position_vectors(
    ephem: rust_ephem.Ephemeris, body: Literal["gcrs", "sun"]
) -> npt.NDArray[np.float64]:
    """Return read-only snapshots for Rust ephemerides; adapters remain live."""
    if type(ephem) in _IMMUTABLE_EPHEMERIDES:
        return _position_snapshot(ephem, body)
    # Test/custom adapters may be mutable; do not silently cache their values.
    return cast(npt.NDArray[np.float64], getattr(ephem, f"{body}_pv").position)
