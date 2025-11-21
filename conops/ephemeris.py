"""
Adapter module to bridge rust-ephem with the existing spacecraft-conops-simulator codebase.

This module provides:
1. Helper functions for working with rust-ephem TLEEphemeris objects
"""

from datetime import datetime, timezone
from typing import Any

import rust_ephem  # type: ignore[import-untyped]


# Helper function for TLEEphemeris
def compute_tle_ephemeris(
    tle: str,
    begin: datetime,
    end: datetime,
    step_size: int = 60,
    **kwargs: Any,
) -> rust_ephem.TLEEphemeris:
    """
    Compute TLE ephemeris using rust-ephem.

    Args:
        tle: TLE string (two-line or three-line format)
        begin: Start datetime (should be timezone-aware)
        end: End datetime (should be timezone-aware)
        step_size: Step size in seconds
        **kwargs: Additional keyword arguments (for compatibility)

    Returns:
        rust-ephem TLEEphemeris object
    """
    # Ensure times are timezone-aware (UTC)
    if begin.tzinfo is None:
        begin = begin.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    # Create rust-ephem TLEEphemeris
    return rust_ephem.TLEEphemeris(tle=tle, begin=begin, end=end, step_size=step_size)
