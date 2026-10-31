"""A moving target's track: its sky position at a series of times."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from ..common import unixtime_from

if TYPE_CHECKING:
    import rust_ephem


class TrackPoint(BaseModel):
    """The target's position at one time."""

    model_config = ConfigDict(frozen=True)

    utime: float = Field(description="The time, in Unix seconds")
    ra: float = Field(description="Right ascension in degrees")
    dec: float = Field(ge=-90.0, le=90.0, description="Declination in degrees")

    @field_validator("utime", mode="before")
    @classmethod
    def _coerce_time(cls, v: float | int | str | datetime) -> float:
        """Accept Unix timestamps, datetimes or ISO-8601 strings."""
        return unixtime_from(v)

    @field_validator("ra")
    @classmethod
    def _wrap_ra(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("right ascension must be finite")
        return float(v % 360.0)

    @field_serializer("utime")
    def _serialize_time(self, v: float) -> str:
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()


def track_from_table(
    times: Sequence[float | int | str | datetime],
    ras: Sequence[float],
    decs: Sequence[float],
) -> list[TrackPoint]:
    """A track from parallel sequences of times, right ascensions and
    declinations, such as a JPL Horizons table."""
    if not len(times) == len(ras) == len(decs):
        raise ValueError("times, ras and decs must have the same length")
    points = [
        TrackPoint(utime=unixtime_from(t), ra=float(ra), dec=float(dec))
        for t, ra, dec in zip(times, ras, decs, strict=True)
    ]
    return sorted(points, key=lambda p: p.utime)


def validate_track(track: Sequence[TrackPoint]) -> list[TrackPoint]:
    """The track in time order, with strictly increasing times."""
    ordered = sorted(track, key=lambda p: p.utime)
    for first, second in zip(ordered, ordered[1:]):
        if second.utime <= first.utime:
            raise ValueError("track times must increase strictly")
    return list(ordered)


def _unwrapped_ra(track: Sequence[TrackPoint]) -> np.ndarray:
    """Right ascensions continuous across the 0/360 degree seam."""
    return np.degrees(np.unwrap(np.radians([p.ra for p in track])))


def position_along(track: Sequence[TrackPoint], utime: float) -> tuple[float, float]:
    """The track's position at ``utime``, in degrees: interpolated between
    the samples around it along the shorter way in right ascension, and
    held at the first or last sample outside the track."""
    if not track:
        raise ValueError("an empty track has no position")
    if len(track) == 1:
        return track[0].ra, track[0].dec
    times = np.array([p.utime for p in track], dtype=np.float64)
    ras = _unwrapped_ra(track)
    decs = np.array([p.dec for p in track], dtype=np.float64)
    t = float(np.clip(utime, times[0], times[-1]))
    ra = float(np.interp(t, times, ras)) % 360.0
    dec = float(np.interp(t, times, decs))
    return ra, dec


def rate_along(track: Sequence[TrackPoint], utime: float) -> float:
    """The track's angular rate at ``utime``, in degrees per second: that
    of the interval the time falls in; zero outside the track or for a
    single sample."""
    if len(track) < 2:
        return 0.0
    times = [p.utime for p in track]
    if utime < times[0] or utime > times[-1]:
        return 0.0
    index = min(
        max(int(np.searchsorted(times, utime, side="right")) - 1, 0), len(track) - 2
    )
    first, second = track[index], track[index + 1]
    ra_first, ra_second = np.radians(first.ra), np.radians(second.ra)
    dec_first, dec_second = np.radians(first.dec), np.radians(second.dec)
    cosine = np.sin(dec_first) * np.sin(dec_second) + np.cos(dec_first) * np.cos(
        dec_second
    ) * np.cos(ra_second - ra_first)
    separation = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return separation / (second.utime - first.utime)


def body_track(
    ephem: rust_ephem.Ephemeris,
    body: str,
    *,
    use_horizons: bool = True,
    spice_kernel: str | None = None,
) -> list[TrackPoint]:
    """The track of a Solar System body over an ephemeris, at every sample
    of it, looked up through ``rust_ephem``: from JPL Horizons by default,
    or from a SPICE kernel. ``body`` is a name or NAIF id Horizons or the
    kernel knows."""
    positions = ephem.get_body(
        body, spice_kernel=spice_kernel, use_horizons=use_horizons
    )
    ras = [float(v) for v in positions.ra.deg]
    decs = [float(v) for v in positions.dec.deg]
    times = [float(t.timestamp()) for t in ephem.timestamp]
    return track_from_table(times, ras, decs)
