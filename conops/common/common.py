import os
import time
from datetime import datetime, timezone

import numpy as np

from .enums import ACSMode

# Make sure we are working in UTC times
os.environ["TZ"] = "UTC"
time.tzset()


def givename(ra: float, dec: float, stem: str = "") -> str:
    # Convert RA/Dec (in degrees) into generic "JHHMM.m+/-DDMM" format
    rapart = "J%02d%04.1f" % (np.floor(ra / 15), 60 * ((ra / 15) - np.floor(ra / 15)))
    decpart = "%02d%02d" % (
        np.floor(abs(dec)),
        round(60 * (abs(dec) - np.floor(abs(dec)))),
    )

    if np.sign(dec) == -1:
        name = f"{rapart}-{decpart}"
    else:
        name = f"{rapart}+{decpart}"

    if stem != "":
        name = f"{stem} {name}"

    return name


def unixtime2date(utime: float) -> str:
    """Converts Unix time to date string of format YYYY-DDD-HH:MM:SS"""
    dt = datetime.fromtimestamp(utime, tz=timezone.utc)
    return f"{dt.year:04d}-{dt.timetuple().tm_yday:03d}-{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"


def ics_date_conv(date: str) -> float:
    """Convert the date format used in the ICS to standard UNIX time"""
    x = date.replace("/", " ").replace("-", " ").replace(":", " ").split()
    base = time.mktime((int(x[0]), 1, 0, 0, 0, 0, 0, 0, 0))
    return base + (
        (int(x[1])) * 86400 + (float(x[2])) * 3600 + float(x[3]) * 60 + float(x[4])
    )


def unixtime2yearday(utime: float) -> tuple[int, int]:
    """Converts Unix time to year and day of year"""
    dt = datetime.fromtimestamp(utime, tz=timezone.utc)
    return dt.year, dt.timetuple().tm_yday


def dtutcfromtimestamp(timestamp: float) -> datetime:
    """Return a timezone-aware UTC datetime from a unix timestamp"""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def normalize_acs_mode(mode: "ACSMode | int | None") -> "ACSMode | None":
    """Normalize ACS mode to ACSMode enum, handling int conversion gracefully.

    Args:
        mode: Raw mode value from housekeeping or ACS

    Returns:
        Normalized ACSMode enum, or None if conversion fails
    """
    if mode is None:
        return None
    # Import here to avoid circular imports
    from .enums import ACSMode

    if isinstance(mode, ACSMode):
        return mode
    try:
        return ACSMode(mode)
    except ValueError:
        return None
