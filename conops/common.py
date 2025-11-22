import os
import time
from datetime import datetime, timezone
from enum import Enum

import numpy as np

# Make sure we are working in UTC times
os.environ["TZ"] = "UTC"
time.tzset()


class ACSMode(int, Enum):
    """Spacecraft ACS Modes"""

    SCIENCE = 0
    SLEWING = 1
    SAA = 2
    PASS = 3
    CHARGING = 4
    SAFE = 5


def givename(ra, dec, stem=""):
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


def unixtime2date(utime):
    """Converts Unix time to date string of format YYYY-DDD-HH:MM:SS"""
    lt = time.localtime(utime)
    return "%4d-%03d-%02d:%02d:%02d" % (lt[0], lt[7], lt[3], lt[4], lt[5])


def ics_date_conv(date):
    """Convert the date format used in the ICS to standard UNIX time"""
    x = date.replace("/", " ").replace("-", " ").replace(":", " ").split()
    base = time.mktime((int(x[0]), 1, 0, 0, 0, 0, 0, 0, 0))
    return base + (
        (int(x[1])) * 86400 + (float(x[2])) * 3600 + float(x[3]) * 60 + float(x[4])
    )


def unixtime2yearday(utime):
    """Converts Unix time to date string of format YYYY-DDD-HH:MM:SS"""
    lt = time.localtime(utime)
    return lt[0], lt[7]


def dtutcfromtimestamp(timestamp: float) -> datetime:
    """Return a timezone-aware UTC datetime from a unix timestamp"""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)
