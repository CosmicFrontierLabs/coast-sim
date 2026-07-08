from __future__ import annotations

from importlib import resources

import numpy as np
import rust_ephem
from shapely import Point, Polygon

from ..common import dtutcfromtimestamp, find_boundaries


def _load_saa_polygon() -> Polygon:
    """Load the SAA boundary polygon (longitude, latitude pairs in degrees)."""
    data_file = resources.files("conops") / "data" / "saa_polygon.csv"
    with resources.as_file(data_file) as path:
        points = np.loadtxt(path, delimiter=",", skiprows=1)
    return Polygon(points)


class SAA:
    """South Atlantic Anomaly (SAA) calculation and tracking for spacecraft."""

    ephem: rust_ephem.Ephemeris | None
    year: int | None
    day: int | None
    saatimes: np.ndarray
    calculated: bool
    saapoly: Polygon

    def __init__(self, year: int | None = None, day: int | None = None) -> None:
        self.year = year
        self.day = day
        self.ephem = None
        self.saatimes = np.array([]).reshape(
            0, 2
        )  # Empty 2D array for [start, end] pairs
        self.calculated = False

        self.saapoly = _load_saa_polygon()

    def insaa_calc(self, utime: float) -> bool:
        """For a given time, are we inside the BAT SAA polygon"""
        if self.ephem is None:
            raise ValueError("Ephemeris must be set before checking SAA status")

        i = self.ephem.index(dtutcfromtimestamp(utime))
        long = self.ephem.longitude_deg[i]
        lat = self.ephem.latitude_deg[i]

        return bool(self.saapoly.contains(Point(long, lat)))

    def calc(self) -> None:
        """
        Calculate the SAA times based on the ephemeris data.
        This method determines the time intervals when the BAT is inside the SAA
        region by analyzing the satellite ephemeris data and checking the
        corresponding geographic coordinates against the SAA polygon.
        """
        if self.ephem is None:
            raise ValueError("Ephemeris must be set before calculating SAA times")

        ephem_utime = [dt.timestamp() for dt in self.ephem.timestamp]
        inside = np.array([self.insaa_calc(t) for t in ephem_utime])

        # starts/ends are half-open [start, end) index pairs; find_boundaries
        # also captures SAA intervals already active at the start of the
        # ephemeris window, or still active at its end, rather than dropping them.
        starts, ends = find_boundaries(inside)
        if len(starts) == 0:
            self.saatimes = np.array([]).reshape(0, 2)
        else:
            self.saatimes = np.array(
                [
                    [ephem_utime[start], ephem_utime[end - 1]]
                    for start, end in zip(starts, ends)
                ]
            )
        self.calculated = True

    def _ensure_calculated(self) -> None:
        if not self.calculated:
            self.calc()

    def get_saa_times(self) -> np.ndarray:
        self._ensure_calculated()
        return self.saatimes

    def insaa(self, utime: float) -> bool:
        """
        Check if the given UTC time is within an SAA interval.

        Args:
            utime (float): The UTC time to check.

        Returns:
            True if the time is within an SAA interval, False otherwise.
        """
        self._ensure_calculated()

        for start, end in self.saatimes:
            if start <= utime <= end:
                return True
        return False

    def get_next_saa_time(self, utime: float) -> tuple[float, float] | None:
        """
        Get the next SAA time interval after the given utime.
        Returns:
            tuple: (start, end) of the next SAA interval, or None if there is no
            upcoming SAA interval.
        """
        self._ensure_calculated()

        for start, end in self.saatimes:
            if start > utime:
                return (start, end)

        return None
