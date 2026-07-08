from __future__ import annotations

from importlib import resources

import numpy as np
import rust_ephem
from shapely import Point, Polygon

from ..common import dtutcfromtimestamp


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

        diff = np.diff(inside.astype(int))
        # Starts are where diff goes from 0 to 1 (so diff is 1)
        start_indices = np.where(diff == 1)[0]
        # Exits are where diff goes from 1 to 0 (so diff is -1)
        end_indices = np.where(diff == -1)[0]

        saatimes_list = []

        for start, end in zip(start_indices, end_indices):
            # The start index from np.diff is the point *before* the transition.
            # So we need to add 1 to get the first point inside the SAA.
            # The end index is also the point *before* the transition, so we take
            # that time as the last point inside the SAA.
            saatimes_list.append([ephem_utime[start + 1], ephem_utime[end]])
        self.saatimes = np.array(saatimes_list)
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
