from __future__ import annotations

from typing import Literal

import rust_ephem

from ..common import givename, unixtime2date
from ..config import Constraint, MissionConfig
from ..simulation.saa import SAA


class PlanEntry:
    """Class to define a entry in the Plan"""

    ra: float
    dec: float
    roll: float
    begin: float
    end: float
    windows: list[list[float]]
    ephem: rust_ephem.Ephemeris | None
    constraint: Constraint
    config: MissionConfig
    merit: float
    saa: SAA | None
    slewpath: tuple[list[float], list[float]]

    def __init__(
        self,
        config: MissionConfig | None = None,
        exptime: int = 1000,
    ) -> None:
        # Extract config parameters from Config object
        if config is None:
            raise ValueError("Config must be provided to PlanEntry")
        self.config = config
        self.constraint = config.constraint
        self.acs_config = config.spacecraft_bus.attitude_control

        assert self.constraint is not None, "Constraint must be set for PlanEntry class"
        self.ephem = self.constraint.ephem
        assert self.ephem is not None, "Ephemeris must be set for PlanEntry class"
        assert self.acs_config is not None, "ACS config must be set for PlanEntry class"
        self.name = ""
        self.ra = 0.0
        self.dec = 0.0
        self.roll = -1.0
        self.begin = 0  # start of window, not observation
        self.slewtime = 0
        self.insaa = 0
        self.end = 0
        self.obsid = 0

        self.saa = None
        self.merit = 101
        self.windows = list()
        self.obstype = "PPT"
        self.slewpath = ([], [])
        self.slewdist = 0.0
        self.ss_min = 1000
        self.ss_max = 1e6
        self._exptime = exptime
        self._exporig = exptime

    @property
    def exptime(self) -> int:
        return self._exptime

    @exptime.setter
    def exptime(self, t: int) -> None:
        if self._exptime is None:
            self._exporig = t
        self._exptime = t

    def copy(self) -> PlanEntry:
        """Create a copy of this class"""
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __str__(self) -> str:
        return f"{unixtime2date(self.begin)} Target: {self.name} ({self.obsid}) Exp: {self.exposure}s "

    @property
    def exposure(self) -> int:  # (),excludesaa=False):
        self.insaa = 0
        return int(
            self.end - self.begin - self.slewtime - self.insaa
        )  # always an integer number of seconds

    @exposure.setter
    def exposure(self, value: int) -> None:
        """Setter for exposure - accepts but ignores the value since exposure is computed."""
        pass

    def givename(self, stem: str = "") -> None:
        self.name = givename(self.ra, self.dec, stem=stem)

    def visibility(
        self,
    ) -> int:
        """Calculate the visibility windows for a target for a given day(s).

        Note: year, day, length, and hires parameters are kept for backwards
        compatibility but are no longer used. The visibility is calculated over
        the entire ephemeris time range.
        """

        assert self.config.constraint is not None, (
            "Constraint must be set to calculate visibility"
        )
        assert self.ephem is not None, "Ephemeris must be set to calculate visibility"

        # Calculate the visibility of this target
        in_constraint = self.constraint.constraint.evaluate(
            ephemeris=self.ephem,
            target_ra=self.ra,  # already in degrees
            target_dec=self.dec,
        )
        # Construct the visibility windows

        self.windows = [
            [v.start_time.timestamp(), v.end_time.timestamp()]
            for v in in_constraint.visibility
        ]

        return 0

    def visible(self, begin: float, end: float) -> list[float] | Literal[False]:
        """Is the target visible between these two times, if yes, return the visibility window"""
        for window in self.windows:
            if begin >= window[0] and end <= window[1]:
                return window
        return False

    def ra_dec(self, utime: float) -> tuple[float, float] | list[int]:
        """Return Spacecraft RA/Dec for any time during the current PPT"""
        if utime >= self.begin and utime <= self.end:
            return self.ra, self.dec
        else:
            return [-1, -1]

    def calc_slewtime(
        self,
        lastra: float,
        lastdec: float,
    ) -> int:
        """Calculate time to slew between 2 coordinates, given in degrees.

        Uses the AttitudeControlSystem configuration for accurate slew time
        calculation with bang-bang control profile.
        """

        # Use the more accurate slew distance instead of angular distance
        self.predict_slew(lastra, lastdec)

        # Calculate slew time using AttitudeControlSystem
        slewtime = round(self.acs_config.slew_time(self.slewdist))

        return slewtime

    def predict_slew(self, lastra: float, lastdec: float) -> None:
        """Calculate great circle slew distance and path using ACS configuration."""
        self.slewdist, self.slewpath = self.acs_config.predict_slew(
            lastra, lastdec, self.ra, self.dec, steps=20
        )
