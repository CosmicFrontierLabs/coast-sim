from __future__ import annotations

import numpy as np
import rust_ephem

from conops.simulation.saa import SAA

from ..common import givename, roll_over_angle, unixtime2date
from ..config import AttitudeControlSystem, Constraint
from ..simulation.saa import SAA


class PlanEntry:
    """Class to define a entry in the Plan"""

    ra: float
    dec: float
    roll: float
    begin: float
    end: float
    windows: list[list[float]]
    ephem: rust_ephem.TLEEphemeris | None
    constraint: Constraint | None
    merit: float
    saa: SAA | None

    def __init__(
        self,
        constraint: Constraint | None = None,
        acs_config: AttitudeControlSystem | None = None,
    ) -> None:
        self.constraint = constraint
        assert self.constraint is not None, "Constraint must be set for Pass class"
        self.ephem = self.constraint.ephem
        assert self.ephem is not None, "Ephemeris must be set for Pass class"
        self.acs_config = acs_config
        assert self.acs_config is not None, "ACS config must be set for PlanEntry class"
        self.name = ""
        # self.targetid = 0
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
        self.slewpath = False
        self.slewdist = False
        self.ssmin = 1000
        self.ssmax = 1e6
        self.constraint = constraint

    def copy(self):
        """Create a copy of this class"""
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    @property
    def targetid(self):
        return self.obsid & 0xFFFFFF

    @targetid.setter
    def targetid(self, value):
        self.obsid = value + (self.segment << 24)

    @property
    def segment(self):
        return self.obsid >> 24

    @segment.setter
    def segment(self, value):
        self.obsid = self.targetid + (value << 24)

    def __str__(self):
        return f"{unixtime2date(self.begin)} Target: {self.name} ({self.targetid}/{self.segment}) Exp: {self.exposure}s "

    @property
    def exposure(self):  # (),excludesaa=False):
        self.insaa = 0

        # if self.saa is not False:
        #     for saatime in np.arange(self.begin + self.slewtime, self.end, 1):
        #         if self.saa.insaa(saatime):
        #             self.insaa += 1

        return int(
            self.end - self.begin - self.slewtime - self.insaa
        )  # always an integer number of seconds

    @exposure.setter
    def exposure(self, value):
        """Setter for exposure - accepts but ignores the value since exposure is computed."""
        pass

    def givename(self, stem=""):
        self.name = givename(self.ra, self.dec, stem=stem)

    def visibility(
        self,
    ) -> int:
        """Calculate the visibility windows for a target for a given day(s).

        Note: year, day, length, and hires parameters are kept for backwards
        compatibility but are no longer used. The visibility is calculated over
        the entire ephemeris time range.
        """

        assert self.constraint is not None, (
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

    def visible(self, begin, end):
        """Is the target visible between these two times, if yes, return the visibility window"""
        for window in self.windows:
            if begin >= window[0] and end <= window[1]:
                return window
        return False

    def slew_ra_dec(self, utime):
        """Return the RA/Dec of Spacecraft after t seconds into a slew. Assumes linear rate of slew."""
        t = utime - self.begin
        ras = roll_over_angle(self.slewpath[0])

        ra = np.interp(t, self.slewsecs, ras) % 360
        dec = np.interp(t, self.slewsecs, self.slewpath[1])
        return ra, dec

    def in_slew(self, utime):
        """Are we slewing right now?"""
        if utime >= self.begin and utime < self.begin + self.slewtime:
            return True
        else:
            return False

    def ra_dec(self, utime):
        """Return Spacecraft RA/Dec for any time during the current PPT"""
        if utime >= self.begin and utime <= self.end:
            if self.in_slew(utime):
                return self.slew_ra_dec(utime)
            else:
                return self.ra, self.dec
        else:
            return [-1, -1]

    def calc_slewtime(
        self, lastra, lastdec, no_update=False, phil=False, distance=False
    ):
        """Calculate time to slew between 2 coordinates, given in degrees.

        Uses the AttitudeControlSystem configuration for accurate slew time
        calculation with bang-bang control profile.
        """

        if distance is False:
            # Use the more accurate slew distance instead of angular distance
            if self.slewdist is False:
                self.predict_slew(lastra, lastdec)
            distance = self.slewdist

        # Calculate slew time using AttitudeControlSystem
        slewtime = round(self.acs_config.slew_time(distance))

        if not no_update:
            self.slewtime = slewtime

        return slewtime

    def predict_slew(self, lastra, lastdec):
        """Calculate great circle slew distance and path using ACS configuration."""
        self.slewdist, self.slewpath = self.acs_config.predict_slew(
            lastra, lastdec, self.ra, self.dec, steps=20
        )
