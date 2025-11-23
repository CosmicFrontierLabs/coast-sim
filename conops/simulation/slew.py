from typing import Any

import numpy as np
import rust_ephem

from ..common import roll_over_angle, unixtime2date
from ..config import AttitudeControlSystem, Constraint


class Slew:
    """Class defines a Spacecraft Slew. Calculates slew time and slew
    path (currently great circle only from simplicity)"""

    ephem: rust_ephem.TLEEphemeris | None
    constraint: Constraint | None
    slewstart: float
    slewend: float
    startra: float
    startdec: float
    endra: float
    enddec: float
    slewtime: float
    slewpath: np.ndarray
    slewdist: float
    obstype: str
    obsid: int
    mode: int
    slewrequest: float
    at: Any  # FIXME: should be Pointing | None but this causes circular import
    acs_config: AttitudeControlSystem | None

    def __init__(
        self,
        constraint: Constraint | None = None,
        acs_config: AttitudeControlSystem | None = None,
    ):
        self.constraint = constraint
        assert self.constraint is not None, "Constraint must be set for Slew class"
        self.ephem = self.constraint.ephem
        assert self.ephem is not None, "Ephemeris must be set for Slew class"

        # Store ACS configuration if provided
        self.acs_config = acs_config
        assert self.acs_config is not None, "ACS config must be set for Slew class"

        self.slewrequest = 0  # When was the slew requested
        self.slewstart = 0
        self.slewend = 0
        self.startra = 0
        self.startdec = 0
        self.endra = 0
        self.enddec = 0
        self.slewtime = 0
        self.slewpath = np.array([])
        self.slewsecs = np.array([])
        self.slewdist = 0

        self.obstype = "PPT"
        self.obsid = 0
        self.mode = 0
        self.at = False  # What's the target associated with this slew?

    def __str__(self):
        return f"Slew from {self.startra:.3f},{self.startdec:.3f} to {self.endra},{self.enddec} @ {unixtime2date(self.slewstart)}"

    def is_slewing(self, utime):
        """For a given utime, are we slewing?"""
        if utime >= self.slewend or utime < self.slewstart:
            return False
        else:
            return True

    def ra_dec(self, utime):
        return self.slew_ra_dec(utime)

    def slew_ra_dec(self, utime):
        """Return RA/Dec at time using bang-bang slew profile when configured.

        If an AttitudeControlSystem config is present, advance along the
        great-circle path according to bang-bang kinematics (accel → cruise → decel).
        Otherwise, fall back to legacy linear time interpolation.
        """
        # Time since slew start
        t = utime - self.slewstart
        if t <= 0:
            return self.startra, self.startdec

        # If we don't have a computed path yet, or no ACS config, use legacy behavior
        has_path = (
            hasattr(self, "slewpath")
            and isinstance(self.slewpath, (tuple, list))
            and len(self.slewpath) == 2
            and len(self.slewpath[0]) > 0
        )
        has_slewsecs = hasattr(self, "slewsecs") and len(self.slewsecs) > 0

        if not self.acs_config or not has_path or self.slewdist <= 0:
            # Legacy linear interpolation path
            if has_path and has_slewsecs:
                ras = roll_over_angle(self.slewpath[0])
                ra = np.interp(t, self.slewsecs, ras) % 360
                dec = np.interp(t, self.slewsecs, self.slewpath[1])
            else:
                # No valid path, return start position
                ra = self.startra
                dec = self.startdec
            return ra, dec

        total_dist = float(self.slewdist)
        motion_time = self.acs_config.motion_time(total_dist)
        tau = max(0.0, min(float(t), motion_time))
        s = self.acs_config.s_of_t(total_dist, tau)
        f = 0.0 if total_dist == 0 else max(0.0, min(1.0, s / total_dist))

        # Interpolate along the great-circle path using fraction f
        ra_path, dec_path = self.slewpath
        n = len(ra_path)
        # Protect against pathological small N
        if n <= 1:
            ra = float(ra_path[0]) % 360
            dec = float(dec_path[0])
            return ra, dec

        idx = f * (n - 1)
        x = np.arange(n, dtype=float)
        ras = roll_over_angle(ra_path)
        ra = np.interp(idx, x, ras) % 360
        dec = np.interp(idx, x, dec_path)
        return ra, dec

    def calc_slewtime(self):
        """Calculate time to slew between 2 coordinates, given in degrees.

        Uses the AttitudeControlSystem configuration for accurate slew time
        calculation with bang-bang control profile.
        """
        # Calculate slew distance along great circle path
        self.predict_slew()
        distance = self.slewdist

        # Handle invalid distances
        if np.isnan(distance) or distance < 0:
            print(
                "WARNING: Invalid slew distance: %s (start=%s,%s end=%s,%s)",
                distance,
                self.startra,
                self.startdec,
                self.endra,
                self.enddec,
            )
            distance = 0.0
            self.slewdist = 0.0

        # Calculate slew time using AttitudeControlSystem
        self.slewtime = round(self.acs_config.slew_time(distance))

        self.slewend = self.slewstart + self.slewtime
        return self.slewtime

    def predict_slew(self):
        """Calculate great circle slew distance and path using ACS configuration."""
        self.slewdist, self.slewpath = self.acs_config.predict_slew(
            self.startra, self.startdec, self.endra, self.enddec, steps=20
        )
