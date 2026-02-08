from typing import TYPE_CHECKING

import numpy as np
import rust_ephem

from ..common import roll_over_angle, unixtime2date
from ..config import AttitudeControlSystem, Constraint, MissionConfig

if TYPE_CHECKING:
    from ..targets.pointing import Pointing


class Slew:
    """Class defines a Spacecraft Slew. Calculates slew time and slew
    path (currently great circle only from simplicity)"""

    ephem: rust_ephem.Ephemeris
    constraint: Constraint
    config: MissionConfig
    slewstart: float
    slewend: float
    startra: float
    startdec: float
    startroll: float
    endra: float
    enddec: float
    endroll: float
    slewtime: float
    slewpath: tuple[list[float], list[float]]
    slewsecs: list[float]
    slewdist: float
    obstype: str
    obsid: int
    mode: int
    slewrequest: float
    at: "Pointing | None"  # In quotes to avoid circular import
    acs_config: AttitudeControlSystem

    def __init__(
        self,
        config: MissionConfig | None = None,
    ):
        # Handle both old and new parameter styles for backward compatibility
        assert config is not None, "MissionConfig must be passed for Slew"
        self.constraint = config.constraint
        self.acs_config = config.spacecraft_bus.attitude_control

        assert self.constraint is not None, "Constraint must be set for Slew class"
        assert self.constraint.ephem is not None, "Ephemeris must be set for Slew class"

        self.ephem = self.constraint.ephem

        # Store ACS configuration if provided
        assert self.acs_config is not None, "ACS config must be set for Slew class"

        self.slewrequest = 0  # When was the slew requested
        self.slewstart = 0
        self.slewend = 0
        self.startra = 0
        self.startdec = 0
        self.startroll = 0
        self.endra = 0
        self.enddec = 0
        self.endroll = 0
        self.slewtime = 0
        self.slewdist = 0

        self.obstype = "PPT"
        self.obsid = 0
        self.mode = 0
        self.at = None  # What's the target associated with this slew?

    def __str__(self) -> str:
        return f"Slew from {self.startra:.3f},{self.startdec:.3f},{self.startroll:.3f}° to {self.endra:.3f},{self.enddec:.3f},{self.endroll:.3f}° @ {unixtime2date(self.slewstart)}"

    def is_slewing(self, utime: float) -> bool:
        """For a given utime, are we slewing?"""
        if utime >= self.slewend or utime < self.slewstart:
            return False
        else:
            return True

    def ra_dec(self, utime: float) -> tuple[float, float]:
        return self.slew_ra_dec(utime)

    def roll(self, utime: float) -> float:
        """Return roll angle at the given time during the slew."""
        return self.slew_roll(utime)

    def slew_ra_dec(self, utime: float) -> tuple[float, float]:
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

        if not self.acs_config or not has_path or self.slewdist <= 0:
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

    def slew_roll(self, utime: float) -> float:
        """Return roll angle at time using linear interpolation during slew."""
        # Time since slew start
        t = utime - self.slewstart
        if t <= 0:
            return self.startroll

        # If no slew time calculated, return start roll
        if self.slewtime <= 0:
            return self.startroll

        # Linear interpolation between start and end roll
        f = max(0.0, min(1.0, t / self.slewtime))
        roll_diff = self.endroll - self.startroll

        # Handle roll angle wrapping (roll is periodic every 360 degrees)
        # Choose the shortest path
        if roll_diff > 180:
            roll_diff -= 360
        elif roll_diff < -180:
            roll_diff += 360

        return (self.startroll + f * roll_diff) % 360

    def calc_slewtime(self) -> float:
        """Calculate time to slew between 2 coordinates, given in degrees.

        Uses the AttitudeControlSystem configuration for accurate slew time
        calculation with bang-bang control profile.
        """
        # Calculate slew distance along great circle path
        self.predict_slew()
        distance = self.slewdist

        # Handle invalid distances
        if np.isnan(distance) or distance < 0:
            raise ValueError(
                f"Invalid slew distance: {distance} (start={self.startra},{self.startdec} end={self.endra},{self.enddec})"
            )

        # Calculate slew time using AttitudeControlSystem
        self.slewtime = round(self.acs_config.slew_time(distance))

        self.slewend = self.slewstart + self.slewtime
        return self.slewtime

    def predict_slew(self) -> None:
        """Calculate great circle slew distance and path using ACS configuration."""

        # Use 100 steps for the path to ensure accurate interpolation.
        # With fewer steps, linear RA/Dec interpolation between coarse points
        # can cause apparent rates to exceed the configured max near celestial poles.
        self.slewdist, self.slewpath = self.acs_config.predict_slew(
            self.startra, self.startdec, self.endra, self.enddec, steps=100
        )
