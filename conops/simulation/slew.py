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
    endra: float
    enddec: float
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
        self.endra = 0
        self.enddec = 0
        self.slewtime = 0
        self.slewdist = 0

        # Optional overrides for dynamics (deg/s^2, deg/s)
        self._accel_override: float | None = None
        self._vmax_override: float | None = None

        self.obstype = "PPT"
        self.obsid = 0
        self.mode = 0
        self.at = None  # What's the target associated with this slew?

    def __str__(self) -> str:
        return f"Slew from {self.startra:.3f},{self.startdec:.3f} to {self.endra},{self.enddec} @ {unixtime2date(self.slewstart)}"

    def is_slewing(self, utime: float) -> bool:
        """For a given utime, are we slewing?"""
        if utime >= self.slewend or utime < self.slewstart:
            return False
        else:
            return True

    def ra_dec(self, utime: float) -> tuple[float, float]:
        return self.slew_ra_dec(utime)

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
        motion_time = self.acs_config.motion_time(
            total_dist, accel=self._accel_override, vmax=self._vmax_override
        )
        tau = max(0.0, min(float(t), motion_time))
        s = self.acs_config.s_of_t(
            total_dist, tau, accel=self._accel_override, vmax=self._vmax_override
        )
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
        # Use any accel/vmax overrides when computing slew time (e.g. wheel-limited)
        self.slewtime = round(
            self.acs_config.slew_time(
                distance, accel=self._accel_override, vmax=self._vmax_override
            )
        )

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

        # Estimate an approximate rotation axis for this slew in body frame.
        # Use the great-circle endpoints to form unit vectors and take their
        # cross-product. This axis is used by ACS to project requested
        # scalar accelerations/torques into a 3D torque vector for wheel mapping.
        # start vector
        ra0 = float(self.startra)
        dec0 = float(self.startdec)
        ra1 = float(self.endra)
        dec1 = float(self.enddec)
        # convert to radians
        r0 = np.deg2rad(ra0)
        d0 = np.deg2rad(dec0)
        r1 = np.deg2rad(ra1)
        d1 = np.deg2rad(dec1)
        v0 = np.array([np.cos(d0) * np.cos(r0), np.cos(d0) * np.sin(r0), np.sin(d0)])
        v1 = np.array([np.cos(d1) * np.cos(r1), np.cos(d1) * np.sin(r1), np.sin(d1)])
        axis = np.cross(v0, v1)
        nrm = np.linalg.norm(axis)
        if nrm <= 1e-12:
            # No slew (or nearly no slew): leave a stable default axis and return.
            self.rotation_axis = (0.0, 0.0, 1.0)
            return
        axis = axis / nrm
        self.rotation_axis = (float(axis[0]), float(axis[1]), float(axis[2]))
