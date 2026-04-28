from typing import TYPE_CHECKING

import numpy as np
import rust_ephem

from ..common import roll_over_angle, unixtime2date
from ..common.common import dtutcfromtimestamp
from ..common.enums import ObsType, SlewAlgorithm
from ..common.vector import (
    sun_avoiding_waypoint,
)
from ..config import AttitudeControlSystem, Constraint, MissionConfig
from ..config.constants import SUN_OCCULT

if TYPE_CHECKING:
    from ..targets.pointing import Pointing


class Slew:
    """Class defines a Spacecraft Slew. Calculates slew time and slew path.

    Supports two path algorithms selected via the ACS configuration:
    - QUATERNION (default): full SO(3) SLERP coupling pointing and roll changes.
    - SUN_AVOIDING: quaternion SLERP with automatic detour around the Sun.
    """

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
    obstype: ObsType
    obsid: int
    mode: int
    slewrequest: float
    at: "Pointing | None"  # In quotes to avoid circular import
    acs_config: AttitudeControlSystem
    # Quaternion SLERP: intermediate roll values along the path
    _quat_roll_path: list[float]

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

        self.obstype = ObsType.PPT
        self.obsid = 0
        self.mode = 0
        self.at = None  # What's the target associated with this slew?
        self._quat_roll_path: list[float] = []  # roll path for QUATERNION algorithm

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
        pre-computed path according to bang-bang kinematics (accel → cruise → decel).
        The path may be a great-circle arc, quaternion SLERP, or sun-avoiding arc
        depending on the configured slew algorithm.
        """
        t = utime - self.slewstart
        if t <= 0:
            return self.startra, self.startdec

        has_path = (
            hasattr(self, "slewpath")
            and isinstance(self.slewpath, (tuple, list))
            and len(self.slewpath) == 2
            and len(self.slewpath[0]) > 0
        )

        if not self.acs_config or not has_path or self.slewdist <= 0:
            return self.startra, self.startdec

        total_dist = float(self.slewdist)
        motion_time = self.acs_config.motion_time(total_dist)
        tau = max(0.0, min(float(t), motion_time))
        s = self.acs_config.s_of_t(total_dist, tau)
        f = 0.0 if total_dist == 0 else max(0.0, min(1.0, s / total_dist))

        ra_path, dec_path = self.slewpath
        n = len(ra_path)
        if n <= 1:
            return float(ra_path[0]) % 360, float(dec_path[0])

        idx = f * (n - 1)
        x = np.arange(n, dtype=float)
        ras = roll_over_angle(ra_path)
        ra = np.interp(idx, x, ras) % 360
        dec = np.interp(idx, x, dec_path)
        return ra, dec

    def slew_roll(self, utime: float) -> float:
        """Return roll angle at time during slew, drawn from the SLERP path."""
        t = utime - self.slewstart
        if t <= 0:
            return self.startroll
        if self.slewtime <= 0:
            return self.startroll

        if self._quat_roll_path:
            total_dist = float(self.slewdist)
            motion_time = self.acs_config.motion_time(total_dist)
            tau = max(0.0, min(float(t), motion_time))
            s = self.acs_config.s_of_t(total_dist, tau)
            f = 0.0 if total_dist == 0 else max(0.0, min(1.0, s / total_dist))
            n = len(self._quat_roll_path)
            idx = f * (n - 1)
            rolls = roll_over_angle(self._quat_roll_path)
            return float(np.interp(idx, np.arange(n, dtype=float), rolls)) % 360

        # Fallback: shortest-path linear interpolation
        f = max(0.0, min(1.0, t / self.slewtime))
        roll_diff = self.endroll - self.startroll
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
        """Compute slew distance and path according to the configured algorithm.

        QUATERNION (default):
            Full SO(3) SLERP.  The RA/Dec path is derived from the boresight
            component of the SLERP; roll is stored in _quat_roll_path so
            slew_roll() returns the properly coupled value.

        SUN_AVOIDING:
            Quaternion SLERP with an automatic detour when the direct arc
            would cross within the configured Sun exclusion angle.  Falls back
            to plain QUATERNION when no violation is detected.

        In all cases self.slewdist is the total angular distance (degrees) and
        self.slewpath is the (ra_list, dec_list) path used for interpolation.
        """
        steps = 100
        if self.acs_config.slew_algorithm == SlewAlgorithm.SUN_AVOIDING:
            self._predict_slew_sun_avoiding(steps)
        else:
            self._predict_slew_quaternion(steps)

    def _predict_slew_quaternion(self, steps: int) -> None:
        """Compute slew path via full quaternion SLERP."""
        from ..common.vector import quaternion_slew_path

        ras, decs, rolls = quaternion_slew_path(
            self.startra,
            self.startdec,
            self.startroll,
            self.endra,
            self.enddec,
            self.endroll,
            steps=steps,
        )
        self.slewpath = (ras, decs)
        self._quat_roll_path = rolls
        # Great-circle distance is still the correct metric for slew time
        from ..common import separation
        from ..config.constants import DTOR

        self.slewdist = (
            separation(
                [self.startra * DTOR, self.startdec * DTOR],
                [self.endra * DTOR, self.enddec * DTOR],
            )
            / DTOR
        )

    def _predict_slew_sun_avoiding(self, steps: int) -> None:
        """Compute sun-avoiding slew path using quaternion SLERP segments.

        If the direct arc violates the Sun exclusion zone, a waypoint is
        inserted and the path is built from two consecutive SLERP segments.
        Falls back to plain quaternion SLERP when no violation is detected.
        """
        # Get sun position at slew start time from the ephemeris
        try:
            idx = self.ephem.index(dtutcfromtimestamp(self.slewstart))
            sun_ra = float(self.ephem.sun_ra_deg[idx])
            sun_dec = float(self.ephem.sun_dec_deg[idx])
        except Exception:
            self._predict_slew_quaternion(steps)
            return

        # Determine the sun exclusion angle from the constraint config
        min_sun_angle = float(SUN_OCCULT)
        sc = self.constraint.sun_constraint
        if sc is not None:
            raw = getattr(sc, "min_angle", None)
            if raw is not None:
                min_sun_angle = float(raw)

        waypoint = sun_avoiding_waypoint(
            self.startra,
            self.startdec,
            self.endra,
            self.enddec,
            sun_ra,
            sun_dec,
            min_sun_angle,
        )

        if waypoint is None:
            # No violation – plain quaternion SLERP
            self._predict_slew_quaternion(steps)
            return

        w_ra, w_dec = waypoint

        # Estimate roll at waypoint by linear interpolation of the total arc fraction
        from ..common import separation
        from ..config.constants import DTOR

        dist1 = (
            separation(
                [self.startra * DTOR, self.startdec * DTOR],
                [w_ra * DTOR, w_dec * DTOR],
            )
            / DTOR
        )
        dist2 = (
            separation(
                [w_ra * DTOR, w_dec * DTOR],
                [self.endra * DTOR, self.enddec * DTOR],
            )
            / DTOR
        )
        total = dist1 + dist2
        frac = dist1 / total if total > 0 else 0.5
        roll_diff = self.endroll - self.startroll
        if roll_diff > 180:
            roll_diff -= 360
        elif roll_diff < -180:
            roll_diff += 360
        w_roll = (self.startroll + frac * roll_diff) % 360

        # Build two SLERP segments; split steps proportionally
        steps1 = max(1, round(steps * frac))
        steps2 = max(1, steps - steps1)

        from ..common.vector import quaternion_slew_path

        ras1, decs1, rolls1 = quaternion_slew_path(
            self.startra,
            self.startdec,
            self.startroll,
            w_ra,
            w_dec,
            w_roll,
            steps=steps1,
        )
        ras2, decs2, rolls2 = quaternion_slew_path(
            w_ra,
            w_dec,
            w_roll,
            self.endra,
            self.enddec,
            self.endroll,
            steps=steps2,
        )

        # Concatenate (drop duplicate waypoint at segment junction)
        all_ras = ras1 + ras2[1:]
        all_decs = decs1 + decs2[1:]
        all_rolls = rolls1 + rolls2[1:]

        self.slewpath = (all_ras, all_decs)
        self._quat_roll_path = all_rolls
        self.slewdist = dist1 + dist2
