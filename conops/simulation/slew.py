from typing import TYPE_CHECKING

import numpy as np
import rust_ephem

from ..common import dtutcfromtimestamp, roll_over_angle, separation, unixtime2date
from ..common.enums import ObsType, SlewAlgorithm
from ..common.vector import (
    attitude_to_quat,
    constraint_avoiding_waypoint,
    quaternion_slew_path,
)
from ..config import AttitudeControlSystem, Constraint, MissionConfig
from ..config.constants import DTOR

if TYPE_CHECKING:
    from ..targets.pointing import Pointing


class Slew:
    """Class defines a Spacecraft Slew. Calculates slew time and slew path.

    Supports two path algorithms selected via the ACS configuration:
    - QUATERNION (default): full SO(3) SLERP coupling pointing and roll changes.
    - CONSTRAINT_AVOIDING: generalized constraint-avoiding SLERP.
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

        CONSTRAINT_AVOIDING:
            Generalized constraint-avoiding SLERP using the combined rust-ephem
            constraint configuration.  Routes around any combination of Sun,
            Earth, Moon, and other exclusion zones.  Falls back to QUATERNION
            when no constraint violation is detected on the direct arc.

        In all cases self.slewdist is the total angular distance (degrees) and
        self.slewpath is the (ra_list, dec_list) path used for interpolation.
        """
        steps = 100
        if self.acs_config.slew_algorithm == SlewAlgorithm.CONSTRAINT_AVOIDING:
            self._predict_slew_constraint_avoiding(steps)
        else:
            self._predict_slew_quaternion(steps)

    def _predict_slew_quaternion(self, steps: int) -> None:
        """Compute slew path via full quaternion SLERP."""

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

        # Compute quaternion angular distance (accounts for both pointing and roll)
        # This is more accurate than great-circle distance for maneuvers with roll changes
        q1 = attitude_to_quat(self.startra, self.startdec, self.startroll)
        q2 = attitude_to_quat(self.endra, self.enddec, self.endroll)

        # Quaternion dot product (ensure shortest path)
        dot = float(np.dot(q1, q2))
        if dot < 0:
            dot = -dot
        dot = min(dot, 1.0)  # Clamp for numerical stability

        # Angular distance in SO(3): theta = arccos(dot)
        # For unit quaternions, the rotation angle is 2*arccos(q·q')
        theta_rad = np.arccos(dot)
        self.slewdist = float(np.rad2deg(2 * theta_rad))

    def _predict_slew_constraint_avoiding(self, steps: int) -> None:
        """Compute constraint-avoiding slew path using quaternion SLERP segments.

        Uses the ACS slew_constraint if configured, otherwise falls back to the
        spacecraft's general pointing constraint.  If any constraint is violated
        (Sun, Earth, Moon, etc.), a waypoint is inserted and the path is built
        from two consecutive SLERP segments.  Falls back to plain quaternion SLERP
        when no violation is detected.
        """
        # Determine which constraint to use: ACS slew_constraint or spacecraft constraint
        slew_constraint = (
            self.acs_config.slew_constraint
            if self.acs_config.slew_constraint is not None
            else self.constraint.constraint
        )

        # Create a constraint check function for the waypoint algorithm
        def check_constraint(ra: float, dec: float, time: float) -> bool:
            """Return True if the constraint is violated at this pointing."""
            if slew_constraint is None:
                return False
            # Round time to nearest ephemeris step to ensure it exists in ephemeris
            # rust-ephem's in_constraint requires exact timestamp matches
            step_size: float = getattr(self.ephem, "step_size", 60)
            rounded_time = round(time / step_size) * step_size
            dt = dtutcfromtimestamp(rounded_time)
            return bool(
                slew_constraint.in_constraint(
                    ephemeris=self.ephem,
                    target_ra=ra,
                    target_dec=dec,
                    time=dt,
                    target_roll=None,
                )
            )

        # Use the generalized constraint-avoiding waypoint algorithm
        waypoint = constraint_avoiding_waypoint(
            self.startra,
            self.startdec,
            self.endra,
            self.enddec,
            self.slewstart,
            check_constraint,
        )

        if waypoint is None:
            # No violation – plain quaternion SLERP
            self._predict_slew_quaternion(steps)
            return

        w_ra, w_dec = waypoint

        # Estimate roll at waypoint by linear interpolation of the total arc fraction

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
