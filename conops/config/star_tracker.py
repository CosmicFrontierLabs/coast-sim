"""Star Tracker configuration and constraint management.

This module provides configuration and constraint checking for star trackers mounted
on spacecraft. It supports:
- Multiple star trackers with arbitrary orientations relative to spacecraft body
- Hard constraints (e.g., sun/earth avoidance - star tracker cannot look there)
- Soft constraints (e.g., degraded performance zones - star tracker works but suboptimally)
- Mode-dependent lock requirements
- Minimum number of required functional trackers
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from .constraint import Constraint


class StarTrackerOrientation(BaseModel):
    """Orientation of a star tracker relative to spacecraft body frame.

    The orientation is defined as a rotation from the spacecraft body +Z axis
    (boresight) to the star tracker boresight. Rotations are specified using
    Euler angles: first roll (rotation about X), then pitch (rotation about Y),
    then yaw (rotation about Z).

    Attributes
    ----------
    roll : float
        Rotation about spacecraft X-axis (body frame), in degrees.
    pitch : float
        Rotation about spacecraft Y-axis (body frame), in degrees.
    yaw : float
        Rotation about spacecraft Z-axis (body frame), in degrees.
    """

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def to_rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Convert Euler angles to rotation matrix (intrinsic ZYX convention).

        Returns
        -------
        ndarray
            3x3 rotation matrix transforming vectors from spacecraft body frame
            to star tracker frame.
        """
        # Convert degrees to radians
        roll_rad = np.deg2rad(self.roll)
        pitch_rad = np.deg2rad(self.pitch)
        yaw_rad = np.deg2rad(self.yaw)

        # Rotation matrices
        rx: npt.NDArray[np.float64] = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll_rad), -np.sin(roll_rad)],
                [0, np.sin(roll_rad), np.cos(roll_rad)],
            ]
        )

        ry: npt.NDArray[np.float64] = np.array(
            [
                [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                [0, 1, 0],
                [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
            ]
        )

        rz: npt.NDArray[np.float64] = np.array(
            [
                [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                [0, 0, 1],
            ]
        )

        # Compose: rz @ ry @ rx (intrinsic ZYX)
        return rz @ ry @ rx

    def transform_pointing(self, ra_deg: float, dec_deg: float) -> tuple[float, float]:
        """Transform a pointing from spacecraft frame to star tracker frame.

        Args
        ----
        ra_deg : float
            Right ascension in spacecraft frame, degrees
        dec_deg : float
            Declination in spacecraft frame, degrees

        Returns
        -------
        tuple[float, float]
            (ra, dec) in star tracker frame, degrees
        """
        # Convert RA/Dec to vector
        ra_rad = np.deg2rad(ra_deg)
        dec_rad = np.deg2rad(dec_deg)
        v_body = np.array(
            [
                np.cos(dec_rad) * np.cos(ra_rad),
                np.cos(dec_rad) * np.sin(ra_rad),
                np.sin(dec_rad),
            ]
        )

        # Transform to star tracker frame
        rot_matrix = self.to_rotation_matrix()
        v_st = rot_matrix @ v_body

        # Convert back to RA/Dec
        ra_st_rad = np.arctan2(v_st[1], v_st[0])
        dec_st_rad = np.arcsin(v_st[2])

        return np.rad2deg(ra_st_rad), np.rad2deg(dec_st_rad)


class StarTracker(BaseModel):
    """Configuration for a single star tracker.

    Attributes
    ----------
    name : str
        Name/identifier for this star tracker
    orientation : StarTrackerOrientation
        Orientation relative to spacecraft body frame
    hard_constraint : Constraint, optional
        Hard constraint - star tracker cannot look in these directions
    soft_constraint : Constraint, optional
        Soft constraint - star tracker works but may have degraded performance
    modes_require_lock : list[int]
        List of operational modes (if any) that require this ST to have a lock.
        Empty list means lock is not required in any mode. None means lock is
        required in all modes. Examples:
        - [0, 2]: lock required in modes 0 and 2
        - []: lock never required
        - None: lock required in all modes
    """

    name: str = "StarTracker"
    orientation: StarTrackerOrientation = Field(default_factory=StarTrackerOrientation)
    hard_constraint: Constraint | None = None
    soft_constraint: Constraint | None = None
    modes_require_lock: list[int] | None = None

    def in_hard_constraint(self, ra_deg: float, dec_deg: float, utime: float) -> bool:
        """Check if pointing violates hard constraint.

        Args
        ----
        ra_deg : float
            Right ascension in spacecraft frame, degrees
        dec_deg : float
            Declination in spacecraft frame, degrees
        utime : float
            Unix timestamp

        Returns
        -------
        bool
            True if pointing violates hard constraint, False otherwise
        """
        if self.hard_constraint is None:
            return False

        # Transform pointing to star tracker frame
        ra_st, dec_st = self.orientation.transform_pointing(ra_deg, dec_deg)
        return self.hard_constraint.in_constraint(ra_st, dec_st, utime)

    def in_soft_constraint(self, ra_deg: float, dec_deg: float, utime: float) -> bool:
        """Check if pointing violates soft constraint.

        Args
        ----
        ra_deg : float
            Right ascension in spacecraft frame, degrees
        dec_deg : float
            Declination in spacecraft frame, degrees
        utime : float
            Unix timestamp

        Returns
        -------
        bool
            True if pointing violates soft constraint, False otherwise
        """
        if self.soft_constraint is None:
            return False

        # Transform pointing to star tracker frame
        ra_st, dec_st = self.orientation.transform_pointing(ra_deg, dec_deg)
        return self.soft_constraint.in_constraint(ra_st, dec_st, utime)

    def requires_lock_in_mode(self, mode: int | None = None) -> bool:
        """Check if this star tracker must have a lock in the given mode.

        Args
        ----
        mode : int, optional
            Operational mode. If None, uses nominal mode.

        Returns
        -------
        bool
            True if lock is required in this mode, False otherwise
        """
        if self.modes_require_lock is None:
            # None means required in all modes
            return True
        if mode is None:
            # Nominal mode
            return False
        return mode in self.modes_require_lock


class StarTrackerConfiguration(BaseModel):
    """Configuration for star tracker subsystem on spacecraft.

    Manages multiple star trackers with independent orientations and constraints.

    Attributes
    ----------
    star_trackers : list[StarTracker]
        List of star trackers on the spacecraft
    min_functional_trackers : int
        Minimum number of star trackers that must not violate hard constraints
        for pointing to be valid. If 0, hard constraints are not enforced.
        Default is 1 (at least one star tracker must be functional).
    """

    star_trackers: list[StarTracker] = Field(default_factory=list)
    min_functional_trackers: int = 1

    def num_trackers(self) -> int:
        """Get the number of star trackers.

        Returns
        -------
        int
            Number of star trackers configured
        """
        return len(self.star_trackers)

    def trackers_violating_hard_constraints(
        self, ra_deg: float, dec_deg: float, utime: float
    ) -> int:
        """Count how many star trackers violate hard constraints.

        Args
        ----
        ra_deg : float
            Right ascension in spacecraft frame, degrees
        dec_deg : float
            Declination in spacecraft frame, degrees
        utime : float
            Unix timestamp

        Returns
        -------
        int
            Number of star trackers that violate hard constraints
        """
        count = 0
        for st in self.star_trackers:
            if st.in_hard_constraint(ra_deg, dec_deg, utime):
                count += 1
        return count

    def any_tracker_violating_soft_constraints(
        self, ra_deg: float, dec_deg: float, utime: float
    ) -> bool:
        """Check if any star tracker violates soft constraints.

        Args
        ----
        ra_deg : float
            Right ascension in spacecraft frame, degrees
        dec_deg : float
            Declination in spacecraft frame, degrees
        utime : float
            Unix timestamp

        Returns
        -------
        bool
            True if any star tracker violates soft constraints
        """
        for st in self.star_trackers:
            if st.in_soft_constraint(ra_deg, dec_deg, utime):
                return True
        return False

    def is_pointing_valid(
        self, ra_deg: float, dec_deg: float, utime: float, mode: int | None = None
    ) -> bool:
        """Check if pointing is valid considering hard constraints and minimum trackers.

        A pointing is valid if at least min_functional_trackers star trackers do not
        violate hard constraints.

        Args
        ----
        ra_deg : float
            Right ascension in spacecraft frame, degrees
        dec_deg : float
            Declination in spacecraft frame, degrees
        utime : float
            Unix timestamp
        mode : int, optional
            Operational mode (used for lock requirement checking)

        Returns
        -------
        bool
            True if pointing is valid, False otherwise
        """
        if len(self.star_trackers) == 0:
            # No star trackers configured - allow all pointings
            return True

        violations = self.trackers_violating_hard_constraints(ra_deg, dec_deg, utime)
        functional_trackers = len(self.star_trackers) - violations

        return functional_trackers >= self.min_functional_trackers

    def check_soft_constraint_degradation(
        self, ra_deg: float, dec_deg: float, utime: float
    ) -> bool:
        """Check if pointing results in any soft constraint violations.

        Soft constraints represent performance degradation, not hard failures.

        Args
        ----
        ra_deg : float
            Right ascension in spacecraft frame, degrees
        dec_deg : float
            Declination in spacecraft frame, degrees
        utime : float
            Unix timestamp

        Returns
        -------
        bool
            True if any star tracker will operate at degraded performance
        """
        return self.any_tracker_violating_soft_constraints(ra_deg, dec_deg, utime)

    def get_tracker_by_name(self, name: str) -> StarTracker | None:
        """Get a star tracker by name.

        Args
        ----
        name : str
            Name of the star tracker

        Returns
        -------
        StarTracker or None
            The star tracker if found, None otherwise
        """
        for st in self.star_trackers:
            if st.name == name:
                return st
        return None
