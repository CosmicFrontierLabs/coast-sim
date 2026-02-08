"""Star Tracker configuration and constraint management.

This module provides configuration and constraint checking for star trackers mounted
on spacecraft. It supports:
- Multiple star trackers with arbitrary orientations relative to spacecraft body
- Hard constraints (e.g., sun/earth avoidance - star tracker cannot look there)
- Soft constraints (e.g., degraded performance zones - star tracker works but suboptimally)
- Mode-dependent lock requirements
- Minimum number of required functional trackers

Helper Functions:
    create_star_tracker_vector: Convert Euler angles (roll/pitch/yaw) to boresight
        vector for convenient creation of oriented star trackers.

Example:
    >>> from conops.config import StarTracker, StarTrackerOrientation
    >>> from conops.config.star_tracker import create_star_tracker_vector
    >>>
    >>> # Create with explicit boresight vector
    >>> ori1 = StarTrackerOrientation(boresight=(1.0, 0.0, 0.0))
    >>>
    >>> # Or create using Euler angles for convenience
    >>> boresight = create_star_tracker_vector(roll_deg=0, pitch_deg=45, yaw_deg=0)
    >>> ori2 = StarTrackerOrientation(boresight=boresight)
    >>>
    >>> st = StarTracker(name="ST1", orientation=ori2)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, field_validator

from ..common.vector import rotvec, vecnorm
from .constraint import Constraint


def create_star_tracker_vector(
    roll_deg: float = 0.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0
) -> tuple[float, float, float]:
    """Create a star tracker boresight vector from Euler angles.

    Converts roll, pitch, yaw angles to a unit boresight vector using the ZYX
    Euler angle convention (yaw about Z, then pitch about Y, then roll about X).

    Args:
        roll_deg: Rotation about X axis (degrees). Default is 0.
        pitch_deg: Rotation about Y axis (degrees). Default is 0.
        yaw_deg: Rotation about Z axis (degrees). Default is 0.

    Returns:
        Boresight vector (x, y, z) as a unit vector pointing in the direction
        from the combined rotations applied to the default +X (1, 0, 0) direction.

    Example:
        create_star_tracker_vector(0, 0, 0)  # Returns (1, 0, 0) - default boresight
        create_star_tracker_vector(0, 90, 0)  # Returns (0, 0, 1) - points along +Z
        create_star_tracker_vector(0, 0, 90)  # Returns (0, 1, 0) - points along +Y
    """
    # Convert angles to radians
    roll_rad = np.deg2rad(roll_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad = np.deg2rad(yaw_deg)

    # Start with default boresight along +X
    boresight = np.array([1.0, 0.0, 0.0])

    # Apply rotations in ZYX order: yaw (Z), then pitch (Y), then roll (X)
    boresight = rotvec(3, yaw_rad, boresight)  # Yaw around Z
    boresight = rotvec(2, pitch_rad, boresight)  # Pitch around Y
    boresight = rotvec(1, roll_rad, boresight)  # Roll around X

    # Normalize to ensure unit vector
    boresight = vecnorm(boresight)

    return tuple(boresight)


class StarTrackerOrientation(BaseModel):
    """Orientation of a star tracker boresight in spacecraft body frame.

    The boresight direction defines where the star tracker looks. It is represented
    as a unit normal vector in the spacecraft body frame, similar to solar panels.

    Attributes:
        boresight: Star tracker boresight direction as a unit vector (x, y, z) where:
            - +x is the spacecraft pointing direction (boresight/forward)
            - +y is the spacecraft "up" direction
            - +z completes the right-handed coordinate system
            Default (1, 0, 0) points along spacecraft boresight.
    """

    boresight: tuple[float, float, float] = Field(
        default=(1.0, 0.0, 0.0),
        description="Star tracker boresight direction as unit vector in body frame",
    )

    @field_validator("boresight")
    @classmethod
    def validate_unit_vector(
        cls, v: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Validate that boresight is a unit vector."""
        magnitude = np.sqrt(sum(x**2 for x in v))
        if magnitude < 0.99 or magnitude > 1.01:
            raise ValueError(
                f"Boresight must be a unit vector. Got magnitude {magnitude}"
            )
        return v

    def to_rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Convert boresight vector to rotation matrix.

        Returns a 3x3 rotation matrix that transforms vectors from spacecraft body frame
        to star tracker frame, where the star tracker +X axis is aligned with the boresight.

        Returns:
            3x3 rotation matrix transforming vectors from spacecraft body frame
            to star tracker frame.
        """
        # Boresight is the first column of the rotation matrix
        bore = np.array(self.boresight, dtype=np.float64)

        # Construct orthonormal basis
        # Choose second axis perpendicular to boresight
        # If boresight is not parallel to Z, use cross product with Z
        if abs(bore[2]) < 0.99:  # Not aligned with Z
            # Second axis: perpendicular to both boresight and Z-axis
            second = np.cross(np.array([0.0, 0.0, 1.0]), bore)
        else:
            # Boresight is parallel to Z, use X-axis instead
            second = np.cross(np.array([1.0, 0.0, 0.0]), bore)

        # Normalize second axis
        second = second / np.linalg.norm(second)

        # Third axis: perpendicular to both boresight and second
        third = np.cross(bore, second)
        third = third / np.linalg.norm(third)

        # Compose rotation matrix: columns are the new basis vectors
        return np.column_stack([bore, second, third])

    def transform_pointing(
        self, ra_deg: float, dec_deg: float, roll_deg: float = 0.0
    ) -> tuple[float, float]:
        """Transform spacecraft frame pointing to star tracker's local frame.

        Given a pointing direction in the spacecraft frame (RA/Dec), this method
        computes what that same direction looks like from the star tracker's
        perspective, accounting for:
        1. The spacecraft's roll rotation about its boresight axis
        2. The star tracker's offset orientation (boresight vector) relative to
           the spacecraft body

        This is used to convert pointing commands into the star tracker's local
        coordinates for constraint checking (e.g., sun avoidance).

        Args:
            ra_deg: Right ascension of pointing in spacecraft frame (degrees)
            dec_deg: Declination of pointing in spacecraft frame (degrees)
            roll_deg: Spacecraft roll angle about its +X (boresight) axis (degrees).
                Default is 0. Roll rotates the RA/Dec pointing in the Y-Z plane.

        Returns:
            (ra, dec) of the pointing in the star tracker's local frame (degrees).
            This is what RA/Dec the star tracker would need to look at in its own
            frame of reference to observe the spacecraft's input pointing direction.

        Example:
            If star tracker boresight points along +X (default), and we point the
            spacecraft at (RA=45°, Dec=30°) with roll=0°, we get back (45°, 30°).
            If the same spacecraft pointing happens while roll=90°, the star tracker
            frame sees a different (RA, Dec).
        """
        # Convert RA/Dec to vector in spacecraft frame
        ra_rad = np.deg2rad(ra_deg)
        dec_rad = np.deg2rad(dec_deg)
        v_sc = np.array(
            [
                np.cos(dec_rad) * np.cos(ra_rad),
                np.cos(dec_rad) * np.sin(ra_rad),
                np.sin(dec_rad),
            ]
        )

        # Apply spacecraft roll to get vector in spacecraft body frame
        roll_rad = np.deg2rad(roll_deg)
        cos_roll = np.cos(roll_rad)
        sin_roll = np.sin(roll_rad)
        roll_matrix: npt.NDArray[np.float64] = np.array(
            [
                [1, 0, 0],
                [0, cos_roll, -sin_roll],
                [0, sin_roll, cos_roll],
            ]
        )
        v_body = roll_matrix @ v_sc

        # Transform to star tracker frame
        rot_matrix = self.to_rotation_matrix()
        v_st = rot_matrix @ v_body

        # Convert back to RA/Dec
        ra_st_rad = np.arctan2(v_st[1], v_st[0])
        dec_st_rad = np.arcsin(v_st[2])

        return np.rad2deg(ra_st_rad), np.rad2deg(dec_st_rad)


class StarTracker(BaseModel):
    """Configuration for a single star tracker.

    Attributes:
        name: Name/identifier for this star tracker
        orientation: Orientation relative to spacecraft body frame
        hard_constraint: Hard constraint - star tracker cannot look in these directions
        soft_constraint: Soft constraint - star tracker works but may have degraded performance
        modes_require_lock: List of operational modes (if any) that require this ST to have a lock.
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

    def in_hard_constraint(
        self, ra_deg: float, dec_deg: float, utime: float, roll_deg: float = 0.0
    ) -> bool:
        """Check if pointing violates hard constraint.

        Args:
            ra_deg: Right ascension in spacecraft frame, degrees
            dec_deg: Declination in spacecraft frame, degrees
            utime: Unix timestamp
            roll_deg: Spacecraft roll angle in degrees. Default is 0.

        Returns:
            True if pointing violates hard constraint, False otherwise
        """
        if self.hard_constraint is None:
            return False

        # Transform pointing to star tracker frame
        ra_st, dec_st = self.orientation.transform_pointing(ra_deg, dec_deg, roll_deg)
        return self.hard_constraint.in_constraint(ra_st, dec_st, utime)

    def in_soft_constraint(
        self, ra_deg: float, dec_deg: float, utime: float, roll_deg: float = 0.0
    ) -> bool:
        """Check if pointing violates soft constraint.

        Args:
            ra_deg: Right ascension in spacecraft frame, degrees
            dec_deg: Declination in spacecraft frame, degrees
            utime: Unix timestamp
            roll_deg: Spacecraft roll angle in degrees. Default is 0.

        Returns:
            True if pointing violates soft constraint, False otherwise
        """
        if self.soft_constraint is None:
            return False

        # Transform pointing to star tracker frame
        ra_st, dec_st = self.orientation.transform_pointing(ra_deg, dec_deg, roll_deg)
        return self.soft_constraint.in_constraint(ra_st, dec_st, utime)

    def requires_lock_in_mode(self, mode: int | None = None) -> bool:
        """Check if this star tracker must have a lock in the given mode.

        Args:
            mode: Operational mode. If None, uses nominal mode.

        Returns:
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

    Attributes:
        star_trackers: List of star trackers on the spacecraft
        min_functional_trackers: Minimum number of star trackers that must not violate hard
            constraints for pointing to be valid. If 0, hard constraints are not enforced.
            Default is 1 (at least one star tracker must be functional).
    """

    star_trackers: list[StarTracker] = Field(default_factory=list)
    min_functional_trackers: int = 1

    def num_trackers(self) -> int:
        """Get the number of star trackers.

        Returns:
            Number of star trackers configured
        """
        return len(self.star_trackers)

    def trackers_violating_hard_constraints(
        self, ra_deg: float, dec_deg: float, utime: float, roll_deg: float = 0.0
    ) -> int:
        """Count how many star trackers violate hard constraints.

        Args:
            ra_deg: Right ascension in spacecraft frame, degrees
            dec_deg: Declination in spacecraft frame, degrees
            utime: Unix timestamp
            roll_deg: Spacecraft roll angle in degrees. Default is 0.

        Returns:
            Number of star trackers that violate hard constraints
        """
        count = 0
        for st in self.star_trackers:
            if st.in_hard_constraint(ra_deg, dec_deg, utime, roll_deg):
                count += 1
        return count

    def any_tracker_violating_soft_constraints(
        self, ra_deg: float, dec_deg: float, utime: float, roll_deg: float = 0.0
    ) -> bool:
        """Check if any star tracker violates soft constraints.

        Args:
            ra_deg: Right ascension in spacecraft frame, degrees
            dec_deg: Declination in spacecraft frame, degrees
            utime: Unix timestamp
            roll_deg: Spacecraft roll angle in degrees. Default is 0.

        Returns:
            True if any star tracker violates soft constraints
        """
        for st in self.star_trackers:
            if st.in_soft_constraint(ra_deg, dec_deg, utime, roll_deg):
                return True
        return False

    def is_pointing_valid(
        self,
        ra_deg: float,
        dec_deg: float,
        utime: float,
        roll_deg: float = 0.0,
        mode: int | None = None,
    ) -> bool:
        """Check if pointing is valid considering hard constraints and minimum trackers.

        A pointing is valid if at least min_functional_trackers star trackers do not
        violate hard constraints.

        Args:
            ra_deg: Right ascension in spacecraft frame, degrees
            dec_deg: Declination in spacecraft frame, degrees
            utime: Unix timestamp
            roll_deg: Spacecraft roll angle in degrees. Default is 0.
            mode: Operational mode (used for lock requirement checking)

        Returns:
            True if pointing is valid, False otherwise
        """
        if len(self.star_trackers) == 0:
            # No star trackers configured - allow all pointings
            return True

        violations = self.trackers_violating_hard_constraints(
            ra_deg, dec_deg, utime, roll_deg
        )
        functional_trackers = len(self.star_trackers) - violations

        return functional_trackers >= self.min_functional_trackers

    def check_soft_constraint_degradation(
        self, ra_deg: float, dec_deg: float, utime: float, roll_deg: float = 0.0
    ) -> bool:
        """Check if pointing results in any soft constraint violations.

        Soft constraints represent performance degradation, not hard failures.

        Args:
            ra_deg: Right ascension in spacecraft frame, degrees
            dec_deg: Declination in spacecraft frame, degrees
            utime: Unix timestamp
            roll_deg: Spacecraft roll angle in degrees. Default is 0.

        Returns:
            True if any star tracker will operate at degraded performance
        """
        return self.any_tracker_violating_soft_constraints(
            ra_deg, dec_deg, utime, roll_deg
        )

    def get_tracker_by_name(self, name: str) -> StarTracker | None:
        """Get a star tracker by name.

        Args:
            name: Name of the star tracker

        Returns:
            The star tracker if found, None otherwise
        """
        for st in self.star_trackers:
            if st.name == name:
                return st
        return None
