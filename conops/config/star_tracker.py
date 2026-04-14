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

from functools import cached_property

import numpy as np
import numpy.typing as npt
import rust_ephem
from pydantic import Field, field_validator
from rust_ephem import AtLeastConstraint, EarthLimbConstraint, SunConstraint
from rust_ephem.constraints import ConstraintConfig

from ..common.enums import ACSMode
from ..common.vector import radec2vec, rotvec, vec2radec, vecnorm
from ._base import ConfigModel
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


class StarTrackerOrientation(ConfigModel):
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
        """Build the rotation matrix whose columns are the star tracker basis vectors.

        The three columns of the returned matrix are, in order: the boresight
        (ST +X axis), an arbitrarily-chosen perpendicular axis (ST +Y), and their
        cross-product (ST +Z), all expressed in **spacecraft body-frame**
        coordinates.

        Interpretation
        --------------
        Because the columns are the ST basis vectors written in body-frame
        coordinates, the matrix ``R`` maps from **ST frame to body frame**::

            v_body = R @ v_st

        To transform a body-frame vector into star-tracker-frame coordinates
        (i.e. to obtain the components that the star tracker "sees"), use the
        **transpose** (which equals the inverse for an orthonormal matrix)::

            v_st = R.T @ v_body

        This is the convention used in :meth:`transform_pointing`.

        Returns
        -------
        ndarray, shape (3, 3)
            Orthonormal rotation matrix (R_ST←body⁻¹, equivalently R_body←ST).
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
        second = vecnorm(second)

        # Third axis: perpendicular to both boresight and second
        third = np.cross(bore, second)
        third = vecnorm(third)

        # Compose rotation matrix: columns are the new basis vectors
        return np.column_stack([bore, second, third])

    def transform_pointing(
        self, ra_deg: float, dec_deg: float, roll_deg: float = 0.0
    ) -> tuple[float, float]:
        """Transform spacecraft attitude to star-tracker inertial pointing.

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
            If the star tracker boresight points along +X (default), and we point the
            spacecraft at (RA=45°, Dec=30°), the tracker boresight inertial direction
            (and thus the returned (RA, Dec)) is unchanged by roll about +X; roll=0°
            and roll=90° give the same (RA, Dec) for the boresight. Roll only changes
            the apparent (RA, Dec) for trackers whose boresight is offset from +X.
        """
        # Spacecraft boresight (+X body axis) in inertial coordinates.
        ra_rad = np.deg2rad(ra_deg)
        dec_rad = np.deg2rad(dec_deg)
        x_hat = radec2vec(ra_rad, dec_rad)

        # Build the body Y/Z basis around boresight using sky north as reference.
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y0 = np.cross(ref, x_hat)
        if np.linalg.norm(y0) < 1e-12:
            # Near celestial poles, choose a different reference for numerical stability.
            y0 = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), x_hat)
        y0 = vecnorm(y0)
        z0 = vecnorm(np.cross(x_hat, y0))

        # Roll is a position-angle rotation about boresight (+X).
        roll_rad = np.deg2rad(roll_deg)
        c = np.cos(roll_rad)
        s = np.sin(roll_rad)
        y_hat = y0 * c - z0 * s
        z_hat = y0 * s + z0 * c

        # Tracker boresight in inertial frame: linear combination of body axes.
        b = np.asarray(self.boresight, dtype=np.float64)
        v_st = b[0] * x_hat + b[1] * y_hat + b[2] * z_hat

        # Convert back to RA/Dec
        ra_st_rad, dec_st_rad = vec2radec(v_st)

        return np.rad2deg(ra_st_rad), np.rad2deg(dec_st_rad)


class StarTracker(ConfigModel):
    """Configuration for a single star tracker.

    Attributes:
        name: Name/identifier for this star tracker
        orientation: Orientation relative to spacecraft body frame
        hard_constraint: Hard constraint - star tracker cannot look in these directions
        soft_constraint: Soft constraint - star tracker works but may have degraded performance
    """

    name: str = "StarTracker"
    orientation: StarTrackerOrientation = Field(default_factory=StarTrackerOrientation)
    hard_constraint: Constraint | None = None
    soft_constraint: Constraint | None = None

    def set_ephem(self, ephem: rust_ephem.Ephemeris) -> None:
        """Set ephemeris on constraint objects.

        Args:
            ephem: Ephemeris object for constraint calculations
        """
        if self.hard_constraint is not None:
            self.hard_constraint.ephem = ephem
        if self.soft_constraint is not None:
            self.soft_constraint.ephem = ephem

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
        return self.hard_constraint.in_constraint(
            ra_st, dec_st, utime, target_roll=roll_deg
        )

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
        return self.soft_constraint.in_constraint(
            ra_st, dec_st, utime, target_roll=roll_deg
        )


class StarTrackerConfiguration(ConfigModel):
    """Configuration for star tracker subsystem on spacecraft.

    Manages multiple star trackers with independent orientations and constraints.

    Attributes:
        star_trackers: List of star trackers on the spacecraft
        min_functional_trackers: Minimum number of star trackers that must be
            outside their **soft** constraint zone for the pointing to be
            considered valid for science (i.e. operationally functional /
            tracking well).  A tracker in its soft constraint is degraded but
            not physically endangered.  If 0, soft constraints are not enforced.
            Default is 1.
            Hard constraints (physical safety keep-outs) are always OR-combined
            and enforced unconditionally — ``min_functional_trackers`` does not
            apply to them.
        modes_require_lock: Operational modes that require star tracker lock quality
            (i.e. where **soft** constraints are enforced as a science-quality check).
            Hard constraints — absolute health-and-safety keep-outs — are always
            enforced in every mode regardless of this setting.
            None means soft constraints are enforced in all modes (conservative default).
            Empty list means soft constraints are never enforced.
            E.g. [ACSMode.SCIENCE] means soft constraints are only checked during science.
        startracker_constraint: Computed observing constraint built from all star
            tracker soft constraints, with boresight offsets applied.
    """

    star_trackers: list[StarTracker] = Field(default_factory=list)
    min_functional_trackers: int = 1
    modes_require_lock: list[ACSMode] | None = None

    def requires_lock_in_mode(self, mode: ACSMode | None = None) -> bool:
        """Check whether star tracker lock is required in the given operational mode.

        Args:
            mode: Operational mode integer. If None, checks nominal/unspecified mode.

        Returns:
            True if lock is required in this mode, False otherwise.
        """
        if self.modes_require_lock is None:
            # None means required in all modes
            return True
        if mode is None:
            # Nominal mode not in an explicit list
            return False
        return mode in self.modes_require_lock

    @staticmethod
    def _boresight_to_euler_deg(
        boresight: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Convert boresight unit vector to azimuth/elevation angles for boresight_offset.

        Returns (roll=0, pitch, yaw) angles such that
        ``boresight_offset(roll, pitch, yaw)`` of a constraint shifts the
        constraint from its natural +X direction to the direction of
        ``boresight`` in the spacecraft body frame.

        Concretely: yaw is the azimuthal angle of the boresight in the body
        xy-plane, and pitch is its elevation above that plane — the standard
        spherical-coordinate decomposition that ``boresight_offset`` expects.
        Roll about the boresight is underdetermined; we set it to 0.
        """
        x, y, z = vecnorm(np.asarray(boresight, dtype=np.float64))
        yaw_deg = float(np.rad2deg(np.arctan2(y, x)))
        pitch_deg = float(np.rad2deg(np.arctan2(z, np.hypot(x, y))))
        roll_deg = 0.0
        return roll_deg, pitch_deg, yaw_deg

    @cached_property
    def startracker_hard_constraint(self) -> ConstraintConfig | None:
        """Combined hard constraint from all star trackers.

        Hard constraints are OR-combined: if **any** tracker is in its hard
        exclusion zone, the pointing is invalid.  These are absolute
        health-and-safety keep-outs (sensor blinding, Earth-limb staring, etc.)
        and are always enforced regardless of ``modes_require_lock``.

        ``min_functional_trackers`` does **not** apply here — a tracker being
        physically endangered is never acceptable, regardless of redundancy.
        """
        combined: ConstraintConfig | None = None

        for st in self.star_trackers:
            if st.hard_constraint is None:
                continue

            base_constraint = st.hard_constraint.constraint
            if base_constraint is None:
                continue

            roll_deg, pitch_deg, yaw_deg = self._boresight_to_euler_deg(
                st.orientation.boresight
            )
            offset_constraint = base_constraint.boresight_offset(
                roll_deg=roll_deg,
                pitch_deg=pitch_deg,
                yaw_deg=yaw_deg,
            )

            if combined is None:
                combined = offset_constraint
            else:
                combined = combined | offset_constraint

        return combined

    @cached_property
    def startracker_constraint(self) -> ConstraintConfig | None:
        """Combined observing constraint from all star tracker soft constraints.

        Each star tracker soft constraint is wrapped with a boresight offset so it is
        evaluated in the star tracker frame but queried in spacecraft pointing frame.
        The combined violation condition is based on ``min_functional_trackers``:
        the constraint is violated only when enough trackers violate their soft
        constraints that the minimum functional requirement can no longer be met.
        """
        offset_constraints: list[ConstraintConfig] = []
        if not self.requires_lock_in_mode(ACSMode.SCIENCE):
            return None
        required_trackers = self.star_trackers
        total_trackers = len(required_trackers)

        for st in required_trackers:
            if st.soft_constraint is None:
                continue

            base_constraint = st.soft_constraint.constraint
            if base_constraint is None:
                continue
            roll_deg, pitch_deg, yaw_deg = self._boresight_to_euler_deg(
                st.orientation.boresight
            )
            offset_constraint = base_constraint.boresight_offset(
                roll_deg=roll_deg,
                pitch_deg=pitch_deg,
                yaw_deg=yaw_deg,
            )
            offset_constraints.append(offset_constraint)

        if not offset_constraints:
            return None

        if self.min_functional_trackers <= 0:
            return None

        required_violations = total_trackers - self.min_functional_trackers + 1
        if required_violations <= 0:
            return None

        if required_violations > len(offset_constraints):
            return None

        return AtLeastConstraint(
            min_violated=required_violations,
            constraints=offset_constraints,
        )

    def set_ephem(self, ephem: rust_ephem.Ephemeris) -> None:
        """Set ephemeris on all star tracker constraint objects.

        Args:
            ephem: Ephemeris object for constraint calculations
        """
        for st in self.star_trackers:
            st.set_ephem(ephem)

    def num_trackers(self) -> int:
        """Get the number of star trackers.

        Returns:
            Number of star trackers configured
        """
        return len(self.star_trackers)

    def trackers_violating_hard_constraints(
        self,
        ra_deg: float,
        dec_deg: float,
        utime: float,
        roll_deg: float = 0.0,
        mode: ACSMode | None = None,
    ) -> int:
        """Count how many star trackers violate hard constraints.

        Hard constraints are absolute health-and-safety keep-outs (e.g. blinding
        a sensor with the Sun) and are **always** evaluated regardless of
        ``modes_require_lock``.  The ``mode`` parameter is accepted for API
        compatibility but has no effect here.

        Args:
            ra_deg: Right ascension in spacecraft frame, degrees
            dec_deg: Declination in spacecraft frame, degrees
            utime: Unix timestamp
            roll_deg: Spacecraft roll angle in degrees. Default is 0.
            mode: Unused; accepted for API compatibility only.

        Returns:
            Number of star trackers that violate hard constraints
        """
        count = 0
        for st in self.star_trackers:
            if st.in_hard_constraint(ra_deg, dec_deg, utime, roll_deg):
                count += 1
        return count

    def any_tracker_violating_soft_constraints(
        self,
        ra_deg: float,
        dec_deg: float,
        utime: float,
        roll_deg: float = 0.0,
        mode: ACSMode | None = None,
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
        if not self.requires_lock_in_mode(mode):
            return False
        for st in self.star_trackers:
            if st.in_soft_constraint(ra_deg, dec_deg, utime, roll_deg):
                return True
        return False

    def trackers_violating_soft_constraints(
        self,
        ra_deg: float,
        dec_deg: float,
        utime: float,
        roll_deg: float = 0.0,
        mode: ACSMode | None = None,
    ) -> int:
        """Count how many star trackers violate soft constraints.

        A tracker violating its soft constraint is degraded — not operating at
        full science quality.  This count drives ``star_tracker_functional_count``
        in Housekeeping telemetry: a tracker that is in soft constraint is
        *not* considered functional.

        Args:
            ra_deg: Right ascension in spacecraft frame, degrees
            dec_deg: Declination in spacecraft frame, degrees
            utime: Unix timestamp
            roll_deg: Spacecraft roll angle in degrees. Default is 0.
            mode: Operational mode; soft constraints are skipped in modes where
                lock is not required (per ``modes_require_lock``).

        Returns:
            Number of star trackers violating soft constraints.
        """
        if not self.requires_lock_in_mode(mode):
            return 0
        count = 0
        for st in self.star_trackers:
            if st.in_soft_constraint(ra_deg, dec_deg, utime, roll_deg):
                count += 1
        return count

    def is_pointing_valid(
        self,
        ra_deg: float,
        dec_deg: float,
        utime: float,
        roll_deg: float = 0.0,
        mode: ACSMode | None = None,
    ) -> bool:
        """Check if pointing is valid considering hard constraints and the
        minimum number of operational trackers.

        A pointing is valid when both conditions hold:

        1. **No** star tracker is in a hard constraint (absolute health-and-safety
           keep-out; always enforced regardless of ``modes_require_lock``).
        2. The number of trackers counted as *functional* is at least
           ``min_functional_trackers``.  A tracker is considered non-functional
           when it is in its soft-constraint zone **and** lock is required in the
           current ``mode`` (per ``modes_require_lock``).  Soft constraints
           represent science-quality degradation, not safety, so they are skipped
           in modes where lock quality is not required.

        Args:
            ra_deg: Right ascension in spacecraft frame, degrees
            dec_deg: Declination in spacecraft frame, degrees
            utime: Unix timestamp
            roll_deg: Spacecraft roll angle in degrees. Default is 0.
            mode: Operational mode. Controls which trackers must have a lock
                (see :attr:`~StarTrackerConfiguration.modes_require_lock`). Pass ``None``
                for nominal/unspecified mode.

        Returns:
            True if pointing is valid, False otherwise
        """
        if len(self.star_trackers) == 0:
            # No star trackers configured - allow all pointings
            return True

        # Hard constraints are absolute health-and-safety keep-outs — always
        # enforced regardless of modes_require_lock.
        hard_violations = sum(
            1
            for st in self.star_trackers
            if st.in_hard_constraint(ra_deg, dec_deg, utime, roll_deg)
        )
        if hard_violations > 0:
            return False

        # Soft constraints represent science-quality degradation and are only
        # enforced in modes where star tracker lock is required.
        if not self.requires_lock_in_mode(mode):
            return True

        soft_violations = sum(
            1
            for st in self.star_trackers
            if st.in_soft_constraint(ra_deg, dec_deg, utime, roll_deg)
        )
        functional_trackers = len(self.star_trackers) - soft_violations
        required_functional = min(self.min_functional_trackers, len(self.star_trackers))
        return functional_trackers >= required_functional

    def check_soft_constraint_degradation(
        self,
        ra_deg: float,
        dec_deg: float,
        utime: float,
        roll_deg: float = 0.0,
        mode: ACSMode | None = None,
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
            ra_deg,
            dec_deg,
            utime,
            roll_deg,
            mode=mode,
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


class DefaultStarTrackerConfiguration(StarTrackerConfiguration):
    """Star tracker configuration pre-populated with a single default star tracker.

    The default tracker is oriented along +X (aligned with the spacecraft boresight)
    with soft constraints for Earth limb (min_angle=0) and Sun (min_angle=20 deg)
    avoidance.
    """

    star_trackers: list[StarTracker] = Field(
        default_factory=lambda: [
            StarTracker(
                orientation=StarTrackerOrientation(boresight=(1.0, 0.0, 0.0)),
                soft_constraint=Constraint(
                    earth_constraint=EarthLimbConstraint(min_angle=0),
                    sun_constraint=SunConstraint(min_angle=20),
                ),
            )
        ]
    )
