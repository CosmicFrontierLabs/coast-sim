import numpy as np
from pydantic import ConfigDict, Field, field_validator
from rust_ephem.constraints import ConstraintConfig

from ..common import separation
from ..common.enums import SlewAlgorithm
from ._base import ConfigModel
from .constants import DTOR
from .momentum import StoredMomentumConfig


class AttitudeControlSystem(ConfigModel):
    """
    Attitude Control System (ACS) configuration parameters.

    Defines slew performance characteristics including acceleration,
    maximum slew rate, accuracy, and settling time.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    slew_acceleration: float = Field(
        default=0.5, description="Maximum angular acceleration in degrees/second²"
    )
    max_slew_rate: float = Field(
        default=0.25, description="Maximum slew rate in degrees/second (15 deg/min)"
    )
    slew_acceleration_body: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "Optional (+X, +Y, +Z) body-axis angular acceleration limits in "
            "degrees/second². Values define an ellipsoidal coupled-axis envelope "
            "and override slew_acceleration when the maneuver axis is known."
        ),
    )
    max_slew_rate_body: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "Optional (+X, +Y, +Z) body-axis slew-rate limits in degrees/second. "
            "Values define an ellipsoidal coupled-axis envelope and override "
            "max_slew_rate when the maneuver axis is known."
        ),
    )
    slew_accuracy: float = Field(
        default=0.01, description="Pointing accuracy after slew completion in degrees"
    )
    settle_time: float = Field(
        default=120.0, description="Time to settle after slew completion in seconds"
    )
    stored_momentum: StoredMomentumConfig = Field(
        default_factory=StoredMomentumConfig,
        description="Optional planning-level stored-momentum tracking configuration.",
    )
    slew_algorithm: SlewAlgorithm = Field(
        default=SlewAlgorithm.QUATERNION,
        description=(
            "Algorithm used to compute slew paths. "
            "'quaternion' (default): full SO(3) SLERP coupling pointing and roll. "
            "'constraint_avoiding': quaternion SLERP with automatic detour around "
            "any configured slew constraint (uses slew_constraint if set, otherwise "
            "falls back to the spacecraft's general pointing constraint)."
        ),
    )
    slew_constraint: ConstraintConfig | None = Field(
        default=None,
        description=(
            "Optional rust-ephem ConstraintConfig for slew path planning. "
            "When set and slew_algorithm is CONSTRAINT_AVOIDING, this constraint "
            "is used to determine waypoints during slews. If None, the spacecraft's "
            "general pointing constraint is used instead. This allows different "
            "constraints for slewing vs. science pointing (e.g., stricter Earth limb "
            "avoidance during slews, or relaxed Sun angle limits for quick transits)."
        ),
    )
    gsp_tracking_phase_step_deg: float = Field(
        default=5.0,
        gt=0.0,
        le=180.0,
        description=(
            "Ground-station-pass roll phase search increment in degrees. "
            "Smaller values search more candidate tracking attitudes at higher CPU cost."
        ),
    )

    @field_validator("slew_acceleration_body", "max_slew_rate_body")
    @classmethod
    def _validate_body_axis_limits(
        cls, limits: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        if limits is None:
            return None
        if not all(np.isfinite(value) and value > 0.0 for value in limits):
            raise ValueError("body-axis slew limits must be finite and positive")
        return float(limits[0]), float(limits[1]), float(limits[2])

    @staticmethod
    def _effective_directional_limit(
        scalar_limit: float,
        body_limits: tuple[float, float, float] | None,
        rotation_axis_body: tuple[float, float, float] | np.ndarray | None,
    ) -> float:
        """Resolve an ellipsoidal body-axis envelope along a maneuver axis."""
        if body_limits is None:
            return float(scalar_limit)
        if rotation_axis_body is None:
            raise ValueError(
                "rotation_axis_body is required when body-axis slew limits are configured"
            )

        axis = np.asarray(rotation_axis_body, dtype=np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("rotation_axis_body must contain three finite values")
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            raise ValueError("rotation_axis_body must have nonzero magnitude")
        unit_axis = axis / norm
        limits = np.asarray(body_limits, dtype=np.float64)
        return float(1.0 / np.linalg.norm(unit_axis / limits))

    @property
    def direction_dependent_slew(self) -> bool:
        """Whether any slew kinematic limit depends on the maneuver axis."""
        return (
            self.slew_acceleration_body is not None
            or self.max_slew_rate_body is not None
        )

    def effective_slew_acceleration(
        self,
        rotation_axis_body: tuple[float, float, float] | np.ndarray | None = None,
    ) -> float:
        """Return the acceleration limit along a body-frame maneuver axis."""
        return self._effective_directional_limit(
            self.slew_acceleration,
            self.slew_acceleration_body,
            rotation_axis_body,
        )

    def effective_max_slew_rate(
        self,
        rotation_axis_body: tuple[float, float, float] | np.ndarray | None = None,
    ) -> float:
        """Return the slew-rate limit along a body-frame maneuver axis."""
        return self._effective_directional_limit(
            self.max_slew_rate,
            self.max_slew_rate_body,
            rotation_axis_body,
        )

    def motion_time(
        self,
        angle_deg: float,
        rotation_axis_body: tuple[float, float, float] | np.ndarray | None = None,
    ) -> float:
        """Time to complete the motion (excluding settle) under bang-bang control."""
        if angle_deg <= 0:
            return 0.0
        a = self.effective_slew_acceleration(rotation_axis_body)
        vmax = self.effective_max_slew_rate(rotation_axis_body)
        if a <= 0 or vmax <= 0:
            return 0.0
        t_accel = vmax / a
        d_accel = 0.5 * a * t_accel**2
        if 2 * d_accel >= angle_deg:
            # Triangular profile
            t_peak = (angle_deg / a) ** 0.5
            return float(2 * t_peak)
        # Trapezoidal profile
        d_cruise = angle_deg - 2 * d_accel
        t_cruise = d_cruise / vmax
        return float(2 * t_accel + t_cruise)

    def max_motion_angle(
        self,
        duration_s: float,
        rotation_axis_body: tuple[float, float, float] | np.ndarray | None = None,
    ) -> float:
        """Maximum rest-to-rest angular motion in the available time."""
        if duration_s <= 0:
            return 0.0
        a = self.effective_slew_acceleration(rotation_axis_body)
        vmax = self.effective_max_slew_rate(rotation_axis_body)
        if a <= 0 or vmax <= 0:
            return 0.0
        t_accel = vmax / a
        if duration_s <= 2 * t_accel:
            return float(0.25 * a * duration_s**2)
        return float(vmax * (duration_s - t_accel))

    def s_of_t(
        self,
        angle_deg: float,
        t: float,
        rotation_axis_body: tuple[float, float, float] | np.ndarray | None = None,
    ) -> float:
        """Distance traveled (deg) along the slew after t seconds under bang-bang control.

        Clamps to [0, angle_deg] and ignores settle time (i.e., assumes t is measured
        from slew start; after motion is done, returns full angle).
        """
        if angle_deg <= 0 or t <= 0:
            return 0.0
        a = self.effective_slew_acceleration(rotation_axis_body)
        vmax = self.effective_max_slew_rate(rotation_axis_body)
        if a <= 0 or vmax <= 0:
            return min(max(0.0, t * vmax), angle_deg)  # best-effort fallback

        # Determine profile
        t_accel = vmax / a
        d_accel = 0.5 * a * t_accel**2
        if 2 * d_accel >= angle_deg:
            # Triangular
            t_peak = (angle_deg / a) ** 0.5
            motion_time = 2 * t_peak
            tau = max(0.0, min(float(t), motion_time))
            if tau <= t_peak:
                s = 0.5 * a * tau**2
            else:
                s = angle_deg - 0.5 * a * (motion_time - tau) ** 2
            return float(max(0.0, min(angle_deg, s)))

        # Trapezoidal
        d_cruise = angle_deg - 2 * d_accel
        t_cruise = d_cruise / vmax
        motion_time = 2 * t_accel + t_cruise
        tau = max(0.0, min(float(t), motion_time))
        if tau <= t_accel:
            s = 0.5 * a * tau**2
        elif tau <= t_accel + t_cruise:
            s = d_accel + vmax * (tau - t_accel)
        else:
            t_dec = tau - (t_accel + t_cruise)
            s = d_accel + d_cruise + vmax * t_dec - 0.5 * a * t_dec**2
        return float(max(0.0, min(angle_deg, s)))

    def slew_time(
        self,
        angle_deg: float,
        rotation_axis_body: tuple[float, float, float] | np.ndarray | None = None,
    ) -> float:
        """Total slew time (motion + settle) using bang-bang control."""
        if angle_deg <= 0 or np.isnan(angle_deg):
            return 0.0
        return self.motion_time(angle_deg, rotation_axis_body) + self.settle_time

    def predict_slew(
        self,
        startra: float,
        startdec: float,
        endra: float,
        enddec: float,
    ) -> tuple[float, tuple[list[float], list[float]]]:
        """Calculate slew distance and endpoint path for scheduling purposes.

        Args:
            startra: Starting RA in degrees
            startdec: Starting Dec in degrees
            endra: Ending RA in degrees
            enddec: Ending Dec in degrees

        Returns:
            Tuple of (slew_distance, slew_path) where slew_path is (ra_array, dec_array)
        """
        slewdist = (
            separation([startra * DTOR, startdec * DTOR], [endra * DTOR, enddec * DTOR])
            / DTOR
        )
        slewpath: tuple[list[float], list[float]] = (
            [startra, endra],
            [startdec, enddec],
        )
        return slewdist, slewpath
