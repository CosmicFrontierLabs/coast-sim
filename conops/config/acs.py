import numpy as np
from pydantic import ConfigDict, Field, field_validator
from rust_ephem.constraints import ConstraintConfig

from ..common import separation
from ..common.enums import SlewAlgorithm
from ..common.motion import RestToRestMotion
from ..common.numerics import ROUNDOFF
from ._base import ConfigModel
from .constants import DTOR
from .momentum import StoredMomentumConfig


def scheduled_slew_time(duration_s: float) -> int:
    """Convert a physical duration to whole seconds without ending it early."""
    return 0 if duration_s <= 0.0 else int(np.ceil(duration_s))


class SlewRequirement(ConfigModel):
    """A slew requirement class: a slew of up to a great-circle angle
    completes, settled, within a time."""

    name: str = Field(description="The class's name, such as 'small' or 'large'")
    max_great_circle_deg: float = Field(
        gt=0.0,
        le=180.0,
        description=(
            "The widest slew of the class, as the great-circle angle between "
            "the pointings it joins, in degrees"
        ),
    )
    max_seconds: float = Field(
        gt=0.0, description="The most a slew of the class may take, settled, in seconds"
    )


class ThermalSettlingStep(ConfigModel):
    """One step of the thermal settling table: a slew that moves the Sun by
    up to an angle in the body frame settles in a time."""

    max_sun_vector_change_deg: float = Field(
        gt=0.0,
        le=180.0,
        description=(
            "The widest change of the Sun's direction in the body frame, between "
            "the slew's start and end attitudes, the step covers, in degrees"
        ),
    )
    settle_seconds: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        description="The settle after such a slew, in seconds",
    )


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
    slew_requirements: list[SlewRequirement] = Field(
        default_factory=list,
        description=(
            "Slew requirement classes, each the time a slew of up to a "
            "great-circle angle may take, settled, in ascending angle. A check on "
            "the slews the ACS model flies, reported by the plan validator and the "
            "statistics, not a constraint on them; empty for no check"
        ),
    )
    settle_time: float = Field(
        default=120.0, description="Time to settle after slew completion in seconds"
    )
    thermal_settling: list[ThermalSettlingStep] = Field(
        default_factory=list,
        description=(
            "Settle time by the change of the Sun's direction in the body frame "
            "a slew makes, in ascending angle: the settle after a slew is that of "
            "the first step wide enough for its change, or the last step's beyond "
            "the table. Empty for the constant settle_time"
        ),
    )
    max_roll_from_optimum_deg: float | None = Field(
        default=None,
        ge=0.0,
        le=180.0,
        description=(
            "How far a commanded roll may differ from the power-optimal roll, "
            "in degrees either way; None: any roll the constraints allow"
        ),
    )
    slew_before_visible: bool = Field(
        default=False,
        description=(
            "Whether a slew to a science target that is not visible yet may "
            "start before it becomes visible, timed to arrive as it does. If "
            "False, the slew waits until the target is visible."
        ),
    )
    idle_min_hold_s: float = Field(
        default=300.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Minimum safe idle dwell after arrival, in addition to a conservative escape-slew reserve.",
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
        if norm < ROUNDOFF:
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

    def at_rate_fraction(self, fraction: float) -> "AttitudeControlSystem":
        """These limits with the slew rates scaled by ``fraction``, as a slew
        flown slower for its momentum is bounded; acceleration and settling
        are unchanged. ``self`` at a fraction of 1."""
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("a slew rate fraction must be finite and within (0, 1]")
        if fraction == 1.0:
            return self
        return self.model_copy(
            update={
                "max_slew_rate": self.max_slew_rate * fraction,
                "max_slew_rate_body": (
                    tuple(x * fraction for x in self.max_slew_rate_body)
                    if self.max_slew_rate_body is not None
                    else None
                ),
            }
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

    def effective_max_slew_rates(self, rotation_axes_body: np.ndarray) -> np.ndarray:
        """Slew-rate limits along each body-frame axis in an array of them.

        ``rotation_axes_body`` has shape ``(..., 3)``; the result has shape
        ``(...)``. A zero axis, which is no rotation, gets the scalar limit.
        """
        axes = np.asarray(rotation_axes_body, dtype=np.float64)
        if self.max_slew_rate_body is None:
            return np.full(axes.shape[:-1], float(self.max_slew_rate))
        norms = np.linalg.norm(axes, axis=-1)
        still = norms < ROUNDOFF
        unit_axes = axes / np.where(still, 1.0, norms)[..., None]
        limits = np.asarray(self.max_slew_rate_body, dtype=np.float64)
        with np.errstate(divide="ignore"):
            rates = 1.0 / np.linalg.norm(unit_axes / limits, axis=-1)
        return np.where(still, float(self.max_slew_rate), rates)

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
        return RestToRestMotion(angle_deg, a, vmax).duration

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

        return RestToRestMotion(angle_deg, a, vmax).at(t)[0]

    def slew_time(
        self,
        angle_deg: float,
        rotation_axis_body: tuple[float, float, float] | np.ndarray | None = None,
        sun_vector_change_deg: float | None = None,
    ) -> float:
        """Total slew time (motion + settle) using bang-bang control. The
        settle is the thermal settling table's for the Sun-vector change when
        one is given, else ``settle_time``."""
        if angle_deg <= 0 or np.isnan(angle_deg):
            return 0.0
        settle = (
            self.settle_time
            if sun_vector_change_deg is None
            else self.settle_seconds(sun_vector_change_deg)
        )
        return self.motion_time(angle_deg, rotation_axis_body) + settle

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

    @field_validator("slew_requirements")
    @classmethod
    def _classes_in_ascending_angle(
        cls, value: list[SlewRequirement]
    ) -> list[SlewRequirement]:
        angles = [r.max_great_circle_deg for r in value]
        if len(set(angles)) != len(angles):
            raise ValueError("slew requirement classes must have distinct angles")
        if len({r.name for r in value}) != len(value):
            raise ValueError("slew requirement classes must have distinct names")
        return sorted(value, key=lambda r: r.max_great_circle_deg)

    @field_validator("thermal_settling")
    @classmethod
    def _steps_in_ascending_angle(
        cls, value: list[ThermalSettlingStep]
    ) -> list[ThermalSettlingStep]:
        angles = [step.max_sun_vector_change_deg for step in value]
        if len(set(angles)) != len(angles):
            raise ValueError("thermal settling steps must have distinct angles")
        return sorted(value, key=lambda step: step.max_sun_vector_change_deg)

    def settle_seconds(self, sun_vector_change_deg: float) -> float:
        """The settle after a slew that moves the Sun's direction in the body
        frame by ``sun_vector_change_deg``: the first step of the thermal
        settling table wide enough for it, the last step's beyond the table,
        and ``settle_time`` without a table."""
        if not self.thermal_settling:
            return float(self.settle_time)
        for step in self.thermal_settling:
            if sun_vector_change_deg <= step.max_sun_vector_change_deg + 1e-9:
                return float(step.settle_seconds)
        return float(self.thermal_settling[-1].settle_seconds)

    @property
    def max_settle_seconds(self) -> float:
        """The longest settle any slew can take: the table's longest step, or
        ``settle_time`` without a table. What a bound on a slew's duration
        must allow for."""
        if not self.thermal_settling:
            return float(self.settle_time)
        return max(float(step.settle_seconds) for step in self.thermal_settling)

    def slew_class(self, great_circle_deg: float) -> SlewRequirement | None:
        """The requirement class a slew of ``great_circle_deg`` falls in: the
        narrowest class wide enough for it; None without a table or for a
        slew wider than the widest class."""
        for requirement in self.slew_requirements:
            if great_circle_deg <= requirement.max_great_circle_deg + 1e-9:
                return requirement
        return None
