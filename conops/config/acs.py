from typing import Any

import numpy as np
from pydantic import BaseModel

from ..common import great_circle, separation
from .constants import DTOR


class WheelSpec(BaseModel):
    """Specification for a single reaction wheel.

    Defines orientation, torque/momentum limits, and power characteristics.
    """

    orientation: tuple[float, float, float] = (1.0, 0.0, 0.0)
    max_torque: float = 0.0  # N*m
    max_momentum: float = 0.0  # N*m*s
    name: str = ""
    idle_power_w: float = 5.0  # Watts when spinning but not applying torque
    torque_power_coeff: float = 50.0  # Additional power per N*m of torque


class MagnetorquerSpec(BaseModel):
    """Specification for a single magnetorquer.

    Defines orientation, dipole strength, and power characteristics.
    """

    orientation: tuple[float, float, float] = (1.0, 0.0, 0.0)
    dipole_strength: float = 0.0  # A*m^2
    power_draw: float = 0.0  # Watts when active
    name: str = ""


class AttitudeControlSystem(BaseModel):
    """
    Attitude Control System (ACS) configuration parameters.

    Slew performance is derived from wheel physics (torque, momentum, MOI).
    Optional rate/acceleration caps can be set for operational constraints
    (e.g., star tracker blur limits, thermal constraints).
    """

    # Optional operational caps (None = no limit, use full wheel capability)
    # These are UPPER BOUNDS, not targets - actual performance is derived from physics
    max_slew_rate: float | None = (
        None  # deg/s - operational cap (e.g., star tracker limit)
    )
    max_slew_accel: float | None = (
        None  # deg/s^2 - operational cap (e.g., power budget)
    )

    # Legacy parameter names (deprecated, mapped to new names for compatibility)
    slew_acceleration: float | None = None  # DEPRECATED: use max_slew_accel
    settle_time: float = 0.0  # DEPRECATED: no longer used, slew ends when motion ends

    slew_accuracy: float = 0.01  # deg - pointing accuracy after slew
    # Simple reaction wheel support (optional)
    wheel_enabled: bool = False
    # Legacy single-wheel params (kept for compatibility)
    wheel_max_torque: float = 0.0  # N*m - maximum torque a wheel assembly can apply
    wheel_max_momentum: float = 0.0  # N*m*s - wheel momentum storage capacity
    # Multi-wheel definition: list of wheels with orientation and per-wheel params
    wheels: list[WheelSpec | dict[str, Any]] = []
    # Spacecraft rotational inertia per principal axis (Ixx, Iyy, Izz) in kg*m^2
    spacecraft_moi: tuple[float, float, float] = (5.0, 5.0, 5.0)
    # Magnetorquer definitions (optional) for finite momentum unloading
    magnetorquers: list[MagnetorquerSpec | dict[str, Any]] = []
    magnetorquer_bfield_T: float = 3e-5  # representative LEO field magnitude (Tesla)
    # If True, allow MTQ bleed during SCIENCE; default keeps MTQs off in SCIENCE
    mtq_bleed_in_science: bool = False
    # Disturbance modeling inputs (drag/SRP/gg/magnetic)
    cp_offset_body: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )  # CoP minus CoM (m) in body frame
    residual_magnetic_moment: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )  # A*m^2 in body frame
    drag_area_m2: float = 0.0  # effective drag cross-section (m^2)
    drag_coeff: float = 2.2  # ballistic drag coefficient
    solar_area_m2: float = 0.0  # illuminated area for solar pressure (m^2)
    solar_reflectivity: float = (
        1.0  # momentum factor (1+r): 1.0=absorbing, 2.0=reflective
    )
    use_msis_density: bool = (
        False  # if True, attempt to use pymsis/nrlmsise-00 for density
    )
    msis_f107: float = 200.0  # solar flux (sfu), higher default for active cycle
    msis_f107a: float = 180.0  # 81-day average
    msis_ap: float = 12.0  # geomagnetic index (Ap), quiet-to-moderate
    msis_density_scale: float = 1.0  # optional multiplier for MSIS density
    # Disturbance torque in body frame (N*m), applied continuously
    disturbance_torque_body: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # If True, raise ValueError when wheel config has rank < 3 (not fully controllable)
    strict_wheel_validation: bool = False

    def get_accel_cap(self) -> float | None:
        """Get the effective acceleration cap (handles legacy parameter name)."""
        # Prefer new name, fall back to legacy
        if self.max_slew_accel is not None:
            return self.max_slew_accel
        return self.slew_acceleration

    def motion_time(
        self, angle_deg: float, accel: float | None = None, vmax: float | None = None
    ) -> float:
        """Time to complete the motion under bang-bang control.

        Args:
            angle_deg: Slew distance in degrees
            accel: Acceleration in deg/s² (required for physics-derived slews)
            vmax: Max rate in deg/s (required for physics-derived slews)

        Returns:
            Motion time in seconds (0 if inputs invalid)
        """
        if angle_deg <= 0:
            return 0.0
        # Use provided values or fall back to caps (for legacy compatibility)
        a = accel if accel is not None else self.get_accel_cap()
        v = vmax if vmax is not None else self.max_slew_rate
        if a is None or v is None or a <= 0 or v <= 0:
            return 0.0
        a = float(a)
        v = float(v)
        t_accel = v / a
        d_accel = 0.5 * a * t_accel**2
        if 2 * d_accel >= angle_deg:
            # Triangular profile
            t_peak = (angle_deg / a) ** 0.5
            return float(2 * t_peak)
        # Trapezoidal profile
        d_cruise = angle_deg - 2 * d_accel
        t_cruise = d_cruise / v
        return float(2 * t_accel + t_cruise)

    def s_of_t(
        self,
        angle_deg: float,
        t: float,
        accel: float | None = None,
        vmax: float | None = None,
    ) -> float:
        """Distance traveled (deg) along the slew after t seconds under bang-bang control.

        Args:
            angle_deg: Total slew distance in degrees
            t: Time since slew start in seconds
            accel: Acceleration in deg/s² (required for physics-derived slews)
            vmax: Max rate in deg/s (required for physics-derived slews)

        Returns:
            Distance traveled in degrees, clamped to [0, angle_deg]
        """
        if angle_deg <= 0 or t <= 0:
            return 0.0
        # Use provided values or fall back to caps (for legacy compatibility)
        a = accel if accel is not None else self.get_accel_cap()
        v = vmax if vmax is not None else self.max_slew_rate
        if a is None or v is None or a <= 0 or v <= 0:
            # Fallback: constant rate if we have vmax
            if v is not None and v > 0:
                return min(max(0.0, t * v), angle_deg)
            return 0.0
        a = float(a)
        v = float(v)

        # Determine profile
        t_accel = v / a
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
        t_cruise = d_cruise / v
        motion_time = 2 * t_accel + t_cruise
        tau = max(0.0, min(float(t), motion_time))
        if tau <= t_accel:
            s = 0.5 * a * tau**2
        elif tau <= t_accel + t_cruise:
            s = d_accel + v * (tau - t_accel)
        else:
            t_dec = tau - (t_accel + t_cruise)
            s = d_accel + d_cruise + v * t_dec - 0.5 * a * t_dec**2
        return float(max(0.0, min(angle_deg, s)))

    def slew_time(
        self, angle_deg: float, accel: float | None = None, vmax: float | None = None
    ) -> float:
        """Total slew time using bang-bang control.

        Note: settle_time is deprecated and no longer added. Slew ends when motion ends.
        The wheels naturally hold attitude after the slew completes.

        Args:
            angle_deg: Slew distance in degrees
            accel: Acceleration in deg/s² (required for physics-derived slews)
            vmax: Max rate in deg/s (required for physics-derived slews)

        Returns:
            Slew time in seconds
        """
        if angle_deg <= 0 or np.isnan(angle_deg):
            return 0.0
        return self.motion_time(angle_deg, accel=accel, vmax=vmax)

    def predict_slew(
        self,
        startra: float,
        startdec: float,
        endra: float,
        enddec: float,
        steps: int = 20,
    ) -> tuple[float, tuple[list[float], list[float]]]:
        """Calculate great circle slew distance and path.

        Args:
            startra: Starting RA in degrees
            startdec: Starting Dec in degrees
            endra: Ending RA in degrees
            enddec: Ending Dec in degrees
            steps: Number of steps in the path

        Returns:
            Tuple of (slew_distance, slew_path) where slew_path is (ra_array, dec_array)
        """
        slewdist = (
            separation([startra * DTOR, startdec * DTOR], [endra * DTOR, enddec * DTOR])
            / DTOR
        )
        slewpath = great_circle(startra, startdec, endra, enddec, steps)
        return slewdist, slewpath

    def get_achievable_slew_performance(
        self,
    ) -> tuple[float | None, float | None, dict[str, float], dict[str, float]]:
        """Compute achievable slew acceleration and rate from wheel physics.

        Returns:
            Tuple of (min_accel, min_rate, accel_by_axis, rate_by_axis) where:
            - min_accel: Minimum achievable acceleration across all axes (deg/s²), or None
            - min_rate: Minimum achievable rate across all axes (deg/s), or None
            - accel_by_axis: Dict mapping axis name to achievable accel
            - rate_by_axis: Dict mapping axis name to achievable rate
        """
        # Build wheel list from either multi-wheel config or legacy params
        wheel_specs: list[tuple[np.ndarray, float, float]] = []  # (orientation, τ, H)

        if self.wheels:
            for w in self.wheels:
                if isinstance(w, dict):
                    orient = np.array(w.get("orientation", [1, 0, 0]), dtype=float)
                    torque = float(w.get("max_torque", 0.0))
                    momentum = float(w.get("max_momentum", 0.0))
                else:
                    orient = np.array(w.orientation, dtype=float)
                    torque = w.max_torque
                    momentum = w.max_momentum
                norm = np.linalg.norm(orient)
                if norm > 1e-9:
                    orient = orient / norm
                wheel_specs.append((orient, torque, momentum))
        elif self.wheel_max_torque > 0 or self.wheel_max_momentum > 0:
            # Legacy single-wheel config - assume 3-axis orthogonal
            for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                wheel_specs.append(
                    (
                        np.array(axis, dtype=float),
                        self.wheel_max_torque,
                        self.wheel_max_momentum,
                    )
                )

        if not wheel_specs:
            return None, None, {}, {}

        # Get MOI for each principal axis
        moi = np.array(self.spacecraft_moi, dtype=float)

        # Compute achievable accel and rate for each principal axis
        principal_axes = [
            np.array([1, 0, 0], dtype=float),
            np.array([0, 1, 0], dtype=float),
            np.array([0, 0, 1], dtype=float),
        ]
        axis_names = ["X", "Y", "Z"]

        accel_by_axis: dict[str, float] = {}
        rate_by_axis: dict[str, float] = {}

        for i, (principal_axis, name) in enumerate(zip(principal_axes, axis_names)):
            i_axis = float(moi[i])
            if i_axis <= 0:
                continue

            total_torque = 0.0
            total_momentum = 0.0
            for orient, torque, momentum in wheel_specs:
                proj = abs(float(np.dot(orient, principal_axis)))
                total_torque += torque * proj
                total_momentum += momentum * proj

            if total_torque > 0:
                accel_rad = total_torque / i_axis
                accel_by_axis[name] = accel_rad * (180.0 / np.pi)

            if total_momentum > 0:
                rate_rad = total_momentum / i_axis
                rate_by_axis[name] = rate_rad * (180.0 / np.pi)

        min_accel = min(accel_by_axis.values()) if accel_by_axis else None
        min_rate = min(rate_by_axis.values()) if rate_by_axis else None

        return min_accel, min_rate, accel_by_axis, rate_by_axis

    def validate_wheel_capabilities(self) -> list[str]:
        """Validate that configured slew caps are achievable with wheel specs.

        Computes achievable acceleration and rate from wheel physics and compares
        against any configured operational caps. Returns warnings if caps exceed
        what the wheels can physically deliver.

        Returns:
            List of warning messages (empty if all caps are achievable)
        """
        warnings: list[str] = []

        # Get configured caps
        accel_cap = self.get_accel_cap()
        rate_cap = self.max_slew_rate

        # If no caps configured, nothing to validate
        if accel_cap is None and rate_cap is None:
            return warnings

        # Get achievable performance from wheel physics
        min_accel, min_rate, accel_by_axis, rate_by_axis = (
            self.get_achievable_slew_performance()
        )

        if not accel_by_axis and not rate_by_axis:
            if accel_cap is not None or rate_cap is not None:
                warnings.append(
                    "Slew caps configured but no wheels defined - "
                    "caps cannot be validated"
                )
            return warnings

        def format_axes(axes: list[str]) -> str:
            if len(axes) == 1:
                return f"{axes[0]}-axis"
            return "/".join(axes) + "-axes"

        # Compare caps against achievable values
        tol = 1e-9
        if accel_cap is not None and min_accel is not None:
            if accel_cap > min_accel:
                limiting_axes = [
                    name
                    for name, val in accel_by_axis.items()
                    if abs(val - min_accel) < tol
                ]
                warnings.append(
                    f"max_slew_accel={accel_cap:.3f} deg/s² exceeds wheel capability "
                    f"of {min_accel:.3f} deg/s² (limited by {format_axes(limiting_axes)}). "
                    f"Slews will use {min_accel:.3f} deg/s²."
                )

        if rate_cap is not None and min_rate is not None:
            if rate_cap > min_rate:
                limiting_axes = [
                    name
                    for name, val in rate_by_axis.items()
                    if abs(val - min_rate) < tol
                ]
                warnings.append(
                    f"max_slew_rate={rate_cap:.3f} deg/s exceeds wheel capability "
                    f"of {min_rate:.3f} deg/s (limited by {format_axes(limiting_axes)}). "
                    f"Slews will use {min_rate:.3f} deg/s."
                )

        return warnings
