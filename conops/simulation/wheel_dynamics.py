"""Wheel dynamics simulation for reaction wheel momentum and torque management.

This module encapsulates all physics related to reaction wheel angular momentum,
torque allocation, and momentum conservation tracking. It separates physical
dynamics from operational ACS logic.
"""

from __future__ import annotations

import logging
from math import pi, sqrt
from typing import TYPE_CHECKING, Any

import numpy as np

from .reaction_wheel import ReactionWheel
from .torque_allocator import allocate_wheel_torques, build_wheel_orientation_matrix

if TYPE_CHECKING:
    from .disturbance import DisturbanceModel

_logger = logging.getLogger(__name__)


class WheelDynamics:
    """Physics simulation for reaction wheel momentum and torque.

    Manages:
    - Reaction wheel momentum state
    - Spacecraft body angular momentum tracking
    - Torque allocation across wheels
    - Momentum conservation validation
    - Magnetorquer desaturation

    Conservation law:
        H_total = H_wheels + H_body
        dH_total/dt = τ_external (only external torques change total momentum)
    """

    def __init__(
        self,
        wheels: list[ReactionWheel],
        inertia_matrix: np.ndarray,
        magnetorquers: list[dict[str, Any]] | None = None,
        disturbance_model: DisturbanceModel | None = None,
        momentum_margin: float = 0.9,
        budget_margin: float = 0.85,
        conservation_tolerance: float = 0.1,
    ) -> None:
        """Initialize wheel dynamics subsystem.

        Args:
            wheels: List of ReactionWheel instances.
            inertia_matrix: 3x3 spacecraft inertia matrix (kg·m²).
            magnetorquers: Optional list of magnetorquer dicts with
                'orientation', 'dipole_strength', 'power_draw' keys.
            disturbance_model: Optional disturbance model for torque estimation.
            momentum_margin: Fraction of max momentum to use as safe limit (0-1).
            budget_margin: Fraction of headroom required for slew feasibility.
            conservation_tolerance: Fractional tolerance for conservation checks.
        """
        self._wheels = wheels
        self.inertia_matrix = np.array(inertia_matrix, dtype=float)
        self.magnetorquers = magnetorquers or []
        self.disturbance_model = disturbance_model

        # Configuration
        self._momentum_margin = momentum_margin
        self._budget_margin = budget_margin
        self._conservation_tolerance = conservation_tolerance

        # Spacecraft body angular momentum (3-vector in body frame, N·m·s)
        # Conservation: H_total = H_wheels + H_body; dH_total/dt = τ_external
        self.body_momentum = np.zeros(3, dtype=float)

        # Track cumulative external impulse for conservation validation
        self._cumulative_external_impulse = np.zeros(3, dtype=float)

        # Initial total momentum (set after first update)
        self._initial_total_momentum: np.ndarray | None = None

        # Slew momentum tracking
        self._slew_momentum_at_start: np.ndarray | None = None
        self._slew_external_impulse_at_start: np.ndarray | None = None
        self._slew_expected_delta_h: np.ndarray | None = None

        # MTQ state tracking
        self.mtq_power_w: float = 0.0
        self._last_mtq_proj_max: float = 0.0
        self._last_mtq_torque_mag: float = 0.0
        self._last_mtq_torque_vec: np.ndarray = np.zeros(3, dtype=float)
        self._last_mtq_bleed_torque_mag: float = 0.0

        # Precompute static wheel properties for torque allocation
        # (orientation matrix and max momentum/torque don't change during simulation)
        self._rebuild_wheel_cache()

    @property
    def wheels(self) -> list[ReactionWheel]:
        """Get the list of reaction wheels."""
        return self._wheels

    @wheels.setter
    def wheels(self, value: list[ReactionWheel]) -> None:
        """Set wheels and rebuild cached orientation matrix."""
        self._wheels = value
        # Rebuild cache if we've already initialized (avoid during __init__)
        if hasattr(self, "_e_mat"):
            self._rebuild_wheel_cache()

    def _rebuild_wheel_cache(self) -> None:
        """Rebuild cached wheel orientation matrix and static properties."""
        self._e_mat, self._max_moms, self._max_torques = build_wheel_orientation_matrix(
            self._wheels
        )

    # -------------------------------------------------------------------------
    # Momentum State Queries
    # -------------------------------------------------------------------------

    def get_total_wheel_momentum(self) -> np.ndarray:
        """Compute total wheel angular momentum vector in body frame.

        Returns:
            H_wheels = Σ (wheel_i.current_momentum * wheel_i.axis)
            as a 3D numpy array in N·m·s.
        """
        h_total = np.zeros(3, dtype=float)
        for w in self.wheels:
            try:
                axis = np.array(w.orientation, dtype=float)
                nrm = np.linalg.norm(axis)
                if nrm > 0:
                    axis = axis / nrm
                mom = float(getattr(w, "current_momentum", 0.0))
                h_total += mom * axis
            except Exception:
                continue
        return h_total

    def get_total_system_momentum(self) -> np.ndarray:
        """Compute total angular momentum of the system (wheels + body).

        Returns:
            H_total = H_wheels + H_body as a 3D numpy array in N·m·s.
        """
        return self.get_total_wheel_momentum() + self.body_momentum

    def get_body_momentum_magnitude(self) -> float:
        """Return magnitude of spacecraft body momentum (should be ~0 when settled)."""
        return float(np.linalg.norm(self.body_momentum))

    def get_headroom_along_axis(self, axis: np.ndarray) -> float:
        """Compute available momentum headroom projected onto an axis.

        Args:
            axis: Unit vector representing the rotation axis.

        Returns:
            Total available momentum headroom (N·m·s) along the axis,
            accounting for current wheel momentum and the safety margin.
        """
        axis = np.array(axis, dtype=float)
        nrm = np.linalg.norm(axis)
        if nrm <= 0:
            return 0.0
        axis = axis / nrm

        headroom = 0.0
        for w in self.wheels:
            try:
                w_axis = np.array(w.orientation, dtype=float)
                w_nrm = np.linalg.norm(w_axis)
                if w_nrm > 0:
                    w_axis = w_axis / w_nrm
                proj = abs(np.dot(w_axis, axis))
                max_mom = float(getattr(w, "max_momentum", 0.0)) * self._momentum_margin
                curr_mom = abs(float(getattr(w, "current_momentum", 0.0)))
                headroom += proj * max(0.0, max_mom - curr_mom)
            except Exception:
                continue
        return headroom

    def get_max_momentum_fraction(self) -> float:
        """Return maximum momentum fraction across all wheels (0-1)."""
        max_frac = 0.0
        for w in self.wheels:
            mm = float(getattr(w, "max_momentum", 0.0))
            cm = float(getattr(w, "current_momentum", 0.0))
            if mm > 0:
                max_frac = max(max_frac, abs(cm) / mm)
        return max_frac

    def get_inertia_about_axis(self, axis: np.ndarray) -> float:
        """Compute moment of inertia about a given axis.

        Args:
            axis: Rotation axis (will be normalized).

        Returns:
            Effective MOI (kg·m²) about the axis: I_eff = axis^T · I · axis
        """
        axis = np.array(axis, dtype=float)
        nrm = np.linalg.norm(axis)
        if nrm <= 0:
            return 0.0
        axis = axis / nrm
        return float(axis.dot(self.inertia_matrix.dot(axis)))

    # -------------------------------------------------------------------------
    # Torque Application (Momentum-Conserving)
    # -------------------------------------------------------------------------

    def allocate_torques(
        self,
        desired_torque: np.ndarray,
        dt: float,
        use_weights: bool = False,
        bias_gain: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Allocate wheel torques to achieve a desired body torque.

        Args:
            desired_torque: Desired torque vector in body frame (N·m).
            dt: Time step (seconds).
            use_weights: If True, penalize wheels with high momentum.
            bias_gain: Null-space gain for driving momentum toward zero.

        Returns:
            Tuple of (raw_torques, allowed_torques, actual_torque, clamped).
        """
        return allocate_wheel_torques(
            self.wheels,
            desired_torque,
            dt,
            use_weights=use_weights,
            bias_gain=bias_gain,
            mom_margin=self._momentum_margin,
            e_mat=self._e_mat,
            max_moms=self._max_moms,
            max_torques=self._max_torques,
        )

    def apply_wheel_torques(self, taus_allowed: np.ndarray, dt: float) -> np.ndarray:
        """Apply wheel torques while conserving total angular momentum.

        When wheels apply torque τ for time dt:
        - Wheel momentum changes by +τ*dt (along wheel axis)
        - Spacecraft body momentum changes by -τ*dt (Newton's 3rd law reaction)

        Args:
            taus_allowed: Per-wheel torques (N·m), one per wheel.
            dt: Time step (seconds).

        Returns:
            Total torque vector applied to spacecraft body (3D, N·m).
        """
        if dt <= 0:
            return np.zeros(3, dtype=float)

        total_impulse_on_body = np.zeros(3, dtype=float)

        for i, w in enumerate(self.wheels):
            tau = float(taus_allowed[i])
            if tau == 0:
                continue

            # Get wheel axis (normalized)
            try:
                axis = np.array(w.orientation, dtype=float)
                nrm = np.linalg.norm(axis)
                if nrm > 0:
                    axis = axis / nrm
                else:
                    axis = np.array([1.0, 0.0, 0.0])
            except Exception:
                axis = np.array([1.0, 0.0, 0.0])

            # Apply torque to wheel (updates wheel momentum)
            # Returns actual momentum change (may be less if wheel saturated)
            actual_delta_h = w.apply_torque(tau, dt)

            # Reaction impulse on spacecraft body (Newton's 3rd law)
            # Use actual momentum change, not requested, to conserve total momentum
            impulse_on_body = -actual_delta_h * axis
            total_impulse_on_body += impulse_on_body

        # Update spacecraft body momentum with actual impulse
        self.body_momentum += total_impulse_on_body

        # Return equivalent average torque for compatibility
        return total_impulse_on_body / dt if dt > 0 else np.zeros(3, dtype=float)

    def apply_external_torque(
        self, external_torque: np.ndarray, dt: float, source: str = "external"
    ) -> None:
        """Apply external torque to the system (changes total momentum).

        External torques (disturbances, magnetorquers) change total system
        momentum. They act on the spacecraft body, not the wheels.

        Args:
            external_torque: Torque vector in body frame (N·m).
            dt: Time step (seconds).
            source: Description for logging (e.g., "disturbance", "MTQ").
        """
        if dt <= 0:
            return

        impulse = external_torque * dt
        self.body_momentum += impulse
        self._cumulative_external_impulse += impulse

        _logger.debug(
            "External torque (%s): τ=[%.4e, %.4e, %.4e] N·m, "
            "ΔH=[%.4e, %.4e, %.4e] N·m·s, H_body=[%.4e, %.4e, %.4e]",
            source,
            external_torque[0],
            external_torque[1],
            external_torque[2],
            impulse[0],
            impulse[1],
            impulse[2],
            self.body_momentum[0],
            self.body_momentum[1],
            self.body_momentum[2],
        )

    def apply_control_torque(
        self,
        desired_torque: np.ndarray,
        dt: float,
        use_weights: bool = False,
        bias_gain: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Allocate and apply control torque in one step.

        Convenience method that allocates torques and applies them with
        momentum conservation.

        Args:
            desired_torque: Desired torque vector in body frame (N·m).
            dt: Time step (seconds).
            use_weights: If True, penalize wheels with high momentum.
            bias_gain: Null-space gain for driving momentum toward zero.

        Returns:
            Tuple of (allowed_torques, actual_torque, clamped).
        """
        _, taus_allowed, t_actual, clamped = self.allocate_torques(
            desired_torque, dt, use_weights=use_weights, bias_gain=bias_gain
        )
        self.apply_wheel_torques(taus_allowed, dt)
        return taus_allowed, t_actual, clamped

    # -------------------------------------------------------------------------
    # Magnetorquer Desaturation
    # -------------------------------------------------------------------------

    def apply_magnetorquer_desat(
        self,
        b_field_body: np.ndarray,
        dt: float,
    ) -> float:
        """Bleed wheel momentum using magnetorquers.

        Physics: MTQ applies external torque τ_mtq = m × B to the spacecraft.
        This external torque changes total system momentum. During desaturation,
        the wheels slow down to release their excess momentum to the environment.

        Args:
            b_field_body: Magnetic field vector in body frame (Tesla).
            dt: Time step (seconds).

        Returns:
            Total MTQ power draw (Watts).
        """
        if not self.magnetorquers or dt <= 0:
            self.mtq_power_w = 0.0
            self._last_mtq_proj_max = 0.0
            self._last_mtq_torque_mag = 0.0
            self._last_mtq_torque_vec = np.zeros(3, dtype=float)
            self._last_mtq_bleed_torque_mag = 0.0
            return 0.0

        # Build total torque vector from magnetorquers using m x B
        t_mtq = np.zeros(3, dtype=float)
        total_power = 0.0
        mtq_axes: list[np.ndarray] = []
        mtq_limits: list[float] = []

        for m in self.magnetorquers:
            try:
                v = np.array(m.get("orientation", (1.0, 0.0, 0.0)), dtype=float)
            except Exception:
                v = np.array([1.0, 0.0, 0.0])
            vn = np.linalg.norm(v)
            if vn <= 0:
                v = np.array([1.0, 0.0, 0.0])
            else:
                v = v / vn

            dipole_max = float(m.get("dipole_strength", m.get("dipole", 0.0)))
            mtq_axes.append(v)
            mtq_limits.append(abs(dipole_max))
            total_power += float(m.get("power_draw", 0.0))

        if mtq_axes and self.wheels:
            b_vec = np.array(b_field_body, dtype=float)
            b_norm = np.linalg.norm(b_vec)
            if b_norm > 0:
                wheel_momentum = np.zeros(3, dtype=float)
                for w in self.wheels:
                    try:
                        axis = np.array(w.orientation, dtype=float)
                    except Exception:
                        axis = np.array([1.0, 0.0, 0.0])
                    an = np.linalg.norm(axis)
                    if an <= 0:
                        axis = np.array([1.0, 0.0, 0.0])
                    else:
                        axis = axis / an
                    wheel_momentum += axis * float(getattr(w, "current_momentum", 0.0))

                if np.linalg.norm(wheel_momentum) > 0:
                    # Command torque aligned with wheel momentum so dm sign matches mom.
                    tau_des = wheel_momentum / dt
                    m_des = np.cross(b_vec, tau_des) / (b_norm * b_norm)
                    axes_mat = np.column_stack(mtq_axes)
                    try:
                        m_cmd = np.linalg.lstsq(axes_mat, m_des, rcond=None)[0]
                    except Exception:
                        m_cmd = axes_mat.T @ m_des
                    for i, limit in enumerate(mtq_limits):
                        if limit > 0:
                            m_cmd[i] = float(np.clip(m_cmd[i], -limit, limit))
                        else:
                            m_cmd[i] = 0.0
                    m_vec = axes_mat @ m_cmd
                    t_mtq = np.cross(m_vec, b_vec)

        if not self.wheels:
            self.mtq_power_w = total_power
            self._last_mtq_proj_max = 0.0
            self._last_mtq_torque_mag = float(np.linalg.norm(t_mtq))
            self._last_mtq_torque_vec = np.array(t_mtq, dtype=float)
            self._last_mtq_bleed_torque_mag = 0.0
            return total_power

        # Project MTQ torque onto each wheel axis and bleed momentum toward zero
        max_proj = 0.0
        actual_impulse = np.zeros(3, dtype=float)

        for w in self.wheels:
            try:
                axis = np.array(w.orientation, dtype=float)
            except Exception:
                axis = np.array([1.0, 0.0, 0.0])
            an = np.linalg.norm(axis)
            if an <= 0:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = axis / an

            tau_w = float(np.dot(t_mtq, axis))
            max_proj = max(max_proj, abs(tau_w))

            if tau_w == 0:
                continue

            dm = tau_w * dt
            mom = float(getattr(w, "current_momentum", 0.0))

            # Only apply MTQ if it would REDUCE momentum magnitude.
            # MTQ reduces momentum when dm has same sign as mom:
            #   mom > 0, dm > 0: new_mom = mom - dm < mom (reduces)
            #   mom < 0, dm < 0: new_mom = mom - dm > mom (reduces magnitude)
            # If signs differ, MTQ would INCREASE momentum - skip this wheel.
            if mom == 0.0:
                continue  # No momentum to bleed
            if (mom > 0 and dm <= 0) or (mom < 0 and dm >= 0):
                continue  # Would increase momentum, skip

            # Wheel releases momentum when external torque opposes its direction
            new_mom = mom - dm

            # Clamp to prevent overshoot past zero during desaturation
            if mom > 0:
                new_mom = max(0.0, new_mom)
            elif mom < 0:
                new_mom = min(0.0, new_mom)

            # Track actual momentum change (momentum leaving the wheel)
            actual_dm = mom - new_mom
            actual_impulse += actual_dm * axis
            w.current_momentum = new_mom

        # The actual wheel momentum change is the momentum transferred out of the
        # system via MTQ. Track this as external impulse for conservation.
        # Note: actual_impulse is the momentum LEAVING the wheels (positive when
        # wheels slow down), so this represents momentum dumped to environment.
        self._cumulative_external_impulse -= actual_impulse

        self.mtq_power_w = total_power
        self._last_mtq_proj_max = float(max_proj)
        self._last_mtq_torque_mag = float(np.linalg.norm(t_mtq))
        self._last_mtq_torque_vec = np.array(t_mtq, dtype=float)
        self._last_mtq_bleed_torque_mag = (
            float(np.linalg.norm(actual_impulse) / dt) if dt > 0 else 0.0
        )

        _logger.debug(
            "MTQ desat: τ_mtq=[%.4e, %.4e, %.4e] N·m, "
            "ΔH_wheel=[%.4e, %.4e, %.4e] N·m·s, ext_impulse=[%.4e, %.4e, %.4e]",
            t_mtq[0],
            t_mtq[1],
            t_mtq[2],
            -actual_impulse[0],
            -actual_impulse[1],
            -actual_impulse[2],
            -actual_impulse[0],
            -actual_impulse[1],
            -actual_impulse[2],
        )

        return total_power

    # -------------------------------------------------------------------------
    # Slew Momentum Budget
    # -------------------------------------------------------------------------

    def compute_slew_peak_momentum(
        self,
        slew_distance_deg: float,
        rotation_axis: np.ndarray,
        accel_deg_s2: float,
        max_rate_deg_s: float | None = None,
    ) -> float:
        """Compute peak angular momentum required for a slew.

        For a bang-bang slew profile, peak angular velocity occurs at midpoint.
        For triangular profile: ω_peak = sqrt(angle * accel).
        For trapezoidal profile (when max_rate is limiting): ω_peak = max_rate.

        Args:
            slew_distance_deg: Slew distance in degrees.
            rotation_axis: Unit vector for rotation axis.
            accel_deg_s2: Angular acceleration in deg/s².
            max_rate_deg_s: Maximum slew rate in deg/s. If provided, caps the
                peak angular velocity (trapezoidal profile).

        Returns:
            Peak momentum magnitude (N·m·s).
        """
        if slew_distance_deg <= 0 or accel_deg_s2 <= 0:
            return 0.0

        # Normalize axis
        axis = np.array(rotation_axis, dtype=float)
        nrm = np.linalg.norm(axis)
        if nrm <= 0:
            return 0.0
        axis = axis / nrm

        # Get MOI along axis
        i_axis = self.get_inertia_about_axis(axis)
        if i_axis <= 0:
            return 0.0

        # Peak rate for triangular profile: ω_peak = sqrt(angle * accel)
        angle_rad = slew_distance_deg * (pi / 180.0)
        accel_rad = accel_deg_s2 * (pi / 180.0)
        omega_peak = sqrt(angle_rad * accel_rad)

        # Cap at max slew rate if provided (trapezoidal profile)
        if max_rate_deg_s is not None and max_rate_deg_s > 0:
            max_rate_rad = max_rate_deg_s * (pi / 180.0)
            omega_peak = min(omega_peak, max_rate_rad)

        return i_axis * omega_peak

    def check_slew_momentum_budget(
        self,
        slew_distance_deg: float,
        rotation_axis: np.ndarray,
        accel_deg_s2: float,
        slew_time_s: float,
        disturbance_torque_mag: float = 0.0,
        max_rate_deg_s: float | None = None,
    ) -> tuple[bool, str]:
        """Check if wheels have sufficient momentum headroom for a slew.

        Args:
            slew_distance_deg: Slew distance in degrees.
            rotation_axis: Unit vector for rotation axis.
            accel_deg_s2: Angular acceleration in deg/s².
            slew_time_s: Estimated slew duration (seconds).
            disturbance_torque_mag: Estimated disturbance torque magnitude (N·m).
            max_rate_deg_s: Maximum slew rate in deg/s (for trapezoidal profile).

        Returns:
            Tuple of (feasible, message).
        """
        if not self.wheels:
            return True, "No wheels configured"

        # Compute peak momentum needed
        h_peak = self.compute_slew_peak_momentum(
            slew_distance_deg, rotation_axis, accel_deg_s2, max_rate_deg_s
        )
        if h_peak <= 0:
            return True, "Zero momentum slew"

        # Normalize axis for headroom check
        axis = np.array(rotation_axis, dtype=float)
        nrm = np.linalg.norm(axis)
        if nrm > 0:
            axis = axis / nrm
        else:
            axis = np.array([0.0, 0.0, 1.0])

        # Get current headroom along slew axis
        headroom = self.get_headroom_along_axis(axis)

        # Disturbance momentum over slew duration
        h_disturbance = disturbance_torque_mag * slew_time_s

        # Total momentum needed with margin
        h_required = (h_peak + h_disturbance) / self._budget_margin

        if headroom >= h_required:
            return True, (
                f"Budget OK: need {h_required:.4f} N·m·s, have {headroom:.4f} N·m·s"
            )
        else:
            deficit = h_required - headroom
            return False, (
                f"Insufficient momentum budget: need {h_required:.4f} N·m·s "
                f"(peak={h_peak:.4f}, disturbance={h_disturbance:.4f}), "
                f"have {headroom:.4f} N·m·s (deficit={deficit:.4f})"
            )

    def record_slew_start(self) -> None:
        """Record wheel momentum state at the start of a slew."""
        self._slew_momentum_at_start = self.get_total_wheel_momentum().copy()
        self._slew_external_impulse_at_start = self._cumulative_external_impulse.copy()

    # -------------------------------------------------------------------------
    # Conservation Validation
    # -------------------------------------------------------------------------

    def check_conservation(self, utime: float) -> tuple[bool, str | None]:
        """Verify momentum conservation: H_total - H_initial = ∫τ_external dt.

        Args:
            utime: Current time (for logging).

        Returns:
            Tuple of (passed, warning_message). passed is True if conservation
            is satisfied within tolerance. warning_message is None if passed,
            otherwise contains the violation details.
        """
        if self._initial_total_momentum is None:
            # Initialize on first call
            self._initial_total_momentum = self.get_total_system_momentum().copy()
            return True, None

        h_current = self.get_total_system_momentum()
        h_expected = self._initial_total_momentum + self._cumulative_external_impulse
        error = h_current - h_expected
        error_mag = float(np.linalg.norm(error))

        # Tolerance: fraction of current momentum magnitude or absolute minimum
        h_mag = float(np.linalg.norm(h_current))
        tolerance = max(self._conservation_tolerance * h_mag, 1e-6)

        if error_mag > tolerance:
            warning = (
                f"Momentum conservation violation at t={utime:.1f}: "
                f"error=[{error[0]:.4e}, {error[1]:.4e}, {error[2]:.4e}] N·m·s "
                f"(mag={error_mag:.4e}, tol={tolerance:.4e})"
            )
            _logger.warning(warning)
            return False, warning

        return True, None

    def reset_conservation_tracking(self) -> None:
        """Reset conservation tracking state (e.g., at simulation start)."""
        self._initial_total_momentum = None
        self._cumulative_external_impulse = np.zeros(3, dtype=float)

    # -------------------------------------------------------------------------
    # Axis Acceleration/Rate Limits
    # -------------------------------------------------------------------------

    def get_axis_accel_limit(self, axis: np.ndarray, motion_time: float) -> float:
        """Compute maximum achievable acceleration about a rotation axis.

        Args:
            axis: Rotation axis (will be normalized).
            motion_time: Duration of motion (seconds), for momentum limiting.

        Returns:
            Maximum acceleration (deg/s²) that wheels can produce.
        """
        if not self.wheels:
            return 0.0

        # Normalize axis
        axis = np.array(axis, dtype=float)
        nrm = np.linalg.norm(axis)
        if nrm <= 0:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / nrm

        # Inertia about axis
        i_axis = self.get_inertia_about_axis(axis)
        if i_axis <= 0:
            return 0.0

        # Compute max torque along axis from all wheels
        max_torque = 0.0
        for w in self.wheels:
            try:
                w_axis = np.array(w.orientation, dtype=float)
                w_nrm = np.linalg.norm(w_axis)
                if w_nrm > 0:
                    w_axis = w_axis / w_nrm
                else:
                    continue
            except Exception:
                continue

            proj = abs(np.dot(w_axis, axis))
            wt = float(getattr(w, "max_torque", 0.0))

            # Also check momentum-limited torque
            mm = float(getattr(w, "max_momentum", 0.0)) * self._momentum_margin
            cm = abs(float(getattr(w, "current_momentum", 0.0)))
            avail = max(0.0, mm - cm)
            mom_limited_torque = avail / motion_time if motion_time > 0 else wt

            effective_torque = min(wt, mom_limited_torque)
            max_torque += proj * effective_torque

        if max_torque <= 0:
            return 0.0

        # accel (rad/s²) = torque / I, convert to deg/s²
        return (max_torque / i_axis) * (180.0 / pi)

    def get_axis_rate_limit(self, axis: np.ndarray) -> float:
        """Compute maximum achievable angular rate about an axis.

        Based on wheel momentum capacity and current loading.

        Args:
            axis: Rotation axis (will be normalized).

        Returns:
            Maximum angular rate (deg/s).
        """
        # Normalize axis
        axis = np.array(axis, dtype=float)
        nrm = np.linalg.norm(axis)
        if nrm <= 0:
            return 0.0
        axis = axis / nrm

        # Get inertia about axis
        i_axis = self.get_inertia_about_axis(axis)
        if i_axis <= 0:
            return 0.0

        # Available momentum headroom (with budget margin for conservative limit)
        # Budget margin ensures computed rates will pass can_execute_slew check
        headroom = self.get_headroom_along_axis(axis) * self._budget_margin

        # Max rate: ω = H / I
        omega_rad = headroom / i_axis
        return omega_rad * (180.0 / pi)

    def compute_slew_params(
        self,
        axis: np.ndarray,
        distance_deg: float,
        accel_cap: float | None = None,
        rate_cap: float | None = None,
    ) -> tuple[float, float, float]:
        """Compute physics-derived slew parameters for a given axis and distance.

        This is the primary interface for determining slew timing. It derives
        acceleration and rate from wheel capabilities, then applies optional
        operational caps.

        Args:
            axis: Rotation axis (will be normalized)
            distance_deg: Slew distance in degrees
            accel_cap: Optional acceleration cap in deg/s² (operational limit)
            rate_cap: Optional rate cap in deg/s (operational limit)

        Returns:
            Tuple of (accel, rate, motion_time) where:
                accel: Effective acceleration in deg/s²
                rate: Effective max rate in deg/s
                motion_time: Slew duration in seconds
        """
        if distance_deg <= 0:
            return 0.0, 0.0, 0.0

        # Get physics-derived limits from wheel capabilities
        #
        # Torque-limited: α_torque = τ_max / I
        # Momentum-limited: derived from peak momentum constraint
        #   For triangular bang-bang: ω_peak = sqrt(α × distance)
        #   Constraint: I × ω_peak ≤ headroom
        #   Solving: α ≤ (headroom / I)² / distance = rate_limit² / distance
        #
        # This closed-form solution is self-consistent (no iteration needed).

        # Torque-limited acceleration (pass motion_time=0 to ignore momentum limit)
        torque_limited_accel = self.get_axis_accel_limit(axis, 0.0)

        # Rate limit from momentum headroom
        physics_rate = self.get_axis_rate_limit(axis)

        if torque_limited_accel <= 0 or physics_rate <= 0:
            return 0.0, 0.0, 0.0

        # Determine effective rate (capped by operational limit if provided)
        if rate_cap is not None and rate_cap > 0:
            effective_rate = min(physics_rate, rate_cap)
        else:
            effective_rate = physics_rate

        # Momentum-limited acceleration only applies to triangular profiles.
        # For trapezoidal profiles (rate capped below triangular peak), there's
        # no momentum constraint on accel since peak momentum = I * rate_cap,
        # which is already within limits if rate_cap <= physics_rate.
        #
        # Triangular profile peak rate = sqrt(accel * distance)
        # If torque_limited_accel gives triangular peak > effective_rate,
        # we'll use trapezoidal and don't need momentum limit on accel.
        triangular_peak = (torque_limited_accel * distance_deg) ** 0.5
        if triangular_peak > effective_rate:
            # Will use trapezoidal profile - no momentum constraint on accel
            physics_accel = torque_limited_accel
        else:
            # Will use triangular profile - need momentum constraint
            # accel_limit = effective_rate² / distance ensures peak = effective_rate
            momentum_limited_accel = (effective_rate**2) / distance_deg
            physics_accel = min(torque_limited_accel, momentum_limited_accel)

        if physics_accel <= 0:
            return 0.0, 0.0, 0.0

        # Apply optional acceleration cap
        if accel_cap is not None and accel_cap > 0:
            accel = min(physics_accel, accel_cap)
        else:
            accel = physics_accel

        # Rate is already computed as effective_rate
        rate = effective_rate

        # Compute motion time using bang-bang profile
        # Triangular profile if we can't reach max rate
        t_accel = rate / accel
        d_accel = 0.5 * accel * t_accel**2

        if 2 * d_accel >= distance_deg:
            # Triangular profile - peak rate is sqrt(accel * distance)
            t_peak = (distance_deg / accel) ** 0.5
            motion_time = 2 * t_peak
            # Actual peak rate for this profile
            rate = accel * t_peak
        else:
            # Trapezoidal profile
            d_cruise = distance_deg - 2 * d_accel
            t_cruise = d_cruise / rate
            motion_time = 2 * t_accel + t_cruise

        return float(accel), float(rate), float(motion_time)

    def can_execute_slew(
        self,
        axis: np.ndarray,
        distance_deg: float,
        accel_cap: float | None = None,
        rate_cap: float | None = None,
    ) -> tuple[bool, str]:
        """Check if a slew can be executed with current wheel state.

        Args:
            axis: Rotation axis
            distance_deg: Slew distance in degrees
            accel_cap: Optional acceleration cap
            rate_cap: Optional rate cap

        Returns:
            Tuple of (can_execute, reason) where reason explains any failure.
        """
        if distance_deg <= 0:
            return True, "No slew needed"

        accel, rate, motion_time = self.compute_slew_params(
            axis, distance_deg, accel_cap, rate_cap
        )

        if accel <= 0:
            return False, "Insufficient torque capacity for slew axis"
        if rate <= 0:
            return False, "Insufficient momentum headroom for slew"

        # Check if we have enough momentum headroom for the peak rate.
        # available_h already has _momentum_margin applied, so we apply _budget_margin
        # on top to ensure we don't cut it too close.
        i_axis = self.get_inertia_about_axis(axis)
        required_h = rate * (pi / 180.0) * i_axis
        available_h = self.get_headroom_along_axis(axis)
        usable_h = available_h * self._budget_margin

        if required_h > usable_h:
            return (
                False,
                f"Peak momentum {required_h:.3f} exceeds usable headroom {usable_h:.3f} "
                f"(available={available_h:.3f}, budget_margin={self._budget_margin})",
            )

        return True, "OK"

    # -------------------------------------------------------------------------
    # Telemetry
    # -------------------------------------------------------------------------

    def get_wheel_states(self) -> list[dict[str, float | str]]:
        """Return current state of all wheels for telemetry."""
        states: list[dict[str, float | str]] = []
        for w in self.wheels:
            states.append(
                {
                    "name": getattr(w, "name", "wheel"),
                    "momentum": float(getattr(w, "current_momentum", 0.0)),
                    "max_momentum": float(getattr(w, "max_momentum", 0.0)),
                    "max_torque": float(getattr(w, "max_torque", 0.0)),
                    "last_torque": float(getattr(w, "last_torque_applied", 0.0)),
                    "power_draw": float(w.power_draw())
                    if hasattr(w, "power_draw")
                    else 0.0,
                }
            )
        return states

    def get_momentum_summary(self) -> dict[str, Any]:
        """Return momentum state summary for telemetry."""
        h_wheels = self.get_total_wheel_momentum()
        h_body = self.body_momentum
        h_total = h_wheels + h_body

        return {
            "wheel_momentum": h_wheels.tolist(),
            "body_momentum": h_body.tolist(),
            "total_momentum": h_total.tolist(),
            "wheel_momentum_mag": float(np.linalg.norm(h_wheels)),
            "body_momentum_mag": float(np.linalg.norm(h_body)),
            "total_momentum_mag": float(np.linalg.norm(h_total)),
            "max_wheel_fraction": self.get_max_momentum_fraction(),
            "cumulative_external_impulse": self._cumulative_external_impulse.tolist(),
        }
