from __future__ import annotations

from math import pi
from typing import Optional


class ReactionWheel:
    """Simple reaction wheel model.

    - Tracks `current_momentum` (N*m*s) stored in the wheel assembly.
    - Enforces `max_torque` (N*m) and `max_momentum` (N*m*s).
    - Provides helper to compute an accel limit (deg/s^2) given spacecraft inertia.
    - When a planned slew is executed, `reserve_impulse` can be used to
      estimate whether the wheel has enough headroom; it will return an
      adjusted torque if necessary.
    """

    def __init__(
        self,
        max_torque: float = 0.0,
        max_momentum: float = 0.0,
        orientation: tuple[float, float, float] | None = None,
        current_momentum: float = 0.0,
        name: str | None = None,
    ) -> None:
        def _to_float(val, default=0.0):
            try:
                return float(val)
            except Exception:
                return default

        # store raw values for debugging; coerce to floats for math use
        self.max_torque_raw = max_torque
        self.max_momentum_raw = max_momentum
        self.max_torque = _to_float(max_torque)
        self.max_momentum = _to_float(max_momentum)
        # Wheel orientation as unit vector in spacecraft body frame
        self.orientation = tuple(orientation) if orientation is not None else (1.0, 0.0, 0.0)
        self.current_momentum = _to_float(current_momentum)
        self.name = name or "wheel"

    def accel_limit_deg(self, moi: float) -> float:
        """Return maximum spacecraft angular acceleration (deg/s^2)
        the wheel torque can produce assuming `torque = max_torque`.
        """
        try:
            moi_f = float(moi)
        except Exception:
            # If moi cannot be interpreted (e.g., a Mock in tests), do not
            # artificially limit acceleration; allow ACS-configured accel
            # to take precedence by returning +inf when wheel torque exists.
            return float("inf") if self.max_torque > 0 else 0.0

        if moi_f <= 0 or self.max_torque <= 0:
            return 0.0
        # torque (N*m) / I (kg*m^2) => rad/s^2; convert to deg/s^2
        return (self.max_torque / moi_f) * (180.0 / pi)

    def reserve_impulse(self, requested_torque: float, motion_time: float) -> float:
        """Given a requested constant torque (N*m) and motion duration (s),
        return a torque possibly reduced so the wheel won't exceed its momentum
        capacity when integrating torque over time (impulse = torque * time).
        """
        if self.max_momentum <= 0 or motion_time <= 0:
            return min(self.max_torque, abs(requested_torque)) * (1 if requested_torque >= 0 else -1)

        available = self.max_momentum - abs(self.current_momentum)
        if available <= 0:
            return 0.0
        needed = abs(requested_torque) * motion_time
        if needed <= available:
            return requested_torque
        # scale torque down so needed == available
        scale = available / (needed)
        adjusted = requested_torque * scale
        # also enforce torque capability
        if abs(adjusted) > self.max_torque:
            adjusted = self.max_torque if adjusted >= 0 else -self.max_torque
        return adjusted

    def apply_torque(self, torque: float, dt: float) -> None:
        """Apply torque for duration dt, updating stored wheel momentum.
        Note: in reality sign convention depends on direction; we treat positive
        torque as increasing stored momentum magnitude.
        """
        self.current_momentum += torque * dt
        # clamp to capacity
        if self.current_momentum > self.max_momentum:
            self.current_momentum = self.max_momentum
        if self.current_momentum < -self.max_momentum:
            self.current_momentum = -self.max_momentum
