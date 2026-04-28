import numpy as np
from pydantic import Field

from ..common import great_circle, separation
from ..common.enums import SlewAlgorithm
from ._base import ConfigModel
from .constants import DTOR


class AttitudeControlSystem(ConfigModel):
    """
    Attitude Control System (ACS) configuration parameters.

    Defines slew performance characteristics including acceleration,
    maximum slew rate, accuracy, and settling time.
    """

    slew_acceleration: float = Field(
        default=0.5, description="Maximum angular acceleration in degrees/second²"
    )
    max_slew_rate: float = Field(
        default=0.25, description="Maximum slew rate in degrees/second (15 deg/min)"
    )
    slew_accuracy: float = Field(
        default=0.01, description="Pointing accuracy after slew completion in degrees"
    )
    settle_time: float = Field(
        default=120.0, description="Time to settle after slew completion in seconds"
    )
    slew_algorithm: SlewAlgorithm = Field(
        default=SlewAlgorithm.GREAT_CIRCLE,
        description=(
            "Algorithm used to compute slew paths. "
            "'great_circle' (default): shortest great-circle arc. "
            "'quaternion': full SO(3) SLERP coupling pointing and roll. "
            "'sun_avoiding': great-circle with automatic Sun-exclusion detour."
        ),
    )

    def motion_time(self, angle_deg: float) -> float:
        """Time to complete the motion (excluding settle) under bang-bang control."""
        if angle_deg <= 0:
            return 0.0
        a = float(self.slew_acceleration)
        vmax = float(self.max_slew_rate)
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

    def s_of_t(self, angle_deg: float, t: float) -> float:
        """Distance traveled (deg) along the slew after t seconds under bang-bang control.

        Clamps to [0, angle_deg] and ignores settle time (i.e., assumes t is measured
        from slew start; after motion is done, returns full angle).
        """
        if angle_deg <= 0 or t <= 0:
            return 0.0
        a = float(self.slew_acceleration)
        vmax = float(self.max_slew_rate)
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

    def slew_time(self, angle_deg: float) -> float:
        """Total slew time (motion + settle) using bang-bang control."""
        if angle_deg <= 0 or np.isnan(angle_deg):
            return 0.0
        return self.motion_time(angle_deg) + self.settle_time

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
