import numpy as np
from pydantic import BaseModel

from ..common import great_circle, separation
from .constants import DTOR


class AttitudeControlSystem(BaseModel):
    """
    Attitude Control System (ACS) configuration parameters.

    Defines slew performance characteristics including acceleration,
    maximum slew rate, accuracy, and settling time.
    """

    slew_acceleration: float = 0.5  # deg/s^2 - maximum angular acceleration
    max_slew_rate: float = 0.25  # deg/s (15 deg/min)
    slew_accuracy: float = 0.01  # deg - pointing accuracy after slew
    settle_time: float = 120.0  # seconds - time to settle after slew

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
