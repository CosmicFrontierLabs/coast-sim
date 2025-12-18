from datetime import datetime

import numpy as np
import numpy.typing as npt
import rust_ephem
from pydantic import BaseModel, Field

from ..common import dtutcfromtimestamp, separation


def get_slice_indices(
    time: datetime | list[datetime], ephemeris: rust_ephem.Ephemeris
) -> np.ndarray:
    """
    Find indices in ephemeris that match the given times.

    Args:
        time: Python datetime object or list of datetime objects
        ephemeris: Ephemeris adapter object with index method

    Returns:
        Array of indices into ephemeris
    """
    if isinstance(time, datetime):
        # Single time - find closest match
        idx = ephemeris.index(time)
        return np.array([idx])
    else:
        # Multiple times - find closest match for each
        indices = []
        for t in time:
            indices.append(ephemeris.index(t))
        return np.array(indices)


class SolarPanel(BaseModel):
    """
    Configuration for a single solar panel element.

    Attributes:
        name (str): Name/identifier for the panel.
        gimbled (bool): Whether this panel is gimbled.
        sidemount (bool): Whether the panel is side-mounted (normal ~90° from boresight).
        cant_x (float): Cant angle around X-axis (deg), one of two orthogonal tilts.
        cant_y (float): Cant angle around Y-axis (deg), one of two orthogonal tilts.
        azimuth_deg (float): Structural placement angle around boresight/X (deg).
            0° = +Y (side), 90° = +Z, 180° = -Y, 270° = -Z. This places the
            panel around the spacecraft circumference; roll adds on top of this.
        max_power (float): Maximum electrical power output at full illumination (W).
        conversion_efficiency (Optional[float]): Optional per-panel efficiency.
            If not provided, array-level efficiency is used.
    """

    # Class-level eclipse constraint (stateless, shared across all instances)
    _eclipse_constraint = rust_ephem.EclipseConstraint()

    name: str = "Panel"
    gimbled: bool = False
    sidemount: bool = True
    cant_x: float = 0.0  # degrees
    cant_y: float = 0.0  # degrees
    azimuth_deg: float = 0.0  # degrees around boresight/X
    max_power: float = 800.0  # Watts at full illumination
    conversion_efficiency: float | None = None

    def panel_illumination_fraction(
        self,
        time: datetime | list[datetime] | float,
        ephem: rust_ephem.Ephemeris,
        ra: float,
        dec: float,
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the fraction of sunlight on this solar panel.

        Args:
            time: Unix timestamp, datetime object, or list of datetime objects
            ephem: Ephemeris object
            ra: Current spacecraft RA in degrees
            dec: Current spacecraft Dec in degrees

        Returns:
            float or np.ndarray: Fraction of panel illumination (0.0 to 1.0)
        """
        # Convert unix time to datetime if needed
        if isinstance(time, (int, float)):
            time = [dtutcfromtimestamp(time)]
            scalar = True
        elif isinstance(time, datetime):
            time = [time]
            scalar = True
        else:
            scalar = False

        # Get the array index of the ephemeris for this time
        try:
            i = get_slice_indices(time=time[0] if scalar else time, ephemeris=ephem)
        except Exception as e:
            print(f"Error getting slice for time={time}, ephem={ephem}: {e}")
            raise

        # Use EclipseConstraint to determine if spacecraft is in eclipse
        # EclipseConstraint returns True when IN eclipse, so we need to invert it
        if scalar:
            in_eclipse = self._eclipse_constraint.in_constraint(
                ephemeris=ephem, target_ra=0.0, target_dec=0.0, time=time[0]
            )
            not_in_eclipse = np.array([not in_eclipse])
        else:
            result = self._eclipse_constraint.evaluate(
                ephemeris=ephem, target_ra=0.0, target_dec=0.0, times=time
            )
            not_in_eclipse = ~np.array(result.constraint_array)

        # Gimbled panels: always point at sun when not in eclipse
        if self.gimbled:
            frac = not_in_eclipse.astype(float)
            if scalar:
                return float(frac[0])
            return frac

        # Non-gimbled panels: compute illumination based on cant, azimuth, and pointing
        # Calculate sun angle using vector separation (expects radians)
        sun_ra_rad = np.deg2rad(ephem.sun[i].ra.deg)
        sun_dec_rad = np.deg2rad(ephem.sun[i].dec.deg)
        target_ra_rad = np.deg2rad(ra)
        target_dec_rad = np.deg2rad(dec)

        # Calculate angle between boresight and sun
        sunangle = np.rad2deg(
            separation([sun_ra_rad, sun_dec_rad], [target_ra_rad, target_dec_rad])
        )

        if self.sidemount:
            # Side-mounted panel with optimal roll assumption
            # The panel normal is perpendicular to boresight (90°).
            # For side-mounted panels, only cant_x is relevant because the cant_y component
            # does not affect the tilt toward or away from the boresight in the side-mount geometry.
            # This is a deliberate change from previous behavior, where both cant_x and cant_y
            # were combined. If this is not the intended behavior, consider reverting to using
            # np.hypot(self.cant_x, self.cant_y) here.
            panel_offset_angle = 90.0 - self.cant_x
        else:
            # Body-mounted panel: panel normal aligned with boresight, with cant offset
            cant_mag = np.hypot(self.cant_x, self.cant_y)
            panel_offset_angle = 0 + cant_mag

        # Calculate panel illumination for this panel
        panel_sun_angle = 180 - sunangle - panel_offset_angle
        panel = np.cos(np.radians(panel_sun_angle))

        # Apply azimuthal constraint for side-mounted panels
        # With optimal roll, the spacecraft orients to maximize total power
        # but panels at different azimuthal positions around the spacecraft
        # cannot all receive optimal illumination simultaneously
        if self.sidemount and self.azimuth_deg != 0.0:
            # Panels at non-zero azimuth receive reduced illumination
            # based on their angular position around the spacecraft
            # cos(azimuth) gives the projection factor
            azimuth_rad = np.deg2rad(self.azimuth_deg)
            azimuth_factor = np.abs(np.cos(azimuth_rad))
            panel = panel * azimuth_factor

        panel = np.clip(panel * not_in_eclipse, a_min=0, a_max=None)

        if scalar:
            return float(panel[0])
        return np.array(panel)


class SolarPanelSet(BaseModel):
    """
    Model that describes the solar panel configuration and power generation

    Represents the spacecraft solar panel set (array) and power generation model.

    Attributes:
        name (str): Name for the solar panel array.
        panels (list[SolarPanel]): List of panel elements, each with its own config.
        conversion_efficiency (float): Default array-level efficiency if a panel
            does not override it.
    """

    name: str = "Default Solar Panel"
    panels: list[SolarPanel] = Field(default_factory=lambda: [SolarPanel()])

    # Array-level default efficiency
    conversion_efficiency: float = 0.95

    @property
    def sidemount(self) -> bool:
        """Return True if any panel is side-mounted. This is a hack right now
        to get the optimum charging pointing calculation to work correctly.

        FIXME: This should be handled better in the future.
        """
        for p in self.panels:
            if p.sidemount:
                return True
        return False

    def _effective_panels(self) -> list[SolarPanel]:
        """Return the configured panels for this set."""
        return self.panels

    def panel_illumination_fraction(
        self,
        time: datetime | list[datetime] | float,
        ephem: rust_ephem.Ephemeris,
        ra: float,
        dec: float,
    ) -> float | np.ndarray:
        """Calculate the weighted average fraction of sunlight on the solar panel set.

        Combines illumination from all panels weighted by their max_power.

        Args:
            time: Unix timestamp, datetime, or list of datetimes
            ephem: Ephemeris object
            ra: Current spacecraft RA in degrees
            dec: Current spacecraft Dec in degrees

        Returns:
            float or np.ndarray: Weighted average fraction of panel illumination (0.0 to 1.0)
        """
        # Convert unix time for scalar detection
        scalar = isinstance(time, (int, float))

        panels = self._effective_panels()
        total_max = sum(p.max_power for p in panels)

        # If we have no panels or total max power is zero, return zeros with correct shape
        if not panels or total_max <= 0:
            # Get array shape from first panel call
            if scalar:
                return 0.0

            # Return zeros consistent with the input type using isinstance checks
            if isinstance(time, (int, float, datetime)):
                return 0.0
            if isinstance(time, np.ndarray):
                return np.zeros(time.shape, dtype=float)
            if isinstance(time, (list, tuple)):
                return np.zeros(len(time), dtype=float)
            # Fallback for other sequence-like objects
            try:
                return np.zeros(len(time), dtype=float)
            except Exception:
                return 0.0

        # Accumulate weighted illumination from each panel
        illum_accum = None
        for p in panels:
            panel_illum = p.panel_illumination_fraction(
                time=time, ephem=ephem, ra=ra, dec=dec
            )
            weight = p.max_power / total_max
            if illum_accum is None:
                illum_accum = panel_illum * weight
            else:
                illum_accum = illum_accum + (panel_illum * weight)

        # Should never be None since we have at least one panel
        assert illum_accum is not None
        return illum_accum

    def power(
        self,
        time: datetime | list[datetime] | float,
        ra: float,
        dec: float,
        ephem: rust_ephem.Ephemeris,
    ) -> float | np.ndarray:
        """Calculate the power generated by the solar panel set.

        Sums power from all panels, each weighted by illumination, max_power, and efficiency.

        Args:
            time: Unix timestamp, datetime, or list of datetimes
            ra: Current spacecraft RA in degrees
            dec: Current spacecraft Dec in degrees
            ephem: Ephemeris object

        Returns:
            float or np.ndarray: Power generated by the solar panels in Watts
        """
        scalar = isinstance(time, (int, float))
        panels = self._effective_panels()

        # Accumulate power across panels
        power_accum = None
        for p in panels:
            eff = (
                p.conversion_efficiency
                if p.conversion_efficiency is not None
                else self.conversion_efficiency
            )
            panel_illum = p.panel_illumination_fraction(
                time=time, ephem=ephem, ra=ra, dec=dec
            )
            panel_power = panel_illum * p.max_power * eff
            if power_accum is None:
                power_accum = panel_power
            else:
                power_accum = power_accum + panel_power

        if power_accum is None:
            return 0.0 if scalar else np.array([0.0])

        return power_accum

    def illumination_and_power(
        self,
        time: datetime | list[datetime] | float,
        ra: float,
        dec: float,
        ephem: rust_ephem.Ephemeris,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Calculate both illumination fraction and power in a single call.

        This is more efficient than calling panel_illumination_fraction() and power()
        separately when both values are needed, as it avoids duplicate calculations.

        Args:
            time: Unix timestamp, datetime, or list of datetimes
            ra: Current spacecraft RA in degrees
            dec: Current spacecraft Dec in degrees
            ephem: Ephemeris object

        Returns:
            tuple: (illumination_fraction, power_watts)
        """
        illum_accum: float | np.ndarray | None = None
        power_accum: float | np.ndarray | None = None

        panels = self._effective_panels()
        total_max = sum(p.max_power for p in panels)

        if not panels or total_max <= 0:
            if isinstance(time, (datetime, float)):
                return 0.0, 0.0
            # Get array shape from a dummy call
            dummy_time = (
                [dtutcfromtimestamp(time)]
                if isinstance(time, (datetime, float))
                else time
            )
            dummy_panel = panels[0] if panels else SolarPanel()
            dummy_result = dummy_panel.panel_illumination_fraction(
                time=dummy_time, ephem=ephem, ra=ra, dec=dec
            )
            shape = dummy_result.shape if hasattr(dummy_result, "shape") else (1,)
            return np.zeros(shape), np.zeros(shape)

        # Calculate illumination and power for each panel
        if isinstance(time, (float, datetime)):
            illum_accum = 0
            power_accum = 0
        else:
            illum_accum = np.zeros(len(time))
            power_accum = np.zeros(len(time))

        for p in panels:
            eff = (
                p.conversion_efficiency
                if p.conversion_efficiency is not None
                else self.conversion_efficiency
            )
            panel_illum = p.panel_illumination_fraction(
                time=time, ephem=ephem, ra=ra, dec=dec
            )
            weight = p.max_power / total_max
            panel_power = panel_illum * p.max_power * eff

            illum_accum = illum_accum + (panel_illum * weight)
            power_accum = power_accum + panel_power

        return illum_accum, power_accum

    def optimal_charging_pointing(
        self, time: float, ephem: rust_ephem.Ephemeris
    ) -> tuple[float, float]:
        """Find optimal RA/Dec pointing for maximum solar panel illumination.

        For side-mounted panels, the optimal pointing is perpendicular to the Sun.
        For body-mounted panels, the optimal pointing is directly at the Sun.

        Args:
            time: Unix timestamp
            ephem: Ephemeris object

        Returns:
            tuple: (ra, dec) in degrees for optimal charging pointing
        """
        # Get sun position
        index = ephem.index(dtutcfromtimestamp(time))
        sun_ra = ephem.sun[index].ra.deg
        sun_dec = ephem.sun[index].dec.deg

        if self.sidemount:
            # For side-mounted panels, point perpendicular to sun (90 degrees away)
            # This maximizes illumination on the side panels
            # Point at sun RA + 90 degrees, same dec
            optimal_ra = (sun_ra + 90.0) % 360.0
            optimal_dec = sun_dec
        else:
            # For body-mounted panels, point directly at sun
            optimal_ra = sun_ra
            optimal_dec = sun_dec

        return optimal_ra, optimal_dec


# class SolarPanelConstraint(BaseModel):
#     """
#     For a given RA/Dec and time, determine if the solar panel constraint is
#     violated. Solar panel constraint is defined as the angle between the Sun
#     and the normal vector of the solar panel being within a given range.

#     Parameters
#     ----------
#     min_angle
#         The minimum angle of the Sun from solar panel normal vector.

#     max_angle
#         The maximum angle of the Sun from solar panel normal vector.

#     Methods
#     -------
#     __call__(coord, ephemeris, sun_radius_angle=None)
#         Checks if a given coordinate is inside the constraint.

#     """

#     name: str = "Panel"
#     short_name: Literal["Panel"] = "Panel"
#     solar_panel: SolarPanelSet = Field(..., description="Solar panel configuration")
#     min_angle: float | None = Field(
#         default=None, ge=0, le=180, description="Minimum angle of Sun from the panel"
#     )
#     max_angle: float | None = Field(
#         default=None, ge=0, le=180, description="Maximum angle of Sun from the panel"
#     )

#     def __call__(
#         self, time: Time, ephemeris: Any, coordinate: SkyCoord
#     ) -> np.typing.NDArray[np.bool_]:
#         """
#         Check if a given coordinate and set of times is inside the solar panel constraint.

#         Parameters
#         ----------
#         coordinate : SkyCoord
#             The coordinate to check. SkyCoord object with RA/Dec in degrees.
#         time : Time
#             The time to check. Array-like Time object.
#         ephemeris : Ephemeris
#             The ephemeris object.

#         Returns
#         -------
#         bool : np.ndarray[np.bool_]
#             Array of booleans. `True` if the coordinate is inside the
#             constraint, `False` otherwise.

#         """
#         # Find a slice what the part of the ephemeris that we're using
#         i = get_slice_indices(time=time, ephemeris=ephemeris)

#         # Calculate the panel illumination angle
#         panel_illumination = self.solar_panel.panel_illumination_fraction(
#             time=ephemeris.timestamp[i], coordinate=coordinate, ephem=ephemeris
#         )
#         panel_angle = np.arccos(panel_illumination) * u.rad

#         # Check if the spacecraft is in eclipse
#         in_eclipse = (
#             ephemeris.sun[i].separation(ephemeris.earth[i])
#             <= ephemeris.earth_radius_angle[i]
#         )

#         # Set the panel angle to 0 if in eclipse, as we don't care about the
#         # angle of the Sun on the panel if there's no Sun.
#         panel_angle[in_eclipse] = 0 * u.rad

#         # Construct the constraint based on the minimum and maximum angles
#         in_constraint = np.zeros(len(ephemeris.sun[i]), dtype=bool)

#         if self.min_angle is not None:
#             in_constraint |= panel_angle < self.min_angle * u.deg

#         if self.max_angle is not None:
#             in_constraint |= panel_angle > self.max_angle * u.deg

#         # Return the result as True or False, or an array of True/False
#         return in_constraint
