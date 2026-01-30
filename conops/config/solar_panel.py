from datetime import datetime

import numpy as np
import numpy.typing as npt
import rust_ephem
from pydantic import BaseModel, Field, PrivateAttr

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
        roll: float = 0.0,
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the fraction of sunlight on this solar panel.

        Args:
            time: Unix timestamp, datetime object, or list of datetime objects
            ephem: Ephemeris object
            ra: Current spacecraft RA in degrees
            dec: Current spacecraft Dec in degrees
            roll: Spacecraft roll angle in degrees (rotation about boresight axis)

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
        sun_ra_rad = np.deg2rad(ephem.sun_ra_deg[i])
        sun_dec_rad = np.deg2rad(ephem.sun_dec_deg[i])
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
            # based on their angular position around the spacecraft.
            # The roll angle adjusts the effective azimuth position of the panel.
            # cos(azimuth + roll) gives the projection factor
            effective_azimuth_rad = np.deg2rad(self.azimuth_deg + roll)
            azimuth_factor = np.abs(np.cos(effective_azimuth_rad))
            panel = panel * azimuth_factor

        panel = np.clip(panel * not_in_eclipse, a_min=0, a_max=None)

        if scalar:
            return float(panel[0])
        return np.array(panel)


# Cached SolarPanel instance for accessing eclipse constraint
_ECLIPSE_PANEL_CACHE: SolarPanel | None = None


def _get_eclipse_constraint() -> rust_ephem.EclipseConstraint:
    """Get the eclipse constraint, using SolarPanel's for test compatibility."""
    global _ECLIPSE_PANEL_CACHE
    # Access via an instance to work around Pydantic class attribute interception
    # This allows tests to patch SolarPanel._eclipse_constraint
    if _ECLIPSE_PANEL_CACHE is None:
        _ECLIPSE_PANEL_CACHE = SolarPanel()
    return _ECLIPSE_PANEL_CACHE._eclipse_constraint


class _PanelGeometry:
    """Pre-computed panel geometry arrays for vectorized calculations."""

    __slots__ = (
        "gimbled",
        "sidemount",
        "cant_x",
        "cant_y",
        "azimuth_rad",
        "max_power",
        "efficiency",
        "weights",
        "panel_offset_angle",
    )

    def __init__(
        self,
        gimbled: npt.NDArray[np.bool_],
        sidemount: npt.NDArray[np.bool_],
        cant_x: npt.NDArray[np.float64],
        cant_y: npt.NDArray[np.float64],
        azimuth_rad: npt.NDArray[np.float64],
        max_power: npt.NDArray[np.float64],
        efficiency: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        panel_offset_angle: npt.NDArray[np.float64],
    ) -> None:
        self.gimbled = gimbled
        self.sidemount = sidemount
        self.cant_x = cant_x
        self.cant_y = cant_y
        self.azimuth_rad = azimuth_rad
        self.max_power = max_power
        self.efficiency = efficiency
        self.weights = weights
        self.panel_offset_angle = panel_offset_angle


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

    # Cached panel geometry for vectorized calculations
    _geometry_cache: _PanelGeometry | None = PrivateAttr(default=None)

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

    def _get_geometry(self) -> _PanelGeometry:
        """Get or compute cached panel geometry arrays."""
        if self._geometry_cache is not None:
            return self._geometry_cache

        panels = self._effective_panels()
        n = len(panels)

        gimbled = np.array([p.gimbled for p in panels], dtype=bool)
        sidemount = np.array([p.sidemount for p in panels], dtype=bool)
        cant_x = np.array([p.cant_x for p in panels], dtype=np.float64)
        cant_y = np.array([p.cant_y for p in panels], dtype=np.float64)
        azimuth_rad = np.deg2rad([p.azimuth_deg for p in panels])
        max_power = np.array([p.max_power for p in panels], dtype=np.float64)
        efficiency = np.array(
            [
                p.conversion_efficiency
                if p.conversion_efficiency is not None
                else self.conversion_efficiency
                for p in panels
            ],
            dtype=np.float64,
        )

        total_max = max_power.sum()
        weights = max_power / total_max if total_max > 0 else np.zeros(n)

        # Pre-compute panel offset angles
        # Sidemount: 90 - cant_x
        # Body-mount: hypot(cant_x, cant_y)
        panel_offset_angle = np.where(
            sidemount, 90.0 - cant_x, np.hypot(cant_x, cant_y)
        )

        self._geometry_cache = _PanelGeometry(
            gimbled=gimbled,
            sidemount=sidemount,
            cant_x=cant_x,
            cant_y=cant_y,
            azimuth_rad=azimuth_rad,
            max_power=max_power,
            efficiency=efficiency,
            weights=weights,
            panel_offset_angle=panel_offset_angle,
        )
        return self._geometry_cache

    def panel_illumination_fraction(
        self,
        time: datetime | list[datetime] | float,
        ephem: rust_ephem.Ephemeris,
        ra: float,
        dec: float,
        roll: float = 0.0,
    ) -> float | np.ndarray:
        """Calculate the weighted average fraction of sunlight on the solar panel set.

        Combines illumination from all panels weighted by their max_power.

        Args:
            time: Unix timestamp, datetime, or list of datetimes
            ephem: Ephemeris object
            ra: Current spacecraft RA in degrees
            dec: Current spacecraft Dec in degrees
            roll: Spacecraft roll angle in degrees (rotation about boresight axis)

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
                time=time, ephem=ephem, ra=ra, dec=dec, roll=roll
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
        roll: float = 0.0,
    ) -> float | np.ndarray:
        """Calculate the power generated by the solar panel set.

        Sums power from all panels, each weighted by illumination, max_power, and efficiency.

        Args:
            time: Unix timestamp, datetime, or list of datetimes
            ra: Current spacecraft RA in degrees
            dec: Current spacecraft Dec in degrees
            ephem: Ephemeris object
            roll: Spacecraft roll angle in degrees (rotation about boresight axis)

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
                time=time, ephem=ephem, ra=ra, dec=dec, roll=roll
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
        roll: float = 0.0,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Calculate both illumination fraction and power in a single call.

        This is a vectorized implementation that computes all panels efficiently
        by looking up sun position and eclipse state only once per call.

        Args:
            time: Unix timestamp, datetime, or list of datetimes
            ra: Current spacecraft RA in degrees
            dec: Current spacecraft Dec in degrees
            ephem: Ephemeris object
            roll: Spacecraft roll angle in degrees (rotation about boresight axis)

        Returns:
            tuple: (illumination_fraction, power_watts)
        """
        panels = self._effective_panels()
        if not panels:
            if isinstance(time, (float, int, datetime)):
                return 0.0, 0.0
            return np.zeros(len(time)), np.zeros(len(time))

        # Get cached panel geometry
        geom = self._get_geometry()

        # Handle time conversion - we only need scalar case for DITL
        if isinstance(time, (int, float)):
            dt = dtutcfromtimestamp(time)
            scalar = True
        elif isinstance(time, datetime):
            dt = time
            scalar = True
        else:
            # List of times - fall back to per-panel loop for now
            # (vectorizing across both panels AND times is more complex)
            return self._illumination_and_power_loop(time, ra, dec, ephem)

        # Get ephemeris index ONCE
        idx = ephem.index(dt)

        # Check eclipse ONCE (use SolarPanel's constraint for test compatibility)
        in_eclipse = _get_eclipse_constraint().in_constraint(
            ephemeris=ephem, target_ra=0.0, target_dec=0.0, time=dt
        )

        if in_eclipse:
            # In eclipse - no illumination for any panel
            return (0.0, 0.0) if scalar else (np.array([0.0]), np.array([0.0]))

        # Get sun position from pre-computed arrays (no SkyCoord overhead)
        sun_ra_deg = ephem.sun_ra_deg[idx]
        sun_dec_deg = ephem.sun_dec_deg[idx]

        # Compute sun angle (same for all panels at this pointing)
        sun_ra_rad = np.deg2rad(sun_ra_deg)
        sun_dec_rad = np.deg2rad(sun_dec_deg)
        target_ra_rad = np.deg2rad(ra)
        target_dec_rad = np.deg2rad(dec)
        sunangle = np.rad2deg(
            separation([sun_ra_rad, sun_dec_rad], [target_ra_rad, target_dec_rad])
        )

        # Vectorized panel illumination calculation
        # panel_sun_angle = 180 - sunangle - panel_offset_angle
        panel_sun_angle = 180.0 - sunangle - geom.panel_offset_angle
        panel_illum = np.cos(np.radians(panel_sun_angle))

        # Apply azimuthal constraint for side-mounted panels with non-zero azimuth
        # cos(azimuth) gives the projection factor
        azimuth_factor = np.abs(np.cos(geom.azimuth_rad))
        # Only apply to sidemount panels (azimuth doesn't affect body-mount)
        panel_illum = np.where(
            geom.sidemount, panel_illum * azimuth_factor, panel_illum
        )

        # Gimbled panels get full illumination when not in eclipse
        panel_illum = np.where(geom.gimbled, 1.0, panel_illum)

        # Clip negative illumination to zero
        panel_illum = np.maximum(panel_illum, 0.0)

        # Compute weighted illumination and power
        weighted_illum = float(np.sum(panel_illum * geom.weights))
        total_power = float(np.sum(panel_illum * geom.max_power * geom.efficiency))

        return weighted_illum, total_power

    def _illumination_and_power_loop(
        self,
        time: list[datetime],
        ra: float,
        dec: float,
        ephem: rust_ephem.Ephemeris,
        roll: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback loop-based implementation for list of times."""
        panels = self._effective_panels()
        total_max = sum(p.max_power for p in panels)

        illum_accum = np.zeros(len(time))
        power_accum = np.zeros(len(time))

        for p in panels:
            eff = (
                p.conversion_efficiency
                if p.conversion_efficiency is not None
                else self.conversion_efficiency
            )
            panel_illum = p.panel_illumination_fraction(
                time=time, ephem=ephem, ra=ra, dec=dec, roll=roll
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
        # Get sun position from pre-computed arrays
        index = ephem.index(dtutcfromtimestamp(time))
        sun_ra = ephem.sun_ra_deg[index]
        sun_dec = ephem.sun_dec_deg[index]

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
