from datetime import datetime
from typing import cast

import numpy as np
import numpy.typing as npt
import rust_ephem
from pydantic import BaseModel, Field, PrivateAttr

from ..common import dtutcfromtimestamp


def get_ephemeris_indices(
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
        normal (tuple[float, float, float]): Panel normal vector in spacecraft body frame.
            Defined as (x, y, z) where:
            - +x is the spacecraft pointing direction (boresight)
            - +y is the spacecraft "up" direction
            - +z completes the right-handed coordinate system
            Should be a unit vector for proper illumination calculations.
            Use create_solar_panel_vector() to generate vectors for common mount types.
        max_power (float): Maximum electrical power output at full illumination (W).
        conversion_efficiency (Optional[float]): Optional per-panel efficiency.
            If not provided, array-level efficiency is used.
    """

    # Class-level eclipse constraint (stateless, shared across all instances)
    _eclipse_constraint = rust_ephem.EclipseConstraint()

    name: str = Field(
        default="Panel", description="Name/identifier for the solar panel"
    )
    gimbled: bool = Field(default=False, description="Whether the panel is gimbled")
    normal: tuple[float, float, float] = Field(
        default=(0.0, 1.0, 0.0),
        description="Panel normal vector in spacecraft body frame",
    )
    max_power: float = Field(
        default=800.0, description="Maximum power output at full illumination in Watts"
    )
    conversion_efficiency: float | None = Field(
        default=None,
        description="Optional per-panel efficiency (uses array-level if not specified)",
    )

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
        from ..common import scbodyvector

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
            indices = get_ephemeris_indices(
                time=time[0] if scalar else time, ephemeris=ephem
            )
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

        # Non-gimbled panels: compute illumination based on panel normal vector
        # Get sun position in body frame for each time
        normal = np.array(self.normal, dtype=np.float64)

        # For each time, get sun vector and compute dot product with normal
        illum = np.zeros(len(indices))
        for idx, time_idx in enumerate(indices):
            # Get sun position vector from ephemeris
            sunvec = ephem.sun_pv.position[time_idx] - ephem.gcrs_pv.position[time_idx]

            # Convert sun vector to body frame
            sun_body = scbodyvector(
                np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(roll), sunvec
            )

            # Illumination is the dot product of normal with sun direction (scaled by sun vector magnitude)
            # Normalize sun vector for proper dot product
            sun_mag = np.linalg.norm(sun_body)
            if sun_mag > 0:
                sun_normalized = sun_body / sun_mag
                illum[idx] = np.dot(normal, sun_normalized)
            else:
                illum[idx] = 0.0

        # Clip negative illumination to zero and apply eclipse constraint
        illum = np.clip(illum * not_in_eclipse, a_min=0, a_max=None)

        if scalar:
            return float(illum[0])
        # Return with added fudge for mypy type checker
        return cast(npt.NDArray[np.float64], illum)


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


def create_solar_panel_vector(
    mount: str | None = None,
    cant_z: float = 0.0,
    cant_perp: float = 0.0,
    cant_x: float | None = None,
    cant_y: float | None = None,
    azimuth_deg: float | None = None,
) -> tuple[float, float, float]:
    """
    Create a unit normal vector for a solar panel based on mount type and cant angles.

    Supports both new and old style parameter configurations for backward compatibility.
    Only one parameter style may be used at a time - either old style OR new style, not both.

    Args:
        mount: Mount type (new style): 'sidemount', 'aftmount', or 'boresight'.
        cant_z: Cant angle around the spacecraft Z-axis in degrees (new style yaw-like rotation).
        cant_perp: Cant angle in degrees around the axis perpendicular to the panel mounting
            direction (new style pitch-like rotation):
            - For 'sidemount': rotates around X-axis.
            - For 'aftmount': rotates around Y-axis.
            - For 'boresight': rotates around Y-axis.
        cant_x: Cant angle around X-axis in degrees (old style), one of two orthogonal tilts.
        cant_y: Cant angle around Y-axis in degrees (old style), one of two orthogonal tilts.
        azimuth_deg: Structural placement angle around boresight/X in degrees (old style).
            0° = +Y (side), 90° = +Z, 180° = -Y, 270° = -Z. This places the
            panel around the spacecraft circumference; roll adds on top of this.

    Returns:
        Unit normal vector (x, y, z) in the spacecraft body frame.

    Mount types:
        - 'sidemount': Panel nominally faces +Y (spacecraft "up").
        - 'aftmount': Panel nominally faces -X (spacecraft "back").
        - 'boresight': Panel nominally faces +X (spacecraft forward/pointing).
    Examples:
        # New style - Sidemount panel with 30° yaw and 15° pitch
        normal = create_solar_panel_vector('sidemount', cant_z=30.0, cant_perp=15.0)

        # New style - Boresight panel tilted backward 45°
        normal = create_solar_panel_vector('boresight', cant_perp=-45.0)

        # Old style - Panel at 0° azimuth (+Y) with 30° cant around X and 15° cant around Y
        normal = create_solar_panel_vector(cant_x=30.0, cant_y=15.0, azimuth_deg=0.0)
    """

    # Validate that only one parameter style is used
    old_style_provided = (
        cant_x is not None or cant_y is not None or azimuth_deg is not None
    )
    new_style_provided = mount is not None

    if old_style_provided and new_style_provided:
        raise ValueError(
            "Cannot mix old style parameters (cant_x, cant_y, azimuth_deg) "
            "with new style parameters (mount, cant_z, cant_perp). "
            "Use either old style OR new style, not both."
        )

    # Check if old style parameters are provided
    if old_style_provided:
        # Use old style parameters
        if cant_x is None:
            cant_x = 0.0
        if cant_y is None:
            cant_y = 0.0
        if azimuth_deg is None:
            azimuth_deg = 0.0

        # Convert old style to rotation matrix approach
        # azimuth_deg determines the base orientation around the boresight
        # cant_x and cant_y are additional tilts

        theta_x = np.radians(cant_x)
        theta_y = np.radians(cant_y)
        azimuth_rad = np.radians(azimuth_deg)

        # Compute base vector continuously around the boresight (X-axis)
        # azimuth_deg: 0° = +Y, 90° = +Z, 180° = -Y, 270° = -Z
        base_x = 0.0
        base_y = np.cos(azimuth_rad)
        base_z = np.sin(azimuth_rad)

        # Apply cant angles
        # First cant around X-axis (theta_x)
        y_after_x = base_y * np.cos(theta_x) - base_z * np.sin(theta_x)
        z_after_x = base_y * np.sin(theta_x) + base_z * np.cos(theta_x)

        # Then cant around Y-axis (theta_y)
        x_final = base_x * np.cos(theta_y) + z_after_x * np.sin(theta_y)
        y_final = y_after_x
        z_final = -base_x * np.sin(theta_y) + z_after_x * np.cos(theta_y)

        return (x_final, y_final, z_final)

    else:
        # Use new style parameters
        if mount is None:
            mount = "sidemount"

        theta_z = np.radians(cant_z)
        theta_perp = np.radians(cant_perp)

        if mount == "sidemount":
            # Start with +Y (0, 1, 0)
            # First rotate around Z axis
            x_after_z = -np.sin(theta_z)
            y_after_z = np.cos(theta_z)

            # Then rotate around X axis (pitch)
            x = x_after_z
            y = y_after_z * np.cos(theta_perp)
            z = y_after_z * np.sin(theta_perp)

        elif mount == "aftmount":
            # Start with -X (-1, 0, 0)
            # First rotate around Z axis
            x_after_z = -np.cos(theta_z)
            y_after_z = -np.sin(theta_z)

            # Then rotate around Y axis (pitch)
            x = x_after_z * np.cos(theta_perp)
            y = y_after_z
            z = -x_after_z * np.sin(theta_perp)

        elif mount == "boresight":
            # Start with +X (1, 0, 0)
            # First rotate around Z axis
            x_after_z = np.cos(theta_z)
            y_after_z = np.sin(theta_z)

            # Then rotate around Y axis (pitch)
            x = x_after_z * np.cos(theta_perp)
            y = y_after_z
            z = x_after_z * np.sin(theta_perp)

        else:
            raise ValueError(f"Unknown mount type: {mount}")

        return (x, y, z)


class _PanelGeometry:
    """Pre-computed panel geometry arrays for vectorized calculations."""

    __slots__ = (
        "gimbled",
        "normal",
        "max_power",
        "efficiency",
        "weights",
    )

    def __init__(
        self,
        gimbled: npt.NDArray[np.bool_],
        normal: npt.NDArray[np.float64],
        max_power: npt.NDArray[np.float64],
        efficiency: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> None:
        self.gimbled = gimbled
        self.normal = normal  # shape (P, 3)
        self.max_power = max_power
        self.efficiency = efficiency
        self.weights = weights


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

    name: str = Field(
        default="Default Solar Panel", description="Name for the solar panel array"
    )
    panels: list[SolarPanel] = Field(
        default_factory=lambda: [SolarPanel()],
        description="List of individual solar panel configurations",
    )

    # Array-level default efficiency
    conversion_efficiency: float = Field(
        default=0.95,
        description="Default array-level conversion efficiency if panel does not override",
    )

    # Cached panel geometry for vectorized calculations
    _geometry_cache: _PanelGeometry | None = PrivateAttr(default=None)

    @property
    def sidemount(self) -> bool:
        """DEPRECATED: Return True if any panel is primarily side-mounted (y-component dominant).

        This is kept for backwards compatibility. With the new normal vector approach,
        panels can have arbitrary orientations.
        """
        for p in self.panels:
            n = p.normal
            # Check if y-component is dominant (side-mounted characteristic)
            if abs(n[1]) > abs(n[0]) and abs(n[1]) > abs(n[2]):
                return True
        return False

    def _get_geometry(self) -> _PanelGeometry:
        """Get or compute cached panel geometry arrays."""
        if self._geometry_cache is not None:
            return self._geometry_cache

        panels = self.panels
        n = len(panels)

        gimbled = np.array([p.gimbled for p in panels], dtype=bool)
        normal = np.array([p.normal for p in panels], dtype=np.float64)  # shape (P, 3)
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

        self._geometry_cache = _PanelGeometry(
            gimbled=gimbled,
            normal=normal,
            max_power=max_power,
            efficiency=efficiency,
            weights=weights,
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

        panels = self.panels
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
        panels = self.panels

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
        from ..common import scbodyvector

        panels = self.panels
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

        # Get sun vector in body frame
        sunvec = ephem.sun_pv.position[idx] - ephem.gcrs_pv.position[idx]  # km
        sun_body = scbodyvector(
            np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(roll), sunvec
        )

        # Normalize sun vector
        sun_mag = np.linalg.norm(sun_body)
        if sun_mag > 0:
            sun_normalized = sun_body / sun_mag
        else:
            # No sun direction - return zero illumination
            return (0.0, 0.0) if scalar else (np.array([0.0]), np.array([0.0]))

        # Vectorized panel illumination calculation: dot product of normal with sun direction
        # illum per panel: shape (P,)
        panel_illum = np.dot(geom.normal, sun_normalized)

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
        panels = self.panels
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

        Analyzes panel normal vectors to determine optimal pointing:
        - Panels with dominant Y component (side-mounted): point perpendicular to sun
        - Panels with dominant Z component (body-mounted): point directly at sun
        - Mixed arrays: uses weighted average approach

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

        # Analyze panel normal vectors to determine optimal pointing
        panels = self.panels

        # Check dominant axis of panel normals (weighted by max_power)
        total_power = sum(p.max_power for p in panels)
        if total_power <= 0:
            # No physical panels - default to pointing at sun
            return sun_ra, sun_dec

        # Compute weighted average normal vector
        avg_normal = np.zeros(3)
        for p in panels:
            weight = p.max_power / total_power
            avg_normal += np.array(p.normal) * weight

        # Determine dominant axis
        abs_normal = np.abs(avg_normal)
        dominant_axis = np.argmax(abs_normal)

        if dominant_axis == 1:  # Y is dominant (side-mounted-like)
            # Point perpendicular to sun (90 degrees away in RA)
            optimal_ra = (sun_ra + 90.0) % 360.0
            optimal_dec = sun_dec
        else:  # X (boresight) or Z is dominant (body-mounted-like)
            # Point directly at sun
            optimal_ra = sun_ra
            optimal_dec = sun_dec

        return optimal_ra, optimal_dec
