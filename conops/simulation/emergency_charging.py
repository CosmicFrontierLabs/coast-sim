"""Emergency battery charging functionality for spacecraft operations."""

from typing import TYPE_CHECKING

import numpy as np
import rust_ephem

from conops.config.battery import Battery

if TYPE_CHECKING:
    from ..ditl.ditl_log import DITLLog
    from ..targets import Pointing

from ..common import unixtime2date
from ..common.vector import angular_separation
from ..config import MissionConfig


class EmergencyCharging:
    """
    Manages emergency battery charging operations.

    This class handles the creation and management of emergency charging
    pointings when battery state of charge falls below acceptable levels.

    The class provides two methods for finding valid charging pointings:
    1. _find_valid_pointing(): General method that evaluates actual solar panel
       illumination for multiple candidate pointings and selects the best one.
       Works for any panel configuration (side-mounted or body-mounted).

    2. _find_valid_pointing_sidemount(): Optimized method for side-mounted panels
       that exploits the geometric fact that all pointings 90° from the Sun are
       equally power-positive.

    Optional slew limiting:
        Set max_slew_deg during initialization to constrain charging pointings
        to be within a specified angular distance from the current pointing.
        This minimizes slew time and energy expenditure during emergency charging.

    Example:
        # Without slew limit (new API — pass the full Config object)
        ec = EmergencyCharging(config=config, starting_obsid=999000)

        # With 45° slew limit
        ec = EmergencyCharging(
            config=config,
            starting_obsid=999000,
            max_slew_deg=45.0,
        )
    """

    def __init__(
        self,
        config: MissionConfig | None = None,
        starting_obsid: int = 999000,
        max_slew_deg: float | None = None,
        sidemount: bool = False,
        log: "DITLLog | None" = None,
    ):
        """
        Initialize emergency charging manager.

        Args:
            config: MissionConfig object containing all spacecraft configuration

            starting_obsid: Starting obsid for charging observations (default: 999000)
            max_slew_deg: Maximum slew distance in degrees from current pointing (default: None = no limit)
        """
        # Handle both old and new parameter styles for backward compatibility

        assert config is not None, "Config must be set for EmergencyCharging"
        self.config = config
        self.constraint = config.constraint
        self.solar_panel = config.solar_panel
        self.acs_config = config.spacecraft_bus.attitude_control

        self.next_charging_obsid = starting_obsid
        self.current_charging_ppt: "Pointing | None" = None
        self.max_slew_deg = max_slew_deg
        self.sidemount = sidemount
        self._charging_suppressed_due_to_eclipse = False
        self.log = log

    def _log_or_print(self, utime: float, event_type: str, description: str) -> None:
        """Log event to DITLLog if available, otherwise print."""
        if self.log is not None:
            self.log.log_event(
                utime=utime, event_type=event_type, description=description
            )
        else:
            print(f"{unixtime2date(utime)} {description}")

    def create_charging_pointing(
        self,
        utime: float,
        ephem: rust_ephem.Ephemeris,
        lastra: float = 0.0,
        lastdec: float = 0.0,
    ) -> "Pointing | None":
        """
        Create an emergency charging pointing to recover battery charge.

        Only creates charging pointing when:
        - Spacecraft is in sunlight (not in eclipse)
        - A valid pointing exists that doesn't violate constraints
        - If max_slew_deg is set, prioritizes pointings within slew limit

        Args:
            utime: Current unix timestamp
            ephem: Ephemeris object for spacecraft position/eclipse info
            lastra: Current pointing RA in degrees
            lastdec: Current pointing Dec in degrees

        Returns:
            Pointing object for emergency charging, or None if not possible
        """
        # Check if we're in sunlight (eclipse check)
        if not self._is_in_sunlight(utime, ephem):
            self._log_or_print(
                utime, "ERROR", "Cannot start emergency charging: in eclipse"
            )
            return None

        # Get optimal charging pointing from solar panel
        optimal_ra, optimal_dec = self.solar_panel.optimal_charging_pointing(
            utime, ephem
        )

        # Find valid pointing that doesn't violate constraints
        if self.sidemount:
            charging_ra, charging_dec = self._find_valid_pointing_sidemount(
                optimal_ra, optimal_dec, utime, lastra, lastdec
            )
        else:
            charging_ra, charging_dec = self._find_valid_pointing(
                optimal_ra, optimal_dec, utime, ephem, lastra, lastdec
            )

        if charging_ra is None or charging_dec is None:
            self._log_or_print(utime, "ERROR", "No valid charging pointing found")
            return None

        # Create the charging PPT
        charging_ppt = self._create_pointing(charging_ra, charging_dec, utime)

        self._log_or_print(
            utime,
            "CHARGING",
            f"Starting EMERGENCY CHARGING pointing at RA={charging_ra:.2f}, Dec={charging_dec:.2f}, obsid={charging_ppt.obsid}",
        )

        self.current_charging_ppt = charging_ppt
        return charging_ppt

    def initiate_emergency_charging(
        self,
        utime: float,
        ephem: rust_ephem.Ephemeris,
        lastra: float,
        lastdec: float,
        current_ppt: "Pointing | None",
    ) -> "Pointing | None":
        """Terminate current science PPT (if any) and create a charging PPT.

        This encapsulates the initiation path so callers can delegate the
        battery-alert transition. Callers are still responsible for updating
        their own "active PPT" reference and enqueuing any ACS slews.

        Returns the created charging Pointing or None if not possible.
        """
        if current_ppt is not None and not getattr(current_ppt, "done", False):
            self._log_or_print(
                utime,
                "ERROR",
                "BATTERY ALERT: Terminating science observation for emergency charging",
            )
            current_ppt.end = utime
            current_ppt.done = True

        return self.create_charging_pointing(utime, ephem, lastra, lastdec)

    def _is_in_sunlight(self, utime: float, ephem: rust_ephem.Ephemeris) -> bool:
        """
        Check if spacecraft is in sunlight (not in eclipse).

        Args:
            utime: Current unix timestamp
            ephem: Ephemeris object

        Returns:
            True if in sunlight, False if in eclipse
        """
        return not self.constraint.in_eclipse(ra=0, dec=0, time=utime)

    def _find_valid_pointing(
        self,
        optimal_ra: float,
        optimal_dec: float,
        utime: float,
        ephem: rust_ephem.Ephemeris,
        current_ra: float = 0.0,
        current_dec: float = 0.0,
    ) -> tuple[float | None, float | None]:
        """
        Find a valid pointing that doesn't violate constraints.

        Tries the optimal pointing first. If that violates constraints,
        searches for alternative pointings that maintain good solar panel
        illumination by exploring different combinations of RA and Dec offsets.
        If max_slew_deg is set, prioritizes pointings within the slew limit.

        Args:
            optimal_ra: Optimal RA for charging
            optimal_dec: Optimal Dec for charging
            utime: Current unix timestamp
            ephem: Ephemeris object for calculating panel illumination
            current_ra: Current pointing RA in degrees
            current_dec: Current pointing Dec in degrees

        Returns:
            Tuple of (ra, dec) if valid pointing found, or (None, None) if not
        """
        # Validate optimal pointing
        if not self.constraint.in_constraint(optimal_ra, optimal_dec, utime):
            # Check if within slew limit
            if self.max_slew_deg is not None:
                slew = angular_separation(
                    current_ra, current_dec, optimal_ra, optimal_dec
                )
                if slew > self.max_slew_deg:
                    self._log_or_print(
                        utime,
                        "CHARGING",
                        f"Optimal charging pointing requires {slew:.1f}° slew (limit: {self.max_slew_deg:.1f}°), searching for closer alternative",
                    )
                else:
                    return optimal_ra, optimal_dec
            else:
                return optimal_ra, optimal_dec

        self._log_or_print(
            utime,
            "CHARGING",
            "Emergency charging pointing violates constraints, searching for alternative",
        )

        # Search strategy: Try multiple RA/Dec combinations and select the one
        # with best solar panel illumination that doesn't violate constraints
        best_ra = None
        best_dec = None
        best_illumination = 0.0

        # For side-mounted panels, explore RA offsets while keeping Dec similar
        # For body-mounted panels, we need to maintain pointing near the Sun

        # Generate candidate pointings:
        # 1. RA offsets with same Dec (±30, ±60, ±90, ±120, ±150, ±180 degrees)
        # 2. Dec offsets with same RA (±10, ±20, ±30 degrees, clamped to [-90, 90])
        # 3. Combined RA and Dec offsets for more options

        candidates = []

        # RA offsets only
        for ra_offset in [30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]:
            alt_ra = (optimal_ra + ra_offset) % 360.0
            candidates.append((alt_ra, optimal_dec))

        # Dec offsets only
        for dec_offset in [10, -10, 20, -20, 30, -30]:
            alt_dec = np.clip(optimal_dec + dec_offset, -90.0, 90.0)
            if abs(alt_dec - optimal_dec) > 0.1:  # Only if actually different
                candidates.append((optimal_ra, alt_dec))

        # Combined offsets (smaller range to avoid too many candidates)
        for ra_offset in [45, -45, 90, -90, 135, -135]:
            for dec_offset in [15, -15, 30, -30]:
                alt_ra = (optimal_ra + ra_offset) % 360.0
                alt_dec = np.clip(optimal_dec + dec_offset, -90.0, 90.0)
                candidates.append((alt_ra, alt_dec))

        # Evaluate each candidate
        for alt_ra, alt_dec in candidates:
            # Check if this pointing violates constraints
            if self.constraint.in_constraint(alt_ra, alt_dec, utime):
                continue  # Skip constrained pointings

            # Check slew distance if limit is set
            if self.max_slew_deg is not None:
                slew = angular_separation(current_ra, current_dec, alt_ra, alt_dec)
                if slew > self.max_slew_deg:
                    continue  # Skip pointings beyond slew limit

            # Calculate solar panel illumination for this pointing
            illumination = self.solar_panel.panel_illumination_fraction(
                time=utime, ra=alt_ra, dec=alt_dec, ephem=ephem
            )

            # Ensure we have a float (should be scalar for single time)
            if isinstance(illumination, np.ndarray):
                illumination = float(illumination[0])

            # Keep track of the best unconstrained pointing
            if illumination > best_illumination:
                best_illumination = illumination
                best_ra = alt_ra
                best_dec = alt_dec

        if best_ra is not None:
            self._log_or_print(
                utime,
                "CHARGING",
                f"Found alternative charging pointing at RA={best_ra:.2f}, Dec={best_dec:.2f} with {best_illumination:.1%} illumination",
            )
            return best_ra, best_dec

        return None, None

    def _find_valid_pointing_sidemount(
        self,
        sun_ra: float,
        sun_dec: float,
        utime: float,
        current_ra: float | None = None,
        current_dec: float | None = None,
    ) -> tuple[float | None, float | None]:
        """
        Find a valid pointing for side-mounted solar panels.

        For side-mounted panels, any pointing 90° away from the Sun is power-positive.
        This method searches the great circle of pointings perpendicular to the Sun
        vector to find one that doesn't violate constraints.
        If max_slew_deg is set, prioritizes pointings within the slew limit.

        Strategy:
        1. Generate candidates along the great circle 90° from Sun
        2. Test each for constraint violations and slew distance
        3. Return the closest valid pointing within slew limit, or best option if none within limit

        Args:
            sun_ra: Sun's right ascension in degrees
            sun_dec: Sun's declination in degrees
            utime: Current unix timestamp
            current_ra: Current pointing RA in degrees
            current_dec: Current pointing Dec in degrees

        Returns:
            Tuple of (ra, dec) if valid pointing found, or (None, None) if not
        """
        # Generate candidate pointings on the great circle 90° from the Sun
        # We'll sample at regular intervals around this circle
        candidates = []

        # For side-mounted panels, we want pointings perpendicular to Sun vector
        # The set of all such pointings forms a great circle

        # Strategy: Sample pointings by rotating around the Sun vector
        # Convert Sun position to unit vector
        sun_ra_rad = np.radians(sun_ra)
        sun_dec_rad = np.radians(sun_dec)

        sun_x = np.cos(sun_dec_rad) * np.cos(sun_ra_rad)
        sun_y = np.cos(sun_dec_rad) * np.sin(sun_ra_rad)
        sun_z = np.sin(sun_dec_rad)
        sun_vec = np.array([sun_x, sun_y, sun_z])

        # Create two orthogonal vectors perpendicular to Sun
        # First perpendicular vector: use cross product with z-axis (or x if Sun near pole)
        if abs(sun_z) < 0.9:
            perp1 = np.cross(sun_vec, [0, 0, 1])
        else:
            perp1 = np.cross(sun_vec, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)

        # Second perpendicular vector: cross Sun with first perpendicular
        perp2 = np.cross(sun_vec, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        # Sample angles around the circle (every 15 degrees)
        for angle_deg in range(0, 360, 15):
            angle_rad = np.radians(angle_deg)

            # Point on the great circle 90° from Sun
            pointing_vec = np.cos(angle_rad) * perp1 + np.sin(angle_rad) * perp2

            # Convert back to RA/Dec
            x, y, z = pointing_vec
            dec_rad = np.arcsin(z)
            ra_rad = np.arctan2(y, x)

            candidate_ra = np.degrees(ra_rad) % 360.0
            candidate_dec = np.degrees(dec_rad)

            # Verify this is actually 90° from Sun (within numerical precision)
            sep = np.degrees(np.arccos(np.clip(np.dot(pointing_vec, sun_vec), -1, 1)))
            if abs(sep - 90.0) < 1.0:  # Within 1 degree of 90°
                candidates.append((candidate_ra, candidate_dec))

        # Test each candidate for constraint violations
        # If slew limit is set, find the closest valid pointing within limit
        best_candidate = None
        best_slew = float("inf")

        for candidate_ra, candidate_dec in candidates:
            if self.constraint.in_constraint(candidate_ra, candidate_dec, utime):
                continue  # Skip constrained pointings

            # If no current position provided, return first valid pointing
            if current_ra is None or current_dec is None:
                self._log_or_print(
                    utime,
                    "CHARGING",
                    f"Found side-mount charging pointing at RA={candidate_ra:.2f}, Dec={candidate_dec:.2f} (90° from Sun at RA={sun_ra:.2f}, Dec={sun_dec:.2f})",
                )
                return candidate_ra, candidate_dec

            # Calculate slew distance
            slew = angular_separation(
                current_ra, current_dec, candidate_ra, candidate_dec
            )

            # If no slew limit, return first valid pointing
            if self.max_slew_deg is None:
                self._log_or_print(
                    utime,
                    "CHARGING",
                    f"Found side-mount charging pointing at RA={candidate_ra:.2f}, Dec={candidate_dec:.2f} (90° from Sun at RA={sun_ra:.2f}, Dec={sun_dec:.2f})",
                )
                return candidate_ra, candidate_dec

            # If within slew limit, track the closest one
            if slew <= self.max_slew_deg and slew < best_slew:
                best_candidate = (candidate_ra, candidate_dec)
                best_slew = slew

        # Return the closest valid pointing within slew limit
        if best_candidate is not None:
            candidate_ra, candidate_dec = best_candidate
            self._log_or_print(
                utime,
                "CHARGING",
                f"Found side-mount charging pointing at RA={candidate_ra:.2f}, Dec={candidate_dec:.2f} (90° from Sun, {best_slew:.1f}° slew)",
            )
            return best_candidate

        self._log_or_print(
            utime,
            "ERROR",
            "No valid side-mount charging pointing found (all perpendicular pointings violate constraints)",
        )
        return None, None

    def _create_pointing(self, ra: float, dec: float, utime: float) -> "Pointing":
        """
        Create a Pointing object for emergency charging.

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            utime: Current unix timestamp

        Returns:
            Configured Pointing object
        """
        from ..targets import Pointing

        charging_ppt = Pointing(
            config=self.config,
            ra=ra,
            dec=dec,
            name=f"EMERGENCY_CHARGE_{self.next_charging_obsid}",
            obsid=self.next_charging_obsid,
            exptime=86400,
        )
        self.next_charging_obsid += 1
        charging_ppt.begin = int(utime)
        charging_ppt.end = int(utime + 86400)  # Set far end time
        charging_ppt.done = False

        return charging_ppt

    def clear_current_charging(self) -> None:
        """Clear the current charging PPT reference."""
        self.current_charging_ppt = None

    def is_charging_active(self) -> bool:
        """
        Check if emergency charging is currently active.

        Returns:
            True if charging is active, False otherwise
        """
        return self.current_charging_ppt is not None

    def check_termination(
        self, utime: float, battery: Battery, ephem: rust_ephem.Ephemeris
    ) -> str | None:
        """Evaluate whether the current emergency charging pointing should terminate.

        Returns a string reason or None if charging should continue. Reasons:
        - 'battery_recharged': Battery alert cleared
        - 'constraint': Current charging pointing violates constraints
        - 'eclipse': Spacecraft entered eclipse (suppression of restart recommended)
        """
        # No active charging pointing
        if self.current_charging_ppt is None:
            return None

        # Battery recovered
        if not battery.battery_alert:
            return "battery_recharged"

        # Constraint violation (e.g., occultation)
        if self.constraint.in_constraint(
            self.current_charging_ppt.ra, self.current_charging_ppt.dec, utime
        ):
            return "constraint"

        # Entered eclipse
        if not self._is_in_sunlight(utime, ephem):
            return "eclipse"

        return None

    def terminate_current_charging(self, utime: float) -> None:
        """Mark the current charging pointing as ended and clear reference."""
        if self.current_charging_ppt is None:
            return
        self.current_charging_ppt.end = utime
        self.current_charging_ppt.done = True
        self.clear_current_charging()

    def should_initiate_charging(
        self, utime: float, ephem: rust_ephem.Ephemeris, battery_alert: bool
    ) -> bool:
        """Determine if emergency charging should be initiated.

        Returns True if charging should start, False otherwise.
        Manages eclipse suppression internally to avoid thrashing.
        """
        if not battery_alert:
            return False

        in_sunlight = self._is_in_sunlight(utime, ephem)

        if self._charging_suppressed_due_to_eclipse and not in_sunlight:
            # Still in eclipse; skip initiating charging
            return False

        if not in_sunlight:
            # First time noticing eclipse: set suppression
            self._log_or_print(
                utime,
                "CHARGING",
                "Battery alert in eclipse - suppressing charging until sunlight",
            )
            self._charging_suppressed_due_to_eclipse = True
            return False

        # Sunlight available; clear suppression and allow charging
        self._charging_suppressed_due_to_eclipse = False
        return True
