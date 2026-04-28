from enum import Enum, auto


class ACSMode(int, Enum):
    """Spacecraft ACS Modes"""

    SCIENCE = 0
    SLEWING = 1
    SAA = 2
    PASS = 3
    CHARGING = 4
    SAFE = 5


class ChargeState(int, Enum):
    """Battery Charging States"""

    NOT_CHARGING = 0
    CHARGING = 1
    TRICKLE = 2


class ACSCommandType(Enum):
    """Types of commands that can be queued for the ACS."""

    SLEW_TO_TARGET = auto()
    START_PASS = auto()
    END_PASS = auto()
    START_BATTERY_CHARGE = auto()
    END_BATTERY_CHARGE = auto()
    ENTER_SAFE_MODE = auto()


class ObsType(str, Enum):
    """Observation / slew type codes used throughout the simulation."""

    PPT = "PPT"  # Pre-Programmed Target (default scheduled observation)
    AT = "AT"  # Astronomical Target (queue-scheduled observation)
    TOO = "TOO"  # Target of Opportunity (unplanned high-priority observation)
    SAFE = "SAFE"  # Safe-mode pointing
    CHARGE = "CHARGE"  # Battery-charging / sun-pointing maneuver
    GSP = "GSP"  # Ground Station Pass


class AntennaType(str, Enum):
    """Antenna mounting and pointing configuration."""

    OMNI = "omni"  # Omnidirectional antenna
    FIXED = "fixed"  # Fixed pointing antenna
    GIMBALED = "gimbaled"  # Gimbaled (steerable) antenna


class Polarization(str, Enum):
    """Antenna polarization type."""

    LINEAR_HORIZONTAL = "linear_horizontal"
    LINEAR_VERTICAL = "linear_vertical"
    CIRCULAR_RIGHT = "circular_right"  # RHCP
    CIRCULAR_LEFT = "circular_left"  # LHCP
    DUAL = "dual"  # Supports multiple polarizations


class SlewAlgorithm(str, Enum):
    """Algorithm used to compute spacecraft slew paths.

    GREAT_CIRCLE: Shortest great-circle arc (existing behaviour). RA/Dec are
        interpolated independently, roll linearly. Fast and adequate for most
        slews that are far from the Sun.

    QUATERNION: Full 3-DOF SLERP in SO(3).  Couples pointing and roll changes
        through the shortest rotation path in quaternion space, giving a
        physically more accurate attitude trajectory.

    SUN_AVOIDING: Great-circle path with an automatic detour when the direct
        arc would violate the configured Sun exclusion angle.  Falls back to
        GREAT_CIRCLE when no violation is detected.
    """

    GREAT_CIRCLE = "great_circle"
    QUATERNION = "quaternion"
    SUN_AVOIDING = "sun_avoiding"
