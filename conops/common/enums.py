from enum import Enum, auto
from typing import Literal

DITLEventType = Literal[
    "ACS",
    "CHARGING",
    "CONSTRAINT",
    "ERROR",
    "INFO",
    "OBSERVATION",
    "PASS",
    "QUEUE",
    "RADIATOR_HARD_CONSTRAINT",
    "SAFE",
    "SCHEDULER",
    "SLEW",
    "STAR_TRACKER_HARD_CONSTRAINT",
    "STAR_TRACKER_SOFT_CONSTRAINT",
    "TELESCOPE_HARD_CONSTRAINT",
    "TOO",
]
"""Closed set of DITLEvent.event_type category strings used across the simulator."""


class BoresightAxis(str, Enum):
    """Spacecraft body axis that is the primary pointing/boresight direction.

    Determines which body-frame axis aligns with the target when the spacecraft
    is "pointing at" a given (RA, Dec).  The default is ``+X``, which is the
    historical COASTSim convention.  Other common values are ``+Z`` (nadir-pointing
    Earth-observation buses) and ``+Y``.
    """

    PLUS_X = "+X"
    PLUS_Y = "+Y"
    PLUS_Z = "+Z"
    MINUS_X = "-X"
    MINUS_Y = "-Y"
    MINUS_Z = "-Z"

    def __str__(self) -> str:
        return self.value


class ACSMode(int, Enum):
    """Spacecraft ACS Modes"""

    SCIENCE = 0
    SLEWING = 1
    SAA = 2
    PASS = 3
    CHARGING = 4
    SAFE = 5
    IDLE = 6


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
    IDLE = "IDLE"  # Constraint-safe idle hold


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

    QUATERNION (default): Full 3-DOF SLERP in SO(3).  Couples pointing and
        roll changes through the shortest rotation path in quaternion space,
        giving a physically accurate attitude trajectory.

    CONSTRAINT_AVOIDING: Generalized constraint-avoiding SLERP using the
        combined rust-ephem constraint configuration.  Routes around any
        combination of Sun, Earth, Moon, and other exclusion zones.  Falls
        back to QUATERNION when no constraint violation is detected on the
        direct arc.
    """

    QUATERNION = "quaternion"
    CONSTRAINT_AVOIDING = "constraint_avoiding"
