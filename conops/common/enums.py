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
