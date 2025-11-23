from enum import Enum


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
