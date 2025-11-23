from enum import Enum, auto
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from ..simulation.passes import Pass
    from ..simulation.slew import Slew


class ACSCommandType(Enum):
    """Types of commands that can be queued for the ACS."""

    SLEW_TO_TARGET = auto()
    START_PASS = auto()
    END_PASS = auto()
    START_BATTERY_CHARGE = auto()
    END_BATTERY_CHARGE = auto()
    ENTER_SAFE_MODE = auto()


class ACSCommand(BaseModel):
    """A command to be executed by the ACS state machine."""

    command_type: ACSCommandType
    execution_time: float
    slew: "Slew | Pass | None" = None
    ra: float | None = None
    dec: float | None = None
    obsid: int | None = None
    obstype: str = "PPT"

    model_config = ConfigDict(arbitrary_types_allowed=True)
