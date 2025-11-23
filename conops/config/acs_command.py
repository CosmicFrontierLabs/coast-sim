from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ..common import ACSCommandType

if TYPE_CHECKING:
    from ..simulation.passes import Pass
    from ..simulation.slew import Slew


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
