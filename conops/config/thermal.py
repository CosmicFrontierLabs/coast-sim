from pydantic import Field

from ._base import ConfigModel
from .power import PowerDraw


class Heater(ConfigModel):
    """Simple model of a spacecraft heater that draws power depending on the mode.

    Heaters typically draw more power during eclipse when there is no solar heating,
    requiring active thermal management to maintain temperature.
    """

    name: str = Field(description="Name/identifier for the heater")
    power_draw: PowerDraw = Field(
        description="Power draw characteristics for the heater"
    )

    def power(self, mode: int | None = None, in_eclipse: bool = False) -> float:
        """Get the heater power in the given mode and eclipse state.

        Args:
            mode: Operational mode (None for nominal)
            in_eclipse: Whether spacecraft is in eclipse

        Returns:
            Heater power draw in watts
        """
        return self.power_draw.power(mode, in_eclipse=in_eclipse)
