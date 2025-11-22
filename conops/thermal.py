from pydantic import BaseModel

from .power import PowerDraw


class Heater(BaseModel):
    """Simple model of a spacecraft heater that draws power depending on the mode.

    Heaters typically draw more power during eclipse when there is no solar heating,
    requiring active thermal management to maintain temperature.
    """

    name: str
    power_draw: PowerDraw

    def power(self, mode: int | None = None, in_eclipse: bool = False) -> float:
        """Get the heater power in the given mode and eclipse state.

        Args:
            mode: Operational mode (None for nominal)
            in_eclipse: Whether spacecraft is in eclipse

        Returns:
            Heater power draw in watts
        """
        return self.power_draw.power(mode, in_eclipse=in_eclipse)
