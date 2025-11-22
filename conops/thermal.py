from pydantic import BaseModel

from .power import PowerDraw


class Heater(BaseModel):
    """Simple model of a spacecraft heater that draws power depending on the mode."""

    name: str
    power: PowerDraw

    def heat_power(self, mode: int | None = None) -> float:
        """Get the heater power in the given mode."""
        return self.power.power(mode)
