from pydantic import BaseModel

from .spacecraft_bus import PowerDraw
from .thermal import Heater


class Instrument(BaseModel):
    """
    A model representing a spacecraft instrument with power consumption characteristics.

    This class defines an instrument's basic properties including its name and power
    draw specifications. It provides methods to query power consumption in different
    operational modes.

    Attributes:
        name (str): The name of the instrument. Defaults to "Default Instrument".
        power_draw (PowerDraw): The power draw characteristics of the instrument,
            including nominal power, peak power, and mode-specific power settings.
            Defaults to a PowerDraw with 50W nominal and 100W peak power.

    Methods:
        power(mode): Returns the power draw for the specified operational mode.

    Example:
        >>> instrument = Instrument(name="Camera", power_draw=PowerDraw(nominal_power=75))
        >>> instrument.power()
        75.0
    """

    name: str = "Default Instrument"
    power_draw: PowerDraw = PowerDraw(nominal_power=50, peak_power=100, power_mode={})
    heater: Heater | None = None

    def power(self, mode: int | None = None) -> float:
        """Get the power draw for the instrument in the given mode."""
        base_power = self.power_draw.power(mode)
        heater_power = self.heater.heat_power(mode) if self.heater else 0.0
        return base_power + heater_power


class InstrumentSet(BaseModel):
    """
    A collection of instruments that can be operated together.

    This class manages multiple Instrument instances and provides aggregate
    operations across all instruments in the set.

    Attributes:
        instruments (list[Instrument]): A list of Instrument objects. Defaults to
            a single default Instrument instance.

    Methods:
        power(mode): Calculate the total power consumption across all instruments.

    Example:
        >>> instrument_set = InstrumentSet(instruments=[instrument1, instrument2])
        >>> instrument_set.power()
        125.0
    """

    instruments: list[Instrument] = [Instrument()]

    def power(self, mode: int | None = None) -> float:
        """Get the total power draw for all instruments in the given mode."""
        return sum(instrument.power(mode) for instrument in self.instruments)
