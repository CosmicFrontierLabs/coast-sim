from pydantic import BaseModel, Field

from .spacecraft_bus import PowerDraw
from .thermal import Heater


class DataGeneration(BaseModel):
    """
    A model representing data generation characteristics for an instrument.

    This class defines how an instrument generates data during observations,
    either at a constant rate or per observation.

    Attributes:
        rate_gbps (float): Data generation rate in Gigabits per second when active.
            Defaults to 0.0 (no data generation).
        per_observation_gb (float): Fixed amount of data generated per observation in Gb.
            If non-zero, this takes precedence over rate_gbps. Defaults to 0.0.

    Example:
        >>> # Instrument that generates 0.1 Gbps continuously
        >>> data_gen = DataGeneration(rate_gbps=0.1)
        >>> # Instrument that generates 5 Gb per observation
        >>> data_gen2 = DataGeneration(per_observation_gb=5.0)
    """

    rate_gbps: float = Field(
        default=0.0, ge=0.0, description="Data generation rate in Gbps when active"
    )
    per_observation_gb: float = Field(
        default=0.0,
        ge=0.0,
        description="Fixed data generated per observation in Gb",
    )

    def data_generated(self, duration_seconds: float) -> float:
        """Calculate data generated over a given duration.

        Args:
            duration_seconds: Duration of data generation in seconds.

        Returns:
            float: Amount of data generated in Gigabits.

        Note:
            If per_observation_gb is set, it returns that value regardless of duration.
            Otherwise, returns rate_gbps * duration_seconds.
        """
        if self.per_observation_gb > 0:
            return self.per_observation_gb
        return self.rate_gbps * duration_seconds


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
    data_generation: DataGeneration = DataGeneration()

    def power(self, mode: int | None = None, in_eclipse: bool = False) -> float:
        """Get the power draw for the instrument in the given mode.

        Args:
            mode: Operational mode (None for nominal)
            in_eclipse: Whether spacecraft is in eclipse

        Returns:
            Total power draw in watts
        """
        base_power = self.power_draw.power(mode, in_eclipse=in_eclipse)
        heater_power = (
            self.heater.power(mode, in_eclipse=in_eclipse) if self.heater else 0.0
        )
        return base_power + heater_power


class Payload(BaseModel):
    """
    A collection of payload that can be operated together.

    This class manages multiple Instrument instances and provides aggregate
    operations across all instruments in the payload.

    Attributes:
        payload (list[Instrument]): A list of Instrument objects. Defaults to
            a single default Instrument instance.

    Methods:
        power(mode): Calculate the total power consumption across all instruments.

    Example:
        >>> payload = Payload(payload=[instrument1, instrument2])
        >>> payload.power()
        125.0
    """

    payload: list[Instrument] = [Instrument()]

    def power(self, mode: int | None = None, in_eclipse: bool = False) -> float:
        """Get the total power draw for all instruments in the payload in the given mode.

        Args:
            mode: Operational mode (None for nominal)
            in_eclipse: Whether spacecraft is in eclipse

        Returns:
            Total power draw in watts
        """
        return sum(
            instrument.power(mode, in_eclipse=in_eclipse) for instrument in self.payload
        )

    def total_data_rate_gbps(self) -> float:
        """Get the total data generation rate across all instruments.

        Returns:
            float: Total data generation rate in Gigabits per second.
        """
        return sum(instrument.data_generation.rate_gbps for instrument in self.payload)

    def data_generated(self, duration_seconds: float) -> float:
        """Calculate total data generated by all instruments over a duration.

        Args:
            duration_seconds: Duration of data generation in seconds.

        Returns:
            float: Total amount of data generated in Gigabits.

        Note:
            For instruments with per_observation_gb set, that amount is added once.
            For instruments with rate_gbps, the amount is calculated based on duration.
        """
        total_data = 0.0
        for instrument in self.payload:
            total_data += instrument.data_generation.data_generated(duration_seconds)
        return total_data
