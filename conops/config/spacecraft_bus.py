from pydantic import BaseModel

from .acs import AttitudeControlSystem
from .power import PowerDraw
from .thermal import Heater


class SpacecraftBus(BaseModel):
    name: str = "Default Bus"
    power_draw: PowerDraw = PowerDraw()
    attitude_control: AttitudeControlSystem = AttitudeControlSystem()
    heater: Heater | None = None

    def power(self, mode: int | None = None, in_eclipse: bool = False) -> float:
        """Get the power draw for the spacecraft bus in the given mode.

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
