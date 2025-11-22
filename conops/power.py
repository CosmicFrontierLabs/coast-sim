from pydantic import BaseModel


class PowerDraw(BaseModel):
    """
    Power draw characteristics for a given subsystem, with option to specify
    different power draws based on operational modes.
    """

    nominal_power: float = 200
    peak_power: float = 300
    power_mode: dict[int, float] = {}

    def power(self, mode: int | None = None) -> float:
        if mode is None:
            return self.nominal_power
        return self.power_mode.get(mode, self.nominal_power)
