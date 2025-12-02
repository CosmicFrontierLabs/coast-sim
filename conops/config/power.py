from pydantic import BaseModel


class PowerDraw(BaseModel):
    """
    Power draw characteristics for a given subsystem, with option to specify
    different power draws based on operational modes.

    Supports eclipse-aware power consumption where different power levels can be
    specified for eclipse conditions (e.g., heaters drawing more power in eclipse).
    """

    nominal_power: float = 50  # Watts
    peak_power: float = 300
    power_mode: dict[int, float] = {}
    eclipse_power: float | None = None
    eclipse_power_mode: dict[int, float] = {}

    def power(self, mode: int | None = None, in_eclipse: bool = False) -> float:
        """Get power draw for the given mode and eclipse state.

        Args:
            mode: Operational mode (None for nominal)
            in_eclipse: Whether spacecraft is in eclipse

        Returns:
            Power draw in watts
        """
        # If in eclipse and eclipse power is configured, use eclipse values
        if in_eclipse:
            if mode is not None and mode in self.eclipse_power_mode:
                return self.eclipse_power_mode[mode]
            if self.eclipse_power is not None:
                return self.eclipse_power

        # Otherwise use normal power values
        if mode is None:
            return self.nominal_power
        return self.power_mode.get(mode, self.nominal_power)
