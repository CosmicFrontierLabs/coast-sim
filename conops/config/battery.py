from typing import Any

from pydantic import Field, PrivateAttr, model_validator

from ..common import ChargeState
from ._base import ConfigModel


class Battery(ConfigModel):
    """It's a fake battery"""

    # Battery size - 20 Ah Voltage = 28V
    # Power drain - 253 W (daily average) - peak power = 416 w
    # Solar panel power - area = 2.0 m^2 -- solar constant = 1353 w/m^2 --
    # efficiency = 29.5%  = ~800W charge rate
    name: str = Field(default="Default Battery", description="Battery name/identifier")
    amphour: float = Field(
        default=20, description="Battery capacity in Ampere-hours (Ah)"
    )
    voltage: float = Field(default=28, description="Nominal battery voltage (V)")
    watthour: float = Field(
        default=560, description="Total energy storage capacity (Wh)"
    )
    emergency_recharge: bool = Field(
        default=False, description="Flag indicating emergency recharge mode"
    )
    max_depth_of_discharge: float = Field(
        default=0.3, description="Maximum allowed depth of discharge (0.0-1.0)"
    )
    recharge_threshold: float = Field(
        default=0.95, description="Battery level fraction to stop charging (0.0-1.0)"
    )
    charge_level: float = Field(
        default=0, description="Current battery charge level (Wh)"
    )
    emergency_charging_min_illumination: float = Field(
        default=1.0,
        description="Illumination threshold for emergency charging (0.0-1.0)",
    )

    _last_charge_power: float = PrivateAttr(default=0.0)

    @model_validator(mode="before")
    @classmethod
    def set_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Set derived default values"""
        if "watthour" not in values:
            values["watthour"] = values.get("amphour", 20) * values.get("voltage", 28)

        return values

    def model_post_init(self, __context: object) -> None:
        self.charge_level = self.watthour

    @property
    def charge_state(self) -> ChargeState:
        """Get the current charging state of the battery.

        Returns:
            ChargeState.NOT_CHARGING: No charging occurring
            ChargeState.CHARGING: Battery is being charged and not at full capacity
            ChargeState.TRICKLE: Battery is at 100% capacity and charging is occurring
        """
        if self._last_charge_power <= 0:
            return ChargeState.NOT_CHARGING
        elif self.battery_level >= 1.0:
            return ChargeState.TRICKLE
        else:
            return ChargeState.CHARGING

    @property
    def battery_alert(self) -> bool:
        """Is the battery in an alert status caused by discharge"""
        # Depth of discharge > max_depth_of_discharge, start an emergency recharge state
        if self.below_minimum_charge_level:
            self.emergency_recharge = True
            return True

        # Alert is True when battery level is below recharge threshold
        if self.battery_level < self.recharge_threshold:
            self.emergency_recharge = True
            return True
        else:
            self.emergency_recharge = False
            return False

    @property
    def minimum_charge_level(self) -> float:
        """Minimum allowed state of charge before the depth-of-discharge limit is exceeded."""
        return 1.0 - self.max_depth_of_discharge

    @property
    def below_minimum_charge_level(self) -> bool:
        """True when state of charge is below the configured depth-of-discharge floor."""
        return self.battery_level < self.minimum_charge_level

    def charge(self, power: float, period: float) -> None:
        """Charge the battery with <power> Watts for <period> seconds"""
        self._last_charge_power = power
        if self.charge_level < self.watthour:
            # Battery is not fully charged
            wattsec = power * period
            self.charge_level += wattsec / 3600  # watthours
            # Check if battery is more than 100% full
            if self.charge_level > self.watthour:
                self.charge_level = self.watthour

    def drain(self, power: float, period: float) -> bool:
        """Drain the battery with <power> Watts for <period> seconds

        Returns:
            bool: True if the drain was successful, False if battery was already empty
        """
        if self.charge_level > 0:
            # Battery has charge
            wattsec = power * period
            self.charge_level -= wattsec / 3600  # watthours
            # Check if battery is drained below 0
            if self.charge_level < 0:
                self.charge_level = 0
            return True
        else:
            # Battery is already empty
            return False

    @property
    def battery_level(self) -> float:
        return self.charge_level / self.watthour
