from pydantic import BaseModel, model_validator

from ..common import ChargeState


class Battery(BaseModel):
    """It's a fake battery"""

    # Battery size - 20 Ah Voltage = 28V
    # Power drain - 253 W (daily average) - peak power = 416 w
    # Solar panel power - area = 2.0 m^2 -- solar constant = 1353 w/m^2 --
    # efficiency = 29.5%  = ~800W charge rate
    name: str = "Default Battery"
    amphour: float = 20  # amphour
    voltage: float = 28  # Volts
    watthour: float = 560  # 20 * 28
    emergency_recharge: bool = False
    max_depth_of_discharge: float = 0.3  # Maximum allowed depth of discharge (30%)
    recharge_threshold: float = (
        0.95  # Threshold at which emergency recharge ends (95% SOC)
    )
    charge_level: float = 0  # Current charge level in watthours

    @model_validator(mode="before")
    @classmethod
    def set_defaults(cls, values):
        """Set derived default values"""
        if "watthour" not in values:
            values["watthour"] = values.get("amphour", 20) * values.get("voltage", 28)

        return values

    def __init__(self, **data):
        super().__init__(**data)

        self.charge_level = self.watthour  # Start fully charged
        self._last_charge_power = 0.0  # Track last charge power for state determination

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
    def battery_alert(self):
        """Is the battery in an alert status caused by discharge"""
        # Calculate minimum allowed charge level from max depth of discharge
        min_charge_level = 1.0 - self.max_depth_of_discharge

        # Depth of discharge > max_depth_of_discharge, start an emergency recharge state
        if self.battery_level < min_charge_level:
            self.emergency_recharge = True

        # Alert is True when battery level is below recharge threshold
        if self.battery_level < self.recharge_threshold:
            self.emergency_recharge = True
            return True
        else:
            self.emergency_recharge = False
            return False

    def charge(self, power, period):
        """Charge the battery with <power> Watts for <period> seconds"""
        self._last_charge_power = power
        if self.charge_level < self.watthour:
            # Battery is not fully charged
            wattsec = power * period
            self.charge_level += wattsec / 3600  # watthours
            # Check if battery is more than 100% full
            if self.charge_level > self.watthour:
                self.charge_level = self.watthour

    def drain(self, power, period):
        """Charge the battery with <power> Watts for <period> seconds"""
        if self.charge_level > 0:
            # Battery is not fully charged
            wattsec = power * period
            self.charge_level -= wattsec / 3600  # watthours
            # Check if battery is more than 100% full
            if self.charge_level < 0:
                self.charge_level = 0
        else:
            return False

    @property
    def battery_level(self):
        return self.charge_level / self.watthour
