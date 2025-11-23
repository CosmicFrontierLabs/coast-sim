from math import isclose

import pytest

from conops import Battery
from conops.common.enums import ChargeState

"""Unit tests for conops.battery.Battery"""


class TestBatteryValidation:
    def test_validator_sets_watthour_calculation(self, batt_20wh):
        b = batt_20wh
        assert b.watthour == 20

    def test_init_charge_level_equals_watthour(self, batt_20wh):
        b = batt_20wh
        assert isclose(b.charge_level, b.watthour)

    def test_battery_level_is_full_on_init(self, batt_20wh):
        b = batt_20wh
        assert b.battery_level == pytest.approx(1.0)

    def test_provided_watthour_is_preserved_value(self, batt_20wh):
        b = Battery(amphour=2, voltage=10, watthour=123.0)
        assert b.watthour == 123.0

    def test_provided_watthour_sets_initial_charge_level(self, batt_20wh):
        b = Battery(amphour=2, voltage=10, watthour=123.0)
        assert isclose(b.charge_level, 123.0)


class TestBatteryChargeAndDrain:
    def test_initial_charge_level_full(self, batt_1wh):
        b = batt_1wh
        assert isclose(b.charge_level, 1.0)

    def test_charge_increases_to_cap(self, batt_1wh):
        b = batt_1wh
        b.charge_level = 0.25
        b.charge(power=3600, period=1)  # Charge 1 Wh -> should cap at 1.0
        assert isclose(b.charge_level, 1.0)

    def test_charge_does_not_exceed_watthour_on_subsequent_charge(self, batt_1wh):
        b = batt_1wh
        b.charge_level = 0.25
        b.charge(power=3600, period=1)
        b.charge(power=7200, period=1)
        assert isclose(b.charge_level, 1.0)

    def test_drain_reduces_charge_level_to_zero(self, batt_1wh):
        b = batt_1wh
        b.drain(power=3600, period=1)
        assert isclose(b.charge_level, 0.0)

    def test_drain_return_false_when_empty(self, batt_1wh):
        b = batt_1wh
        b.drain(power=3600, period=1)
        assert b.drain(power=3600, period=1) is False

    def test_battery_level_fraction_when_setting_charge_level(self, batt_20wh):
        b = batt_20wh
        b.charge_level = 10.0
        assert isclose(b.battery_level, 0.5)

    def test_battery_level_fraction_decreases_after_drain(self, batt_20wh):
        b = batt_20wh
        b.charge_level = 10.0
        b.drain(power=3600, period=1)
        assert isclose(b.battery_level, 9.0 / 20.0)


class TestBatteryAlerts:
    def test_default_battery_level_full(self, default_battery):
        b = default_battery
        assert b.battery_level == pytest.approx(1.0)

    def test_default_battery_alert_is_false(self, default_battery):
        b = default_battery
        assert b.battery_alert is False

    def test_default_emergency_recharge_is_false(self, default_battery):
        b = default_battery
        assert b.emergency_recharge is False

    def test_battery_level_below_max_depth_of_discharge(self, default_battery):
        b = default_battery
        b.charge_level = b.watthour * 0.2  # 20% charge level (below 70% minimum)
        min_allowed_charge = 1.0 - b.max_depth_of_discharge  # 0.7
        assert b.battery_level < min_allowed_charge

    def test_battery_alert_true_when_below_max_depth_of_discharge(
        self, default_battery
    ):
        b = default_battery
        b.charge_level = b.watthour * 0.6
        assert b.battery_alert is True

    def test_emergency_recharge_true_when_below_max_depth_of_discharge(
        self, default_battery
    ):
        b = default_battery
        b.charge_level = b.watthour * 0.6
        # Trigger battery_alert to set emergency_recharge state
        _ = b.battery_alert
        assert b.emergency_recharge is True

    def test_battery_level_below_recharge_threshold(self, default_battery):
        b = default_battery
        b.charge_level = b.watthour * 0.94
        assert b.battery_level < b.recharge_threshold

    def test_battery_alert_true_when_below_recharge_threshold(self, default_battery):
        b = default_battery
        # First discharge below max_depth_of_discharge to trigger emergency_recharge
        b.charge_level = b.watthour * 0.6
        _ = b.battery_alert
        # Now recharge to test the recharge_threshold condition
        b.charge_level = b.watthour * 0.94
        assert b.battery_alert is True

    def test_emergency_recharge_true_when_below_recharge_threshold(
        self, default_battery
    ):
        b = default_battery
        # First discharge below max_depth_of_discharge to trigger emergency_recharge
        b.charge_level = b.watthour * 0.6
        _ = b.battery_alert
        # Now recharge to test the recharge_threshold condition
        b.charge_level = b.watthour * 0.94
        assert b.emergency_recharge is True

    def test_battery_alert_cleared_above_recharge_threshold(self, default_battery):
        b = default_battery
        b.charge_level = b.watthour * 0.96
        assert b.battery_alert is False

    def test_emergency_recharge_cleared_above_recharge_threshold(self, default_battery):
        b = default_battery
        b.charge_level = b.watthour * 0.96
        assert b.emergency_recharge is False


class TestBatteryChargeState:
    def test_initial_charge_state_not_charging(self, batt_20wh):
        b = batt_20wh
        assert b.charge_state == ChargeState.NOT_CHARGING

    def test_charge_state_charging_when_power_positive_and_not_full(self, batt_20wh):
        b = batt_20wh
        b.charge_level = 10.0  # Not full
        b.charge(power=100, period=1)
        assert b.charge_state == ChargeState.CHARGING

    def test_charge_state_trickle_when_full_and_charging(self, batt_20wh):
        b = batt_20wh
        # Already full, charge again
        b.charge(power=100, period=1)
        assert b.charge_state == ChargeState.TRICKLE

    def test_charge_state_not_charging_when_power_zero(self, batt_20wh):
        b = batt_20wh
        b.charge_level = 10.0
        b.charge(power=0, period=1)
        assert b.charge_state == ChargeState.NOT_CHARGING

    def test_drain_clamps_charge_level_to_zero_when_over_drained(self, batt_1wh):
        b = batt_1wh
        b.drain(power=3600, period=2)  # Drain 2 Wh from 1 Wh battery
        assert isclose(b.charge_level, 0.0)
        assert isclose(b.battery_level, 0.0)
