import pytest

from conops.acs import ACS


def test_fault_management_adds_default_battery_threshold(base_config):
    assert "battery_level" in base_config.fault_management.thresholds


def test_fault_management_yellow_state_and_accumulation(base_config):
    fm = base_config.fault_management
    acs = ACS(constraint=base_config.constraint, config=base_config)
    # Simulate battery level between yellow and red
    base_config.battery.charge_level = base_config.battery.watthour * (
        fm.thresholds["battery_level"].yellow - 0.01
    )
    fm.check(
        {"battery_level": base_config.battery.battery_level},
        utime=1000.0,
        step_size=60.0,
        acs=acs,
    )
    stats = fm.statistics()["battery_level"]
    assert stats["current"] == "yellow"
    assert stats["yellow_seconds"] == pytest.approx(60.0)
    assert stats["red_seconds"] == 0.0
    assert not acs.in_safe_mode


def test_fault_management_red_triggers_safe_mode(base_config):
    fm = base_config.fault_management
    acs = ACS(constraint=base_config.constraint, config=base_config)
    # Force battery below red limit
    base_config.battery.charge_level = base_config.battery.watthour * (
        fm.thresholds["battery_level"].red - 0.01
    )
    fm.check(
        {"battery_level": base_config.battery.battery_level},
        utime=2000.0,
        step_size=60.0,
        acs=acs,
    )
    # Verify safe mode flag was set
    assert fm.safe_mode_requested
    stats = fm.statistics()["battery_level"]
    assert stats["current"] == "red"
    assert stats["red_seconds"] == pytest.approx(60.0)


def test_fault_management_multiple_cycles_accumulate(base_config):
    fm = base_config.fault_management
    acs = ACS(constraint=base_config.constraint, config=base_config)
    yellow_limit = fm.thresholds["battery_level"].yellow
    # Cycle 1: nominal (no accumulation)
    base_config.battery.charge_level = base_config.battery.watthour * (
        yellow_limit + 0.05
    )
    fm.check(
        {"battery_level": base_config.battery.battery_level},
        utime=3000.0,
        step_size=60.0,
        acs=acs,
    )
    # Cycle 2: yellow
    base_config.battery.charge_level = base_config.battery.watthour * (
        yellow_limit - 0.01
    )
    fm.check(
        {"battery_level": base_config.battery.battery_level},
        utime=3060.0,
        step_size=60.0,
        acs=acs,
    )
    # Cycle 3: yellow again
    fm.check(
        {"battery_level": base_config.battery.battery_level},
        utime=3120.0,
        step_size=60.0,
        acs=acs,
    )
    stats = fm.statistics()["battery_level"]
    assert stats["yellow_seconds"] == pytest.approx(120.0)
    assert stats["red_seconds"] == 0.0
    assert not acs.in_safe_mode
