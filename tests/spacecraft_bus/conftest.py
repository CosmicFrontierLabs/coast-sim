"""Test fixtures for spacecraft_bus subsystem tests."""

import pytest

from conops import AttitudeControlSystem, PowerDraw, SpacecraftBus


@pytest.fixture
def default_power_draw():
    return PowerDraw()


@pytest.fixture
def custom_power_draw():
    return PowerDraw(nominal_power=150, peak_power=250, power_mode={1: 175, 2: 225})


@pytest.fixture
def default_acs():
    return AttitudeControlSystem()


@pytest.fixture
def custom_acs():
    return AttitudeControlSystem(
        slew_acceleration=1.0,
        max_slew_rate=0.5,
        slew_accuracy=0.05,
        settle_time=60.0,
    )


@pytest.fixture
def default_bus():
    return SpacecraftBus()


@pytest.fixture
def custom_bus():
    pd = PowerDraw(nominal_power=150)
    acs = AttitudeControlSystem(slew_acceleration=1.0)
    return SpacecraftBus(name="Custom Bus", power_draw=pd, attitude_control=acs)
