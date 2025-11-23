"""Test fixtures for instrument subsystem tests."""

import pytest

from conops import Instrument, Payload, PowerDraw


# fixtures for single instrument and power draws
@pytest.fixture
def default_instrument():
    return Instrument()


@pytest.fixture
def pd_with_modes():
    return PowerDraw(
        nominal_power=75.0, peak_power=150.0, power_mode={0: 10.0, 1: 20.0}
    )


@pytest.fixture
def cam_instrument(pd_with_modes):
    return Instrument(name="Cam", power_draw=pd_with_modes)


# fixtures for payload tests
@pytest.fixture
def i1_10_20():
    return Instrument(
        power_draw=PowerDraw(nominal_power=10.0, peak_power=20.0, power_mode={})
    )


@pytest.fixture
def i2_20_40():
    return Instrument(
        power_draw=PowerDraw(nominal_power=20.0, peak_power=40.0, power_mode={})
    )


@pytest.fixture
def payload_10_20_and_20_40(i1_10_20, i2_20_40):
    return Payload(payload=[i1_10_20, i2_20_40])


@pytest.fixture
def i1_5_10_mode0():
    return Instrument(
        power_draw=PowerDraw(nominal_power=5.0, peak_power=10.0, power_mode={0: 100.0})
    )


@pytest.fixture
def payload_mixed(i1_5_10_mode0, i2_20_40):
    return Payload(payload=[i1_5_10_mode0, i2_20_40])
