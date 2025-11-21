from math import isclose

import pytest

from conops.instrument import Instrument, InstrumentSet
from conops.spacecraft_bus import PowerDraw


# fixtures for single instruments and power draws
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


# fixtures for instrument set tests
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
def instrument_set_10_20_and_20_40(i1_10_20, i2_20_40):
    return InstrumentSet(instruments=[i1_10_20, i2_20_40])


@pytest.fixture
def i1_5_10_mode0():
    return Instrument(
        power_draw=PowerDraw(nominal_power=5.0, peak_power=10.0, power_mode={0: 100.0})
    )


@pytest.fixture
def instrument_set_mixed(i1_5_10_mode0, i2_20_40):
    return InstrumentSet(instruments=[i1_5_10_mode0, i2_20_40])


class TestInstrument:
    def test_default_instrument_name(self, default_instrument):
        assert default_instrument.name == "Default Instrument"

    def test_default_instrument_nominal_power(self, default_instrument):
        assert isclose(default_instrument.power_draw.nominal_power, 50.0)

    def test_default_instrument_peak_power(self, default_instrument):
        assert isclose(default_instrument.power_draw.peak_power, 100.0)

    def test_default_instrument_power_implicit_nominal(self, default_instrument):
        assert isclose(default_instrument.power(), 50.0)

    def test_default_instrument_power_explicit_none(self, default_instrument):
        assert isclose(default_instrument.power(None), 50.0)

    def test_instrument_nominal_power_with_modes(self, pd_with_modes):
        inst = Instrument(name="Cam", power_draw=pd_with_modes)
        assert isclose(inst.power(), 75.0)

    def test_instrument_nominal_power_with_modes_explicit_none(self, pd_with_modes):
        inst = Instrument(name="Cam", power_draw=pd_with_modes)
        assert isclose(inst.power(None), 75.0)

    def test_instrument_mode_0_power(self, cam_instrument):
        inst = cam_instrument
        assert isclose(inst.power(0), 10.0)

    def test_instrument_mode_1_power(self, cam_instrument):
        inst = cam_instrument
        assert isclose(inst.power(1), 20.0)

    def test_instrument_mode_missing_falls_back_to_nominal(self, cam_instrument):
        inst = cam_instrument
        assert isclose(inst.power(999), 75.0)


class TestInstrumentSet:
    def test_instrument_set_aggregates_nominal_power_initial(
        self, instrument_set_10_20_and_20_40
    ):
        instrument_set = instrument_set_10_20_and_20_40
        assert isclose(instrument_set.power(), 30.0)

    def test_instrument_set_aggregates_nominal_power_after_change(
        self, i1_10_20, i2_20_40
    ):
        instrument_set = InstrumentSet(instruments=[i1_10_20, i2_20_40])
        i1_10_20.power_draw.nominal_power = 15.0
        assert isclose(instrument_set.power(), 35.0)

    def test_instrument_set_aggregates_mode_0_with_mixed_modes(
        self, instrument_set_mixed
    ):
        instrument_set = instrument_set_mixed
        assert isclose(instrument_set.power(0), 120.0)

    def test_instrument_set_aggregates_mode_missing_falls_back_to_nominal(
        self, instrument_set_mixed
    ):
        instrument_set = instrument_set_mixed
        assert isclose(instrument_set.power(99), 25.0)
