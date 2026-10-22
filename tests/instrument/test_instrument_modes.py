"""Instrument modes: an instrument with modes draws its mode's power, the
mode an observation names for it or IDLE otherwise."""

import pytest
from pydantic import ValidationError

from conops.common import ACSMode
from conops.config import Instrument, InstrumentMode, Payload, PowerDraw


def _camera() -> Instrument:
    return Instrument(
        name="camera",
        power_draw=PowerDraw(nominal_power=999.0),
        modes=[
            InstrumentMode(name="IDLE", power_w=10.0),
            InstrumentMode(name="SCIENCE", power_w=80.0, eclipse_power_w=90.0),
            InstrumentMode(name="SAFE", power_w=5.0),
            InstrumentMode(name="FAST", power_w=120.0),
        ],
    )


def _spectrograph() -> Instrument:
    return Instrument(
        name="spectrograph",
        modes=[
            InstrumentMode(name="IDLE", power_w=20.0),
            InstrumentMode(name="SCIENCE", power_w=150.0),
        ],
    )


class TestInstrument:
    def test_modes_need_distinct_names_with_idle_and_science(self) -> None:
        with pytest.raises(ValidationError, match="IDLE"):
            Instrument(modes=[InstrumentMode(name="SCIENCE", power_w=1.0)])
        with pytest.raises(ValidationError, match="SCIENCE"):
            Instrument(modes=[InstrumentMode(name="IDLE", power_w=1.0)])
        with pytest.raises(ValidationError, match="distinct"):
            Instrument(
                modes=[
                    InstrumentMode(name="IDLE", power_w=1.0),
                    InstrumentMode(name="IDLE", power_w=2.0),
                    InstrumentMode(name="SCIENCE", power_w=3.0),
                ]
            )

    def test_an_instrument_with_modes_draws_its_modes_power(self) -> None:
        camera = _camera()
        # No observation: IDLE, whatever the ACS mode; power_draw is unused.
        assert camera.power() == 10.0
        assert camera.power(ACSMode.SLEWING.value) == 10.0
        assert camera.power(ACSMode.SCIENCE.value) == 10.0
        # A mode named like the ACS mode applies in that mode.
        assert camera.power(ACSMode.SAFE.value) == 5.0
        # The observation names the mode, in eclipse too.
        assert camera.power(ACSMode.SCIENCE.value, instrument_mode="SCIENCE") == 80.0
        assert (
            camera.power(ACSMode.SCIENCE.value, True, instrument_mode="SCIENCE") == 90.0
        )
        assert camera.power(ACSMode.SCIENCE.value, instrument_mode="FAST") == 120.0
        assert camera.mode_in(ACSMode.SAFE.value, None) == "SAFE"
        assert camera.mode_in(None, None) == "IDLE"
        with pytest.raises(ValueError, match="no mode"):
            camera.power(instrument_mode="NOPE")

    def test_an_instrument_without_modes_draws_by_acs_mode_as_before(self) -> None:
        legacy = Instrument(
            name="legacy",
            power_draw=PowerDraw(nominal_power=50.0, power_mode={0: 70.0}),
        )
        assert legacy.power() == 50.0
        assert legacy.power(0) == 70.0
        assert legacy.power(0, instrument_mode="SCIENCE") == 70.0


class TestPayload:
    def test_the_payload_sums_each_instruments_mode(self) -> None:
        payload = Payload(instruments=[_camera(), _spectrograph()])
        assert payload.power(ACSMode.SCIENCE.value) == 30.0
        assert (
            payload.power(ACSMode.SCIENCE.value, instrument_modes={"camera": "SCIENCE"})
            == 100.0
        )
        assert (
            payload.power(
                ACSMode.SCIENCE.value,
                instrument_modes={"camera": "FAST", "spectrograph": "SCIENCE"},
            )
            == 270.0
        )
        assert payload.instruments_with_modes() == ["camera", "spectrograph"]

    def test_a_requests_modes_are_checked_against_the_payload(self) -> None:
        payload = Payload(instruments=[_camera(), Instrument(name="legacy")])
        payload.check_instrument_modes({"camera": "FAST", "legacy": "SCIENCE"})
        with pytest.raises(ValueError, match="no instrument"):
            payload.check_instrument_modes({"radar": "SCIENCE"})
        with pytest.raises(ValueError, match="no mode"):
            payload.check_instrument_modes({"camera": "WARM"})
        with pytest.raises(ValueError, match="has no modes"):
            payload.check_instrument_modes({"legacy": "FAST"})
