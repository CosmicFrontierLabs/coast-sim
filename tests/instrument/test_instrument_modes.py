"""Instrument modes: an instrument with modes draws its mode's power, the
mode an observation names for it or IDLE otherwise."""

import pytest
from pydantic import ValidationError

from conops.common import ACSMode
from conops.config import (
    ConcurrencyRules,
    DataGeneration,
    Instrument,
    InstrumentMode,
    Payload,
    PowerDraw,
    ScienceData,
)


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


def _imager() -> Instrument:
    return Instrument(
        name="imager",
        data_generation=DataGeneration(science=ScienceData(rate_gbps=0.01)),
        modes=[
            InstrumentMode(name="IDLE", power_w=1.0),
            InstrumentMode(name="SCIENCE", power_w=2.0),
            InstrumentMode(
                name="FULL", power_w=3.0, science_data=ScienceData(rate_gbps=0.1)
            ),
        ],
    )


class TestScienceData:
    def test_a_mode_sets_what_the_instrument_generates(self) -> None:
        """In SCIENCE the imager generates its default; in FULL its mode's
        rate; an instrument with modes that the observation does not use
        generates nothing; one without modes always generates its own."""
        payload = Payload(
            instruments=[
                _imager(),
                Instrument(
                    name="legacy",
                    data_generation=DataGeneration(
                        science=ScienceData(per_observation_gb=0.5)
                    ),
                ),
            ]
        )
        assert payload.science_generated(100.0) == pytest.approx(1.0 + 0.5)
        assert payload.science_generated(
            100.0, instrument_modes={"imager": "SCIENCE"}
        ) == (pytest.approx(1.0 + 0.5))
        assert payload.science_generated(
            100.0, instrument_modes={"imager": "FULL"}
        ) == (pytest.approx(10.0 + 0.5))
        assert payload.science_generated(100.0, instrument_modes={}) == pytest.approx(
            0.5
        )
        # The target's own data for an instrument wins over its mode's.
        assert payload.science_generated(
            100.0, {"imager": ScienceData(rate_gbps=0.02)}, {"imager": "FULL"}
        ) == pytest.approx(2.0 + 0.5)


class TestConcurrency:
    def _payload(self, **rules: object) -> Payload:
        return Payload(
            instruments=[
                _camera(),
                _spectrograph(),
                Instrument(
                    name="guider",
                    modes=[
                        InstrumentMode(name="IDLE", power_w=1.0),
                        InstrumentMode(name="SCIENCE", power_w=2.0),
                    ],
                ),
            ],
            concurrency=ConcurrencyRules(**rules),  # type: ignore[arg-type]
        )

    def test_rules_name_instruments_the_payload_has(self) -> None:
        with pytest.raises(ValidationError, match="lacks"):
            self._payload(required_guider="radar")
        with pytest.raises(ValidationError, match="lacks"):
            self._payload(exclusive=[("camera", "radar")])
        assert not self._payload().concurrency.active

    def test_at_most_so_many_observe_at_once(self) -> None:
        payload = self._payload(max_observing=2)
        payload.check_concurrency({"camera": "SCIENCE", "guider": "SCIENCE"})
        # IDLE is not observing.
        payload.check_concurrency(
            {"camera": "SCIENCE", "guider": "SCIENCE", "spectrograph": "IDLE"}
        )
        with pytest.raises(ValueError, match="allows 2"):
            payload.check_concurrency(
                {"camera": "FAST", "guider": "SCIENCE", "spectrograph": "SCIENCE"}
            )
        # The default, every instrument with modes, breaks the rule.
        with pytest.raises(ValueError, match="allows 2"):
            payload.check_request({})

    def test_a_guider_is_required_whenever_anything_observes(self) -> None:
        payload = self._payload(required_guider="guider")
        payload.check_concurrency({"camera": "SCIENCE", "guider": "SCIENCE"})
        payload.check_concurrency({})
        with pytest.raises(ValueError, match="must also use 'guider'"):
            payload.check_concurrency({"camera": "SCIENCE"})

    def test_exclusive_instruments_do_not_observe_together(self) -> None:
        payload = self._payload(exclusive=[("camera", "spectrograph")])
        payload.check_concurrency({"camera": "SCIENCE"})
        payload.check_concurrency({"camera": "SCIENCE", "spectrograph": "IDLE"})
        with pytest.raises(ValueError, match="cannot observe together"):
            payload.check_concurrency({"camera": "SCIENCE", "spectrograph": "SCIENCE"})

    def test_a_standby_mode_draws_power_without_observing(self) -> None:
        warm = Instrument(
            name="warm",
            modes=[
                InstrumentMode(name="IDLE", power_w=1.0, observing=True),
                InstrumentMode(name="SCIENCE", power_w=50.0),
                InstrumentMode(name="STANDBY", power_w=20.0, observing=False),
            ],
        )
        assert warm.mode("IDLE") is not None and not warm.mode("IDLE").observing
        payload = Payload(
            instruments=[_camera(), warm],
            concurrency=ConcurrencyRules(
                max_observing=1, exclusive=[("camera", "warm")]
            ),
        )
        assert warm.power(instrument_mode="STANDBY") == 20.0
        payload.check_concurrency({"camera": "SCIENCE", "warm": "STANDBY"})
        with pytest.raises(ValueError, match="allows 1"):
            payload.check_concurrency({"camera": "SCIENCE", "warm": "SCIENCE"})

    def test_an_activity_needs_no_guider(self) -> None:
        payload = self._payload(required_guider="guider")
        with pytest.raises(ValueError, match="must also use"):
            payload.check_concurrency({"camera": "SCIENCE"})
        payload.check_concurrency({"camera": "SCIENCE"}, pointing=False)
