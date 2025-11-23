from math import isclose

from conops import Heater, Instrument, Payload, PowerDraw


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


class TestPayload:
    def test_payload_aggregates_nominal_power_initial(self, payload_10_20_and_20_40):
        payload = payload_10_20_and_20_40
        assert isclose(payload.power(), 30.0)

    def test_payload_aggregates_nominal_power_after_change(self, i1_10_20, i2_20_40):
        payload = Payload(payload=[i1_10_20, i2_20_40])
        i1_10_20.power_draw.nominal_power = 15.0
        assert isclose(payload.power(), 35.0)

    def test_payload_aggregates_mode_0_with_mixed_modes(self, payload_mixed):
        payload = payload_mixed
        assert isclose(payload.power(0), 120.0)

    def test_payload_aggregates_mode_missing_falls_back_to_nominal(self, payload_mixed):
        payload = payload_mixed
        assert isclose(payload.power(99), 25.0)


class TestInstrumentEclipse:
    """Test eclipse-aware power consumption for payload."""

    def test_instrument_with_heater_eclipse(self):
        """Test instrument power with heater in eclipse."""
        instrument = Instrument(
            name="Camera",
            power_draw=PowerDraw(nominal_power=50.0),
            heater=Heater(
                name="Camera Heater",
                power_draw=PowerDraw(nominal_power=8.0, eclipse_power=20.0),
            ),
        )

        # Sunlight: base + heater
        assert instrument.power(in_eclipse=False) == 58.0
        # Eclipse: base + higher heater power
        assert instrument.power(in_eclipse=True) == 70.0

    def test_instrument_base_power_with_eclipse(self):
        """Test instrument base power draw can also be eclipse-aware."""
        instrument = Instrument(
            name="Detector",
            power_draw=PowerDraw(
                nominal_power=30.0,
                eclipse_power=35.0,  # Slightly higher in eclipse
            ),
        )

        assert instrument.power(in_eclipse=False) == 30.0
        assert instrument.power(in_eclipse=True) == 35.0

    def test_instrument_full_eclipse_configuration(self):
        """Test instrument with both base and heater eclipse power."""
        instrument = Instrument(
            name="Spectrometer",
            power_draw=PowerDraw(
                nominal_power=40.0,
                eclipse_power=45.0,
                power_mode={2: 80.0},
                eclipse_power_mode={2: 90.0},
            ),
            heater=Heater(
                name="Spec Heater",
                power_draw=PowerDraw(
                    nominal_power=10.0,
                    eclipse_power=25.0,
                    power_mode={2: 15.0},
                    eclipse_power_mode={2: 30.0},
                ),
            ),
        )

        # Nominal mode
        assert instrument.power(in_eclipse=False) == 50.0  # 40 + 10
        assert instrument.power(in_eclipse=True) == 70.0  # 45 + 25

        # Mode 2
        assert instrument.power(mode=2, in_eclipse=False) == 95.0  # 80 + 15
        assert instrument.power(mode=2, in_eclipse=True) == 120.0  # 90 + 30

    def test_instrument_no_heater_eclipse(self):
        """Test instrument without heater still handles eclipse."""
        instrument = Instrument(
            name="Simple", power_draw=PowerDraw(nominal_power=25.0, eclipse_power=30.0)
        )

        assert instrument.power(in_eclipse=False) == 25.0
        assert instrument.power(in_eclipse=True) == 30.0


class TestPayloadEclipse:
    """Test eclipse-aware power for payloads."""

    def test_payload_eclipse(self):
        """Test that payload passes eclipse to all instruments."""
        inst1 = Instrument(
            name="Cam1", power_draw=PowerDraw(nominal_power=30.0, eclipse_power=40.0)
        )
        inst2 = Instrument(
            name="Cam2", power_draw=PowerDraw(nominal_power=20.0, eclipse_power=25.0)
        )

        inst_set = Payload(payload=[inst1, inst2])

        # Sunlight: 30 + 20 = 50
        assert inst_set.power(in_eclipse=False) == 50.0
        # Eclipse: 40 + 25 = 65
        assert inst_set.power(in_eclipse=True) == 65.0

    def test_payload_with_heaters_eclipse(self):
        """Test payload with heaters in eclipse."""
        inst1 = Instrument(
            name="Detector",
            power_draw=PowerDraw(nominal_power=35.0),
            heater=Heater(
                name="Det Heater",
                power_draw=PowerDraw(nominal_power=5.0, eclipse_power=15.0),
            ),
        )
        inst2 = Instrument(
            name="Processor",
            power_draw=PowerDraw(nominal_power=25.0),
            heater=Heater(
                name="Proc Heater",
                power_draw=PowerDraw(nominal_power=3.0, eclipse_power=10.0),
            ),
        )

        inst_set = Payload(payload=[inst1, inst2])

        # Sunlight: (35+5) + (25+3) = 68
        assert inst_set.power(in_eclipse=False) == 68.0
        # Eclipse: (35+15) + (25+10) = 85
        assert inst_set.power(in_eclipse=True) == 85.0
