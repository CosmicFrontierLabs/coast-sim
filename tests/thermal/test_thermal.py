from conops import Heater, PowerDraw

"""Unit tests for conops.thermal.Heater"""


class TestHeaterInitialization:
    """Test Heater initialization and attributes."""

    def test_heater_has_name(self, simple_heater):
        assert simple_heater.name == "Test Heater"

    def test_heater_has_power_draw(self, simple_heater):
        assert isinstance(simple_heater.power_draw, PowerDraw)

    def test_heater_initialization_with_power_draw(self):
        power = PowerDraw(nominal_power=20.0)
        heater = Heater(name="Custom Heater", power_draw=power)
        assert heater.name == "Custom Heater"
        assert heater.power_draw.nominal_power == 20.0


class TestHeaterPowerNominal:
    """Test heater power consumption with nominal (no mode) settings."""

    def test_heat_power_no_mode_returns_nominal(self, simple_heater):
        assert simple_heater.power() == 10.0

    def test_heat_power_none_mode_returns_nominal(self, simple_heater):
        assert simple_heater.power(mode=None) == 10.0

    def test_heat_power_with_zero_nominal(self):
        power = PowerDraw(nominal_power=0.0)
        heater = Heater(name="Zero Heater", power_draw=power)
        assert heater.power() == 0.0

    def test_heat_power_with_high_nominal(self):
        power = PowerDraw(nominal_power=100.0)
        heater = Heater(name="High Power Heater", power_draw=power)
        assert heater.power() == 100.0


class TestHeaterPowerModes:
    """Test heater power consumption with different operational modes."""

    def test_heat_power_mode_0(self, mode_heater):
        assert mode_heater.power(mode=0) == 5.0

    def test_heat_power_mode_1(self, mode_heater):
        assert mode_heater.power(mode=1) == 10.0

    def test_heat_power_mode_2(self, mode_heater):
        assert mode_heater.power(mode=2) == 12.0

    def test_heat_power_mode_3(self, mode_heater):
        assert mode_heater.power(mode=3) == 20.0

    def test_heat_power_undefined_mode_returns_nominal(self, mode_heater):
        """Mode not in power_mode dict should return nominal power."""
        assert mode_heater.power(mode=99) == 10.0

    def test_heat_power_negative_mode_returns_nominal(self, mode_heater):
        """Negative mode should return nominal power."""
        assert mode_heater.power(mode=-1) == 10.0


class TestHeaterEdgeCases:
    """Test edge cases and special scenarios."""

    def test_heater_with_empty_power_mode_dict(self):
        power = PowerDraw(nominal_power=15.0, power_mode={})
        heater = Heater(name="Empty Mode Heater", power_draw=power)
        assert heater.power(mode=0) == 15.0
        assert heater.power(mode=1) == 15.0

    def test_multiple_heaters_independent(self):
        """Test that multiple heaters are independent."""
        power1 = PowerDraw(nominal_power=10.0)
        power2 = PowerDraw(nominal_power=20.0)
        heater1 = Heater(name="Heater1", power_draw=power1)
        heater2 = Heater(name="Heater2", power_draw=power2)

        assert heater1.power() == 10.0
        assert heater2.power() == 20.0
        assert heater1.name != heater2.name

    def test_heater_with_fractional_power(self):
        power = PowerDraw(nominal_power=7.5, power_mode={1: 3.14159})
        heater = Heater(name="Fractional Heater", power_draw=power)
        assert heater.power() == 7.5
        assert heater.power(mode=1) == 3.14159


class TestHeaterPydanticFeatures:
    """Test Pydantic model features of Heater."""

    def test_heater_model_dump(self, simple_heater):
        """Test that heater can be serialized."""
        data = simple_heater.model_dump()
        assert data["name"] == "Test Heater"
        assert "power_draw" in data

    def test_heater_from_dict(self):
        """Test creating heater from dictionary."""
        data = {
            "name": "Dict Heater",
            "power_draw": {
                "nominal_power": 25.0,
                "peak_power": 30.0,
                "power_mode": {0: 5.0, 1: 15.0},
            },
        }
        heater = Heater(**data)
        assert heater.name == "Dict Heater"
        assert heater.power() == 25.0
        assert heater.power(mode=0) == 5.0
        assert heater.power(mode=1) == 15.0

    def test_heater_model_copy(self, mode_heater):
        """Test that heater can be copied."""
        heater_copy = mode_heater.model_copy()
        assert heater_copy.name == mode_heater.name
        assert heater_copy.power(mode=1) == mode_heater.power(mode=1)


class TestHeaterIntegration:
    """Integration tests for realistic heater usage scenarios."""

    def test_bus_heater_scenario(self):
        """Test a spacecraft bus heater configuration."""
        power = PowerDraw(
            nominal_power=15.0,  # Normal ops
            peak_power=25.0,
            power_mode={
                0: 5.0,  # Safe mode - low power
                1: 15.0,  # Normal ops
                2: 20.0,  # Science mode - higher power
            },
        )
        bus_heater = Heater(name="Bus Heater", power_draw=power)

        # Safe mode
        assert bus_heater.power(mode=0) == 5.0
        # Normal ops
        assert bus_heater.power(mode=1) == 15.0
        # Science mode
        assert bus_heater.power(mode=2) == 20.0

    def test_instrument_heater_scenario(self):
        """Test an instrument heater configuration."""
        power = PowerDraw(
            nominal_power=8.0,  # Standby
            peak_power=12.0,
            power_mode={
                0: 2.0,  # Off/survival
                1: 8.0,  # Standby
                2: 10.0,  # Active observation
            },
        )
        inst_heater = Heater(name="Instrument Heater", power_draw=power)

        assert inst_heater.power(mode=0) == 2.0
        assert inst_heater.power(mode=1) == 8.0
        assert inst_heater.power(mode=2) == 10.0

    def test_multiple_heaters_total_power(self):
        """Test calculating total power from multiple heaters."""
        bus_power = PowerDraw(nominal_power=15.0, power_mode={1: 10.0})
        inst_power = PowerDraw(nominal_power=8.0, power_mode={1: 12.0})

        bus_heater = Heater(name="Bus", power_draw=bus_power)
        inst_heater = Heater(name="Instrument", power_draw=inst_power)

        # Nominal total
        total_nominal = bus_heater.power() + inst_heater.power()
        assert total_nominal == 23.0

        # Mode 1 total
        total_mode1 = bus_heater.power(mode=1) + inst_heater.power(mode=1)
        assert total_mode1 == 22.0


class TestHeaterEclipse:
    """Test eclipse-aware heater power consumption."""

    def test_heater_with_eclipse_power(self):
        """Test heater draws more power in eclipse."""
        power = PowerDraw(
            nominal_power=10.0,
            eclipse_power=25.0,  # More power needed in eclipse
        )
        heater = Heater(name="Eclipse Heater", power_draw=power)

        # In sunlight
        assert heater.power(in_eclipse=False) == 10.0
        # In eclipse
        assert heater.power(in_eclipse=True) == 25.0

    def test_heater_eclipse_with_modes(self):
        """Test eclipse power with operational modes."""
        power = PowerDraw(
            nominal_power=10.0,
            eclipse_power=20.0,
            power_mode={0: 5.0, 1: 10.0, 2: 15.0},
            eclipse_power_mode={0: 10.0, 1: 20.0, 2: 30.0},
        )
        heater = Heater(name="Mode Eclipse Heater", power_draw=power)

        # Sunlight modes
        assert heater.power(mode=0, in_eclipse=False) == 5.0
        assert heater.power(mode=1, in_eclipse=False) == 10.0
        assert heater.power(mode=2, in_eclipse=False) == 15.0

        # Eclipse modes
        assert heater.power(mode=0, in_eclipse=True) == 10.0
        assert heater.power(mode=1, in_eclipse=True) == 20.0
        assert heater.power(mode=2, in_eclipse=True) == 30.0

    def test_heater_no_eclipse_power_falls_back(self):
        """Test that heater without eclipse_power uses normal power in eclipse."""
        power = PowerDraw(nominal_power=10.0)
        heater = Heater(name="No Eclipse Config", power_draw=power)

        # Should use normal power even in eclipse
        assert heater.power(in_eclipse=False) == 10.0
        assert heater.power(in_eclipse=True) == 10.0

    def test_heater_partial_eclipse_mode_config(self):
        """Test eclipse mode config that doesn't cover all modes."""
        power = PowerDraw(
            nominal_power=10.0,
            eclipse_power=20.0,
            power_mode={0: 5.0, 1: 10.0, 2: 15.0},
            eclipse_power_mode={1: 25.0},  # Only mode 1 has eclipse override
        )
        heater = Heater(name="Partial Eclipse", power_draw=power)

        # Mode 0 in eclipse: no eclipse_power_mode, use eclipse_power
        assert heater.power(mode=0, in_eclipse=True) == 20.0
        # Mode 1 in eclipse: has eclipse_power_mode
        assert heater.power(mode=1, in_eclipse=True) == 25.0
        # Mode 2 in eclipse: no eclipse_power_mode, use eclipse_power
        assert heater.power(mode=2, in_eclipse=True) == 20.0

    def test_heater_eclipse_realistic_scenario(self):
        """Test realistic spacecraft heater behavior through eclipse."""
        power = PowerDraw(
            nominal_power=12.0,  # Sunlight standby
            eclipse_power=30.0,  # Eclipse standby - needs more heating
            power_mode={
                0: 5.0,  # Safe mode sunlight
                1: 12.0,  # Normal ops sunlight
                2: 15.0,  # Science mode sunlight
            },
            eclipse_power_mode={
                0: 15.0,  # Safe mode eclipse
                1: 30.0,  # Normal ops eclipse
                2: 35.0,  # Science mode eclipse
            },
        )
        heater = Heater(name="Spacecraft Heater", power_draw=power)

        # Simulate orbit: sunlight -> eclipse -> sunlight
        # In sunlight, normal ops
        assert heater.power(mode=1, in_eclipse=False) == 12.0

        # Enter eclipse, power increases
        assert heater.power(mode=1, in_eclipse=True) == 30.0

        # Back to sunlight
        assert heater.power(mode=1, in_eclipse=False) == 12.0

    def test_heater_eclipse_power_always_higher(self):
        """Verify that eclipse configurations typically draw more power."""
        # Typical configuration where eclipse heating is higher
        power = PowerDraw(
            nominal_power=10.0,
            eclipse_power=25.0,
            power_mode={1: 12.0},
            eclipse_power_mode={1: 28.0},
        )
        heater = Heater(name="Typical Heater", power_draw=power)

        # Eclipse power should be higher
        assert heater.power(in_eclipse=True) > heater.power(in_eclipse=False)
        assert heater.power(mode=1, in_eclipse=True) > heater.power(
            mode=1, in_eclipse=False
        )
