import pytest

from conops.power import PowerDraw
from conops.thermal import Heater

"""Unit tests for conops.thermal.Heater"""


@pytest.fixture
def simple_power_draw():
    """Simple PowerDraw with only nominal power."""
    return PowerDraw(nominal_power=10.0, peak_power=15.0)


@pytest.fixture
def mode_based_power_draw():
    """PowerDraw with mode-specific power values."""
    return PowerDraw(
        nominal_power=10.0,
        peak_power=15.0,
        power_mode={0: 5.0, 1: 10.0, 2: 12.0, 3: 20.0},
    )


@pytest.fixture
def simple_heater(simple_power_draw):
    """Heater with simple power draw."""
    return Heater(name="Test Heater", power=simple_power_draw)


@pytest.fixture
def mode_heater(mode_based_power_draw):
    """Heater with mode-based power draw."""
    return Heater(name="Mode Heater", power=mode_based_power_draw)


class TestHeaterInitialization:
    """Test Heater initialization and attributes."""

    def test_heater_has_name(self, simple_heater):
        assert simple_heater.name == "Test Heater"

    def test_heater_has_power_draw(self, simple_heater):
        assert isinstance(simple_heater.power, PowerDraw)

    def test_heater_initialization_with_power_draw(self):
        power = PowerDraw(nominal_power=20.0)
        heater = Heater(name="Custom Heater", power=power)
        assert heater.name == "Custom Heater"
        assert heater.power.nominal_power == 20.0


class TestHeaterPowerNominal:
    """Test heater power consumption with nominal (no mode) settings."""

    def test_heat_power_no_mode_returns_nominal(self, simple_heater):
        assert simple_heater.heat_power() == 10.0

    def test_heat_power_none_mode_returns_nominal(self, simple_heater):
        assert simple_heater.heat_power(mode=None) == 10.0

    def test_heat_power_with_zero_nominal(self):
        power = PowerDraw(nominal_power=0.0)
        heater = Heater(name="Zero Heater", power=power)
        assert heater.heat_power() == 0.0

    def test_heat_power_with_high_nominal(self):
        power = PowerDraw(nominal_power=100.0)
        heater = Heater(name="High Power Heater", power=power)
        assert heater.heat_power() == 100.0


class TestHeaterPowerModes:
    """Test heater power consumption with different operational modes."""

    def test_heat_power_mode_0(self, mode_heater):
        assert mode_heater.heat_power(mode=0) == 5.0

    def test_heat_power_mode_1(self, mode_heater):
        assert mode_heater.heat_power(mode=1) == 10.0

    def test_heat_power_mode_2(self, mode_heater):
        assert mode_heater.heat_power(mode=2) == 12.0

    def test_heat_power_mode_3(self, mode_heater):
        assert mode_heater.heat_power(mode=3) == 20.0

    def test_heat_power_undefined_mode_returns_nominal(self, mode_heater):
        """Mode not in power_mode dict should return nominal power."""
        assert mode_heater.heat_power(mode=99) == 10.0

    def test_heat_power_negative_mode_returns_nominal(self, mode_heater):
        """Negative mode should return nominal power."""
        assert mode_heater.heat_power(mode=-1) == 10.0


class TestHeaterEdgeCases:
    """Test edge cases and special scenarios."""

    def test_heater_with_empty_power_mode_dict(self):
        power = PowerDraw(nominal_power=15.0, power_mode={})
        heater = Heater(name="Empty Mode Heater", power=power)
        assert heater.heat_power(mode=0) == 15.0
        assert heater.heat_power(mode=1) == 15.0

    def test_multiple_heaters_independent(self):
        """Test that multiple heaters are independent."""
        power1 = PowerDraw(nominal_power=10.0)
        power2 = PowerDraw(nominal_power=20.0)
        heater1 = Heater(name="Heater1", power=power1)
        heater2 = Heater(name="Heater2", power=power2)

        assert heater1.heat_power() == 10.0
        assert heater2.heat_power() == 20.0
        assert heater1.name != heater2.name

    def test_heater_with_fractional_power(self):
        power = PowerDraw(nominal_power=7.5, power_mode={1: 3.14159})
        heater = Heater(name="Fractional Heater", power=power)
        assert heater.heat_power() == 7.5
        assert heater.heat_power(mode=1) == 3.14159


class TestHeaterPydanticFeatures:
    """Test Pydantic model features of Heater."""

    def test_heater_model_dump(self, simple_heater):
        """Test that heater can be serialized."""
        data = simple_heater.model_dump()
        assert data["name"] == "Test Heater"
        assert "power" in data

    def test_heater_from_dict(self):
        """Test creating heater from dictionary."""
        data = {
            "name": "Dict Heater",
            "power": {
                "nominal_power": 25.0,
                "peak_power": 30.0,
                "power_mode": {0: 5.0, 1: 15.0},
            },
        }
        heater = Heater(**data)
        assert heater.name == "Dict Heater"
        assert heater.heat_power() == 25.0
        assert heater.heat_power(mode=0) == 5.0
        assert heater.heat_power(mode=1) == 15.0

    def test_heater_model_copy(self, mode_heater):
        """Test that heater can be copied."""
        heater_copy = mode_heater.model_copy()
        assert heater_copy.name == mode_heater.name
        assert heater_copy.heat_power(mode=1) == mode_heater.heat_power(mode=1)


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
        bus_heater = Heater(name="Bus Heater", power=power)

        # Safe mode
        assert bus_heater.heat_power(mode=0) == 5.0
        # Normal ops
        assert bus_heater.heat_power(mode=1) == 15.0
        # Science mode
        assert bus_heater.heat_power(mode=2) == 20.0

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
        inst_heater = Heater(name="Instrument Heater", power=power)

        assert inst_heater.heat_power(mode=0) == 2.0
        assert inst_heater.heat_power(mode=1) == 8.0
        assert inst_heater.heat_power(mode=2) == 10.0

    def test_multiple_heaters_total_power(self):
        """Test calculating total power from multiple heaters."""
        bus_power = PowerDraw(nominal_power=15.0, power_mode={1: 10.0})
        inst_power = PowerDraw(nominal_power=8.0, power_mode={1: 12.0})

        bus_heater = Heater(name="Bus", power=bus_power)
        inst_heater = Heater(name="Instrument", power=inst_power)

        # Nominal total
        total_nominal = bus_heater.heat_power() + inst_heater.heat_power()
        assert total_nominal == 23.0

        # Mode 1 total
        total_mode1 = bus_heater.heat_power(mode=1) + inst_heater.heat_power(mode=1)
        assert total_mode1 == 22.0
