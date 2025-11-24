import pytest

from conops.config.power import PowerDraw

"""Unit tests for conops.power.PowerDraw"""


@pytest.fixture
def simple_power():
    """Simple PowerDraw with only nominal power."""
    return PowerDraw(nominal_power=100.0, peak_power=150.0)


@pytest.fixture
def mode_power():
    """PowerDraw with mode-specific values."""
    return PowerDraw(
        nominal_power=100.0, peak_power=150.0, power_mode={0: 50.0, 1: 100.0, 2: 120.0}
    )


@pytest.fixture
def eclipse_power():
    """PowerDraw with eclipse-specific values."""
    return PowerDraw(
        nominal_power=100.0,
        eclipse_power=150.0,
        power_mode={0: 50.0, 1: 100.0},
        eclipse_power_mode={0: 80.0, 1: 150.0},
    )


class TestPowerDrawInitialization:
    """Test PowerDraw initialization and defaults."""

    def test_default_values(self):
        power = PowerDraw()
        assert power.nominal_power == 200
        assert power.peak_power == 300
        assert power.power_mode == {}
        assert power.eclipse_power is None
        assert power.eclipse_power_mode == {}

    def test_custom_initialization(self):
        power = PowerDraw(
            nominal_power=75.0,
            peak_power=100.0,
            power_mode={1: 80.0},
            eclipse_power=90.0,
            eclipse_power_mode={1: 95.0},
        )
        assert power.nominal_power == 75.0
        assert power.peak_power == 100.0
        assert power.power_mode == {1: 80.0}
        assert power.eclipse_power == 90.0
        assert power.eclipse_power_mode == {1: 95.0}


class TestPowerDrawNominal:
    """Test nominal power draw behavior."""

    def test_power_no_mode(self, simple_power):
        assert simple_power.power() == 100.0

    def test_power_none_mode(self, simple_power):
        assert simple_power.power(mode=None) == 100.0

    def test_power_no_eclipse(self, simple_power):
        assert simple_power.power(in_eclipse=False) == 100.0


class TestPowerDrawModes:
    """Test mode-specific power behavior."""

    def test_power_mode_0(self, mode_power):
        assert mode_power.power(mode=0) == 50.0

    def test_power_mode_1(self, mode_power):
        assert mode_power.power(mode=1) == 100.0

    def test_power_mode_2(self, mode_power):
        assert mode_power.power(mode=2) == 120.0

    def test_power_undefined_mode_returns_nominal(self, mode_power):
        assert mode_power.power(mode=99) == 100.0

    def test_power_negative_mode_returns_nominal(self, mode_power):
        assert mode_power.power(mode=-1) == 100.0


class TestPowerDrawEclipse:
    """Test eclipse-aware power behavior."""

    def test_eclipse_power_nominal(self, eclipse_power):
        """Test nominal power in eclipse."""
        assert eclipse_power.power(in_eclipse=True) == 150.0
        assert eclipse_power.power(in_eclipse=False) == 100.0

    def test_eclipse_power_with_mode(self, eclipse_power):
        """Test mode-specific eclipse power."""
        # Mode 0
        assert eclipse_power.power(mode=0, in_eclipse=False) == 50.0
        assert eclipse_power.power(mode=0, in_eclipse=True) == 80.0

        # Mode 1
        assert eclipse_power.power(mode=1, in_eclipse=False) == 100.0
        assert eclipse_power.power(mode=1, in_eclipse=True) == 150.0

    def test_eclipse_mode_fallback_to_eclipse_power(self, eclipse_power):
        """Test that undefined eclipse modes fall back to eclipse_power."""
        # Mode 2 has no eclipse_power_mode entry, should use eclipse_power
        assert eclipse_power.power(mode=2, in_eclipse=True) == 150.0

    def test_no_eclipse_power_configured(self, mode_power):
        """Test that without eclipse config, normal power is used."""
        assert mode_power.power(mode=0, in_eclipse=True) == 50.0
        assert mode_power.power(mode=1, in_eclipse=True) == 100.0
        assert mode_power.power(in_eclipse=True) == 100.0

    def test_only_eclipse_power_no_modes(self):
        """Test eclipse_power without eclipse_power_mode."""
        power = PowerDraw(
            nominal_power=50.0, eclipse_power=100.0, power_mode={0: 40.0, 1: 50.0}
        )
        # All modes in eclipse should use eclipse_power
        assert power.power(mode=0, in_eclipse=True) == 100.0
        assert power.power(mode=1, in_eclipse=True) == 100.0
        assert power.power(mode=99, in_eclipse=True) == 100.0

    def test_only_eclipse_modes_no_base_eclipse(self):
        """Test eclipse_power_mode without eclipse_power."""
        power = PowerDraw(
            nominal_power=50.0,
            power_mode={0: 40.0, 1: 50.0},
            eclipse_power_mode={0: 80.0},
        )
        # Mode 0 has eclipse override
        assert power.power(mode=0, in_eclipse=True) == 80.0
        # Mode 1 has no eclipse override, use normal
        assert power.power(mode=1, in_eclipse=True) == 50.0


class TestPowerDrawEdgeCases:
    """Test edge cases and special scenarios."""

    def test_zero_power(self):
        power = PowerDraw(nominal_power=0.0)
        assert power.power() == 0.0
        assert power.power(mode=1) == 0.0

    def test_negative_power(self):
        """Negative power could represent power generation."""
        power = PowerDraw(nominal_power=-10.0, power_mode={1: -5.0})
        assert power.power() == -10.0
        assert power.power(mode=1) == -5.0

    def test_fractional_power(self):
        power = PowerDraw(
            nominal_power=3.14159, eclipse_power=2.71828, power_mode={1: 1.41421}
        )
        assert power.power() == 3.14159
        assert power.power(mode=1) == 1.41421
        assert power.power(in_eclipse=True) == 2.71828


class TestPowerDrawPydanticFeatures:
    """Test Pydantic model features."""

    def test_model_dump(self, eclipse_power):
        data = eclipse_power.model_dump()
        assert data["nominal_power"] == 100.0
        assert data["eclipse_power"] == 150.0
        assert data["power_mode"] == {0: 50.0, 1: 100.0}
        assert data["eclipse_power_mode"] == {0: 80.0, 1: 150.0}

    def test_from_dict(self):
        data = {
            "nominal_power": 55.0,
            "peak_power": 75.0,
            "eclipse_power": 80.0,
            "power_mode": {0: 30.0},
            "eclipse_power_mode": {0: 60.0},
        }
        power = PowerDraw(**data)
        assert power.nominal_power == 55.0
        assert power.eclipse_power == 80.0
        assert power.power(mode=0) == 30.0
        assert power.power(mode=0, in_eclipse=True) == 60.0

    def test_model_copy(self, eclipse_power):
        power_copy = eclipse_power.model_copy()
        assert power_copy.nominal_power == eclipse_power.nominal_power
        assert power_copy.eclipse_power == eclipse_power.eclipse_power
        assert power_copy.power(mode=1, in_eclipse=True) == eclipse_power.power(
            mode=1, in_eclipse=True
        )


class TestPowerDrawRealisticScenarios:
    """Test realistic spacecraft power scenarios."""

    def test_instrument_power_profile(self):
        """Test typical instrument power profile."""
        power = PowerDraw(
            nominal_power=50.0,  # Standby
            peak_power=150.0,  # Peak during observation
            power_mode={
                0: 10.0,  # Off
                1: 50.0,  # Standby
                2: 120.0,  # Active observation
            },
        )

        assert power.power(mode=0) == 10.0
        assert power.power(mode=1) == 50.0
        assert power.power(mode=2) == 120.0

    def test_heater_power_profile(self):
        """Test typical heater power with eclipse."""
        power = PowerDraw(
            nominal_power=15.0,
            eclipse_power=35.0,  # More heating in eclipse
            power_mode={
                0: 5.0,  # Survival mode
                1: 15.0,  # Normal
            },
            eclipse_power_mode={
                0: 20.0,  # Survival in eclipse
                1: 35.0,  # Normal in eclipse
            },
        )

        # Verify eclipse power is higher
        assert power.power(mode=1, in_eclipse=True) > power.power(
            mode=1, in_eclipse=False
        )
        assert power.power(mode=0, in_eclipse=True) > power.power(
            mode=0, in_eclipse=False
        )

    def test_bus_power_profile(self):
        """Test spacecraft bus power profile."""
        power = PowerDraw(
            nominal_power=200.0,
            peak_power=300.0,
            power_mode={
                0: 100.0,  # Safe mode
                1: 200.0,  # Normal ops
                2: 250.0,  # Science mode
            },
        )

        assert power.power(mode=0) == 100.0
        assert power.power(mode=1) == 200.0
        assert power.power(mode=2) == 250.0
