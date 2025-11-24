import pytest

from conops.config.power import PowerDraw


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
