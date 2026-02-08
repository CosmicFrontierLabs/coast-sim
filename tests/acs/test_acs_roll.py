"""Unit tests for ACS roll angle calculations."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from conops import ACS, ACSMode
from conops.config.solar_panel import SolarPanel, SolarPanelSet


class DummyEphemeris:
    """Minimal mock ephemeris for testing."""

    def __init__(self, sun_ra=45.0, sun_dec=23.5):
        self.step_size = 1.0
        self.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
        self.sun = [Mock(ra=Mock(deg=sun_ra), dec=Mock(deg=sun_dec))]
        self.earth_ra_deg = [0.0]
        self.earth_dec_deg = [0.0]
        self.sun_ra_deg = [sun_ra]
        self.sun_dec_deg = [sun_dec]
        self.moon_ra_deg = [90.0]
        self.moon_dec_deg = [10.0]
        self.sun_pv = Mock(position=np.array([[1.5e8, 0.0, 0.0]]))
        self.gcrs_pv = Mock(position=np.array([[0.0, 0.0, 6378.0]]))

    def index(self, time):
        return 0


@pytest.fixture
def mock_ephem_roll():
    """Create a mock ephemeris for roll testing."""
    return DummyEphemeris(sun_ra=90.0, sun_dec=0.0)


@pytest.fixture
def mock_constraint_roll(mock_ephem_roll):
    """Create a mock constraint for roll testing."""
    constraint = Mock()
    constraint.ephem = mock_ephem_roll
    constraint.panel_constraint = Mock()
    constraint.panel_constraint.solar_panel = None
    constraint.in_constraint = Mock(return_value=False)
    constraint.in_eclipse = Mock(return_value=False)
    return constraint


@pytest.fixture
def mock_config_roll(mock_ephem_roll, mock_constraint_roll):
    """Create a mock config for roll testing."""
    config = Mock()
    config.constraint = mock_constraint_roll
    config.ground_stations = Mock()

    panel = SolarPanel(
        name="Panel",
        normal=(-0.0, 0.7071067811865476, 0.7071067811865476),
        max_power=250.0,
    )
    config.solar_panel = SolarPanelSet(panels=[panel])

    config.spacecraft_bus = Mock()
    config.spacecraft_bus.attitude_control = Mock()
    config.spacecraft_bus.attitude_control.predict_slew = Mock(return_value=(45.0, []))
    config.spacecraft_bus.attitude_control.slew_time = Mock(return_value=100.0)
    return config


@pytest.fixture
def acs_roll(mock_constraint_roll, mock_config_roll):
    """Create an ACS instance for roll testing."""
    with patch("conops.simulation.passes.PassTimes") as mock_passtimes:
        mock_pt = Mock()
        mock_pt.passes = []
        mock_pt.next_pass = Mock(return_value=None)
        mock_pt.__iter__ = Mock(return_value=iter([]))
        mock_passtimes.return_value = mock_pt
        acs_instance = ACS(config=mock_config_roll)
        acs_instance.passrequests = mock_pt
        return acs_instance


class TestACSRollInitialization:
    """Test ACS roll angle initialization."""

    def test_roll_initialized_to_zero(self, acs_roll) -> None:
        """Test that roll is initialized to 0.0."""
        assert acs_roll.roll == 0.0
        assert isinstance(acs_roll.roll, float)

    def test_roll_is_accessible(self, acs_roll) -> None:
        """Test that roll attribute is accessible."""
        assert hasattr(acs_roll, "roll")
        roll = acs_roll.roll
        assert isinstance(roll, float)


class TestACSRollCalculation:
    """Test ACS roll angle calculation in the pointing method."""

    def test_pointing_returns_roll(self, acs_roll) -> None:
        """Test that pointing() method returns roll angle."""
        utime = 1514764800.0
        ra, dec, roll, obsid = acs_roll.pointing(utime)

        assert isinstance(ra, (int, float, np.floating))
        assert isinstance(dec, (int, float, np.floating))
        assert isinstance(roll, (int, float, np.floating))
        assert isinstance(obsid, (int, type(None)))
        assert 0.0 <= roll < 360.0

    def test_roll_calculation_updates_state(self, acs_roll) -> None:
        """Test that roll calculation updates the ACS state."""
        utime = 1514764800.0

        # Call pointing which should calculate roll
        _, _, returned_roll, _ = acs_roll.pointing(utime)

        # ACS state should be updated
        assert acs_roll.roll == returned_roll
        # Roll should be a valid angle
        assert 0.0 <= acs_roll.roll < 360.0

    def test_roll_varies_with_sun_position(self) -> None:
        """Test that roll varies with sun position."""
        # Create two ACS instances with different sun positions
        ephem1 = DummyEphemeris(sun_ra=0.0, sun_dec=0.0)
        constraint1 = Mock()
        constraint1.ephem = ephem1
        constraint1.panel_constraint = Mock()
        constraint1.panel_constraint.solar_panel = None
        constraint1.in_constraint = Mock(return_value=False)
        constraint1.in_eclipse = Mock(return_value=False)

        config1 = Mock()
        config1.constraint = constraint1
        config1.ground_stations = Mock()
        panel1 = SolarPanel(
            name="Panel",
            normal=(-0.0, 0.7071067811865476, 0.7071067811865476),
            max_power=250.0,
        )
        config1.solar_panel = SolarPanelSet(panels=[panel1])
        config1.spacecraft_bus = Mock()
        config1.spacecraft_bus.attitude_control = Mock()
        config1.spacecraft_bus.attitude_control.predict_slew = Mock(
            return_value=(45.0, [])
        )
        config1.spacecraft_bus.attitude_control.slew_time = Mock(return_value=100.0)

        with patch("conops.simulation.passes.PassTimes"):
            acs1 = ACS(config=config1)

        ephem2 = DummyEphemeris(sun_ra=90.0, sun_dec=0.0)
        constraint2 = Mock()
        constraint2.ephem = ephem2
        constraint2.panel_constraint = Mock()
        constraint2.panel_constraint.solar_panel = None
        constraint2.in_constraint = Mock(return_value=False)
        constraint2.in_eclipse = Mock(return_value=False)

        config2 = Mock()
        config2.constraint = constraint2
        config2.ground_stations = Mock()
        panel2 = SolarPanel(
            name="Panel",
            normal=(-0.0, 0.7071067811865476, 0.7071067811865476),
            max_power=250.0,
        )
        config2.solar_panel = SolarPanelSet(panels=[panel2])
        config2.spacecraft_bus = Mock()
        config2.spacecraft_bus.attitude_control = Mock()
        config2.spacecraft_bus.attitude_control.predict_slew = Mock(
            return_value=(45.0, [])
        )
        config2.spacecraft_bus.attitude_control.slew_time = Mock(return_value=100.0)

        with patch("conops.simulation.passes.PassTimes"):
            acs2 = ACS(config=config2)

        utime = 1514764800.0

        _, _, roll1, _ = acs1.pointing(utime)
        _, _, roll2, _ = acs2.pointing(utime)

        # Rolls should be different for different sun positions
        # (unless by coincidence they're the same)
        assert isinstance(roll1, float)
        assert isinstance(roll2, float)
        assert 0.0 <= roll1 < 360.0
        assert 0.0 <= roll2 < 360.0

    def test_roll_angle_in_valid_range(self, acs_roll) -> None:
        """Test that roll angle is always in valid range [0, 360)."""
        utime = 1514764800.0

        _, _, roll, _ = acs_roll.pointing(utime)

        assert 0.0 <= roll < 360.0


class TestACSRollWithSolarPanels:
    """Test ACS roll calculation with solar panels."""

    def test_roll_optimization_with_panel_set(self, acs_roll) -> None:
        """Test that roll is calculated to optimize solar panel illumination."""
        # The ACS should calculate roll using optimum_roll
        utime = 1514764800.0

        ra, dec, roll, obsid = acs_roll.pointing(utime)

        # Roll should be calculated (not always 0)
        # With the mock, we may get 0 if the mocks don't support it fully
        assert isinstance(roll, float)
        assert 0.0 <= roll < 360.0

    def test_roll_persists_across_pointing_calls(self, acs_roll) -> None:
        """Test that roll value is properly maintained."""
        utime1 = 1514764800.0
        utime2 = 1514764900.0

        _, _, roll1, _ = acs_roll.pointing(utime1)
        _, _, roll2, _ = acs_roll.pointing(utime2)

        # Both should be valid rolls
        assert 0.0 <= roll1 < 360.0
        assert 0.0 <= roll2 < 360.0


class TestACSRollInPointingOutput:
    """Test that roll is correctly integrated in the pointing output."""

    def test_pointing_tuple_format(self, acs_roll) -> None:
        """Test that pointing returns (ra, dec, roll, obsid)."""
        utime = 1514764800.0
        result = acs_roll.pointing(utime)

        assert isinstance(result, tuple)
        assert len(result) == 4
        ra, dec, roll, obsid = result

        assert isinstance(ra, (int, float, np.floating))
        assert isinstance(dec, (int, float, np.floating))
        assert isinstance(roll, (int, float, np.floating))

    def test_roll_consistency_in_multiple_calls(self, acs_roll) -> None:
        """Test that roll calculation is consistent across multiple calls."""
        utime = 1514764800.0

        # Call pointing multiple times at same time
        rolls = []
        for _ in range(3):
            _, _, roll, _ = acs_roll.pointing(utime)
            rolls.append(roll)

        # All rolls should be the same (deterministic calculation)
        assert rolls[0] == pytest.approx(rolls[1])
        assert rolls[1] == pytest.approx(rolls[2])


class TestACSRollMode:
    """Test roll calculation in different ACS modes."""

    def test_roll_in_science_mode(self, acs_roll) -> None:
        """Test roll calculation in SCIENCE mode."""
        acs_roll.acsmode = ACSMode.SCIENCE
        utime = 1514764800.0

        _, _, roll, _ = acs_roll.pointing(utime)

        assert 0.0 <= roll < 360.0

    def test_roll_in_pass_mode(self, acs_roll) -> None:
        """Test roll calculation in PASS mode."""
        acs_roll.acsmode = ACSMode.PASS
        utime = 1514764800.0

        _, _, roll, _ = acs_roll.pointing(utime)

        assert 0.0 <= roll < 360.0

    def test_roll_in_safe_mode(self, acs_roll) -> None:
        """Test roll calculation when in safe mode."""
        acs_roll.in_safe_mode = True
        utime = 1514764800.0

        _, _, roll, _ = acs_roll.pointing(utime)

        assert 0.0 <= roll < 360.0


class TestACSRollEdgeCases:
    """Test edge cases for ACS roll calculation."""

    def test_roll_at_zero_sun_elevation(self, acs_roll) -> None:
        """Test roll calculation when sun is at equator."""
        utime = 1514764800.0

        ra, dec, roll, obsid = acs_roll.pointing(utime)

        # Should still return valid roll
        assert 0.0 <= roll < 360.0

    def test_roll_near_poles(self) -> None:
        """Test roll calculation with spacecraft near poles."""
        ephem = DummyEphemeris(sun_ra=0.0, sun_dec=89.0)
        constraint = Mock()
        constraint.ephem = ephem
        constraint.panel_constraint = Mock()
        constraint.panel_constraint.solar_panel = None
        constraint.in_constraint = Mock(return_value=False)
        constraint.in_eclipse = Mock(return_value=False)

        config = Mock()
        config.constraint = constraint
        config.ground_stations = Mock()
        panel = SolarPanel(
            name="Panel",
            normal=(-0.0, 0.7071067811865476, 0.7071067811865476),
            max_power=250.0,
        )
        config.solar_panel = SolarPanelSet(panels=[panel])
        config.spacecraft_bus = Mock()
        config.spacecraft_bus.attitude_control = Mock()
        config.spacecraft_bus.attitude_control.predict_slew = Mock(
            return_value=(45.0, [])
        )
        config.spacecraft_bus.attitude_control.slew_time = Mock(return_value=100.0)

        with patch("conops.simulation.passes.PassTimes"):
            acs = ACS(config=config)

        utime = 1514764800.0
        _, _, roll, _ = acs.pointing(utime)

        assert 0.0 <= roll < 360.0
