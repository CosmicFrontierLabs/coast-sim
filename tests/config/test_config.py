"""Tests for conops.config module."""

from unittest.mock import Mock

from conops.battery import Battery
from conops.config import Config
from conops.constraint import Constraint
from conops.groundstation import GroundStationRegistry
from conops.instrument import Payload
from conops.solar_panel import SolarPanelSet
from conops.spacecraft_bus import SpacecraftBus


class TestConfig:
    """Test Config class initialization."""

    def test_config_initialization_and_post_init(self):
        """Test that Config initializes correctly."""
        # Create minimal required objects
        spacecraft_bus = Mock(spec=SpacecraftBus)
        solar_panel = Mock(spec=SolarPanelSet)
        payload = Mock(spec=Payload)
        battery = Mock(spec=Battery)
        constraint = Mock(spec=Constraint)
        ground_stations = Mock(spec=GroundStationRegistry)

        # Create config
        config = Config(
            name="Test Config",
            spacecraft_bus=spacecraft_bus,
            solar_panel=solar_panel,
            payload=payload,
            battery=battery,
            constraint=constraint,
            ground_stations=ground_stations,
        )

        # Verify initialization
        assert config.name == "Test Config"
        assert config.spacecraft_bus == spacecraft_bus
        assert config.solar_panel == solar_panel
        assert config.payload == payload
        assert config.battery == battery
        assert config.constraint == constraint
        assert config.ground_stations == ground_stations

    def test_config_default_name(self):
        """Test that Config uses default name."""
        spacecraft_bus = Mock(spec=SpacecraftBus)
        solar_panel = Mock(spec=SolarPanelSet)
        payload = Mock(spec=Payload)
        battery = Mock(spec=Battery)
        constraint = Mock(spec=Constraint)
        # Create a proper mock for panel_constraint with solar_panel attribute
        panel_constraint_mock = Mock()
        panel_constraint_mock.solar_panel = None  # Initialize to None
        constraint.panel_constraint = panel_constraint_mock
        ground_stations = Mock(spec=GroundStationRegistry)

        config = Config(
            spacecraft_bus=spacecraft_bus,
            solar_panel=solar_panel,
            payload=payload,
            battery=battery,
            constraint=constraint,
            ground_stations=ground_stations,
        )

        assert config.name == "Default Config"
