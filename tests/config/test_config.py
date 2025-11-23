"""Tests for conops.config module."""

from unittest.mock import Mock

from conops import (
    Battery,
    Config,
    Constraint,
    GroundStationRegistry,
    Payload,
    SolarPanelSet,
    SpacecraftBus,
)


class TestConfig:
    """Test Config class initialization."""

    def test_config_sets_name(self, minimal_config):
        """Test that Config sets the provided name."""
        assert minimal_config["config"].name == "Test Config"

    def test_config_sets_spacecraft_bus(self, minimal_config):
        """Test that Config sets spacecraft_bus correctly."""
        assert (
            minimal_config["config"].spacecraft_bus == minimal_config["spacecraft_bus"]
        )

    def test_config_sets_solar_panel(self, minimal_config):
        """Test that Config sets solar_panel correctly."""
        assert minimal_config["config"].solar_panel == minimal_config["solar_panel"]

    def test_config_sets_payload(self, minimal_config):
        """Test that Config sets payload correctly."""
        assert minimal_config["config"].payload == minimal_config["payload"]

    def test_config_sets_battery(self, minimal_config):
        """Test that Config sets battery correctly."""
        assert minimal_config["config"].battery == minimal_config["battery"]

    def test_config_sets_constraint(self, minimal_config):
        """Test that Config sets constraint correctly."""
        assert minimal_config["config"].constraint == minimal_config["constraint"]

    def test_config_sets_ground_stations(self, minimal_config):
        """Test that Config sets ground_stations correctly."""
        assert (
            minimal_config["config"].ground_stations
            == minimal_config["ground_stations"]
        )

    def test_config_default_name(self):
        """Test that Config uses default name."""
        spacecraft_bus = Mock(spec=SpacecraftBus)
        solar_panel = Mock(spec=SolarPanelSet)
        payload = Mock(spec=Payload)
        battery = Mock(spec=Battery)
        constraint = Mock(spec=Constraint)
        panel_constraint_mock = Mock()
        panel_constraint_mock.solar_panel = None
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
