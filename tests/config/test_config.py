"""Tests for conops.config module."""

import json
from unittest.mock import Mock

from conops import (
    Battery,
    Config,
    Constraint,
    FaultManagement,
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

    def test_config_sets_fault_management(self, minimal_config):
        """Test that Config sets fault_management correctly."""
        assert (
            minimal_config["config"].fault_management
            == minimal_config["fault_management"]
        )

    def test_init_fault_management_defaults_none(self):
        """Test init_fault_management_defaults does nothing if fault_management is None."""
        config = Config(
            spacecraft_bus=Mock(spec=SpacecraftBus),
            solar_panel=Mock(spec=SolarPanelSet),
            payload=Mock(spec=Payload),
            battery=Mock(spec=Battery),
            constraint=Mock(spec=Constraint),
            ground_stations=Mock(spec=GroundStationRegistry),
            fault_management=None,
        )
        config.init_fault_management_defaults()
        # No assertions needed, just ensure no errors

    def test_init_fault_management_defaults_adds_threshold(self):
        """Test init_fault_management_defaults adds battery_level threshold if not present."""
        fault_management = FaultManagement()
        battery = Mock(spec=Battery)
        battery.max_depth_of_discharge = 0.8
        config = Config(
            spacecraft_bus=Mock(spec=SpacecraftBus),
            solar_panel=Mock(spec=SolarPanelSet),
            payload=Mock(spec=Payload),
            battery=battery,
            constraint=Mock(spec=Constraint),
            ground_stations=Mock(spec=GroundStationRegistry),
            fault_management=fault_management,
        )
        config.init_fault_management_defaults()
        assert "battery_level" in fault_management.thresholds
        threshold = fault_management.thresholds["battery_level"]
        assert threshold.yellow == 0.8
        assert (
            abs(threshold.red - 0.7) < 1e-10
        )  # Use approximate comparison for floating point

    def test_init_fault_management_defaults_threshold_exists(self):
        """Test init_fault_management_defaults does not add threshold if already present."""
        fault_management = FaultManagement()
        fault_management.add_threshold(
            "battery_level", yellow=0.5, red=0.4, direction="below"
        )
        config = Config(
            spacecraft_bus=Mock(spec=SpacecraftBus),
            solar_panel=Mock(spec=SolarPanelSet),
            payload=Mock(spec=Payload),
            battery=Mock(spec=Battery),
            constraint=Mock(spec=Constraint),
            ground_stations=Mock(spec=GroundStationRegistry),
            fault_management=fault_management,
        )
        config.init_fault_management_defaults()
        # Should still only have the one threshold we added
        assert len(fault_management.thresholds) == 1

    def test_from_json_file(self, tmp_path):
        """Test loading Config from JSON file."""
        json_data = {
            "name": "Test Config",
            "spacecraft_bus": {"mass": 100.0},
            "solar_panel": {"area": 10.0},
            "payload": {"mass": 50.0},
            "battery": {"capacity": 200.0, "max_depth_of_discharge": 0.8},
            "constraint": {"some_constraint": "value"},
            "ground_stations": {},
            "fault_management": None,
        }
        file_path = tmp_path / "config.json"
        with open(file_path, "w") as f:
            json.dump(json_data, f)
        config = Config.from_json_file(str(file_path))
        assert config.name == "Test Config"
        assert config.fault_management is None

    def test_to_json_file(self, tmp_path):
        """Test saving Config to JSON file."""
        config = Config(
            name="Test Config",
            spacecraft_bus=Mock(spec=SpacecraftBus, mass=100.0),
            solar_panel=Mock(spec=SolarPanelSet, area=10.0),
            payload=Mock(spec=Payload, mass=50.0),
            battery=Mock(spec=Battery, capacity=200.0, max_depth_of_discharge=0.8),
            constraint=Mock(spec=Constraint),
            ground_stations=Mock(spec=GroundStationRegistry),
            fault_management=None,
        )
        file_path = tmp_path / "config.json"
        config.to_json_file(str(file_path))
        assert file_path.exists()
        with open(file_path) as f:
            data = json.load(f)
        assert data["name"] == "Test Config"
