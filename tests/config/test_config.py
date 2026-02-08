"""Tests for conops.config module."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import yaml

from conops import (
    Battery,
    Constraint,
    FaultManagement,
    GroundStationRegistry,
    MissionConfig,
    Payload,
    SolarPanelSet,
    SpacecraftBus,
)


class TestConfig:
    """Test Config class initialization."""

    def test_config_sets_name(self, minimal_config: dict[str, Any]) -> None:
        """Test that Config sets the provided name."""
        assert minimal_config["config"].name == "Test Config"

    def test_config_sets_spacecraft_bus(self, minimal_config: dict[str, Any]) -> None:
        """Test that Config sets spacecraft_bus correctly."""
        assert (
            minimal_config["config"].spacecraft_bus == minimal_config["spacecraft_bus"]
        )

    def test_config_sets_solar_panel(self, minimal_config: dict[str, Any]) -> None:
        """Test that Config sets solar_panel correctly."""
        assert minimal_config["config"].solar_panel == minimal_config["solar_panel"]

    def test_config_sets_payload(self, minimal_config: dict[str, Any]) -> None:
        """Test that Config sets payload correctly."""
        assert minimal_config["config"].payload == minimal_config["payload"]

    def test_config_sets_battery(self, minimal_config: dict[str, Any]) -> None:
        """Test that Config sets battery correctly."""
        assert minimal_config["config"].battery == minimal_config["battery"]

    def test_config_sets_constraint(self, minimal_config: dict[str, Any]) -> None:
        """Test that Config sets constraint correctly."""
        assert minimal_config["config"].constraint == minimal_config["constraint"]

    def test_config_sets_ground_stations(self, minimal_config: dict[str, Any]) -> None:
        """Test that Config sets ground_stations correctly."""
        assert (
            minimal_config["config"].ground_stations
            == minimal_config["ground_stations"]
        )

    def test_config_default_name(self) -> None:
        """Test that Config uses default name."""
        spacecraft_bus = Mock(spec=SpacecraftBus)
        solar_panel = Mock(spec=SolarPanelSet)
        payload = Mock(spec=Payload)
        battery = Mock(spec=Battery)
        battery.max_depth_of_discharge = 0.3  # Configure mock battery
        constraint = Mock(spec=Constraint)
        panel_constraint_mock = Mock()
        panel_constraint_mock.solar_panel = None
        constraint.panel_constraint = panel_constraint_mock
        ground_stations = Mock(spec=GroundStationRegistry)

        config = MissionConfig(
            spacecraft_bus=spacecraft_bus,
            solar_panel=solar_panel,
            payload=payload,
            battery=battery,
            constraint=constraint,
            ground_stations=ground_stations,
        )

        assert config.name == "Default Config"

    def test_config_sets_fault_management(self, minimal_config: dict[str, Any]) -> None:
        """Test that Config sets fault_management correctly."""
        assert minimal_config["config"].fault_management is not None
        assert isinstance(minimal_config["config"].fault_management, FaultManagement)

    def test_init_fault_management_defaults_none(self) -> None:
        """Test init_fault_management_defaults does nothing if fault_management is None."""
        battery = Mock(spec=Battery)
        battery.max_depth_of_discharge = 0.3  # Configure mock battery
        config = MissionConfig(
            spacecraft_bus=Mock(spec=SpacecraftBus),
            solar_panel=Mock(spec=SolarPanelSet),
            payload=Mock(spec=Payload),
            battery=battery,
            constraint=Mock(spec=Constraint),
            ground_stations=Mock(spec=GroundStationRegistry),
        )
        # Set fault_management to None after creation to test the validator
        config.fault_management = None
        config.init_fault_management_defaults()
        # No assertions needed, just ensure no errors

    def test_init_fault_management_defaults_adds_threshold(self) -> None:
        """Test init_fault_management_defaults adds battery_level threshold if not present."""
        fault_management = FaultManagement()
        battery = Mock(spec=Battery)
        battery.max_depth_of_discharge = 0.2
        config = MissionConfig(
            spacecraft_bus=Mock(spec=SpacecraftBus),
            solar_panel=Mock(spec=SolarPanelSet),
            payload=Mock(spec=Payload),
            battery=battery,
            constraint=Mock(spec=Constraint),
            ground_stations=Mock(spec=GroundStationRegistry),
            fault_management=fault_management,
        )
        config.init_fault_management_defaults()
        assert any(t.name == "battery_level" for t in fault_management.thresholds)
        threshold = next(
            t for t in fault_management.thresholds if t.name == "battery_level"
        )
        assert threshold.yellow == 0.8
        assert (
            abs(threshold.red - 0.7) < 1e-10
        )  # Use approximate comparison for floating point

    def test_init_fault_management_defaults_threshold_exists(self) -> None:
        """Test init_fault_management_defaults does not add threshold if already present."""
        fault_management = FaultManagement()
        fault_management.add_threshold(
            "battery_level", yellow=0.5, red=0.4, direction="below"
        )
        battery = Mock(spec=Battery)
        battery.max_depth_of_discharge = 0.2
        config = MissionConfig(
            spacecraft_bus=Mock(spec=SpacecraftBus),
            solar_panel=Mock(spec=SolarPanelSet),
            payload=Mock(spec=Payload),
            battery=battery,
            constraint=Mock(spec=Constraint),
            ground_stations=Mock(spec=GroundStationRegistry),
            fault_management=fault_management,
        )
        config.init_fault_management_defaults()
        # Should have battery_level (that we added) and recorder_fill_fraction (added by init)
        assert len(fault_management.thresholds) == 2
        assert any(t.name == "battery_level" for t in fault_management.thresholds)
        assert any(
            t.name == "recorder_fill_fraction" for t in fault_management.thresholds
        )
        # Battery level should have our custom values, not defaults
        battery_threshold = next(
            t for t in fault_management.thresholds if t.name == "battery_level"
        )
        assert battery_threshold.yellow == 0.5
        assert battery_threshold.red == 0.4

    def test_from_json_file(self, tmp_path: Path) -> None:
        """Test loading Config from JSON file."""
        json_data = {
            "name": "Test Config",
            "spacecraft_bus": {"mass": 100.0},
            "solar_panel": {"area": 10.0},
            "payload": {"mass": 50.0},
            "battery": {"capacity": 200.0, "max_depth_of_discharge": 0.8},
            "constraint": {"some_constraint": "value"},
            "ground_stations": {},
        }
        file_path = tmp_path / "config.json"
        with open(file_path, "w") as f:
            json.dump(json_data, f)
        config = MissionConfig.from_json_file(str(file_path))
        assert config.name == "Test Config"
        assert config.fault_management is not None

    def test_to_json_file(self, tmp_path: Path) -> None:
        """Test saving Config to JSON file."""
        config = MissionConfig(
            name="Test Config",
            spacecraft_bus=Mock(spec=SpacecraftBus, mass=100.0),
            solar_panel=Mock(spec=SolarPanelSet, area=10.0),
            payload=Mock(spec=Payload, mass=50.0),
            battery=Mock(spec=Battery, capacity=200.0, max_depth_of_discharge=0.8),
            constraint=Mock(spec=Constraint),
            ground_stations=Mock(spec=GroundStationRegistry),
        )
        file_path = tmp_path / "config.json"
        config.to_json_file(str(file_path))
        assert file_path.exists()
        with open(file_path) as f:
            data = json.load(f)
        assert data["name"] == "Test Config"

    def test_from_yaml_file(self, tmp_path: Path) -> None:
        """Test loading Config from YAML file."""
        yaml_data = {
            "name": "Test Config",
            "spacecraft_bus": {"mass": 100.0},
            "solar_panel": {"area": 10.0},
            "payload": {"mass": 50.0},
            "battery": {"capacity": 200.0, "max_depth_of_discharge": 0.8},
            "constraint": {"some_constraint": "value"},
            "ground_stations": {},
        }
        file_path = tmp_path / "config.yaml"
        with open(file_path, "w") as f:
            yaml.safe_dump(yaml_data, f)
        config = MissionConfig.from_yaml_file(str(file_path))
        assert config.name == "Test Config"
        assert config.fault_management is not None

    def test_to_yaml_file(self, tmp_path: Path) -> None:
        """Test saving Config to YAML file."""
        # Create a realistic config instead of using Mocks
        config = MissionConfig(name="Test Config")
        file_path = tmp_path / "config.yaml"
        config.to_yaml_file(str(file_path))
        assert file_path.exists()
        with open(file_path) as f:
            content = f.read()
        # Check for header comments
        assert "COAST-Sim Mission Configuration File" in content
        assert "Units Legend:" in content
        # Check that data was written (name value may or may not have quotes depending on YAML dumper)
        assert "name: Test Config" in content or 'name: "Test Config"' in content

    def test_yaml_roundtrip(self, tmp_path: Path) -> None:
        """Test that YAML save and load maintains data integrity."""
        # Use example config to test full roundtrip
        import os

        example_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "examples", "example_config.json"
        )
        if os.path.exists(example_path):
            config1 = MissionConfig.from_json_file(example_path)
            yaml_path = tmp_path / "roundtrip.yaml"
            config1.to_yaml_file(str(yaml_path))
            config2 = MissionConfig.from_yaml_file(str(yaml_path))
            # Compare the model dumps to ensure data integrity
            assert config1.model_dump() == config2.model_dump()
