"""Tests for DITLMixin.print_statistics method."""

import io
import sys
from datetime import datetime
from unittest.mock import Mock

from conops import (
    ACSMode,
    Battery,
    Config,
    Constraint,
    DITLMixin,
    DITLStats,
    GroundStationRegistry,
    Payload,
    SolarPanelSet,
    SpacecraftBus,
)


class MockDITL(DITLMixin, DITLStats):
    """Mock DITL class for testing."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # Initialize required lists
        self.ra = []
        self.dec = []
        self.roll = []
        self.mode = []
        self.panel = []
        self.power = []
        self.panel_power = []
        self.batterylevel = []
        self.obsid = []
        self.utime = []


def create_test_config():
    """Create a minimal test config."""
    spacecraft_bus = Mock(spec=SpacecraftBus)
    solar_panel = Mock(spec=SolarPanelSet)
    payload = Mock(spec=Payload)
    battery = Mock(spec=Battery)
    battery.capacity = 100.0
    battery.max_depth_of_discharge = 0.3
    constraint = Mock(spec=Constraint)
    constraint.ephem = Mock()  # Mock ephemeris to satisfy PassTimes
    ground_stations = Mock(spec=GroundStationRegistry)

    config = Config(
        name="Test Spacecraft",
        spacecraft_bus=spacecraft_bus,
        solar_panel=solar_panel,
        payload=payload,
        battery=battery,
        constraint=constraint,
        ground_stations=ground_stations,
    )
    return config


def test_print_statistics_basic():
    """Test that print_statistics runs without errors on basic data."""
    config = create_test_config()

    # Create mock DITL
    ditl = MockDITL(config)

    # Set up basic simulation parameters
    ditl.begin = datetime(2025, 11, 1, 0, 0, 0)
    ditl.end = datetime(2025, 11, 1, 1, 0, 0)
    ditl.step_size = 60

    # Add some sample data
    for i in range(60):
        ditl.utime.append(i * 60)
        ditl.ra.append(180.0 + i * 0.1)
        ditl.dec.append(45.0 + i * 0.05)
        ditl.roll.append(0.0)
        ditl.mode.append(ACSMode.SCIENCE if i % 5 != 0 else ACSMode.SLEWING)
        ditl.panel.append(0.8 if i % 10 < 8 else 0.0)  # Simulate eclipse
        ditl.power.append(50.0 + i * 0.5)
        ditl.panel_power.append(80.0 if i % 10 < 8 else 0.0)
        ditl.batterylevel.append(0.8 - i * 0.001)
        ditl.obsid.append(1000 + (i // 10))

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call print_statistics
    ditl.print_statistics()

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Get the output
    output = captured_output.getvalue()

    # Verify key sections are present
    assert "DITL SIMULATION STATISTICS" in output
    assert "Configuration: Test Spacecraft" in output
    assert "MODE DISTRIBUTION" in output
    assert "OBSERVATION STATISTICS" in output
    assert "POINTING STATISTICS" in output
    assert "POWER AND BATTERY STATISTICS" in output
    assert "Battery Capacity: 100.00 Wh" in output
    assert "SCIENCE" in output
    assert "SLEWING" in output


def test_print_statistics_with_queue():
    """Test that print_statistics handles queue information."""
    config = create_test_config()

    # Create mock DITL
    ditl = MockDITL(config)

    # Set up basic simulation parameters
    ditl.begin = datetime(2025, 11, 1, 0, 0, 0)
    ditl.end = datetime(2025, 11, 1, 1, 0, 0)
    ditl.step_size = 60

    # Add minimal data
    ditl.utime = [0]
    ditl.mode = [ACSMode.SCIENCE]
    ditl.obsid = [1000]
    ditl.batterylevel = [0.8]
    ditl.ra = [180.0]
    ditl.dec = [45.0]

    # Add a mock queue
    from conops import Queue

    ditl.queue = Queue()

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call print_statistics
    ditl.print_statistics()

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Get the output
    output = captured_output.getvalue()

    # Verify queue section is present
    assert "TARGET QUEUE STATISTICS" in output


def test_print_statistics_empty_data():
    """Test that print_statistics handles empty data gracefully."""
    config = create_test_config()

    # Create mock DITL with empty data
    ditl = MockDITL(config)
    ditl.begin = datetime(2025, 11, 1, 0, 0, 0)
    ditl.end = datetime(2025, 11, 1, 1, 0, 0)
    ditl.step_size = 60

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call print_statistics - should not raise any errors
    ditl.print_statistics()

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Get the output
    output = captured_output.getvalue()

    # Verify basic header is present even with empty data
    assert "DITL SIMULATION STATISTICS" in output
    assert "Configuration: Test Spacecraft" in output
