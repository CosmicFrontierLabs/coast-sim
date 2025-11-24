"""Shared fixtures for visualization tests."""

import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from unittest.mock import Mock

import numpy as np

from conops.common import ACSMode


@pytest.fixture
def mock_ditl():
    """Create a mock DITL object with minimal telemetry data for testing."""
    from unittest.mock import Mock

    # Create minimal config with required fields
    config = Mock()
    config.name = "Test Config"

    # Mock battery with max_depth_of_discharge
    config.battery = Mock()
    config.battery.max_depth_of_discharge = 0.2

    # Mock recorder configuration
    config.recorder = Mock()
    config.recorder.capacity_gb = 2.0
    config.recorder.yellow_threshold = 0.8
    config.recorder.red_threshold = 0.95

    # Set observation_categories to None to use defaults
    config.observation_categories = None

    # Create mock DITL with required attributes
    ditl = Mock()
    ditl.config = config

    # Add minimal telemetry data
    ditl.utime = [0, 3600, 7200, 10800]  # 4 time points: 0, 1, 2, 3 hours
    ditl.ra = [0.0, 10.0, 20.0, 30.0]
    ditl.dec = [0.0, 5.0, 10.0, 15.0]
    ditl.mode = [
        ACSMode.SCIENCE.value,
        ACSMode.SLEWING.value,
        ACSMode.SLEWING.value,
        ACSMode.SCIENCE.value,
    ]
    ditl.obsid = [0, 10000, 10001, 0]  # Mix of survey observations and slews
    ditl.panel = [0.8, 0.9, 0.7, 0.6]  # Solar panel illumination
    ditl.power = [150.0, 200.0, 180.0, 160.0]  # Power consumption
    ditl.batterylevel = [0.8, 0.85, 0.82, 0.78]  # Battery levels
    ditl.charge_state = [1, 1, 1, 0]  # Charging states

    # Subsystem power breakdown
    ditl.power_bus = [50.0, 60.0, 55.0, 52.0]
    ditl.power_payload = [100.0, 140.0, 125.0, 108.0]

    # Data management telemetry
    ditl.recorder_volume_gb = [0.0, 0.5, 1.2, 1.8]
    ditl.recorder_fill_fraction = [0.0, 0.05, 0.12, 0.18]
    ditl.recorder_alert = [0, 0, 0, 1]
    ditl.data_generated_gb = [0.0, 0.5, 1.2, 1.8]
    ditl.data_downlinked_gb = [0.0, 0.3, 0.8, 1.4]

    return ditl


@pytest.fixture
def mock_ditl_with_ephem(mock_ditl):
    """Create a mock DITL with ephemeris data for timeline plotting."""
    # Add ephemeris-related attributes needed for timeline plotting
    mock_ditl.ephem = Mock()
    mock_ditl.saa = Mock()
    mock_ditl.passes = Mock()
    mock_ditl.executed_passes = Mock()
    mock_ditl.acs = Mock()  # Add ACS mock

    # Mock the SAA data
    mock_ditl.saa.times = np.array([3600, 7200])  # SAA passages at 1 and 2 hours
    mock_ditl.saa.durations = np.array([600, 600])  # 10 minutes each

    # Mock passes data
    mock_ditl.passes.times = np.array([1800, 5400])  # Ground station passes
    mock_ditl.passes.durations = np.array([300, 300])  # 5 minutes each

    # Mock ACS data - set passrequests to None to avoid iteration
    mock_ditl.acs.passrequests = None

    # Add a mock plan with some entries for timeline plotting
    mock_plan_entry1 = Mock()
    mock_plan_entry1.begin = 0.0
    mock_plan_entry1.end = 1800.0
    mock_plan_entry1.obsid = 10000
    mock_plan_entry1.slewtime = 0.0  # No slew for first observation

    mock_plan_entry2 = Mock()
    mock_plan_entry2.begin = 1800.0
    mock_plan_entry2.end = 3600.0
    mock_plan_entry2.obsid = 0  # Slew
    mock_plan_entry2.slewtime = 120.0

    mock_ditl.plan = [mock_plan_entry1, mock_plan_entry2]

    return mock_ditl
