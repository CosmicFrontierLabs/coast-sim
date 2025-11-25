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
    ditl.utime = [0, 3600, 7200, 10800, 14400]  # 5 time points: 0, 1, 2, 3, 4 hours
    ditl.ra = [0.0, 10.0, 20.0, 30.0, 40.0]
    ditl.dec = [0.0, 5.0, 10.0, 15.0, 20.0]
    ditl.mode = [
        ACSMode.SCIENCE,  # 0
        ACSMode.SAA,  # 1 - SAA passage
        ACSMode.SLEWING,  # 2 - slewing
        ACSMode.SCIENCE,  # 3 - exit charging
        ACSMode.CHARGING,  # 4 - enter charging again, end with charging
    ]
    ditl.obsid = [0, 10000, 10001, 0, 10002]  # Mix of survey observations and slews
    ditl.panel = [0.8, 0.9, 0.7, 0.6, 0.5]  # Solar panel illumination
    ditl.power = [150.0, 200.0, 180.0, 160.0, 140.0]  # Power consumption
    ditl.batterylevel = [0.8, 0.85, 0.82, 0.78, 0.75]  # Battery levels
    ditl.charge_state = [1, 1, 1, 0, 1]  # Charging states

    # Subsystem power breakdown
    ditl.power_bus = [50.0, 60.0, 55.0, 52.0, 48.0]
    ditl.power_payload = [100.0, 140.0, 125.0, 108.0, 92.0]

    # Data management telemetry
    ditl.recorder_volume_gb = [0.0, 0.5, 1.2, 1.8, 2.0]
    ditl.recorder_fill_fraction = [0.0, 0.05, 0.12, 0.18, 0.2]
    ditl.recorder_alert = [0, 0, 0, 1, 2]
    ditl.data_generated_gb = [0.0, 0.5, 1.2, 1.8, 2.5]
    ditl.data_downlinked_gb = [0.0, 0.3, 0.8, 1.4, 2.0]

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
    mock_ditl.constraint = Mock()  # Add constraint mock

    # Mock the SAA data
    mock_ditl.saa.times = np.array([3600, 7200])  # SAA passages at 1 and 2 hours
    mock_ditl.saa.durations = np.array([600, 600])  # 10 minutes each

    # Mock passes data
    mock_ditl.passes.times = np.array([1800, 5400])  # Ground station passes
    mock_ditl.passes.durations = np.array([300, 300])  # 5 minutes each

    # Mock ACS data - set passrequests to have passes for ground station testing
    mock_pass = Mock()
    mock_pass.begin = 1800.0
    mock_pass.length = 300.0
    mock_ditl.acs.passrequests = Mock()
    mock_ditl.acs.passrequests.passes = [mock_pass]

    # Mock constraint with in_eclipse method that returns True for some points
    def mock_in_eclipse(ra, dec, time):
        # Return True for multiple points to cover enter/exit and extending
        return time in [7200, 14400]  # utime[2] and utime[4]

    mock_ditl.constraint.in_eclipse = mock_in_eclipse

    # Add a mock plan with some entries for timeline plotting
    mock_plan_entry1 = Mock()
    mock_plan_entry1.begin = 0.0
    mock_plan_entry1.end = 1800.0
    mock_plan_entry1.obsid = 10000
    mock_plan_entry1.slewtime = 0.0  # No slew for first observation
    mock_plan_entry1.ra = 45.0  # Add RA for sky pointing
    mock_plan_entry1.dec = 30.0  # Add Dec for sky pointing

    mock_plan_entry2 = Mock()
    mock_plan_entry2.begin = 1800.0
    mock_plan_entry2.end = 3600.0
    mock_plan_entry2.obsid = 0  # Slew
    mock_plan_entry2.slewtime = 120.0
    mock_plan_entry2.ra = 90.0
    mock_plan_entry2.dec = -15.0

    # Add a plan entry with zero or negative duration to cover edge cases
    mock_plan_entry3 = Mock()
    mock_plan_entry3.begin = 3600.0
    mock_plan_entry3.end = 3600.0  # Same as begin + slewtime, so duration = 0
    mock_plan_entry3.obsid = 10001
    mock_plan_entry3.slewtime = 0.0
    mock_plan_entry3.ra = 135.0
    mock_plan_entry3.dec = 45.0

    # Add a plan entry with very long duration (> 24 hours) to cover unrealistic duration check
    mock_plan_entry4 = Mock()
    mock_plan_entry4.begin = 7200.0
    mock_plan_entry4.end = 7200.0 + 25 * 3600  # 25 hours later
    mock_plan_entry4.obsid = 10002
    mock_plan_entry4.slewtime = 0.0
    mock_plan_entry4.ra = 180.0
    mock_plan_entry4.dec = -30.0

    mock_ditl.plan = [
        mock_plan_entry1,
        mock_plan_entry2,
        mock_plan_entry3,
        mock_plan_entry4,
    ]

    return mock_ditl
