"""Test fixtures for ditl subsystem tests."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from matplotlib import pyplot as plt

from conops import DITL, ACSMode
from conops.ditl.ditl_mixin import DITLMixin


class DummyEphemeris:
    """Minimal mock ephemeris for testing."""

    def __init__(self):
        from datetime import datetime, timezone

        self.step_size = 1.0
        # Cover 2018 day 331 (Nov 27) for 2 days
        unix_times = np.arange(1543276800, 1543449600, 60)
        self.timestamp = [
            datetime.fromtimestamp(t, tz=timezone.utc) for t in unix_times
        ]
        self.utime = unix_times

    def index(self, time):
        """Mock index method."""
        return 0


@pytest.fixture
def mock_config():
    """Create a minimal mock config with required attributes for DITLMixin."""
    cfg = Mock()
    cfg.name = "test"
    cfg.constraint = Mock()
    cfg.constraint.ephem = Mock()  # DITLMixin asserts this is not None
    cfg.battery = Mock()
    cfg.battery.max_depth_of_discharge = 0.5
    return cfg


@pytest.fixture
def mock_ephem():
    """Create a mock ephemeris object."""
    return DummyEphemeris()


@pytest.fixture
def mock_config_detailed():
    """Create a mock config with all required subsystems."""
    config = Mock()

    # Mock constraint
    config.constraint = Mock()
    config.constraint.ephem = DummyEphemeris()
    config.constraint.panel_constraint = Mock()
    config.constraint.panel_constraint.solar_panel = Mock()
    config.constraint.inoccult = Mock(return_value=False)

    # Mock battery
    config.battery = Mock()
    config.battery.battery_level = 0.8
    config.battery.battery_alert = False
    config.battery.drain = Mock()
    config.battery.charge = Mock()
    config.battery.panel_charge_rate = 100.0

    # Mock spacecraft bus
    config.spacecraft_bus = Mock()
    config.spacecraft_bus.power = Mock(return_value=50.0)

    # Mock payload
    config.payload = Mock()
    config.payload.power = Mock(return_value=30.0)
    config.payload.data_generated = Mock(return_value=0.0)

    # Mock recorder
    config.recorder = Mock()
    config.recorder.current_volume_gb = 0.0
    config.recorder.get_fill_fraction = Mock(return_value=0.0)
    config.recorder.get_alert_level = Mock(return_value="none")
    config.recorder.add_data = Mock()
    config.recorder.remove_data = Mock(return_value=0.0)

    # Mock solar panel
    config.solar_panel = Mock()
    config.solar_panel.power = Mock(return_value=100.0)
    config.solar_panel.panel_illumination_fraction = Mock(return_value=0.5)
    config.solar_panel.illumination_and_power = Mock(return_value=(0.5, 100.0))

    # Mock ground stations
    config.ground_stations = Mock()

    return config


@pytest.fixture
def ditl(mock_config_detailed, mock_ephem):
    """Create a DITL instance with mocked dependencies."""
    with (
        patch("conops.PassTimes") as mock_passtimes,
        patch("conops.ACS") as mock_acs_class,
    ):
        # Mock PassTimes
        mock_pt = Mock()
        mock_pt.passes = []
        mock_pt.get = Mock()
        mock_passtimes.return_value = mock_pt

        # Mock ACS
        mock_acs = Mock()
        mock_acs.ephem = None
        mock_acs.slewing = False
        mock_acs.inpass = False
        mock_acs.saa = None
        mock_acs.pointing = Mock(return_value=(0.0, 0.0, 0.0, 0))
        mock_acs.add_slew = Mock()
        mock_acs.passrequests = mock_pt
        mock_acs.get_mode = Mock(return_value=ACSMode.SCIENCE)
        mock_acs_class.return_value = mock_acs

        ditl = DITL(config=mock_config_detailed)
        ditl.ephem = mock_ephem
        ditl.acs = mock_acs
        ditl.plan = Mock()
        ditl.plan.which_ppt = Mock(
            return_value=Mock(ra=0.0, dec=0.0, obsid=1, obstype="science")
        )

        return ditl


@pytest.fixture
def mock_pass_inst():
    """Mock PassTimes instance."""
    return Mock()


@pytest.fixture
def mock_acs_inst():
    """Mock ACS instance."""
    return Mock()


@pytest.fixture
def ditl_instance(mock_config, mock_pass_inst, mock_acs_inst):
    """Fixture to create a DITLMixin instance with mocked dependencies."""
    with (
        patch("conops.ditl.ditl_mixin.PassTimes") as mock_pass_class,
        patch("conops.ditl.ditl_mixin.ACS") as mock_acs_class,
        patch("conops.ditl.ditl_mixin.Plan") as mock_plan_class,
    ):
        # Set return values for patched classes
        mock_pass_class.return_value = mock_pass_inst
        mock_acs_class.return_value = mock_acs_inst
        mock_plan_class.return_value = Mock()

        ditl = DITLMixin(config=mock_config)
        return ditl, mock_pass_inst, mock_acs_inst


@pytest.fixture
def populated_ditl(ditl_instance):
    """Fixture to populate DITLMixin with sample data for plotting."""
    ditl, _, _ = ditl_instance
    # Populate data arrays of same length
    base_time = 1514764800.0
    ditl.utime = [base_time + i * 60 for i in range(4)]
    ditl.ra = [1.0, 2.0, 3.0, 4.0]
    ditl.dec = [0.5, 0.6, 0.7, 0.8]
    ditl.mode = [0, 1, 2, 3]
    ditl.batterylevel = [0.2, 0.3, 0.4, 0.5]
    ditl.panel = [0.1, 0.2, 0.3, 0.4]
    ditl.power = [5.0, 6.0, 7.0, 8.0]
    ditl.obsid = [0, 1, 2, 3]
    return ditl


@pytest.fixture
def comprehensive_ditl(ditl_instance, mock_config):
    """Fixture to populate DITLMixin with comprehensive data for statistics."""
    ditl, _, _ = ditl_instance
    # Mock battery capacity
    mock_config.battery.watthour = 100.0
    mock_config.battery.max_depth_of_discharge = 0.1

    # Mock recorder capacity
    mock_config.recorder.capacity_gb = 100.0
    mock_config.recorder.yellow_threshold = 0.8
    mock_config.recorder.red_threshold = 0.95

    # Populate comprehensive telemetry data
    base_time = 1514764800.0  # 2018-01-01 00:00:00 UTC
    ditl.utime = [base_time + i * 60 for i in range(10)]
    ditl.mode = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]  # Mix of modes
    ditl.obsid = [0, 10000, 10001, 0, 10002, 10003, 0, 10004, 10005, 0]
    ditl.ra = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ditl.dec = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    ditl.roll = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    ditl.batterylevel = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7]
    ditl.charge_state = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    ditl.power = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    ditl.power_bus = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
    ditl.power_payload = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
    ditl.panel_power = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ditl.panel = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Data management
    ditl.recorder_volume_gb = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    ditl.recorder_fill_fraction = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ditl.recorder_alert = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    ditl.data_generated_gb = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ditl.data_downlinked_gb = [
        0.0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
    ]

    # Mock queue and ACS commands for statistics
    ditl.queue = Mock()
    ditl.queue.targets = [Mock(done=True), Mock(done=False), Mock(done=True)]

    ditl.acs = Mock()
    mock_cmd1 = Mock()
    mock_cmd1.command_type = Mock()
    mock_cmd1.command_type.name = "SLEW"
    mock_cmd2 = Mock()
    mock_cmd2.command_type = Mock()
    mock_cmd2.command_type.name = "OBSERVE"
    ditl.acs.commands = [mock_cmd1, mock_cmd2, mock_cmd1]

    # Mock executed passes
    mock_pass = Mock()
    mock_pass.begin = base_time + 120
    mock_pass.end = base_time + 240
    ditl.executed_passes = Mock()
    ditl.executed_passes.passes = [mock_pass]

    return ditl


@pytest.fixture
def statistics_output(comprehensive_ditl):
    """Fixture to capture output of print_statistics."""
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        comprehensive_ditl.print_statistics()
    return f.getvalue()


@pytest.fixture
def plot_figure(populated_ditl):
    """Fixture to create and yield the plot figure."""
    plt.close("all")
    with patch("matplotlib.pyplot.show"):
        populated_ditl.plot()
    fig = plt.gcf()
    yield fig
    plt.close("all")


@pytest.fixture
def ditl_with_payload_and_recorder(ditl_instance):
    """Fixture to set up ditl with payload and recorder mocks."""
    ditl, _, _ = ditl_instance
    ditl.payload = Mock()
    ditl.payload.data_generated.return_value = 0.1
    ditl.recorder = Mock()
    ditl.recorder.add_data.return_value = None
    ditl.recorder.remove_data.return_value = 0.05
    return ditl


@pytest.fixture
def ditl_with_pass_setup(ditl_instance, mock_config):
    """Fixture to set up ditl with pass-related mocks."""
    ditl, _, _ = ditl_instance
    ditl.payload = Mock()
    ditl.payload.data_generated.return_value = 0.1
    ditl.recorder = Mock()
    ditl.recorder.add_data.return_value = None
    ditl.recorder.remove_data.return_value = 0.05
    mock_pass = Mock()
    mock_pass.in_pass.return_value = True
    mock_pass.station = "station1"
    ditl.acs = Mock()
    ditl.acs.passrequests = Mock()
    ditl.acs.passrequests.passes = [mock_pass]
    mock_station = Mock()
    mock_station.antenna.max_data_rate_mbps = 100.0
    ditl.config.ground_stations = {"station1": mock_station}
    return ditl
