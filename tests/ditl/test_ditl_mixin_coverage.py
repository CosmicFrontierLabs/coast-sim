"""Tests to drive `conops.ditl.ditl_mixin.DITLMixin` to 100% coverage."""

from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import pytest

from conops.common.enums import ACSMode
from conops.config import Config
from conops.ditl.ditl_mixin import DITLMixin


@pytest.fixture
def mock_config():
    cfg = Mock(spec=Config)
    # constraint with ephem required for ACS init
    # Build a minimal ephem with earth[0].ra.deg and dec.deg
    ra = Mock()
    ra.deg = 0.0
    dec = Mock()
    dec.deg = 0.0
    earth_entry = Mock()
    earth_entry.ra = ra
    earth_entry.dec = dec
    ephem = Mock()
    ephem.earth = [earth_entry]
    # Also provide sun and index for safe mode fallbacks (unused here but safe)
    sun_ra = Mock()
    sun_ra.deg = 0.0
    sun_dec = Mock()
    sun_dec.deg = 0.0
    sun_entry = Mock()
    sun_entry.ra = sun_ra
    sun_entry.dec = sun_dec
    ephem.sun = [sun_entry]
    ephem.index = Mock(return_value=0)

    cfg.constraint = Mock()
    cfg.constraint.ephem = ephem
    # subsystems
    cfg.battery = Mock()
    cfg.spacecraft_bus = Mock()
    cfg.spacecraft_bus.attitude_control = Mock()
    cfg.payload = Mock()
    cfg.recorder = Mock()
    # ground stations
    cfg.ground_stations = Mock()
    # solar panel optional
    cfg.solar_panel = None
    return cfg


def test_plot_calls_visualization_show(mock_config):
    mixin = DITLMixin(config=mock_config)
    with patch("conops.visualization.plot_ditl_telemetry") as plot_ditl_telemetry:
        with patch.object(plt, "show") as show:
            mixin.plot()
            plot_ditl_telemetry.assert_called_once()
            show.assert_called_once()


def test_find_current_pass_checks_scheduled_and_executed(mock_config):
    mixin = DITLMixin(config=mock_config)
    utime = 1000.0

    # No passes -> None
    assert mixin._find_current_pass(utime) is None

    # Scheduled passes path
    scheduled = Mock()
    scheduled.in_pass = Mock(return_value=True)
    mixin.acs.passrequests.passes = [scheduled]
    assert mixin._find_current_pass(utime) is scheduled

    # Executed passes fallback path
    mixin.acs.passrequests.passes = []
    executed = Mock()
    executed.in_pass = Mock(return_value=True)
    mixin.executed_passes.passes = [executed]
    assert mixin._find_current_pass(utime) is executed


def test_process_data_management_generates_and_downlinks(mock_config):
    mixin = DITLMixin(config=mock_config)
    utime = 2000.0
    step = 60

    # SCIENCE mode generates data
    mock_config.payload.data_generated = Mock(return_value=0.5)  # Gb per step
    mock_config.recorder.add_data = Mock()
    gen, dl = mixin._process_data_management(utime, ACSMode.SCIENCE, step)
    assert gen == 0.5
    mock_config.recorder.add_data.assert_called_once_with(0.5)
    assert dl == 0.0

    # PASS mode downlinks if in pass with effective rate
    # scheduled pass detection with station + comms config
    pass_obj = Mock()
    pass_obj.in_pass = Mock(return_value=True)
    pass_obj.station = "GS1"
    pass_obj.comms_config = Mock()
    pass_obj.comms_config.get_downlink_rate = Mock(return_value=50.0)  # Mbps
    mixin.acs.passrequests.passes = [pass_obj]
    # ground station
    station = Mock()
    station.supported_bands = Mock(return_value=["X"])  # one band
    station.bands = ["X"]
    station.get_downlink_rate = Mock(return_value=100.0)
    mock_config.ground_stations.get = Mock(return_value=station)
    # recorder remove_data returns the amount actually removed (Gb)
    mock_config.recorder.remove_data = Mock(return_value=0.3)

    gen2, dl2 = mixin._process_data_management(utime, ACSMode.PASS, step)
    # 50 Mbps effective => 50*60 / 1000 / 8 = 0.375 Gb per step, but recorder returns 0.3
    assert gen2 == 0.0
    assert dl2 == 0.3


def test_get_effective_data_rate_branches(mock_config):
    mixin = DITLMixin(config=mock_config)
    station = Mock()
    current_pass = Mock()

    # No comms_config -> use station overall max
    station.get_overall_max_downlink = Mock(return_value=10.0)
    current_pass.comms_config = None
    assert mixin._get_effective_data_rate(station, current_pass) == 10.0

    # No bands on station -> None
    current_pass.comms_config = Mock()
    station.bands = None
    station.supported_bands = Mock(return_value=[])
    assert mixin._get_effective_data_rate(station, current_pass) is None

    # Compute min across common bands and take max
    station.bands = ["S", "X", "Ka"]
    station.supported_bands = Mock(return_value=["S", "X", "Ka"])  # defined
    station.get_downlink_rate = Mock(
        side_effect=lambda b: {"S": 5.0, "X": 50.0, "Ka": 80.0}[b]
    )
    current_pass.comms_config.get_downlink_rate = Mock(
        side_effect=lambda b: {"S": 10.0, "X": 40.0, "Ka": 0.0}[b]
    )
    # Effective per band: S=min(5,10)=5, X=min(50,40)=40, Ka excluded (0)
    assert mixin._get_effective_data_rate(station, current_pass) == 40.0
