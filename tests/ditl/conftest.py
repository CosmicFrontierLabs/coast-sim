"""Test fixtures for ditl subsystem tests."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from conops.common import ACSMode
from conops.ditl import DITL


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
        patch("conops.ditl_mixin.PassTimes") as mock_passtimes,
        patch("conops.ditl_mixin.ACS") as mock_acs_class,
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
        ditl.ppst = Mock()
        ditl.ppst.which_ppt = Mock(
            return_value=Mock(ra=0.0, dec=0.0, obsid=1, obstype="science")
        )

        return ditl
