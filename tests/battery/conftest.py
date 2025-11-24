"""Test fixtures for battery subsystem tests."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from conops import (
    Battery,
    Constraint,
    EmergencyCharging,
    QueueDITL,
    SolarPanel,
    SolarPanelSet,
)


@pytest.fixture
def default_battery():
    """Fixture for a default Battery instance."""
    return Battery()


@pytest.fixture
def battery_with_custom_threshold():
    """Fixture for a Battery instance with custom recharge threshold."""
    return Battery(recharge_threshold=0.90)


@pytest.fixture
def battery_with_dod():
    """Fixture for a Battery instance with max depth of discharge."""
    return Battery(max_depth_of_discharge=0.35)  # Allows discharge to 65% charge level


@pytest.fixture
def battery_with_dod_and_threshold():
    """Fixture for a Battery instance with both max depth of discharge and recharge threshold."""
    return Battery(max_depth_of_discharge=0.35, recharge_threshold=0.95)


@pytest.fixture
def mock_constraint():
    """Create a mock constraint."""
    constraint = Mock(spec=Constraint)
    constraint.inoccult = Mock(return_value=False)
    constraint.ephem = Mock()  # Add ephem for Pointing initialization
    # Add panel_constraint with solar_panel for EmergencyCharging initialization
    constraint.panel_constraint = Mock()
    constraint.panel_constraint.solar_panel = Mock(spec=SolarPanel)
    constraint.in_eclipse = Mock(return_value=False)
    return constraint


@pytest.fixture
def mock_solar_panel():
    """Create a mock solar panel."""
    solar_panel = Mock(spec=SolarPanelSet)
    solar_panel.optimal_charging_pointing = Mock(return_value=(180.0, 0.0))
    solar_panel.panel_illumination_fraction = Mock(return_value=0.8)
    return solar_panel


@pytest.fixture
def mock_acs_config():
    """Create a mock ACS config."""
    return Mock()


@pytest.fixture
def mock_ephem():
    """Create a mock ephemeris."""
    ephem = Mock()
    ephem.index.return_value = 0
    return ephem


@pytest.fixture
def emergency_charging(mock_constraint, mock_solar_panel, mock_acs_config):
    """Create an EmergencyCharging instance."""
    return EmergencyCharging(
        constraint=mock_constraint,
        solar_panel=mock_solar_panel,
        acs_config=mock_acs_config,
        starting_obsid=999000,
    )


@pytest.fixture
def mock_config():
    """Create a mock config with required components."""
    config = Mock()
    config.constraint = Mock(spec=Constraint)
    config.constraint.inoccult = Mock(return_value=False)
    config.constraint.ephem = Mock()  # Add ephem for Pointing initialization
    # Add panel_constraint with solar_panel for EmergencyCharging initialization
    config.constraint.panel_constraint = Mock()
    config.constraint.panel_constraint.solar_panel = Mock(spec=SolarPanel)
    return config


@pytest.fixture
def queue_ditl(mock_config):
    """Create a QueueDITL instance with mocked dependencies."""

    def mock_ditl_init(self, config=None):
        """Mock DITLMixin.__init__ that sets config and calls _init_subsystems."""
        self.config = config
        self._init_subsystems()

    with patch(
        "conops.DITLMixin.__init__",
        side_effect=mock_ditl_init,
        autospec=False,
    ):
        ditl = QueueDITL(config=mock_config)

        # Mock ephemeris
        ditl.ephem = Mock()
        ditl.ephem.index.return_value = np.array([0])

        # Mock sun and earth for eclipse check
        mock_sun = Mock()
        mock_earth = Mock()
        mock_sun.separation.return_value = np.array([2.0])  # Not in eclipse

        # Create mock list-like objects for sun and earth
        sun_list = Mock()
        sun_list.__getitem__ = Mock(return_value=mock_sun)
        earth_list = Mock()
        earth_list.__getitem__ = Mock(return_value=mock_earth)

        ditl.ephem.sun = sun_list
        ditl.ephem.earth = earth_list
        ditl.ephem.earth_radius_angle = np.array([1.0])

        # Mock ACS
        ditl.acs = Mock()
        ditl.acs.solar_panel = Mock()
        ditl.acs.solar_panel.optimal_charging_pointing = Mock(return_value=(180.0, 0.0))

        # Initialize the tracking variables (already done in __init__ but ensure they exist)
        if not hasattr(ditl, "charging_ppt"):
            ditl.charging_ppt = None
        if not hasattr(ditl, "emergency_charging"):
            ditl.emergency_charging = Mock(spec=EmergencyCharging)
            ditl.emergency_charging.next_charging_obsid = 999000

        return ditl


@pytest.fixture
def mock_battery():
    """Create a mock battery."""
    battery = Mock(spec=Battery)
    battery.battery_alert = False
    battery.battery_level = 0.80
    battery.drain = Mock()
    battery.charge = Mock()
    return battery


@pytest.fixture
def batt_20wh():
    """Create a battery with 20 watthours capacity."""
    return Battery(amphour=2, voltage=10, watthour=20)


@pytest.fixture
def batt_1wh():
    """Create a battery with 1 watthour capacity."""
    return Battery(amphour=1, voltage=1, watthour=1)
