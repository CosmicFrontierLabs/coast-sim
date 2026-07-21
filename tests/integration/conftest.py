"""Test fixtures for integration tests."""

from collections.abc import Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest
import rust_ephem
from astropy.time import Time  # type: ignore[import-untyped]

from conops import (
    AttitudeConstraintScope,
    Constraint,
    MissionConfig,
    SolarPanel,
    SolarPanelSet,
)


@pytest.fixture(autouse=True)
def patch_eclipse_constraint() -> Generator[None, None, None]:
    """Patch eclipse constraint to avoid ephemeris lookup errors in all integration tests."""
    mock_constraint = Mock()
    mock_constraint.in_constraint = Mock(return_value=False)
    with patch.object(SolarPanel, "_eclipse_constraint", mock_constraint):
        yield


@pytest.fixture
def mock_ephem_with_pv(sun_ra: float = 90.0, sun_dec: float = 0.0) -> Mock:
    """Create a comprehensive mock ephemeris with PV (position-velocity) data.

    This fixture provides a more complete ephemeris mock with sun/earth position vectors
    and all necessary attributes for solar panel and coordinate calculations.

    Args:
        sun_ra: Sun RA in degrees (default 90°)
        sun_dec: Sun Dec in degrees (default 0°)
    """
    ephem = Mock()
    ephem.__class__ = rust_ephem.Ephemeris

    # Time data - required by eclipse constraint
    start_time = 1514764800.0
    times = np.array([Time(start_time + i * 60, format="unix") for i in range(10)])
    ephem.timestamp = times

    # Sun position arrays
    ephem.sun_ra_deg = np.array([sun_ra] * 10)
    ephem.sun_dec_deg = np.array([sun_dec] * 10)
    ephem.earth_ra_deg = np.array([0.0] * 10)
    ephem.earth_dec_deg = np.array([0.0] * 10)
    ephem.sun = [Mock(ra=Mock(deg=sun_ra), dec=Mock(deg=sun_dec))]
    ephem.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
    ephem.moon_ra_deg = [90.0]
    ephem.moon_dec_deg = [10.0]

    # Position-velocity vectors for coordinate transformations
    ephem.sun_pv = Mock(position=np.array([[1.5e8, 0.0, 0.0]] * 10))
    ephem.gcrs_pv = Mock(position=np.array([[0.0, 0.0, 6378.0]] * 10))

    # Index method for ephemeris lookups
    def mock_index(time_obj: object) -> int:
        if hasattr(time_obj, "unix"):
            for idx, t in enumerate(times):
                if abs(t.unix - time_obj.unix) < 30:
                    return idx
        return 0

    ephem.index = mock_index

    return ephem


@pytest.fixture
def test_config_with_panels(
    mock_ephem_with_pv: Mock, mock_spacecraft_bus: Mock
) -> tuple[Mock, SolarPanelSet]:
    """Create a test configuration with solar panels and mock ephemeris.

    Args:
        mock_ephem_with_pv: Fixture providing mock ephemeris with PV data

    Returns:
        Tuple of (config Mock, SolarPanelSet)
    """
    # Create constraint
    constraint = Mock()
    constraint.__class__ = Constraint
    constraint.ephem = mock_ephem_with_pv
    constraint.constraint = None  # no combined rust-ephem constraint in tests
    constraint.roll_dependent_constraint = None
    constraint.panel_constraint = Mock()
    constraint.in_constraint = Mock(return_value=False)
    constraint.in_star_tracker_hard = Mock(return_value=False)
    constraint.in_radiator_hard = Mock(return_value=False)
    constraint.in_telescope_hard = Mock(return_value=False)
    constraint.in_eclipse = Mock(return_value=False)

    # Create solar panels
    panel_set = SolarPanelSet(
        conversion_efficiency=0.95,
        panels=[
            SolarPanel(name="P1", normal=(1.0, 0.0, 0.0), max_power=500.0),
            SolarPanel(name="P2", normal=(0.0, 1.0, 0.0), max_power=500.0),
        ],
    )
    constraint.panel_constraint.solar_panel = panel_set

    # Create config
    config = Mock()
    # Satisfies isinstance checks (e.g. Slew's pydantic "config" field) without
    # the attribute-restriction that Mock(spec=...) would impose.
    config.__class__ = MissionConfig
    # MissionConfig's init_fault_management_defaults model_validator re-runs
    # whenever this config is embedded as a nested pydantic field elsewhere
    # (e.g. on Slew.config) and iterates fault_management.thresholds, while
    # ACS calls fault_management.events.append(...) directly — so
    # fault_management must be a populated mock, not None.
    config.fault_management = Mock()
    config.fault_management.check = Mock()
    config.fault_management.safe_mode_requested = False
    config.fault_management.events = []
    config.fault_management.thresholds = []
    config.constraint = constraint
    config.ground_stations = Mock()
    config.solar_panel = panel_set
    config.spacecraft_bus = mock_spacecraft_bus
    config.battery = Mock()
    config.battery.max_depth_of_discharge = 0.3
    config.recorder = Mock()
    config.recorder.yellow_threshold = 0.7
    config.recorder.red_threshold = 0.9
    config.attitude_constraint_scopes_for_mode = Mock(
        return_value=[
            AttitudeConstraintScope.HARDWARE_SAFETY,
            AttitudeConstraintScope.IMAGING_QUALITY,
            AttitudeConstraintScope.POWER_GENERATION,
            AttitudeConstraintScope.GROUND_CONTACT,
        ]
    )
    config.spacecraft_bus.boresight_axis = "+X"
    config.boresight_axis = "+X"

    return config, panel_set


@pytest.fixture
def mock_ephem_sun_at_pole() -> Mock:
    """Create mock ephemeris with sun at high declination (north pole)."""
    ephem = Mock()

    start_time = 1514764800.0
    times = np.array([Time(start_time + i * 60, format="unix") for i in range(10)])
    ephem.timestamp = times

    ephem.sun_ra_deg = np.array([0.0] * 10)
    ephem.sun_dec_deg = np.array([89.0] * 10)
    ephem.earth_ra_deg = np.array([0.0] * 10)
    ephem.earth_dec_deg = np.array([0.0] * 10)
    ephem.sun = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=89.0))]
    ephem.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
    ephem.moon_ra_deg = [90.0]
    ephem.moon_dec_deg = [10.0]
    ephem.sun_pv = Mock(position=np.array([[1.5e8, 0.0, 0.0]] * 10))
    ephem.gcrs_pv = Mock(position=np.array([[0.0, 0.0, 6378.0]] * 10))

    def mock_index(time_obj: object) -> int:
        if hasattr(time_obj, "unix"):
            for idx, t in enumerate(times):
                if abs(t.unix - time_obj.unix) < 30:
                    return idx
        return 0

    ephem.index = mock_index
    return ephem


@pytest.fixture
def mock_ephem_sun_behind() -> Mock:
    """Create mock ephemeris with sun behind spacecraft (180° RA)."""
    ephem = Mock()

    start_time = 1514764800.0
    times = np.array([Time(start_time + i * 60, format="unix") for i in range(10)])
    ephem.timestamp = times

    ephem.sun_ra_deg = np.array([180.0] * 10)
    ephem.sun_dec_deg = np.array([0.0] * 10)
    ephem.earth_ra_deg = np.array([0.0] * 10)
    ephem.earth_dec_deg = np.array([0.0] * 10)
    ephem.sun = [Mock(ra=Mock(deg=180.0), dec=Mock(deg=0.0))]
    ephem.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
    ephem.moon_ra_deg = [90.0]
    ephem.moon_dec_deg = [10.0]
    ephem.sun_pv = Mock(position=np.array([[1.5e8, 0.0, 0.0]] * 10))
    ephem.gcrs_pv = Mock(position=np.array([[0.0, 0.0, 6378.0]] * 10))

    def mock_index(time_obj: object) -> int:
        if hasattr(time_obj, "unix"):
            for idx, t in enumerate(times):
                if abs(t.unix - time_obj.unix) < 30:
                    return idx
        return 0

    ephem.index = mock_index
    return ephem
