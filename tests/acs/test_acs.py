"""Unit tests for Attitude Control System (ACS) class."""

from unittest.mock import Mock

import numpy as np
import pytest

from conops import ACS, ACSMode


class DummyEphemeris:
    """Minimal mock ephemeris for testing."""

    def __init__(self):
        self.step_size = 1.0
        self.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
        self.sun = [Mock(ra=Mock(deg=45.0), dec=Mock(deg=23.5))]
        # New direct array access (rust-ephem 0.3.0+)
        self.earth_ra_deg = [0.0]
        self.earth_dec_deg = [0.0]
        self.sun_ra_deg = [45.0]
        self.sun_dec_deg = [23.5]
        self.moon_ra_deg = [90.0]
        self.moon_dec_deg = [10.0]
        self.sun_pv = Mock(position=np.array([[1.5e8, 0.0, 0.0]]))
        self.gcrs_pv = Mock(position=np.array([[0.0, 0.0, 6378.0]]))

    def index(self, time):
        return 0


class TestACSInitialization:
    """Test ACS initialization and properties."""

    def test_acs_initialization(self, acs, mock_constraint) -> None:
        """Test that ACS initializes with correct defaults."""
        assert acs.constraint is mock_constraint
        assert acs.ra == 180.0  # Earth-opposite nadir pointing
        assert acs.dec == 0.0
        assert acs.roll == 0.0
        assert acs.obstype == "PPT"
        assert acs.last_slew is not None  # Initialized with boundary condition slew
        assert acs.last_ppt is None
        assert acs.acsmode == 0
        assert acs.current_pass is None
        assert acs.slew_dists == []

    def test_acs_requires_constraint(self, mock_config) -> None:
        """Test that ACS requires a constraint."""
        mock_config.constraint = None
        with pytest.raises(AssertionError, match="Constraint must be provided"):
            ACS(config=mock_config)

    def test_acs_requires_constraint_with_ephem(self, mock_config) -> None:
        """Test that ACS requires constraint with ephemeris."""
        mock_config.constraint.ephem = None
        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            ACS(config=mock_config)


class TestACSAttributes:
    """Test ACS attribute access and manipulation."""

    def test_obstype_attribute(self, acs) -> None:
        """Test obstype attribute."""
        assert acs.obstype == "PPT"
        acs.obstype = "GSP"
        assert acs.obstype == "GSP"

    def test_constraint_attribute(self, acs, mock_constraint) -> None:
        """Test constraint attribute."""
        assert acs.constraint is mock_constraint

    def test_config_attribute(self, acs, mock_config) -> None:
        """Test config attribute."""
        assert acs.config is mock_config

    def test_solar_panel_attribute(self, acs) -> None:
        """Test solar_panel attribute exists."""
        assert acs.solar_panel is not None

    def test_ephem_attribute(self, acs) -> None:
        """Test ephem attribute exists."""
        assert acs.ephem is not None

    def test_last_slew_attribute(self, acs) -> None:
        """Test last_slew attribute."""
        assert acs.last_slew is not None  # Initialized with boundary condition slew

    def test_last_ppt_attribute(self, acs) -> None:
        """Test last_ppt attribute."""
        assert acs.last_ppt is None

    def test_current_pass_attribute(self, acs) -> None:
        """Test current_pass attribute."""
        assert acs.current_pass is None


class TestACSStateManagement:
    """Test ACS state management."""

    def test_acsmode_initial(self, acs) -> None:
        """Test default acsmode is SCIENCE."""
        assert acs.acsmode == ACSMode.SCIENCE

    def test_acsmode_set_slewing(self, acs) -> None:
        """Test setting acsmode to SLEWING."""
        acs.acsmode = ACSMode.SLEWING
        assert acs.acsmode == ACSMode.SLEWING

    def test_acsmode_set_saa(self, acs) -> None:
        """Test setting acsmode to SAA."""
        acs.acsmode = ACSMode.SAA
        assert acs.acsmode == ACSMode.SAA

    def test_acsmode_set_pass(self, acs) -> None:
        """Test setting acsmode to PASS."""
        acs.acsmode = ACSMode.PASS
        assert acs.acsmode == ACSMode.PASS

    def test_acsmode_set_charging(self, acs) -> None:
        """Test setting acsmode to CHARGING."""
        acs.acsmode = ACSMode.CHARGING
        assert acs.acsmode == ACSMode.CHARGING

    def test_acsmode_return_to_science(self, acs) -> None:
        """Test returning acsmode to SCIENCE."""
        acs.acsmode = ACSMode.SCIENCE
        assert acs.acsmode == ACSMode.SCIENCE

    def test_ra_dec_updates(self, acs) -> None:
        """Test that RA/Dec can be updated."""
        assert acs.ra == 180.0  # Earth-opposite nadir pointing
        assert acs.dec == 0.0
        acs.ra = 45.0
        acs.dec = 30.0
        assert acs.ra == 45.0
        assert acs.dec == 30.0

    def test_roll_updates(self, acs) -> None:
        """Test that roll can be updated."""
        assert acs.roll == 0.0
        acs.roll = 90.0
        assert acs.roll == 90.0

    def test_slew_dists_tracking(self, acs) -> None:
        """Test that slew_dists list is tracked."""
        assert acs.slew_dists == []
        acs.slew_dists.append(45.0)
        assert len(acs.slew_dists) == 1
        assert acs.slew_dists[0] == 45.0

    def test_rapid_state_changes(self, acs) -> None:
        """Test rapid state changes."""
        for _ in range(10):
            for mode in [
                ACSMode.SCIENCE,
                ACSMode.SLEWING,
                ACSMode.SAA,
                ACSMode.PASS,
                ACSMode.CHARGING,
            ]:
                acs.acsmode = mode
                assert acs.acsmode == mode


class TestACSPassRequestManagement:
    """Test pass request management."""

    def test_request_pass_basic(self, acs) -> None:
        """Test requesting a pass."""
        acs.passrequests.passes = []
        mock_pass = Mock()
        mock_pass.begin = 1000.0
        mock_pass.end = 2000.0
        mock_pass.obsid = 5

        acs.request_pass(mock_pass)
        assert mock_pass in acs.passrequests.passes

    def test_request_pass_multiple(self, acs) -> None:
        """Test requesting multiple non-overlapping passes."""
        acs.passrequests.passes = []
        mock_pass1 = Mock()
        mock_pass1.begin = 1000.0
        mock_pass1.end = 2000.0

        mock_pass2 = Mock()
        mock_pass2.begin = 3000.0
        mock_pass2.end = 4000.0

        acs.request_pass(mock_pass1)
        acs.request_pass(mock_pass2)
        assert len(acs.passrequests.passes) >= 2

    def test_request_pass_with_same_pass(self, acs) -> None:
        """Test that non-overlapping passes can be added to list."""
        acs.passrequests.passes = []
        passes_to_add = []
        for i in range(5):
            mock_pass = Mock()
            # Create non-overlapping passes: each starts after the previous one ends
            mock_pass.begin = 1000.0 + (i * 1500)  # Space passes 1500s apart
            mock_pass.end = 2000.0 + (i * 1500)  # Each pass is 1000s long
            passes_to_add.append(mock_pass)
            acs.request_pass(mock_pass)

        assert len(acs.passrequests.passes) == len(passes_to_add)


class TestACSStateTransitions:
    """Test state transitions during operations."""

    def test_full_state_cycle(self, acs) -> None:
        """Test cycling through all acsmode states."""
        for mode in [0, 1, 2, 3, 4]:
            acs.acsmode = mode
            assert acs.acsmode == mode
