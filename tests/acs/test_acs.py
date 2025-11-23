"""Unit tests for Attitude Control System (ACS) class."""

from unittest.mock import Mock

import pytest

from conops.acs import ACS
from conops.common import ACSMode


class DummyEphemeris:
    """Minimal mock ephemeris for testing."""

    def __init__(self):
        self.step_size = 1.0
        self.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]

    def index(self, time):
        return 0


class TestACSInitialization:
    """Test ACS initialization and properties."""

    def test_acs_initialization(self, acs, mock_constraint):
        """Test that ACS initializes with correct defaults."""
        assert acs.constraint is mock_constraint
        assert acs.ra == 0.0
        assert acs.dec == 0.0
        assert acs.roll == 0.0
        assert acs.obstype == "PPT"
        assert acs.last_slew is None
        assert acs.last_ppt is None
        assert acs.acsmode == 0
        assert acs.currentpass is None
        assert acs.slew_dists == []

    def test_acs_requires_constraint(self, mock_config):
        """Test that ACS requires a constraint."""
        with pytest.raises(AssertionError, match="Constraint must be provided"):
            ACS(constraint=None, config=mock_config)

    def test_acs_requires_constraint_with_ephem(self, mock_config):
        """Test that ACS requires constraint with ephemeris."""
        constraint = Mock()
        constraint.ephem = None
        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            ACS(constraint=constraint, config=mock_config)


class TestACSAttributes:
    """Test ACS attribute access and manipulation."""

    def test_obstype_attribute(self, acs):
        """Test obstype attribute."""
        assert acs.obstype == "PPT"
        acs.obstype = "GSP"
        assert acs.obstype == "GSP"

    def test_constraint_attribute(self, acs, mock_constraint):
        """Test constraint attribute."""
        assert acs.constraint is mock_constraint

    def test_config_attribute(self, acs, mock_config):
        """Test config attribute."""
        assert acs.config is mock_config

    def test_solar_panel_attribute(self, acs):
        """Test solar_panel attribute exists."""
        assert acs.solar_panel is not None

    def test_ephem_attribute(self, acs):
        """Test ephem attribute exists."""
        assert acs.ephem is not None

    def test_last_slew_attribute(self, acs):
        """Test last_slew attribute."""
        assert acs.last_slew is None

    def test_last_ppt_attribute(self, acs):
        """Test last_ppt attribute."""
        assert acs.last_ppt is None

    def test_currentpass_attribute(self, acs):
        """Test currentpass attribute."""
        assert acs.currentpass is None


class TestACSStateManagement:
    """Test ACS state management."""

    def test_acsmode_initial(self, acs):
        """Test default acsmode is SCIENCE."""
        assert acs.acsmode == ACSMode.SCIENCE

    def test_acsmode_set_slewing(self, acs):
        """Test setting acsmode to SLEWING."""
        acs.acsmode = ACSMode.SLEWING
        assert acs.acsmode == ACSMode.SLEWING

    def test_acsmode_set_saa(self, acs):
        """Test setting acsmode to SAA."""
        acs.acsmode = ACSMode.SAA
        assert acs.acsmode == ACSMode.SAA

    def test_acsmode_set_pass(self, acs):
        """Test setting acsmode to PASS."""
        acs.acsmode = ACSMode.PASS
        assert acs.acsmode == ACSMode.PASS

    def test_acsmode_set_charging(self, acs):
        """Test setting acsmode to CHARGING."""
        acs.acsmode = ACSMode.CHARGING
        assert acs.acsmode == ACSMode.CHARGING

    def test_acsmode_return_to_science(self, acs):
        """Test returning acsmode to SCIENCE."""
        acs.acsmode = ACSMode.SCIENCE
        assert acs.acsmode == ACSMode.SCIENCE

    def test_ra_dec_updates(self, acs):
        """Test that RA/Dec can be updated."""
        assert acs.ra == 0.0
        assert acs.dec == 0.0
        acs.ra = 45.0
        acs.dec = 30.0
        assert acs.ra == 45.0
        assert acs.dec == 30.0

    def test_roll_updates(self, acs):
        """Test that roll can be updated."""
        assert acs.roll == 0.0
        acs.roll = 90.0
        assert acs.roll == 90.0

    def test_slew_dists_tracking(self, acs):
        """Test that slew_dists list is tracked."""
        assert acs.slew_dists == []
        acs.slew_dists.append(45.0)
        assert len(acs.slew_dists) == 1
        assert acs.slew_dists[0] == 45.0

    def test_rapid_state_changes(self, acs):
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

    def test_request_pass_basic(self, acs):
        """Test requesting a pass."""
        acs.passrequests.passes = []
        mock_pass = Mock()
        mock_pass.begin = 1000.0
        mock_pass.end = 2000.0
        mock_pass.obsid = 5

        acs.request_pass(mock_pass)
        assert mock_pass in acs.passrequests.passes

    def test_request_pass_multiple(self, acs):
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

    def test_request_pass_with_same_pass(self, acs):
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

    def test_full_state_cycle(self, acs):
        """Test cycling through all acsmode states."""
        for mode in [0, 1, 2, 3, 4]:
            acs.acsmode = mode
            assert acs.acsmode == mode
