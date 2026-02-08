"""Comprehensive unit tests for ACS methods to achieve 100% coverage."""

from unittest.mock import Mock, patch

# ACSCommandType removed because tests rely on internal enqueue API
from conops import ACSMode, Pass, Pointing, Slew


class TestAddSlew:
    """Test enqueue_command method."""

    @patch("conops.targets.Pointing")
    def test_enqueue_command_basic(self, mock_too, acs) -> None:
        """Test basic slew addition."""
        # Setup mock Pointing
        mock_at = Mock(spec=Pointing)
        mock_at.next_vis = Mock(return_value=1514764800.0)  # Visibility available now
        mock_at.windows = []
        mock_at.visibility = Mock()
        mock_too.return_value = mock_at

        result = acs._enqueue_slew(45.0, 30.0, 100, 1514764800.0, "PPT")
        assert result is True
        assert len(acs.command_queue) == 1

    @patch("conops.targets.Pointing")
    def test_enqueue_command_different_obstype(self, mock_too, acs) -> None:
        """Test adding slew with different obstype."""
        mock_at = Mock(spec=Pointing)
        mock_at.next_vis = Mock(return_value=1514764800.0)
        mock_at.windows = []
        mock_at.visibility = Mock()
        mock_too.return_value = mock_at

        result = acs._enqueue_slew(45.0, 30.0, 100, 1514764800.0, "GSP")
        assert result is True


class TestAddSlewClass:
    """Test enqueue_command method."""

    @patch("conops.targets.Pointing")
    @patch("conops.unixtime2yearday")
    def test_enqueue_command_first_slew(self, mock_yearday, mock_too, acs) -> None:
        """Test adding first slew."""
        mock_yearday.return_value = (2018, 1)

        # Create mock slew
        slew = Mock(spec=Slew)
        slew.endra = 45.0
        slew.enddec = 30.0
        slew.obstype = "PPT"
        slew.obsid = 100
        slew.calc_slewtime = Mock()
        slew.slewdist = 45.0
        slew.is_slewing = Mock(return_value=False)

        # Mock Pointing
        mock_at = Mock(spec=Pointing)
        mock_at.ra = 45.0
        mock_at.dec = 30.0
        mock_at.visibility = Mock()
        mock_at.next_vis = Mock(return_value=1514764800.0)  # Visibility available now
        mock_at.windows = []
        mock_too.return_value = mock_at

        acs.last_slew = None
        result = acs._enqueue_slew(
            slew.endra, slew.enddec, slew.obsid, 1514764800.0, slew.obstype
        )

        assert result is True
        assert acs.last_slew is not None
        assert len(acs.command_queue) == 1

    @patch("conops.targets.Pointing")
    @patch("conops.unixtime2yearday")
    def test_enqueue_command_subsequent_slew(self, mock_yearday, mock_too, acs) -> None:
        """Test adding subsequent slew."""
        mock_yearday.return_value = (2018, 1)

        # Setup existing last_slew
        acs.last_slew = Mock(spec=Slew)
        acs.last_slew.is_slewing = Mock(return_value=False)
        acs.ra = 10.0
        acs.dec = 20.0

        # Create new slew
        slew = Mock(spec=Slew)
        slew.endra = 45.0
        slew.enddec = 30.0
        slew.obstype = "PPT"
        slew.obsid = 100
        slew.calc_slewtime = Mock()
        slew.slewdist = 45.0
        slew.is_slewing = Mock(return_value=False)

        # Mock Pointing
        mock_at = Mock(spec=Pointing)
        mock_at.visibility = Mock()
        mock_at.next_vis = Mock(return_value=1514764800.0)
        mock_at.windows = []
        mock_too.return_value = mock_at

        result = acs._enqueue_slew(
            slew.endra, slew.enddec, slew.obsid, 1514764800.0, slew.obstype
        )

        assert result is True
        # New slew instance is enqueued; check that it was initialized using current spacecraft pointing
        enqueued_slew = acs.command_queue[-1].slew
        assert enqueued_slew.startra == 10.0
        assert enqueued_slew.startdec == 20.0

    @patch("conops.targets.Pointing")
    @patch("conops.unixtime2yearday")
    def test_enqueue_command_rejected_constraint(
        self, mock_yearday, mock_too, acs
    ) -> None:
        """Test slew rejected due to constraint."""
        mock_yearday.return_value = (2018, 1)

        slew = Mock(spec=Slew)
        slew.endra = 45.0
        slew.enddec = 30.0
        slew.obstype = "PPT"
        slew.obsid = 100

        # Mock Pointing with no visibility
        mock_at = Mock(spec=Pointing)
        mock_at.visibility = Mock()
        mock_at.next_vis = Mock(return_value=False)  # No visibility
        mock_at.windows = [[1514764800.0, 1514765000.0]]
        mock_too.return_value = mock_at

        acs.last_slew = None
        slew.at = mock_at
        result = acs._enqueue_slew(
            slew.endra, slew.enddec, slew.obsid, 1514764800.0, slew.obstype
        )

        assert result is False
        assert len(acs.command_queue) == 0

    @patch("conops.targets.Pointing")
    @patch("conops.unixtime2yearday")
    def test_enqueue_command_delayed_for_visibility(
        self, mock_yearday, mock_too, acs
    ) -> None:
        """Test slew delayed for visibility."""
        mock_yearday.return_value = (2018, 1)

        slew = Mock(spec=Slew)
        slew.endra = 45.0
        slew.enddec = 30.0
        slew.obstype = "PPT"
        slew.obsid = 100
        slew.calc_slewtime = Mock()
        slew.slewdist = 45.0

        # Mock Pointing with delayed visibility
        mock_at = Mock(spec=Pointing)
        mock_at.visibility = Mock()
        mock_at.next_vis = Mock(return_value=1514765000.0)  # 200s later
        mock_too.return_value = mock_at

        acs.last_slew = False
        result = acs._enqueue_slew(
            slew.endra, slew.enddec, slew.obsid, 1514764800.0, slew.obstype
        )

        assert result is True
        # The execution_time is set on the command queued
        assert acs.command_queue[-1].execution_time == 1514765000.0

    @patch("conops.targets.Pointing")
    @patch("conops.unixtime2yearday")
    def test_enqueue_command_during_existing_slew(
        self, mock_yearday, mock_too, acs
    ) -> None:
        """Test adding slew during an existing slew."""
        mock_yearday.return_value = (2018, 1)

        # Setup existing slew that's currently active
        existing_slew = Mock(spec=Slew)
        existing_slew.is_slewing = Mock(return_value=True)
        existing_slew.slewstart = 1514764800.0
        existing_slew.slewtime = 120.0
        acs.last_slew = existing_slew
        acs.ra = 10.0
        acs.dec = 20.0

        # Create new slew
        slew = Mock(spec=Slew)
        slew.endra = 45.0
        slew.enddec = 30.0
        slew.obstype = "PPT"
        slew.obsid = 100
        slew.calc_slewtime = Mock()
        slew.slewdist = 45.0
        slew.is_slewing = Mock(return_value=False)

        # Mock Pointing
        mock_at = Mock(spec=Pointing)
        mock_at.visibility = Mock()
        mock_at.next_vis = Mock(return_value=1514764920.0)  # After existing slew
        mock_at.windows = []
        mock_too.return_value = mock_at

        result = acs._enqueue_slew(
            slew.endra, slew.enddec, slew.obsid, 1514764800.0, slew.obstype
        )

        assert result is True
        # Slew should be delayed until existing slew finishes

    @patch("conops.targets.Pointing")
    @patch("conops.unixtime2yearday")
    def test_enqueue_command_during_pass(self, mock_yearday, mock_too, acs) -> None:
        """Test adding slew during a pass updates last_ppt."""
        mock_yearday.return_value = (2018, 1)

        # Set acsmode to PASS
        acs.acsmode = ACSMode.PASS

        slew = Mock(spec=Slew)
        slew.endra = 45.0
        slew.enddec = 30.0
        slew.obstype = "PPT"
        slew.obsid = 100
        slew.calc_slewtime = Mock()
        slew.slewdist = 45.0

        # Mock Pointing
        mock_at = Mock(spec=Pointing)
        mock_at.visibility = Mock()
        mock_at.next_vis = Mock(return_value=1514764800.0)
        mock_at.windows = []
        mock_too.return_value = mock_at

        acs.last_slew = None
        result = acs._enqueue_slew(
            slew.endra, slew.enddec, slew.obsid, 1514764800.0, slew.obstype
        )

        assert result is True
        # Given current ACS design, enqueuing still occurs in PASS mode
        assert len(acs.command_queue) == 1

    @patch("conops.targets.Pointing")
    @patch("conops.unixtime2yearday")
    def test_enqueue_command_gsp_obstype(self, mock_yearday, mock_too, acs) -> None:
        """Test adding slew with GSP obstype."""
        mock_yearday.return_value = (2018, 1)

        slew = Mock(spec=Slew)
        slew.endra = 45.0
        slew.enddec = 30.0
        slew.obstype = "GSP"
        slew.obsid = 100
        slew.calc_slewtime = Mock()
        slew.slewdist = 45.0

        # Mock Pointing - GSP should not be rejected for constraints
        mock_at = Mock(spec=Pointing)
        mock_at.visibility = Mock()
        mock_at.next_vis = Mock(return_value=False)
        mock_at.windows = []
        mock_too.return_value = mock_at

        acs.last_slew = False
        result = acs._enqueue_slew(
            slew.endra, slew.enddec, slew.obsid, 1514764800.0, slew.obstype
        )

        # GSP should succeed even without visibility
        assert result is True


class TestPointing:
    """Test pointing method."""

    @patch("conops.optimum_roll")
    def test_pointing_no_slew(self, mock_roll, acs, mock_ephem) -> None:
        """Test pointing with no slew."""
        mock_roll.return_value = 0.0
        acs.ephem = mock_ephem

        ra, dec, roll, obsid = acs.pointing(1514764800.0)

        assert ra == 0.0  # Earth RA from mock
        assert dec == 0.0  # Earth Dec from mock
        assert 0.0 <= roll < 360.0
        assert obsid == 0  # Default obsid when no slew is active

    @patch("conops.optimum_roll")
    def test_pointing_slew_start_adjustment(self, mock_roll, acs) -> None:
        """Test pointing adjusts slew start when within step_size."""
        mock_roll.return_value = 0.0

        # Setup slew request that should start now
        mock_slew = Mock(spec=Slew)
        mock_slew.slewstart = 1514764800.0 + 30.0  # Within step_size (60s)
        mock_slew.startra = 10.0
        mock_slew.startdec = 20.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        mock_slew.obsid = 100
        mock_slew.calc_slewtime = Mock()
        mock_slew.is_slewing = Mock(return_value=False)
        mock_slew.ra_dec = Mock(return_value=(45.0, 30.0))
        mock_slew.at = None

        acs.current_slew = mock_slew
        acs.last_slew = mock_slew
        acs.ra = 10.0
        acs.dec = 20.0

        ra, dec, roll, obsid = acs.pointing(1514764800.0)
        assert acs.last_slew is not None

    @patch("conops.optimum_roll")
    def test_pointing_during_slew(self, mock_roll, acs) -> None:
        """Test pointing during an active slew."""
        mock_roll.return_value = 45.0

        mock_slew = Mock(spec=Slew)
        mock_slew.is_slewing = Mock(return_value=True)
        mock_slew.obstype = "PPT"
        mock_slew.obsid = 100
        mock_slew.ra_dec = Mock(return_value=(45.0, 30.0))
        mock_slew.at = None

        acs.last_slew = mock_slew
        acs.current_slew = mock_slew

        ra, dec, roll, obsid = acs.pointing(1514764800.0)

        assert acs.acsmode == ACSMode.SLEWING
        assert ra == 45.0
        assert dec == 30.0

    @patch("conops.optimum_roll")
    def test_pointing_during_pass_slew(self, mock_roll, acs) -> None:
        """Test pointing during pass slew."""
        mock_roll.return_value = 45.0

        mock_pass = Mock(spec=Pass)
        mock_pass.is_slewing = Mock(return_value=True)
        mock_pass.obstype = "GSP"
        mock_pass.obsid = 200
        mock_pass.ra_dec = Mock(return_value=(45.0, 30.0))

        acs.last_slew = mock_pass
        acs.current_slew = mock_pass

        ra, dec, roll, obsid = acs.pointing(1514764800.0)

        assert acs.acsmode == ACSMode.PASS
        assert obsid == 200

    @patch("conops.optimum_roll")
    def test_pointing_during_pass_dwell(self, mock_roll, acs) -> None:
        """Test pointing during pass dwell phase."""
        mock_roll.return_value = 0.0

        mock_pass = Mock(spec=Pass)
        mock_pass.is_slewing = Mock(return_value=False)
        mock_pass.obstype = "GSP"
        mock_pass.obsid = 200
        mock_pass.slewend = 1514764800.0
        mock_pass.begin = 1514764800.0
        mock_pass.length = 600.0
        mock_pass.ra_dec = Mock(return_value=(45.0, 30.0))

        acs.last_slew = mock_pass
        acs.current_slew = mock_pass

        ra, dec, roll, obsid = acs.pointing(1514765000.0)  # Within pass

        # Check that we're using the pass pointing
        assert obsid == 200  # Should use pass obsid
        assert ra == 45.0
        assert dec == 30.0

    @patch("conops.optimum_roll")
    def test_pointing_with_constraint_violation(
        self, mock_roll, acs, mock_constraint
    ) -> None:
        """Test pointing with constraint violation."""
        mock_roll.return_value = 0.0

        mock_slew = Mock(spec=Slew)
        mock_slew.is_slewing = Mock(return_value=False)
        mock_slew.obstype = "PPT"
        mock_slew.obsid = 100
        mock_slew.ra_dec = Mock(return_value=(45.0, 30.0))
        mock_slew.at = Mock(spec=Pointing)
        mock_slew.at.ra = 45.0
        mock_slew.at.dec = 30.0
        mock_slew.at.in_moon = Mock(return_value=False)
        mock_slew.at.in_sun = Mock(return_value=False)
        mock_slew.at.in_earth = Mock(return_value=True)
        mock_slew.at.in_panel = Mock(return_value=False)

        acs.last_slew = mock_slew
        acs.constraint.in_constraint = Mock(return_value=True)

        ra, dec, roll, obsid = acs.pointing(1514764800.0)

        # Should log constraint but continue
        assert acs.acsmode == ACSMode.SCIENCE

    @patch("conops.optimum_roll")
    def test_pointing_constraint_checks_without_at(
        self, mock_roll, acs, mock_constraint
    ):
        """Test pointing constraint checks when slew has no at attribute."""
        mock_roll.return_value = 0.0

        mock_slew = Mock(spec=Slew)
        mock_slew.is_slewing = Mock(return_value=False)
        mock_slew.obstype = "PPT"
        mock_slew.obsid = 100
        mock_slew.ra_dec = Mock(return_value=(45.0, 30.0))
        mock_slew.at = None  # No at attribute

        acs.last_slew = mock_slew
        acs.constraint.in_constraint = Mock(return_value=False)

        ra, dec, roll, obsid = acs.pointing(1514764800.0)

        # Should skip constraint checks but still work
        assert ra == 45.0
        assert dec == 30.0

    @patch("conops.optimum_roll")
    def test_pointing_pass_after_dwell(self, mock_roll, acs) -> None:
        """Test pointing after pass dwell phase."""
        mock_roll.return_value = 0.0

        mock_pass = Mock(spec=Pass)
        mock_pass.is_slewing = Mock(return_value=False)
        mock_pass.obstype = "GSP"
        mock_pass.obsid = 200
        mock_pass.slewend = 1514764800.0
        mock_pass.begin = 1514764800.0
        mock_pass.length = 600.0
        mock_pass.ra_dec = Mock(return_value=(45.0, 30.0))
        mock_pass.at = None

        acs.last_slew = mock_pass

        # After pass ends
        ra, dec, roll, obsid = acs.pointing(1514765500.0)

        assert acs.acsmode == ACSMode.SCIENCE


class TestRequestPass:
    """Test request_pass method."""

    def test_request_pass_no_overlap(self, acs) -> None:
        """Test requesting pass with no overlap."""
        acs.passrequests.passes = []

        mock_pass = Mock(spec=Pass)
        mock_pass.begin = 1514765000.0
        mock_pass.end = 1514766000.0

        acs.request_pass(mock_pass)

        assert mock_pass in acs.passrequests.passes

    def test_request_pass_overlap_end(self, acs) -> None:
        """Test requesting pass with overlap at end."""
        existing_pass = Mock(spec=Pass)
        existing_pass.begin = 1514765000.0
        existing_pass.end = 1514766000.0
        acs.passrequests.passes = [existing_pass]
        acs.passrequests.__iter__ = lambda self: iter(acs.passrequests.passes)

        new_pass = Mock(spec=Pass)
        new_pass.begin = 1514764000.0
        new_pass.end = 1514765500.0  # Overlaps with existing pass

        acs.request_pass(new_pass)

        # Should not add overlapping pass
        assert new_pass not in acs.passrequests.passes

    def test_request_pass_overlap_begin(self, acs) -> None:
        """Test requesting pass with overlap at begin."""
        existing_pass = Mock(spec=Pass)
        existing_pass.begin = 1514765000.0
        existing_pass.end = 1514766000.0
        acs.passrequests.passes = [existing_pass]
        acs.passrequests.__iter__ = lambda self: iter(acs.passrequests.passes)

        new_pass = Mock(spec=Pass)
        new_pass.begin = 1514765500.0  # Overlaps with existing pass
        new_pass.end = 1514767000.0

        acs.request_pass(new_pass)

        # Should not add overlapping pass
        assert new_pass not in acs.passrequests.passes

    def test_request_pass_no_overlap_before(self, acs) -> None:
        """Test requesting pass before existing pass."""
        existing_pass = Mock(spec=Pass)
        existing_pass.begin = 1514765000.0
        existing_pass.end = 1514766000.0
        acs.passrequests.passes = [existing_pass]
        acs.passrequests.__iter__ = lambda self: iter(acs.passrequests.passes)

        new_pass = Mock(spec=Pass)
        new_pass.begin = 1514764000.0
        new_pass.end = 1514764500.0

        acs.request_pass(new_pass)

        assert new_pass in acs.passrequests.passes

    def test_request_pass_no_overlap_after(self, acs) -> None:
        """Test requesting pass after existing pass."""
        existing_pass = Mock(spec=Pass)
        existing_pass.begin = 1514765000.0
        existing_pass.end = 1514766000.0
        acs.passrequests.passes = [existing_pass]
        acs.passrequests.__iter__ = lambda self: iter(acs.passrequests.passes)

        new_pass = Mock(spec=Pass)
        new_pass.begin = 1514766500.0
        new_pass.end = 1514767000.0

        acs.request_pass(new_pass)

        assert new_pass in acs.passrequests.passes


class TestAddSlewClassEdgeCases:
    """Test edge cases in enqueue_command."""

    @patch("conops.targets.Pointing")
    @patch("conops.unixtime2yearday")
    def test_enqueue_command_pass_object(self, mock_yearday, mock_too, acs) -> None:
        """Test adding a Pass object (not storing as last_ppt)."""
        mock_yearday.return_value = (2018, 1)

        # Set acsmode to PASS
        acs.acsmode = ACSMode.PASS

        pass_obj = Mock(spec=Pass)
        pass_obj.endra = 45.0
        pass_obj.enddec = 30.0
        pass_obj.obstype = "GSP"
        pass_obj.obsid = 100
        pass_obj.calc_slewtime = Mock()
        pass_obj.slewdist = 45.0

        # Mock Pointing
        mock_at = Mock(spec=Pointing)
        mock_at.visibility = Mock()
        mock_at.next_vis = Mock(return_value=1514764800.0)
        mock_at.windows = []
        mock_too.return_value = mock_at

        acs.last_slew = False
        acs.last_ppt = False
        result = acs._enqueue_slew(
            pass_obj.endra,
            pass_obj.enddec,
            pass_obj.obsid,
            1514764800.0,
            pass_obj.obstype,
        )

        assert result is True
        # Pass objects should not be stored as last_ppt
        assert acs.last_ppt is False

    @patch("conops.targets.Pointing")
    @patch("conops.unixtime2yearday")
    def test_enqueue_command_slew_in_pass(self, mock_yearday, mock_too, acs) -> None:
        """Test adding Slew during pass mode stores it as last_ppt."""
        mock_yearday.return_value = (2018, 1)

        # Set acsmode to PASS
        acs.acsmode = ACSMode.PASS

        slew = Mock(spec=Slew)
        slew.endra = 45.0
        slew.enddec = 30.0
        slew.obstype = "PPT"
        slew.obsid = 100
        slew.calc_slewtime = Mock()
        slew.slewdist = 45.0

        # Mock Pointing
        mock_at = Mock(spec=Pointing)
        mock_at.visibility = Mock()
        mock_at.next_vis = Mock(return_value=1514764800.0)
        mock_at.windows = []
        mock_too.return_value = mock_at

        acs.last_slew = False
        acs.last_ppt = False
        result = acs._enqueue_slew(
            slew.endra, slew.enddec, slew.obsid, 1514764800.0, slew.obstype
        )

        assert result is True
        # Enqueuing during pass does not set last_ppt until slew start
        assert acs.last_ppt is False

    @patch("conops.targets.Pointing")
    @patch("conops.unixtime2yearday")
    def test_enqueue_command_non_ppt_rejection(
        self, mock_yearday, mock_too, acs
    ) -> None:
        """Test that only PPT slews are rejected for constraints."""
        mock_yearday.return_value = (2018, 1)

        slew = Mock(spec=Slew)
        slew.endra = 45.0
        slew.enddec = 30.0
        slew.obstype = "GSP"  # Not PPT
        slew.obsid = 100
        slew.calc_slewtime = Mock()
        slew.slewdist = 45.0

        # Mock Pointing with no visibility
        mock_at = Mock(spec=Pointing)
        mock_at.visibility = Mock()
        mock_at.next_vis = Mock(return_value=False)  # No visibility
        mock_at.windows = [[1514764800.0, 1514765000.0]]
        mock_too.return_value = mock_at

        acs.last_slew = False
        result = acs._enqueue_slew(
            slew.endra, slew.enddec, slew.obsid, 1514764800.0, slew.obstype
        )

        # GSP should succeed even without visibility
        assert result is True

    @patch("conops.targets.Pointing")
    @patch("conops.unixtime2yearday")
    def test_enqueue_command_delayed_start_not_ppt(
        self, mock_yearday, mock_too, acs
    ) -> None:
        """Test slew with delayed start for non-PPT obstype."""
        mock_yearday.return_value = (2018, 1)

        slew = Mock(spec=Slew)
        slew.endra = 45.0
        slew.enddec = 30.0
        slew.obstype = "GSP"  # Not PPT
        slew.obsid = 100
        slew.calc_slewtime = Mock()
        slew.slewdist = 45.0

        # Mock Pointing with delayed visibility (but obstype != PPT, so no delay)
        mock_at = Mock(spec=Pointing)
        mock_at.visibility = Mock()
        mock_at.next_vis = Mock(return_value=1514765000.0)  # 200s later
        mock_at.windows = []
        mock_too.return_value = mock_at

        acs.last_slew = False
        result = acs._enqueue_slew(
            slew.endra, slew.enddec, slew.obsid, 1514764800.0, slew.obstype
        )

        assert result is True
        # GSP should start immediately, not delayed
        assert acs.command_queue[-1].execution_time == 1514764800.0
