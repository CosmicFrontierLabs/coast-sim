from conops.pointing import Pointing

# Assuming DummyConstraint is defined elsewhere or imported; if not, define it here or import it.
# For this example, I'll assume it's available.


class TestPointingInitialization:
    """Test Pointing initialization and default fields."""

    def test_exptime_is_none(self, pointing):
        assert pointing.exptime is None

    def test_inview_is_false(self, pointing):
        assert pointing.inview is False

    def test_done_is_false(self, pointing):
        assert pointing.done is False

    def test_obstype_is_at(self, pointing):
        assert pointing.obstype == "AT"

    def test_coordinated_is_none(self, pointing):
        assert pointing.coordinated is None

    def test_isat_is_false(self, pointing):
        assert pointing.isat is False


class TestPointingVisibility:
    """Test visibility-related methods on Pointing."""

    def test_is_visible(self, dummy_constraint, acs_config):
        tr = Pointing(constraint=dummy_constraint, acs_config=acs_config)
        # ra/dec values passed through to the constraint; methods just return the stubbed values
        tr.ra = 12.34
        tr.dec = -21.0
        assert tr.is_visible(12345)

    def test_in_sun(self, dummy_constraint, acs_config):
        tr = Pointing(constraint=dummy_constraint, acs_config=acs_config)
        # ra/dec values passed through to the constraint; methods just return the stubbed values
        tr.ra = 12.34
        tr.dec = -21.0
        assert tr.in_sun(12345)

    def test_in_earth(self, dummy_constraint, acs_config):
        tr = Pointing(constraint=dummy_constraint, acs_config=acs_config)
        # ra/dec values passed through to the constraint; methods just return the stubbed values
        tr.ra = 12.34
        tr.dec = -21.0
        assert tr.in_earth(12345)

    def test_in_moon_is_false(self, dummy_constraint, acs_config):
        tr = Pointing(constraint=dummy_constraint, acs_config=acs_config)
        # ra/dec values passed through to the constraint; methods just return the stubbed values
        tr.ra = 12.34
        tr.dec = -21.0
        assert tr.in_moon(12345) is False

    def test_in_panel(self, dummy_constraint, acs_config):
        tr = Pointing(constraint=dummy_constraint, acs_config=acs_config)
        # ra/dec values passed through to the constraint; methods just return the stubbed values
        tr.ra = 12.34
        tr.dec = -21.0
        assert tr.in_panel(12345)


class TestPointingNextVis:
    """Test next_vis and visible methods for visibility window calculation."""

    def test_visible_returns_current_time_if_visible(self, pointing):
        utime = 1000.0
        # Place a visibility window that includes utime
        pointing.windows = [(utime - 1.0, utime + 10.0)]
        # Confirm visible at utime (visible returns the window tuple, not True)
        assert pointing.visible(utime, utime) == (utime - 1.0, utime + 10.0)

    def test_next_vis_returns_current_time_if_visible(self, pointing):
        utime = 1000.0
        # Place a visibility window that includes utime
        pointing.windows = [(utime - 1.0, utime + 10.0)]
        assert pointing.next_vis(utime) == utime

    def test_next_vis_returns_false_when_no_windows(self, pointing):
        pointing.windows = []
        assert pointing.next_vis(10.0) is False

    def test_next_vis_returns_next_start_time_before_first_window(self, pointing):
        pointing.windows = [(5.0, 7.0), (15.0, 20.0)]
        # utime before first window start -> should return 5.0
        assert pointing.next_vis(0.0) == 5.0

    def test_next_vis_returns_next_start_time_between_windows(self, pointing):
        pointing.windows = [(5.0, 7.0), (15.0, 20.0)]
        # utime between windows starts -> should return 15.0
        assert pointing.next_vis(10.0) == 15.0


class TestPointingExposure:
    """Test exposure calculation with and without SAA."""

    # def test_exposure_calculation_without_saa(self):
    #     c = DummyConstraint(step_size=1)
    #     tr = Pointing(constraint=c, acs_config=DummyACSConfig())
    #     tr.begin = 0.0
    #     tr.end = 10.0
    #     tr.slewtime = 2.0
    #     tr.saa = None  # force saatime to 0 in else branch
    #     exposure = tr.exposure
    #     assert exposure == pytest.approx(8.0)

    # def test_exposure_calculation_with_saa(self):
    #     c = DummyConstraint(step_size=1)
    #     tr = Pointing(constraint=c, acs_config=DummyACSConfig())
    #     tr.begin = 0.0
    #     tr.end = 10.0
    #     tr.slewtime = 2.0
    #     # insaa returns 1 for all steps from 2..9 inclusive => 8 steps
    #     tr.saa = DummySAA(value=1)
    #     exposure = tr.exposure
    #     # end - begin - slewtime - saatime => 10 - 0 - 2 - 8 = 0
    #     assert exposure == pytest.approx(0.0)
    #     # saatime should be recorded
    #     assert tr.saatime == pytest.approx(8.0)


class TestPointingStringRepresentation:
    """Test string representation of Pointing."""

    def test_str_contains_target_name_and_id(self, pointing):
        pointing.begin = 0
        pointing.name = "TargetName"
        pointing.targetid = 42
        pointing.ra = 1.2345
        pointing.dec = -0.1234
        pointing.roll = 3.4
        pointing.merit = 7.0
        s = str(pointing)
        # Ensure human-readable components are present
        assert "TargetName (42)" in s

    def test_str_contains_ra(self, pointing):
        pointing.begin = 0
        pointing.name = "TargetName"
        pointing.targetid = 42
        pointing.ra = 1.2345
        pointing.dec = -0.1234
        pointing.roll = 3.4
        pointing.merit = 7.0
        s = str(pointing)
        # Ensure human-readable components are present
        assert "RA=1.2345" in s

    def test_str_contains_dec(self, pointing):
        pointing.begin = 0
        pointing.name = "TargetName"
        pointing.targetid = 42
        pointing.ra = 1.2345
        pointing.dec = -0.1234
        pointing.roll = 3.4
        pointing.merit = 7.0
        s = str(pointing)
        # Ensure human-readable components are present
        assert "Dec" in s or "Dec=" in s  # Accept either formatting presence

    def test_str_contains_merit(self, pointing):
        pointing.begin = 0
        pointing.name = "TargetName"
        pointing.targetid = 42
        pointing.ra = 1.2345
        pointing.dec = -0.1234
        pointing.roll = 3.4
        pointing.merit = 7.0
        s = str(pointing)
        # Ensure human-readable components are present
        assert "Merit=" in s


class TestPointing:
    """Test Pointing class initialization and properties."""

    def test_constraint(self, pointing, constraint):
        assert pointing.constraint == constraint

    def test_obsstart(self, pointing):
        assert pointing.obsstart == 0

    def test_inview_false(self, pointing):
        assert pointing.inview is False

    def test_obstype_at(self, pointing):
        assert pointing.obstype == "AT"

    def test_coordinated_none(self, pointing):
        assert pointing.coordinated is None

    def test_ra_zero(self, pointing):
        assert pointing.ra == 0.0

    def test_dec_zero(self, pointing):
        assert pointing.dec == 0.0

    def test_targetid_zero(self, pointing):
        assert pointing.targetid == 0

    def test_name_fake_target(self, pointing):
        assert pointing.name == "FakeTarget"

    def test_fom_100(self, pointing):
        assert pointing.fom == 100.0  # fom is maintained as legacy alias for merit

    def test_merit_100(self, pointing):
        assert pointing.merit == 100

    def test_exptime_none(self, pointing):
        assert pointing.exptime is None

    def test_saatime_zero(self, pointing):
        assert pointing.saatime == 0

    def test_exptime_setter_initializes_exporig(self, pointing):
        # First set initializes _exporig
        pointing.exptime = 500
        assert pointing.exptime == 500

    def test_exptime_setter_sets_exporig(self, pointing):
        # First set initializes _exporig
        pointing.exptime = 500
        assert pointing._exporig == 500

    def test_exptime_setter_second_set_changes_exptime(self, pointing):
        # First set initializes _exporig
        pointing.exptime = 500
        # Second set doesn't change _exporig
        pointing.exptime = 1000
        assert pointing.exptime == 1000

    def test_exptime_setter_second_set_does_not_change_exporig(self, pointing):
        # First set initializes _exporig
        pointing.exptime = 500
        # Second set doesn't change _exporig
        pointing.exptime = 1000
        assert pointing._exporig == 500

    def test_done_property_initially_false(self, pointing):
        # Set exptime first to avoid None comparison
        pointing.exptime = 100
        # Initially done is False
        assert pointing.done is False

    def test_done_property_setter_true(self, pointing):
        # Set exptime first to avoid None comparison
        pointing.exptime = 100
        # Set done to True
        pointing.done = True
        assert pointing.done is True

    def test_done_property_when_exptime_zero(self, pointing):
        pointing.exptime = 0
        # When exptime <= 0, done should be True
        assert pointing.done is True

    def test_reset_exptime_to_exporig(self, pointing):
        pointing.exptime = 500
        pointing.begin = 100.0
        pointing.end = 200.0
        pointing.slewtime = 10.0
        pointing.done = True
        # Reset
        pointing.reset()
        assert pointing.exptime == 500  # Should be reset to _exporig

    def test_reset_done_to_false(self, pointing):
        pointing.exptime = 500
        pointing.begin = 100.0
        pointing.end = 200.0
        pointing.slewtime = 10.0
        pointing.done = True
        # Reset
        pointing.reset()
        assert pointing.done is False

    def test_reset_begin_to_zero(self, pointing):
        pointing.exptime = 500
        pointing.begin = 100.0
        pointing.end = 200.0
        pointing.slewtime = 10.0
        pointing.done = True
        # Reset
        pointing.reset()
        assert pointing.begin == 0

    def test_reset_end_to_zero(self, pointing):
        pointing.exptime = 500
        pointing.begin = 100.0
        pointing.end = 200.0
        pointing.slewtime = 10.0
        pointing.done = True
        # Reset
        pointing.reset()
        assert pointing.end == 0

    def test_reset_slewtime_to_zero(self, pointing):
        pointing.exptime = 500
        pointing.begin = 100.0
        pointing.end = 200.0
        pointing.slewtime = 10.0
        pointing.done = True
        # Reset
        pointing.reset()
        assert pointing.slewtime == 0
