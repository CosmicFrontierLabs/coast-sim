import numpy as np
import pytest
from pydantic import ValidationError

from conops import PlanEntry


class MockSAA:
    """Mock SAA for testing."""

    def __init__(self, intervals=None):
        if intervals is None:
            self.saatimes = np.array([[150, 160], [250, 260]])
        else:
            self.saatimes = np.array(intervals)

    def insaa(self, utime):
        """Check if time is in SAA."""
        for start, end in self.saatimes:
            if start <= utime <= end:
                return 1
        return 0


class MockTarget:
    """Mock target for testing."""

    def __init__(self, constraint):
        self.constraint = constraint
        self.ra = 0.0
        self.dec = 0.0
        self.starttime = 0
        self.endtime = 0
        self.isat = True
        self.windows = [[100, 200], [300, 400]]

    def constraints(self):
        """Mock constraints calculation."""
        pass


class TestPlanEntryInit:
    def test_init_sets_constraint(self, mock_config):
        """Test PlanEntry initialization sets constraint."""
        pe = PlanEntry(config=mock_config)
        assert pe.constraint is mock_config.constraint

    def test_init_sets_acs_config(self, mock_config):
        """Test PlanEntry initialization sets ACS config."""
        pe = PlanEntry(config=mock_config)
        assert pe.acs_config is mock_config.spacecraft_bus.attitude_control

    def test_init_sets_ephem(self, mock_config):
        """Test PlanEntry initialization sets ephem."""
        pe = PlanEntry(config=mock_config)
        assert pe.ephem is mock_config.constraint.ephem

    def test_init_sets_default_name(self, mock_config):
        """Test PlanEntry initialization sets default name."""
        pe = PlanEntry(config=mock_config)
        assert pe.name == ""

    def test_init_sets_default_ra(self, mock_config):
        """Test PlanEntry initialization sets default ra."""
        pe = PlanEntry(config=mock_config)
        assert pe.ra == 0.0

    def test_init_sets_default_dec(self, mock_config):
        """Test PlanEntry initialization sets default dec."""
        pe = PlanEntry(config=mock_config)
        assert pe.dec == 0.0

    def test_init_sets_default_roll(self, mock_config):
        """Test PlanEntry initialization sets default roll."""
        pe = PlanEntry(config=mock_config)
        assert pe.roll == -1.0

    def test_init_sets_default_begin(self, mock_config):
        """Test PlanEntry initialization sets default begin."""
        pe = PlanEntry(config=mock_config)
        assert pe.begin == 0

    def test_init_sets_default_slewtime(self, mock_config):
        """Test PlanEntry initialization sets default slewtime."""
        pe = PlanEntry(config=mock_config)
        assert pe.slewtime == 0

    def test_init_sets_default_insaa(self, mock_config):
        """Test PlanEntry initialization sets default insaa."""
        pe = PlanEntry(config=mock_config)
        assert pe.insaa == 0

    def test_init_sets_default_end(self, mock_config):
        """Test PlanEntry initialization sets default end."""
        pe = PlanEntry(config=mock_config)
        assert pe.end == 0

    def test_init_sets_default_obsid(self, mock_config):
        """Test PlanEntry initialization sets default obsid."""
        pe = PlanEntry(config=mock_config)
        assert pe.obsid == 0

    def test_init_sets_default_merit(self, mock_config):
        """Test PlanEntry initialization sets default merit."""
        pe = PlanEntry(config=mock_config)
        assert pe.merit == 101

    def test_init_sets_default_windows(self, mock_config):
        """Test PlanEntry initialization sets default windows."""
        pe = PlanEntry(config=mock_config)
        assert pe.windows == []

    def test_init_sets_default_obstype(self, mock_config):
        """Test PlanEntry initialization sets default obstype."""
        pe = PlanEntry(config=mock_config)
        assert pe.obstype == "PPT"

    def test_init_sets_default_slewpath(self, mock_config):
        """Test PlanEntry initialization sets default slewpath."""
        pe = PlanEntry(config=mock_config)
        assert pe.slewpath == ([], [])

    def test_init_sets_default_slewdist(self, mock_config):
        """Test PlanEntry initialization sets default slewdist."""
        pe = PlanEntry(config=mock_config)
        assert pe.slewdist == 0.0

    def test_init_without_config_raises_assertion(self):
        """Test that initialization without constraint raises AssertionError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PlanEntry(config=None)

    def test_init_with_constraint_missing_ephem(self, mock_config):
        """Test that initialization with constraint missing ephem raises ValidationError."""
        mock_config.constraint.ephem = None
        with pytest.raises(ValidationError, match="Ephemeris must be set"):
            PlanEntry(config=mock_config)


class TestPlanEntryCopy:
    def test_copy_creates_different_object(self, plan_entry):
        """Test that copy creates a different object."""
        copied = plan_entry.model_copy()
        assert copied is not plan_entry

    def test_copy_preserves_name(self, plan_entry):
        """Test that copy preserves name."""
        plan_entry.name = "Test Target"
        copied = plan_entry.model_copy()
        assert copied.name == plan_entry.name

    def test_copy_preserves_ra(self, plan_entry):
        """Test that copy preserves ra."""
        plan_entry.ra = 123.45
        copied = plan_entry.model_copy()
        assert copied.ra == plan_entry.ra

    def test_copy_preserves_dec(self, plan_entry):
        """Test that copy preserves dec."""
        plan_entry.dec = 67.89
        copied = plan_entry.model_copy()
        assert copied.dec == plan_entry.dec

    def test_copy_preserves_merit(self, plan_entry):
        """Test that copy preserves merit."""
        plan_entry.merit = 50
        copied = plan_entry.model_copy()
        assert copied.merit == plan_entry.merit

    def test_copy_shallow_copies_constraint(self, plan_entry):
        """Test that copy shallow copies constraint."""
        copied = plan_entry.model_copy()
        assert copied.constraint is plan_entry.constraint


class TestObsidProperties:
    def test_obsid_value(self, plan_entry):
        """Test obsid property value."""
        plan_entry.obsid = 0x12ABCD
        assert plan_entry.obsid == 0x12ABCD


class TestPlanEntryStr:
    def test_str_includes_name(self, plan_entry):
        """Test string representation includes name."""
        plan_entry.name = "TestTarget"
        plan_entry.begin = 1000000000
        plan_entry.end = 1000001000
        plan_entry.slewtime = 100
        plan_entry.obsid = 0x12ABCD

        result = str(plan_entry)
        assert "TestTarget" in result

    def test_str_includes_obsid(self, plan_entry):
        """Test string representation includes obsid."""
        plan_entry.name = "TestTarget"
        plan_entry.begin = 1000000000
        plan_entry.end = 1000001000
        plan_entry.slewtime = 100
        plan_entry.obsid = 0x12ABCD

        result = str(plan_entry)
        assert str(plan_entry.obsid) in result

    def test_str_includes_exposure_time(self, plan_entry):
        """Test string representation includes exposure time."""
        plan_entry.name = "TestTarget"
        plan_entry.begin = 1000000000
        plan_entry.end = 1000001000
        plan_entry.slewtime = 100
        plan_entry.obsid = 0x12ABCD

        result = str(plan_entry)
        assert "900s" in result  # exposure time


class TestExposureProperty:
    def test_exposure_without_saa_calculates_correctly(self, plan_entry):
        """Test exposure calculation without SAA."""
        plan_entry.begin = 1000
        plan_entry.end = 2000
        plan_entry.slewtime = 100
        plan_entry.saa = False

        exposure = plan_entry.exposure
        assert exposure == 900  # 2000 - 1000 - 100

    def test_exposure_without_saa_sets_insaa_to_zero(self, plan_entry):
        """Test exposure sets insaa to zero without SAA."""
        plan_entry.begin = 1000
        plan_entry.end = 2000
        plan_entry.slewtime = 100
        plan_entry.saa = False

        assert plan_entry.insaa == 0

    def test_exposure_with_saa_no_overlap_calculates_correctly(self, plan_entry):
        """Test exposure with SAA but no overlap."""
        plan_entry.begin = 1000
        plan_entry.end = 1100
        plan_entry.slewtime = 10
        plan_entry.saa = MockSAA([[2000, 2100]])

        exposure = plan_entry.exposure
        assert exposure == 90  # 1100 - 1000 - 10

    def test_exposure_with_saa_no_overlap_sets_insaa_to_zero(self, plan_entry):
        """Test exposure sets insaa to zero with SAA but no overlap."""
        plan_entry.begin = 1000
        plan_entry.end = 1100
        plan_entry.slewtime = 10
        plan_entry.saa = MockSAA([[2000, 2100]])

        assert plan_entry.insaa == 0

    # def test_exposure_with_saa_overlap(self, plan_entry):
    #     """Test exposure with SAA overlap."""
    #     plan_entry.begin = 1000
    #     plan_entry.end = 1200
    #     plan_entry.slewtime = 10
    #     plan_entry.saa = MockSAA([[1050, 1070]])  # 20 seconds in SAA

    #     exposure = plan_entry.exposure
    #     # Should subtract SAA time (21 seconds where 1050-1070 inclusive)
    #     assert plan_entry.insaa == 21
    #     assert exposure == 169  # 1200 - 1000 - 10 - 21

    def test_exposure_setter_ignored(self, plan_entry):
        """Test that exposure setter is ignored."""
        plan_entry.begin = 1000
        plan_entry.end = 2000
        plan_entry.slewtime = 100

        original_exposure = plan_entry.exposure
        plan_entry.exposure = 5000  # This should be ignored
        assert plan_entry.exposure == original_exposure


class TestGivename:
    def test_givename_without_stem(self, plan_entry):
        """Test givename without stem."""
        plan_entry.ra = 180.0
        plan_entry.dec = 45.0
        plan_entry.givename()

        assert plan_entry.name.startswith("J")
        assert "+" in plan_entry.name  # Positive declination

    def test_givename_with_stem(self, plan_entry):
        """Test givename with stem."""
        plan_entry.ra = 180.0
        plan_entry.dec = -45.0
        plan_entry.givename(stem="GRB")

        assert plan_entry.name.startswith("GRB")
        assert "-" in plan_entry.name  # Negative declination

    def test_givename_ra_zero_dec_zero(self, plan_entry):
        """Test givename with ra=0.0, dec=0.0."""
        plan_entry.ra = 0.0
        plan_entry.dec = 0.0
        plan_entry.givename()
        assert plan_entry.name != ""
        assert "J" in plan_entry.name

    def test_givename_ra_90_dec_30(self, plan_entry):
        """Test givename with ra=90.0, dec=30.0."""
        plan_entry.ra = 90.0
        plan_entry.dec = 30.0
        plan_entry.givename()
        assert plan_entry.name != ""
        assert "J" in plan_entry.name

    def test_givename_ra_270_dec_neg_60(self, plan_entry):
        """Test givename with ra=270.0, dec=-60.0."""
        plan_entry.ra = 270.0
        plan_entry.dec = -60.0
        plan_entry.givename()
        assert plan_entry.name != ""
        assert "J" in plan_entry.name

    def test_givename_ra_359_9_dec_89_9(self, plan_entry):
        """Test givename with ra=359.9, dec=89.9."""
        plan_entry.ra = 359.9
        plan_entry.dec = 89.9
        plan_entry.givename()
        assert plan_entry.name != ""
        assert "J" in plan_entry.name


class TestVisibility:
    def test_visibility_basic(self, plan_entry):
        """Test visibility calculation."""
        plan_entry.ra = 180.0
        plan_entry.dec = 45.0

        result = plan_entry.visibility()

        assert result == 0
        # Should have calculated windows (actual values depend on ephemeris/constraints)
        assert isinstance(plan_entry.windows, list)

    def test_visibility_multiple_days(self, plan_entry):
        """Test visibility calculation for multiple days."""
        plan_entry.ra = 90.0
        plan_entry.dec = -30.0

        result = plan_entry.visibility()

        assert result == 0
        # Should have calculated windows
        assert isinstance(plan_entry.windows, list)


class TestVisible:
    def test_visible_within_window(self, plan_entry):
        """Test visible returns window when time is within."""
        plan_entry.windows = [[100, 200], [300, 400]]

        window = plan_entry.visible(110, 190)
        assert window == [100, 200]

    def test_visible_outside_windows(self, plan_entry):
        """Test visible returns False when time is outside."""
        plan_entry.windows = [[100, 200], [300, 400]]

        window = plan_entry.visible(210, 290)
        assert window is False

    def test_visible_partial_overlap(self, plan_entry):
        """Test visible returns False when only partial overlap."""
        plan_entry.windows = [[100, 200], [300, 400]]

        window = plan_entry.visible(150, 250)
        assert window is False

    def test_visible_exact_boundaries(self, plan_entry):
        """Test visible with exact window boundaries."""
        plan_entry.windows = [[100, 200], [300, 400]]

        window = plan_entry.visible(100, 200)
        assert window == [100, 200]

    def test_visible_empty_windows(self, plan_entry):
        """Test visible with no windows."""
        plan_entry.windows = []

        window = plan_entry.visible(100, 200)
        assert window is False


class TestRaDec:
    def test_ra_dec_before_observation(self, plan_entry):
        """Test ra_dec before observation starts."""
        plan_entry.begin = 1000
        plan_entry.end = 2000
        plan_entry.slewtime = 100
        plan_entry.ra = 180.0
        plan_entry.dec = 45.0

        ra, dec = plan_entry.ra_dec(999)
        assert ra == -1
        assert dec == -1

    def test_ra_dec_during_slew(self, plan_entry):
        """Test ra_dec during slew."""
        plan_entry.begin = 1000
        plan_entry.end = 2000
        plan_entry.slewtime = 100
        plan_entry.ra = 180.0
        plan_entry.dec = 45.0
        # Note: ra_dec only returns target ra/dec during observation, not during slew
        ra, dec = plan_entry.ra_dec(1050)

        # Should return target ra/dec during observation period
        assert ra == 180.0
        assert dec == 45.0

    def test_ra_dec_after_slew(self, plan_entry):
        """Test ra_dec after slew completes."""
        plan_entry.begin = 1000
        plan_entry.end = 2000
        plan_entry.slewtime = 100
        plan_entry.ra = 180.0
        plan_entry.dec = 45.0

        ra, dec = plan_entry.ra_dec(1500)
        assert ra == 180.0
        assert dec == 45.0

    def test_ra_dec_at_end(self, plan_entry):
        """Test ra_dec at end of observation."""
        plan_entry.begin = 1000
        plan_entry.end = 2000
        plan_entry.slewtime = 100
        plan_entry.ra = 180.0
        plan_entry.dec = 45.0

        ra, dec = plan_entry.ra_dec(2000)
        assert ra == 180.0
        assert dec == 45.0

    def test_ra_dec_after_observation(self, plan_entry):
        """Test ra_dec after observation ends."""
        plan_entry.begin = 1000
        plan_entry.end = 2000
        plan_entry.slewtime = 100
        plan_entry.ra = 180.0
        plan_entry.dec = 45.0

        ra, dec = plan_entry.ra_dec(2001)
        assert ra == -1
        assert dec == -1


class TestCalcSlewtime:
    def test_calc_slewtime_returns_positive_value(self, plan_entry):
        """Test calc_slewtime returns positive value."""
        lastra = 100.0
        lastdec = 30.0
        plan_entry.ra = 150.0
        plan_entry.dec = 60.0

        slewtime = plan_entry.calc_slewtime(lastra, lastdec)
        assert slewtime > 0

    def test_calc_slewtime_does_not_update_self_slewtime(self, plan_entry):
        """Test calc_slewtime does not update self.slewtime."""
        lastra = 100.0
        lastdec = 30.0
        plan_entry.ra = 150.0
        plan_entry.dec = 60.0

        plan_entry.calc_slewtime(lastra, lastdec)
        assert plan_entry.slewtime == 0

    def test_calc_slewtime_sets_slewdist(self, plan_entry):
        """Test calc_slewtime sets slewdist."""
        lastra = 100.0
        lastdec = 30.0
        plan_entry.ra = 150.0
        plan_entry.dec = 60.0

        plan_entry.calc_slewtime(lastra, lastdec)
        assert plan_entry.slewdist > 0

    def test_calc_slewtime_uses_slew_time_from_acs_config(self, plan_entry):
        """Test calc_slewtime uses slew_time from acs_config."""
        lastra = 100.0
        lastdec = 30.0
        plan_entry.ra = 150.0
        plan_entry.dec = 60.0

        slewtime = plan_entry.calc_slewtime(lastra, lastdec)
        assert slewtime == round(plan_entry.acs_config.slew_time(plan_entry.slewdist))


class TestPredictSlew:
    def test_predict_slew_sets_slewdist_not_false(self, plan_entry):
        """Test predict_slew sets slewdist to not False."""
        lastra = 100.0
        lastdec = 30.0
        plan_entry.ra = 150.0
        plan_entry.dec = 60.0

        plan_entry.predict_slew(lastra, lastdec)
        assert plan_entry.slewdist is not False

    def test_predict_slew_sets_slewpath_not_false(self, plan_entry):
        """Test predict_slew sets slewpath to not False."""
        lastra = 100.0
        lastdec = 30.0
        plan_entry.ra = 150.0
        plan_entry.dec = 60.0

        plan_entry.predict_slew(lastra, lastdec)
        assert plan_entry.slewpath is not False

    def test_predict_slew_sets_positive_slewdist(self, plan_entry):
        """Test predict_slew sets positive slewdist."""
        lastra = 100.0
        lastdec = 30.0
        plan_entry.ra = 150.0
        plan_entry.dec = 60.0

        plan_entry.predict_slew(lastra, lastdec)
        assert plan_entry.slewdist > 0

    def test_predict_slew_sets_slewpath_as_tuple_of_two_lists(self, plan_entry):
        """Test predict_slew sets slewpath as tuple of two lists."""
        lastra = 100.0
        lastdec = 30.0
        plan_entry.ra = 150.0
        plan_entry.dec = 60.0

        plan_entry.predict_slew(lastra, lastdec)
        assert len(plan_entry.slewpath) == 2  # (ra_path, dec_path)

    def test_predict_slew_zero_distance(self, plan_entry):
        """Test predict_slew with zero distance."""
        plan_entry.ra = 100.0
        plan_entry.dec = 30.0

        plan_entry.predict_slew(100.0, 30.0)

        assert plan_entry.slewdist == 0.0

    def test_predict_slew_large_distance_sets_positive_slewdist(self, plan_entry):
        """Test predict_slew with large distance sets positive slewdist."""
        plan_entry.ra = 180.0
        plan_entry.dec = 60.0

        plan_entry.predict_slew(0.0, -60.0)
        assert plan_entry.slewdist > 0

    def test_predict_slew_large_distance_sets_significant_slewdist(self, plan_entry):
        """Test predict_slew with large distance sets significant slewdist."""
        plan_entry.ra = 180.0
        plan_entry.dec = 60.0

        plan_entry.predict_slew(0.0, -60.0)
        assert plan_entry.slewdist > 100
