import numpy as np
import pytest

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
    def test_init_with_constraint_and_acs(self, mock_config):
        """Test PlanEntry initialization with valid constraint and ACS."""
        pe = PlanEntry(config=mock_config)
        assert pe.constraint is mock_config.constraint
        assert pe.acs_config is mock_config.spacecraft_bus.attitude_control
        assert pe.ephem is mock_config.constraint.ephem
        assert pe.name == ""
        assert pe.ra == 0.0
        assert pe.dec == 0.0
        assert pe.roll == -1.0
        assert pe.begin == 0
        assert pe.slewtime == 0
        assert pe.insaa == 0
        assert pe.end == 0
        assert pe.obsid == 0
        assert pe.merit == 101
        assert pe.windows == []
        assert pe.obstype == "PPT"
        assert pe.slewpath == ([], [])
        assert pe.slewdist == 0.0

    def test_init_without_config_raises_assertion(self):
        """Test that initialization without constraint raises AssertionError."""
        with pytest.raises(ValueError, match="Config must be provided to PlanEntry"):
            PlanEntry(config=None)

    def test_init_with_constraint_missing_ephem(self, mock_config):
        """Test that initialization with constraint missing ephem raises AssertionError."""
        mock_config.constraint.ephem = None
        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            PlanEntry(config=mock_config)


class TestPlanEntryCopy:
    def test_copy_creates_independent_object(self, plan_entry):
        """Test that copy creates an independent object."""
        plan_entry.name = "Test Target"
        plan_entry.ra = 123.45
        plan_entry.dec = 67.89
        plan_entry.merit = 50

        copied = plan_entry.copy()

        assert copied is not plan_entry
        assert copied.name == plan_entry.name
        assert copied.ra == plan_entry.ra
        assert copied.dec == plan_entry.dec
        assert copied.merit == plan_entry.merit
        assert copied.constraint is plan_entry.constraint  # Shallow copy


class TestObsidProperties:
    def test_obsid_value(self, plan_entry):
        """Test obsid property value."""
        plan_entry.obsid = 0x12ABCD
        assert plan_entry.obsid == 0x12ABCD


class TestPlanEntryStr:
    def test_str_representation(self, plan_entry):
        """Test string representation of PlanEntry."""
        plan_entry.name = "TestTarget"
        plan_entry.begin = 1000000000
        plan_entry.end = 1000001000
        plan_entry.slewtime = 100
        plan_entry.obsid = 0x12ABCD

        result = str(plan_entry)

        assert "TestTarget" in result
        assert str(plan_entry.obsid) in result
        assert "900s" in result  # exposure time


class TestExposureProperty:
    def test_exposure_without_saa(self, plan_entry):
        """Test exposure calculation without SAA."""
        plan_entry.begin = 1000
        plan_entry.end = 2000
        plan_entry.slewtime = 100
        plan_entry.saa = False

        exposure = plan_entry.exposure
        assert exposure == 900  # 2000 - 1000 - 100
        assert plan_entry.insaa == 0

    def test_exposure_with_saa_no_overlap(self, plan_entry):
        """Test exposure with SAA but no overlap."""
        plan_entry.begin = 1000
        plan_entry.end = 1100
        plan_entry.slewtime = 10
        plan_entry.saa = MockSAA([[2000, 2100]])

        exposure = plan_entry.exposure
        assert exposure == 90  # 1100 - 1000 - 10
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

    def test_givename_various_coordinates(self, plan_entry):
        """Test givename with various coordinates."""
        test_cases = [
            (0.0, 0.0),
            (90.0, 30.0),
            (270.0, -60.0),
            (359.9, 89.9),
        ]

        for ra, dec in test_cases:
            plan_entry.ra = ra
            plan_entry.dec = dec
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
    def test_calc_slewtime_basic(self, plan_entry):
        """Test calc_slewtime basic calculation."""
        lastra = 100.0
        lastdec = 30.0
        plan_entry.ra = 150.0
        plan_entry.dec = 60.0

        slewtime = plan_entry.calc_slewtime(lastra, lastdec)

        assert slewtime > 0
        # Note: calc_slewtime no longer updates self.slewtime
        assert plan_entry.slewtime == 0

    def test_calc_slewtime_uses_cached_distance(self, plan_entry):
        """Test calc_slewtime computes distance (predict_slew always recalculates)."""
        lastra = 100.0
        lastdec = 30.0
        plan_entry.ra = 150.0
        plan_entry.dec = 60.0

        slewtime = plan_entry.calc_slewtime(lastra, lastdec)

        # calc_slewtime always calls predict_slew which recalculates distance
        assert plan_entry.slewdist > 0
        assert slewtime == round(plan_entry.acs_config.slew_time(plan_entry.slewdist))


class TestPredictSlew:
    def test_predict_slew_basic(self, plan_entry):
        """Test predict_slew calculation."""
        lastra = 100.0
        lastdec = 30.0
        plan_entry.ra = 150.0
        plan_entry.dec = 60.0

        plan_entry.predict_slew(lastra, lastdec)

        assert plan_entry.slewdist is not False
        assert plan_entry.slewpath is not False
        assert plan_entry.slewdist > 0
        assert len(plan_entry.slewpath) == 2  # (ra_path, dec_path)

    def test_predict_slew_zero_distance(self, plan_entry):
        """Test predict_slew with zero distance."""
        plan_entry.ra = 100.0
        plan_entry.dec = 30.0

        plan_entry.predict_slew(100.0, 30.0)

        assert plan_entry.slewdist == 0.0

    def test_predict_slew_large_distance(self, plan_entry):
        """Test predict_slew with large distance."""
        plan_entry.ra = 180.0
        plan_entry.dec = 60.0

        plan_entry.predict_slew(0.0, -60.0)

        assert plan_entry.slewdist > 0
        # Distance should be significant
        assert plan_entry.slewdist > 100
