import numpy as np


class TestSAAInsaaCalc:
    def test_insaa_calc_true_when_point_inside(self, saa_inside):
        assert saa_inside.insaa_calc(100) is True

    def test_insaa_calc_false_when_point_outside(self, saa_outside):
        assert saa_outside.insaa_calc(100) is False


class TestSAACalcAndInsaa:
    # Single interval tests ----------------------------------------------------------------------------
    def test_calc_sets_calculated_true(self, saa_single):
        saa_single.calc()
        assert saa_single.calculated is True

    def test_calc_sets_single_interval(self, saa_single):
        saa_single.calc()
        np.testing.assert_array_equal(saa_single.saatimes, [[20, 30]])

    def test_insaa_returns_one_for_start_of_interval(self, saa_single):
        saa_single.calc()
        assert saa_single.insaa(20) == 1

    def test_insaa_returns_one_for_middle_of_interval(self, saa_single):
        saa_single.calc()
        assert saa_single.insaa(25) == 1  # Note: corrected to 25 for middle

    def test_insaa_returns_one_for_end_of_interval_inclusive(self, saa_single):
        saa_single.calc()
        assert saa_single.insaa(30) == 1

    def test_insaa_returns_zero_before_interval(self, saa_single):
        saa_single.calc()
        assert saa_single.insaa(10) == 0

    def test_insaa_returns_zero_after_interval(self, saa_single):
        saa_single.calc()
        assert saa_single.insaa(41) == 0

    # Multiple interval tests --------------------------------------------------------------------------
    def test_calc_detects_multiple_intervals(self, saa_multiple):
        saa_multiple.calc()
        np.testing.assert_array_equal(saa_multiple.saatimes, [[2, 3], [5, 6]])

    def test_insaa_returns_one_for_first_interval_start(self, saa_multiple):
        saa_multiple.calc()
        assert saa_multiple.insaa(2) == 1

    def test_insaa_returns_one_for_first_interval_middle(self, saa_multiple):
        saa_multiple.calc()
        assert saa_multiple.insaa(2.5) == 1  # Note: corrected to 2.5 for middle

    def test_insaa_returns_one_for_first_interval_end_inclusive(self, saa_multiple):
        saa_multiple.calc()
        assert saa_multiple.insaa(3) == 1

    def test_insaa_returns_one_for_second_interval_start(self, saa_multiple):
        saa_multiple.calc()
        assert saa_multiple.insaa(5) == 1

    def test_insaa_returns_one_for_second_interval_middle(self, saa_multiple):
        saa_multiple.calc()
        assert saa_multiple.insaa(5.5) == 1  # Note: corrected to 5.5 for middle

    def test_insaa_returns_one_for_second_interval_end_inclusive(self, saa_multiple):
        saa_multiple.calc()
        assert saa_multiple.insaa(6) == 1

    def test_insaa_returns_zero_between_intervals(self, saa_multiple):
        saa_multiple.calc()
        assert saa_multiple.insaa(4) == 0  # Note: changed to 4, assuming integer times
