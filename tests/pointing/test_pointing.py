from conops import Pointing

from .conftest import DummyConstraint


class TestPointingNextVis:
    """Test next_vis method of Pointing."""

    def test_next_vis_returns_same_time_when_in_window(
        self, pointing: Pointing
    ) -> None:
        pointing.windows = [(5.0, 10.0)]
        utime = 7.0
        assert pointing.next_vis(utime) == utime

    def test_next_vis_returns_false_when_no_windows(self, pointing: Pointing) -> None:
        pointing.windows = []
        assert pointing.next_vis(10.0) is False

    def test_next_vis_returns_next_start_time_before_first_window(
        self, pointing: Pointing
    ) -> None:
        pointing.windows = [(5.0, 7.0), (15.0, 20.0)]
        # utime before first window start -> should return 5.0
        assert pointing.next_vis(0.0) == 5.0

    def test_next_vis_returns_next_start_time_between_windows(
        self, pointing: Pointing
    ) -> None:
        pointing.windows = [(5.0, 7.0), (15.0, 20.0)]
        # utime between windows starts -> should return 15.0
        assert pointing.next_vis(10.0) == 15.0


class TestPointingStringRepresentation:
    """Test string representation of Pointing."""

    def test_str_contains_target_name_and_id(self, pointing: Pointing) -> None:
        pointing.begin = 0
        pointing.name = "TargetName"
        pointing.obsid = 42
        pointing.ra = 1.2345
        pointing.dec = -0.1234
        pointing.roll = 3.4
        pointing.merit = 7.0
        s = str(pointing)
        # Ensure human-readable components are present
        assert "TargetName (42)" in s

    def test_str_contains_ra(self, pointing: Pointing) -> None:
        pointing.begin = 0
        pointing.name = "TargetName"
        pointing.obsid = 42
        pointing.ra = 1.2345
        pointing.dec = -0.1234
        pointing.roll = 3.4
        pointing.merit = 7.0
        s = str(pointing)
        # Ensure human-readable components are present
        assert "RA=1.2345" in s

    def test_str_contains_dec(self, pointing: Pointing) -> None:
        pointing.begin = 0
        pointing.name = "TargetName"
        pointing.obsid = 42
        pointing.ra = 1.2345
        pointing.dec = -0.1234
        pointing.roll = 3.4
        pointing.merit = 7.0
        s = str(pointing)
        # Ensure human-readable components are present
        assert "Dec" in s or "Dec=" in s  # Accept either formatting presence

    def test_str_contains_merit(self, pointing: Pointing) -> None:
        pointing.begin = 0
        pointing.name = "TargetName"
        pointing.obsid = 42
        pointing.ra = 1.2345
        pointing.dec = -0.1234
        pointing.roll = 3.4
        pointing.merit = 7.0
        s = str(pointing)
        # Ensure human-readable components are present
        assert "Merit=" in s


class TestPointing:
    """Test Pointing class initialization and properties."""

    def test_constraint(self, pointing: Pointing, constraint: DummyConstraint) -> None:
        assert pointing.constraint == constraint

    def test_obstype_at(self, pointing: Pointing) -> None:
        assert pointing.obstype == "AT"

    def test_ra_zero(self, pointing: Pointing) -> None:
        assert pointing.ra == 0.0

    def test_dec_zero(self, pointing: Pointing) -> None:
        assert pointing.dec == 0.0

    def test_obsid_zero(self, pointing: Pointing) -> None:
        assert pointing.obsid == 0

    def test_name_fake_target(self, pointing: Pointing) -> None:
        assert pointing.name == "FakeTarget"

    def test_fom_100(self, pointing: Pointing) -> None:
        assert pointing.fom == 100.0  # fom is maintained as legacy alias for merit

    def test_merit_100(self, pointing: Pointing) -> None:
        assert pointing.merit == 100

    def test_exptime_none(self, pointing: Pointing) -> None:
        assert pointing.exptime is None

    def test_exptime_setter_initializes_exporig(self, pointing: Pointing) -> None:
        # First set initializes _exporig
        pointing.exptime = 500
        assert pointing.exptime == 500

    def test_exptime_setter_sets_exporig(self, pointing: Pointing) -> None:
        # First set initializes _exporig
        pointing.exptime = 500
        assert pointing._exporig == 500

    def test_exptime_setter_second_set_changes_exptime(
        self, pointing: Pointing
    ) -> None:
        # First set initializes _exporig
        pointing.exptime = 500
        # Second set doesn't change _exporig
        pointing.exptime = 1000
        assert pointing.exptime == 1000

    def test_exptime_setter_second_set_does_not_change_exporig(
        self, pointing: Pointing
    ) -> None:
        # First set initializes _exporig
        pointing.exptime = 500
        # Second set doesn't change _exporig
        pointing.exptime = 1000
        assert pointing._exporig == 500

    def test_done_property_initially_false(self, pointing: Pointing) -> None:
        # Set exptime first to avoid None comparison
        pointing.exptime = 100
        # Initially done is False
        assert pointing.done is False

    def test_done_property_setter_true(self, pointing: Pointing) -> None:
        # Set exptime first to avoid None comparison
        pointing.exptime = 100
        # Set done to True
        pointing.done = True
        assert pointing.done is True

    def test_done_property_when_exptime_zero(self, pointing: Pointing) -> None:
        pointing.exptime = 0
        # When exptime <= 0, done should be True
        assert pointing.done is True

    def test_reset_exptime_to_exporig(self, pointing: Pointing) -> None:
        pointing.exptime = 500
        pointing.begin = 100.0
        pointing.end = 200.0
        pointing.slewtime = 10.0
        pointing.done = True
        # Reset
        pointing.reset()
        assert pointing.exptime == 500  # Should be reset to _exporig

    def test_reset_done_to_false(self, pointing: Pointing) -> None:
        pointing.exptime = 500
        pointing.begin = 100.0
        pointing.end = 200.0
        pointing.slewtime = 10.0
        pointing.done = True
        # Reset
        pointing.reset()
        assert pointing.done is False

    def test_reset_begin_to_zero(self, pointing: Pointing) -> None:
        pointing.exptime = 500
        pointing.begin = 100.0
        pointing.end = 200.0
        pointing.slewtime = 10.0
        pointing.done = True
        # Reset
        pointing.reset()
        assert pointing.begin == 0

    def test_reset_end_to_zero(self, pointing: Pointing) -> None:
        pointing.exptime = 500
        pointing.begin = 100.0
        pointing.end = 200.0
        pointing.slewtime = 10.0
        pointing.done = True
        # Reset
        pointing.reset()
        assert pointing.end == 0

    def test_reset_slewtime_to_zero(self, pointing: Pointing) -> None:
        pointing.exptime = 500
        pointing.begin = 100.0
        pointing.end = 200.0
        pointing.slewtime = 10.0
        pointing.done = True
        # Reset
        pointing.reset()
        assert pointing.slewtime == 0
