from types import SimpleNamespace

from conops.pointing import Pointing


class DummyACSConfig:
    """Dummy ACS configuration for testing."""

    pass


class DummyConstraint:
    def __init__(
        self,
        inoccult_val=False,
        in_sun_val=False,
        in_earth_val=False,
        in_moon_val=False,
        in_panel_val=False,
        step_size=1,
    ):
        self._inoccult = inoccult_val
        self._in_sun = in_sun_val
        self._in_earth = in_earth_val
        self._in_moon = in_moon_val
        self._in_panel = in_panel_val
        self.ephem = SimpleNamespace(step_size=step_size)

    def inoccult(self, ra, dec, utime, hardonly=False):
        return self._inoccult

    def in_sun(self, ra, dec, utime):
        return self._in_sun

    def in_earth(self, ra, dec, utime):
        return self._in_earth

    def in_moon(self, ra, dec, utime):
        return self._in_moon

    def in_panel(self, ra, dec, utime):
        return self._in_panel


class DummySAA:
    def __init__(self, value=0):
        self._value = value

    def insaa(self, t):
        # Return same value for any time step
        return self._value


class TestPointingInitialization:
    """Test Pointing initialization and default fields."""

    def test_default_fields(self):
        c = DummyConstraint()
        tr = Pointing(constraint=c, acs_config=DummyACSConfig())
        assert tr.exptime is None
        assert tr.inview is False
        assert tr.done is False
        assert tr.obstype == "AT"
        assert tr.coordinated is None
        assert tr.isat is False


class TestPointingVisibility:
    """Test visibility-related methods on Pointing."""

    def test_visibility_wrappers_and_is_visible(self):
        c = DummyConstraint(
            inoccult_val=False,
            in_sun_val=True,
            in_earth_val=True,
            in_moon_val=False,
            in_panel_val=True,
        )
        tr = Pointing(constraint=c, acs_config=DummyACSConfig())
        # ra/dec values passed through to the constraint; methods just return the stubbed values
        tr.ra = 12.34
        tr.dec = -21.0
        assert tr.is_visible(12345)
        assert tr.in_sun(12345)
        assert tr.in_earth(12345)
        assert tr.in_moon(12345) is False
        assert tr.in_panel(12345)


class TestPointingNextVis:
    """Test next_vis and visible methods for visibility window calculation."""

    def test_returns_current_time_if_visible(self):
        c = DummyConstraint()
        tr = Pointing(constraint=c, acs_config=DummyACSConfig())
        utime = 1000.0
        # Place a visibility window that includes utime
        tr.windows = [(utime - 1.0, utime + 10.0)]
        # Confirm visible at utime (visible returns the window tuple, not True)
        assert tr.visible(utime, utime) == (utime - 1.0, utime + 10.0)
        assert tr.next_vis(utime) == utime

    def test_returns_false_when_no_windows(self):
        c = DummyConstraint()
        tr = Pointing(constraint=c, acs_config=DummyACSConfig())
        tr.windows = []
        assert tr.next_vis(10.0) is False

    def test_returns_next_start_time(self):
        c = DummyConstraint()
        tr = Pointing(constraint=c, acs_config=DummyACSConfig())
        tr.windows = [(5.0, 7.0), (15.0, 20.0)]
        # utime before first window start -> should return 5.0
        assert tr.next_vis(0.0) == 5.0
        # utime between windows starts -> should return 15.0
        assert tr.next_vis(10.0) == 15.0


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

    def test_str_contains_key_fields(self):
        c = DummyConstraint()
        tr = Pointing(constraint=c, acs_config=DummyACSConfig())
        tr.begin = 0
        tr.name = "TargetName"
        tr.targetid = 42
        tr.ra = 1.2345
        tr.dec = -0.1234
        tr.roll = 3.4
        tr.merit = 7.0
        s = str(tr)
        # Ensure human-readable components are present
        assert "TargetName (42)" in s
        assert "RA=1.2345" in s
        assert "Dec" in s or "Dec=" in s  # Accept either formatting presence
        assert "Merit=" in s


class TestPointing:
    """Test Pointing class initialization and properties."""

    def test_pointing_initialization(self):
        c = DummyConstraint()
        p = Pointing(constraint=c, acs_config=DummyACSConfig())
        assert p.constraint == c
        assert p.obsstart == 0
        assert p.inview is False
        assert p.obstype == "AT"
        assert p.coordinated is None
        assert p.ra == 0.0
        assert p.dec == 0.0
        assert p.targetid == 0
        assert p.name == "FakeTarget"
        assert p.fom == 100.0  # fom is maintained as legacy alias for merit
        assert p.merit == 100
        assert p.exptime is None
        assert p.saatime == 0

    def test_pointing_exptime_setter(self):
        c = DummyConstraint()
        p = Pointing(constraint=c, acs_config=DummyACSConfig())
        # First set initializes _exporig
        p.exptime = 500
        assert p.exptime == 500
        assert p._exporig == 500
        # Second set doesn't change _exporig
        p.exptime = 1000
        assert p.exptime == 1000
        assert p._exporig == 500

    def test_pointing_done_property_setter(self):
        c = DummyConstraint()
        p = Pointing(constraint=c, acs_config=DummyACSConfig())
        # Set exptime first to avoid None comparison
        p.exptime = 100
        # Initially done is False
        assert p.done is False
        # Set done to True
        p.done = True
        assert p.done is True

    def test_pointing_done_property_when_exptime_zero(self):
        c = DummyConstraint()
        p = Pointing(constraint=c, acs_config=DummyACSConfig())
        p.exptime = 0
        # When exptime <= 0, done should be True
        assert p.done is True

    def test_pointing_reset(self):
        c = DummyConstraint()
        p = Pointing(constraint=c, acs_config=DummyACSConfig())
        p.exptime = 500
        p.begin = 100.0
        p.end = 200.0
        p.slewtime = 10.0
        p.done = True
        # Reset
        p.reset()
        assert p.exptime == 500  # Should be reset to _exporig
        assert p.done is False
        assert p.begin == 0
        assert p.end == 0
        assert p.slewtime == 0
