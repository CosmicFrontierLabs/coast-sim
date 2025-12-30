"""Tests for slew edge cases and singularities."""

import numpy as np
import pytest

from conops import AttitudeControlSystem, Constraint, MissionConfig, Slew, SpacecraftBus


class DummyConstraint(Constraint):
    """Minimal constraint for slew testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ephem = object()


def make_slew(
    startra: float = 0.0,
    startdec: float = 0.0,
    endra: float = 10.0,
    enddec: float = 0.0,
) -> Slew:
    """Create a slew with given coordinates."""
    acs = AttitudeControlSystem(
        slew_acceleration=1.0, max_slew_rate=0.5, settle_time=90.0
    )
    constraint = DummyConstraint()
    spacecraft_bus = SpacecraftBus(attitude_control=acs)
    config = MissionConfig(
        name="Test", constraint=constraint, spacecraft_bus=spacecraft_bus
    )
    s = Slew(config=config)
    s.startra = startra
    s.startdec = startdec
    s.endra = endra
    s.enddec = enddec
    return s


class TestZeroDistanceSlew:
    """Tests for slews with zero angular distance."""

    def test_zero_distance_slew_returns_zero_time(self):
        """Slew from point to same point has zero motion time."""
        s = make_slew(startra=45.0, startdec=30.0, endra=45.0, enddec=30.0)
        s.predict_slew()

        assert s.slewdist == pytest.approx(0.0, abs=1e-10)

    def test_zero_distance_slew_time_is_zero(self):
        """Zero distance slew time is zero (no motion needed)."""
        s = make_slew(startra=45.0, startdec=30.0, endra=45.0, enddec=30.0)
        slew_time = s.calc_slewtime()

        # Zero distance means zero slew time
        assert slew_time == pytest.approx(0.0, abs=1.0)

    def test_zero_distance_rotation_axis_defaults_to_z(self):
        """Zero distance slew uses default Z rotation axis."""
        s = make_slew(startra=45.0, startdec=30.0, endra=45.0, enddec=30.0)
        s.predict_slew()

        assert s.rotation_axis == (0.0, 0.0, 1.0)

    def test_zero_distance_ra_dec_returns_start(self):
        """Zero distance slew returns start coordinates at all times."""
        s = make_slew(startra=45.0, startdec=30.0, endra=45.0, enddec=30.0)
        s.slewstart = 0.0
        s.calc_slewtime()

        # At any time during slew, should return start position
        ra, dec = s.slew_ra_dec(50.0)
        assert ra == pytest.approx(45.0, abs=0.01)
        assert dec == pytest.approx(30.0, abs=0.01)


class TestSmallSlew:
    """Tests for very small angular slews."""

    def test_tiny_slew_computes_correctly(self):
        """Very small slew (0.001 deg) computes without error."""
        s = make_slew(startra=45.0, startdec=30.0, endra=45.001, enddec=30.0)
        s.predict_slew()

        assert s.slewdist > 0
        assert s.slewdist < 0.01

    def test_tiny_slew_has_valid_rotation_axis(self):
        """Small slew has normalized rotation axis."""
        s = make_slew(startra=45.0, startdec=30.0, endra=45.001, enddec=30.0)
        s.predict_slew()

        axis = np.array(s.rotation_axis)
        assert np.linalg.norm(axis) == pytest.approx(1.0, rel=1e-6)


class TestAntipodalSlew:
    """Tests for 180-degree slews (antipodal points)."""

    def test_180_degree_slew_across_ra(self):
        """180 degree slew in RA computes correctly."""
        s = make_slew(startra=0.0, startdec=0.0, endra=180.0, enddec=0.0)
        s.predict_slew()

        assert s.slewdist == pytest.approx(180.0, abs=1.0)

    def test_180_degree_slew_has_valid_axis(self):
        """180 degree slew has valid (non-zero) rotation axis."""
        s = make_slew(startra=0.0, startdec=0.0, endra=180.0, enddec=0.0)
        s.predict_slew()

        axis = np.array(s.rotation_axis)
        # Should be normalized
        assert np.linalg.norm(axis) == pytest.approx(1.0, rel=1e-6)

    def test_near_antipodal_slew(self):
        """Near-180-degree slew computes without singularity."""
        s = make_slew(startra=0.0, startdec=0.0, endra=179.99, enddec=0.0)
        s.predict_slew()

        assert s.slewdist > 179.0
        axis = np.array(s.rotation_axis)
        assert np.linalg.norm(axis) == pytest.approx(1.0, rel=1e-6)


class TestPolarSlew:
    """Tests for slews to/from celestial poles."""

    def test_slew_to_north_pole(self):
        """Slew to Dec=+90 computes correctly."""
        s = make_slew(startra=0.0, startdec=0.0, endra=0.0, enddec=90.0)
        s.predict_slew()

        assert s.slewdist == pytest.approx(90.0, abs=1.0)

    def test_slew_to_south_pole(self):
        """Slew to Dec=-90 computes correctly."""
        s = make_slew(startra=0.0, startdec=0.0, endra=0.0, enddec=-90.0)
        s.predict_slew()

        assert s.slewdist == pytest.approx(90.0, abs=1.0)

    def test_slew_from_pole_to_pole(self):
        """Slew from north pole to south pole is 180 degrees."""
        s = make_slew(startra=0.0, startdec=90.0, endra=0.0, enddec=-90.0)
        s.predict_slew()

        assert s.slewdist == pytest.approx(180.0, abs=1.0)

    def test_slew_at_pole_different_ra(self):
        """At the pole, different RA values represent same point."""
        s1 = make_slew(startra=0.0, startdec=90.0, endra=45.0, enddec=80.0)
        s2 = make_slew(startra=180.0, startdec=90.0, endra=45.0, enddec=80.0)

        s1.predict_slew()
        s2.predict_slew()

        # Both should give same slew distance since start is at pole
        assert s1.slewdist == pytest.approx(s2.slewdist, abs=0.1)


class TestRAWrapAround:
    """Tests for slews crossing RA=0/360 boundary."""

    def test_slew_across_ra_zero(self):
        """Slew from RA=350 to RA=10 takes short path."""
        s = make_slew(startra=350.0, startdec=0.0, endra=10.0, enddec=0.0)
        s.predict_slew()

        # Should take 20 degree path, not 340 degree path
        assert s.slewdist == pytest.approx(20.0, abs=1.0)

    def test_slew_across_ra_180(self):
        """Slew from RA=170 to RA=190 computes correctly."""
        s = make_slew(startra=170.0, startdec=0.0, endra=190.0, enddec=0.0)
        s.predict_slew()

        assert s.slewdist == pytest.approx(20.0, abs=1.0)

    def test_ra_wraparound_in_path(self):
        """Slew path handles RA wraparound correctly."""
        s = make_slew(startra=350.0, startdec=0.0, endra=10.0, enddec=0.0)
        s.slewstart = 0.0
        s.calc_slewtime()

        # Mid-slew position should be around RA=0 or 360
        ra_mid, _ = s.slew_ra_dec(s.slewtime / 2)
        # Allow for either 0 or 360 representation
        assert ra_mid < 15 or ra_mid > 355


class TestSlewKinematics:
    """Tests for bang-bang slew kinematics."""

    def test_slew_starts_at_start_position(self):
        """At t=0, slew position equals start coordinates."""
        s = make_slew(startra=10.0, startdec=20.0, endra=50.0, enddec=40.0)
        s.slewstart = 0.0
        s.calc_slewtime()

        ra, dec = s.slew_ra_dec(0.0)
        assert ra == pytest.approx(10.0, abs=0.01)
        assert dec == pytest.approx(20.0, abs=0.01)

    def test_slew_ends_at_end_position(self):
        """After motion completes, slew position equals end coordinates."""
        s = make_slew(startra=10.0, startdec=20.0, endra=50.0, enddec=40.0)
        s.slewstart = 0.0
        s.calc_slewtime()

        # After slew time, should be at end position
        ra, dec = s.slew_ra_dec(s.slewtime + 100.0)
        assert ra == pytest.approx(50.0, abs=0.5)
        assert dec == pytest.approx(40.0, abs=0.5)

    def test_slew_monotonic_progress(self):
        """Distance from start increases monotonically during slew."""
        s = make_slew(startra=0.0, startdec=0.0, endra=90.0, enddec=0.0)
        s.slewstart = 0.0
        s.calc_slewtime()

        prev_dist = 0.0
        for t in np.linspace(0, s.slewtime - 90, 10):  # Up to end of motion
            ra, dec = s.slew_ra_dec(t)
            # Distance from start
            dist = np.sqrt((ra - 0.0) ** 2 + (dec - 0.0) ** 2)
            assert dist >= prev_dist - 0.01  # Allow tiny numerical error
            prev_dist = dist

    def test_slew_before_start_returns_start(self):
        """Before slew starts, position is at start."""
        s = make_slew(startra=10.0, startdec=20.0, endra=50.0, enddec=40.0)
        s.slewstart = 100.0
        s.calc_slewtime()

        ra, dec = s.slew_ra_dec(50.0)  # Before slewstart
        assert ra == pytest.approx(10.0, abs=0.01)
        assert dec == pytest.approx(20.0, abs=0.01)


class TestSlewIsSlewing:
    """Tests for is_slewing() state detection."""

    def test_is_slewing_before_start(self):
        """is_slewing returns False before slew starts."""
        s = make_slew()
        s.slewstart = 100.0
        s.slewend = 200.0

        assert not s.is_slewing(50.0)

    def test_is_slewing_during_slew(self):
        """is_slewing returns True during slew."""
        s = make_slew()
        s.slewstart = 100.0
        s.slewend = 200.0

        assert s.is_slewing(150.0)

    def test_is_slewing_at_start_boundary(self):
        """is_slewing returns True at exact start time."""
        s = make_slew()
        s.slewstart = 100.0
        s.slewend = 200.0

        assert s.is_slewing(100.0)

    def test_is_slewing_at_end_boundary(self):
        """is_slewing returns False at exact end time."""
        s = make_slew()
        s.slewstart = 100.0
        s.slewend = 200.0

        assert not s.is_slewing(200.0)

    def test_is_slewing_after_end(self):
        """is_slewing returns False after slew ends."""
        s = make_slew()
        s.slewstart = 100.0
        s.slewend = 200.0

        assert not s.is_slewing(250.0)


class TestRotationAxis:
    """Tests for rotation axis calculation."""

    def test_rotation_axis_perpendicular_to_endpoints(self):
        """Rotation axis is perpendicular to both start and end vectors."""
        s = make_slew(startra=0.0, startdec=0.0, endra=90.0, enddec=0.0)
        s.predict_slew()

        axis = np.array(s.rotation_axis)

        # Convert start/end to unit vectors
        ra0, dec0 = np.deg2rad(0.0), np.deg2rad(0.0)
        ra1, dec1 = np.deg2rad(90.0), np.deg2rad(0.0)
        v0 = np.array(
            [np.cos(dec0) * np.cos(ra0), np.cos(dec0) * np.sin(ra0), np.sin(dec0)]
        )
        v1 = np.array(
            [np.cos(dec1) * np.cos(ra1), np.cos(dec1) * np.sin(ra1), np.sin(dec1)]
        )

        # Axis should be perpendicular to both
        assert abs(np.dot(axis, v0)) < 1e-10
        assert abs(np.dot(axis, v1)) < 1e-10

    def test_rotation_axis_normalized(self):
        """Rotation axis is unit length."""
        test_cases = [
            (0.0, 0.0, 45.0, 30.0),
            (350.0, -20.0, 10.0, 40.0),
            (0.0, 80.0, 180.0, 80.0),
        ]

        for startra, startdec, endra, enddec in test_cases:
            s = make_slew(
                startra=startra, startdec=startdec, endra=endra, enddec=enddec
            )
            s.predict_slew()

            axis = np.array(s.rotation_axis)
            assert np.linalg.norm(axis) == pytest.approx(1.0, rel=1e-6)
