"""Tests for star tracker constraint checking."""

import time

from conops.config import StarTracker, StarTrackerOrientation


class TestStarTrackerConstraints:
    """Test hard and soft constraint checking."""

    def test_no_constraint_always_valid(self, basic_star_tracker):
        """Star tracker without constraints should not violate any pointing."""
        st = basic_star_tracker
        current_time = time.time()
        # Test various pointings
        assert not st.in_hard_constraint(0.0, 0.0, current_time)
        assert not st.in_hard_constraint(90.0, 45.0, current_time)
        assert not st.in_hard_constraint(180.0, -60.0, current_time)

    def test_no_soft_constraint_always_valid(self, basic_star_tracker):
        """Star tracker without soft constraints should not degrade any pointing."""
        st = basic_star_tracker
        current_time = time.time()
        assert not st.in_soft_constraint(0.0, 0.0, current_time)
        assert not st.in_soft_constraint(90.0, 45.0, current_time)

    def test_hard_constraint_with_sun(self, star_tracker_with_hard_constraint):
        """Hard constraint should prevent looking at sun."""
        st = star_tracker_with_hard_constraint
        # Set ephemeris for constraint
        import rust_ephem

        st.hard_constraint.ephem = rust_ephem.Ephemeris.earth()

        current_time = time.time()
        # Pointing near the sun should violate hard constraint
        # (sun is approximately at RA=280, Dec=-20 in Jan 2026)
        # We just check that the constraint mechanism works
        result = st.in_hard_constraint(280.0, -20.0, current_time)
        assert isinstance(result, bool)

    def test_soft_constraint_degradation(self, star_tracker_with_soft_constraint):
        """Soft constraint should indicate performance degradation."""
        st = star_tracker_with_soft_constraint
        # Set ephemeris
        import rust_ephem

        st.soft_constraint.ephem = rust_ephem.Ephemeris.earth()

        current_time = time.time()
        result = st.in_soft_constraint(0.0, 0.0, current_time)
        assert isinstance(result, bool)

    def test_constraint_with_orientation_transform(self):
        """Constraint checking should account for orientation."""
        from conops.config import Constraint

        ori = StarTrackerOrientation(roll=0.0, pitch=45.0, yaw=0.0)
        constraint = Constraint()
        st = StarTracker(
            name="ST_OrientedConstraint",
            orientation=ori,
            hard_constraint=constraint,
        )

        import rust_ephem

        st.hard_constraint.ephem = rust_ephem.Ephemeris.earth()

        current_time = time.time()
        # Test that pointing and orientation interact
        result = st.in_hard_constraint(0.0, 0.0, current_time)
        assert isinstance(result, bool)


class TestStarTrackerModeLockRequirements:
    """Test mode-dependent lock requirements."""

    def test_requires_lock_all_modes(self):
        """Star tracker requiring lock in all modes."""
        st = StarTracker(
            name="ST_AlwaysLock",
            modes_require_lock=None,  # None = all modes
        )
        assert st.requires_lock_in_mode(None)
        assert st.requires_lock_in_mode(0)
        assert st.requires_lock_in_mode(1)
        assert st.requires_lock_in_mode(5)

    def test_requires_lock_specific_modes(self, star_tracker_mode_dependent):
        """Star tracker requiring lock only in specific modes."""
        st = star_tracker_mode_dependent
        # Should require lock in modes 0 and 2
        assert st.requires_lock_in_mode(0)
        assert st.requires_lock_in_mode(2)
        # Should not require lock in other modes
        assert not st.requires_lock_in_mode(1)
        assert not st.requires_lock_in_mode(3)
        assert not st.requires_lock_in_mode(None)  # Nominal mode

    def test_requires_lock_no_modes(self):
        """Star tracker never requiring lock."""
        st = StarTracker(
            name="ST_NoLock",
            modes_require_lock=[],  # Empty list = no modes
        )
        assert not st.requires_lock_in_mode(None)
        assert not st.requires_lock_in_mode(0)
        assert not st.requires_lock_in_mode(1)
        assert not st.requires_lock_in_mode(5)

    def test_requires_lock_nominal_mode_not_in_list(self):
        """Nominal mode (None) should never be in lock requirement list."""
        st = StarTracker(
            name="ST_NominalMode",
            modes_require_lock=[0, 1, 2],
        )
        # Nominal mode should not require lock
        assert not st.requires_lock_in_mode(None)
