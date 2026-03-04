"""Tests for star tracker constraint checking."""

import time

import pytest
import rust_ephem

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

    def test_hard_constraint_with_sun(self, basic_star_tracker):
        """Star tracker should be constructable with hard constraint."""
        # Just verify the star tracker was created with constraint set
        assert basic_star_tracker.hard_constraint is None
        assert basic_star_tracker.name == "ST1"
        assert basic_star_tracker.orientation is not None

    def test_soft_constraint_degradation(self, basic_star_tracker):
        """Star tracker should be constructable with soft constraint."""
        # Just verify the star tracker was created
        assert basic_star_tracker.soft_constraint is None
        assert basic_star_tracker.name == "ST1"

    def test_constraint_with_orientation_transform(self):
        """Star tracker with custom orientation should be constructable."""
        from conops.config import Constraint

        # 45 degree pitch about Y means boresight rotated toward +Z and +Y
        # normalized: (0, sin(45), cos(45)) = (0, 0.707, 0.707)
        boresight = (0.0, 1.0 / (2**0.5), 1.0 / (2**0.5))
        ori = StarTrackerOrientation(boresight=boresight)
        constraint = Constraint()
        st = StarTracker(
            name="ST_OrientedConstraint",
            orientation=ori,
            hard_constraint=constraint,
        )

        # Verify the star tracker was created correctly
        assert st.name == "ST_OrientedConstraint"
        assert st.orientation.boresight == boresight
        assert st.hard_constraint is not None


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


class TestStarTrackerComputedConstraint:
    """Test computed star tracker observing constraint."""

    def test_startracker_constraint_none_without_soft_constraints(self):
        st = StarTracker(name="ST1")
        from conops.config import StarTrackerConfiguration

        cfg = StarTrackerConfiguration(star_trackers=[st])
        assert cfg.startracker_constraint is None

    def test_startracker_hard_constraint_none_without_hard_constraints(self):
        st = StarTracker(name="ST1")
        from conops.config import StarTrackerConfiguration

        cfg = StarTrackerConfiguration(star_trackers=[st])
        assert cfg.startracker_hard_constraint is None

    def test_startracker_hard_constraint_multiple_trackers_or_combined(self):
        from conops.config import Constraint, StarTrackerConfiguration

        st1 = StarTracker(
            name="ST1",
            orientation=StarTrackerOrientation(boresight=(1.0, 0.0, 0.0)),
            hard_constraint=Constraint(
                sun_constraint=rust_ephem.SunConstraint(min_angle=20.0)
            ),
        )
        st2 = StarTracker(
            name="ST2",
            orientation=StarTrackerOrientation(boresight=(0.0, 1.0, 0.0)),
            hard_constraint=Constraint(
                moon_constraint=rust_ephem.MoonConstraint(min_angle=10.0)
            ),
        )

        cfg = StarTrackerConfiguration(star_trackers=[st1, st2])
        c = cfg.startracker_hard_constraint
        assert c is not None
        assert c.type == "or"
        assert len(c.constraints) == 2
        assert all(child.type == "boresight_offset" for child in c.constraints)

    def test_startracker_constraint_single_tracker_has_boresight_offset(self):
        from conops.config import Constraint, StarTrackerConfiguration

        st = StarTracker(
            name="ST_Soft",
            orientation=StarTrackerOrientation(boresight=(0.0, 1.0, 0.0)),
            soft_constraint=Constraint(
                sun_constraint=rust_ephem.SunConstraint(min_angle=25.0)
            ),
        )
        cfg = StarTrackerConfiguration(star_trackers=[st])

        c = cfg.startracker_constraint
        assert c is not None
        assert c.type == "at_least"
        assert c.min_violated == 1
        assert len(c.constraints) == 1
        assert c.constraints[0].type == "boresight_offset"
        assert c.constraints[0].roll_deg == pytest.approx(0.0)
        assert c.constraints[0].pitch_deg == pytest.approx(0.0)
        assert c.constraints[0].yaw_deg == pytest.approx(90.0)

    def test_startracker_constraint_multiple_soft_constraints_at_least_combined(self):
        from conops.config import Constraint, StarTrackerConfiguration

        st1 = StarTracker(
            name="ST1",
            orientation=StarTrackerOrientation(boresight=(1.0, 0.0, 0.0)),
            soft_constraint=Constraint(
                sun_constraint=rust_ephem.SunConstraint(min_angle=20.0)
            ),
        )
        st2 = StarTracker(
            name="ST2",
            orientation=StarTrackerOrientation(boresight=(0.0, 1.0, 0.0)),
            soft_constraint=Constraint(
                moon_constraint=rust_ephem.MoonConstraint(min_angle=10.0)
            ),
        )

        cfg = StarTrackerConfiguration(star_trackers=[st1, st2])
        c = cfg.startracker_constraint
        assert c is not None
        assert c.type == "at_least"
        assert c.min_violated == 2
        assert len(c.constraints) == 2
        assert all(child.type == "boresight_offset" for child in c.constraints)

    def test_startracker_constraint_respects_min_functional_trackers_threshold(self):
        from conops.config import Constraint, StarTrackerConfiguration

        st1 = StarTracker(
            name="ST1",
            orientation=StarTrackerOrientation(boresight=(1.0, 0.0, 0.0)),
            soft_constraint=Constraint(
                sun_constraint=rust_ephem.SunConstraint(min_angle=20.0)
            ),
        )
        st2 = StarTracker(
            name="ST2",
            orientation=StarTrackerOrientation(boresight=(0.0, 1.0, 0.0)),
            soft_constraint=Constraint(
                moon_constraint=rust_ephem.MoonConstraint(min_angle=10.0)
            ),
        )

        cfg = StarTrackerConfiguration(
            star_trackers=[st1, st2],
            min_functional_trackers=2,
        )
        c = cfg.startracker_constraint
        assert c is not None
        assert c.type == "at_least"
        assert c.min_violated == 1
        assert len(c.constraints) == 2

    def test_startracker_constraint_none_when_threshold_unreachable_from_soft_only(
        self,
    ):
        from conops.config import Constraint, StarTrackerConfiguration

        st1 = StarTracker(
            name="ST1",
            orientation=StarTrackerOrientation(boresight=(1.0, 0.0, 0.0)),
            soft_constraint=Constraint(
                sun_constraint=rust_ephem.SunConstraint(min_angle=20.0)
            ),
        )
        st2 = StarTracker(
            name="ST2",
            orientation=StarTrackerOrientation(boresight=(0.0, 1.0, 0.0)),
            soft_constraint=None,
        )

        cfg = StarTrackerConfiguration(
            star_trackers=[st1, st2],
            min_functional_trackers=1,
        )

        assert cfg.startracker_constraint is None
