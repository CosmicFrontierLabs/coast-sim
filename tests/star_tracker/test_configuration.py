"""Tests for star tracker configuration."""

import time

from conops.config import (
    StarTracker,
    StarTrackerConfiguration,
    StarTrackerOrientation,
)


class TestStarTrackerConfiguration:
    """Test star tracker configuration management."""

    def test_empty_configuration(self, basic_config):
        """Empty configuration should have no trackers."""
        assert basic_config.num_trackers() == 0

    def test_single_tracker_configuration(self, config_single_tracker):
        """Single tracker configuration."""
        assert config_single_tracker.num_trackers() == 1
        assert config_single_tracker.star_trackers[0].name == "ST1"

    def test_multiple_trackers_configuration(self, config_dual_trackers):
        """Multiple tracker configuration."""
        assert config_dual_trackers.num_trackers() == 2
        assert config_dual_trackers.star_trackers[0].name == "ST1"
        assert config_dual_trackers.star_trackers[1].name == "ST2"

    def test_get_tracker_by_name_found(self, config_dual_trackers):
        """Get tracker by name when it exists."""
        st = config_dual_trackers.get_tracker_by_name("ST1")
        assert st is not None
        assert st.name == "ST1"

    def test_get_tracker_by_name_not_found(self, config_dual_trackers):
        """Get tracker by name when it doesn't exist."""
        st = config_dual_trackers.get_tracker_by_name("NonExistent")
        assert st is None

    def test_get_tracker_by_name_empty_config(self, basic_config):
        """Get tracker by name in empty configuration."""
        st = basic_config.get_tracker_by_name("ST1")
        assert st is None


class TestStarTrackerConfigurationValidation:
    """Test pointing validity checking."""

    def test_empty_config_always_valid(self, basic_config):
        """Empty configuration should allow all pointings."""
        current_time = time.time()
        assert basic_config.is_pointing_valid(0.0, 0.0, current_time)
        assert basic_config.is_pointing_valid(90.0, 45.0, current_time)
        assert basic_config.is_pointing_valid(180.0, -60.0, current_time)

    def test_single_tracker_no_constraints_valid(self, config_single_tracker):
        """Single tracker without constraints should allow all pointings."""
        current_time = time.time()
        assert config_single_tracker.is_pointing_valid(0.0, 0.0, current_time)
        assert config_single_tracker.is_pointing_valid(90.0, 45.0, current_time)
        assert config_single_tracker.is_pointing_valid(180.0, -60.0, current_time)

    def test_num_trackers_violating_constraints(self, config_single_tracker):
        """Count trackers violating constraints."""
        current_time = time.time()
        # No constraints, so 0 violations
        violations = config_single_tracker.trackers_violating_hard_constraints(
            0.0, 0.0, current_time
        )
        assert violations == 0

    def test_min_functional_trackers_requirement(
        self, config_triple_trackers_high_requirement
    ):
        """Test minimum functional trackers requirement."""
        config = config_triple_trackers_high_requirement
        assert config.num_trackers() == 3
        assert config.min_functional_trackers == 2

        current_time = time.time()
        # Without constraints, all 3 are functional, so should be valid
        assert config.is_pointing_valid(0.0, 0.0, current_time)

    def test_soft_constraint_checking(self, config_single_tracker):
        """Check soft constraint degradation flag."""
        config = config_single_tracker
        current_time = time.time()
        # No soft constraints, so no degradation
        assert not config.check_soft_constraint_degradation(0.0, 0.0, current_time)


class TestStarTrackerConfigurationConstraintInteraction:
    """Test interaction of multiple trackers with constraints."""

    def test_multiple_trackers_each_checked(self):
        """Each tracker should be independently checked."""

        # Create trackers with different orientations
        st1 = StarTracker(
            name="ST1",
            orientation=StarTrackerOrientation(roll=0.0, pitch=0.0, yaw=0.0),
        )
        st2 = StarTracker(
            name="ST2",
            orientation=StarTrackerOrientation(roll=45.0, pitch=0.0, yaw=0.0),
        )

        config = StarTrackerConfiguration(
            star_trackers=[st1, st2],
            min_functional_trackers=1,
        )

        current_time = time.time()
        # Both should check independently
        violations = config.trackers_violating_hard_constraints(0.0, 0.0, current_time)
        assert violations == 0  # No constraints, no violations

    def test_any_tracker_soft_constraint(self):
        """Check if ANY tracker violates soft constraints."""
        config = StarTrackerConfiguration(
            star_trackers=[
                StarTracker(name="ST1"),
                StarTracker(name="ST2"),
            ],
            min_functional_trackers=1,
        )

        current_time = time.time()
        # No soft constraints, so should return False
        result = config.any_tracker_violating_soft_constraints(0.0, 0.0, current_time)
        assert not result


class TestMinimumFunctionalTrackers:
    """Test minimum functional tracker requirements."""

    def test_min_one_of_one(self):
        """Minimum 1 tracker required, only 1 available."""
        config = StarTrackerConfiguration(
            star_trackers=[StarTracker(name="ST1")],
            min_functional_trackers=1,
        )
        current_time = time.time()
        assert config.is_pointing_valid(0.0, 0.0, current_time)

    def test_min_two_of_two(self):
        """Minimum 2 trackers required, 2 available."""
        config = StarTrackerConfiguration(
            star_trackers=[
                StarTracker(name="ST1"),
                StarTracker(name="ST2"),
            ],
            min_functional_trackers=2,
        )
        current_time = time.time()
        assert config.is_pointing_valid(0.0, 0.0, current_time)

    def test_min_two_of_three(self):
        """Minimum 2 trackers required, 3 available."""
        config = StarTrackerConfiguration(
            star_trackers=[
                StarTracker(name="ST1"),
                StarTracker(name="ST2"),
                StarTracker(name="ST3"),
            ],
            min_functional_trackers=2,
        )
        current_time = time.time()
        # All functional, should be valid
        assert config.is_pointing_valid(0.0, 0.0, current_time)

    def test_min_zero_trackers(self):
        """Minimum 0 trackers required (relaxed requirement)."""
        from conops.config import Constraint

        constraint = Constraint()
        config = StarTrackerConfiguration(
            star_trackers=[
                StarTracker(
                    name="ST1",
                    hard_constraint=constraint,
                ),
            ],
            min_functional_trackers=0,  # Allow all pointings
        )

        import rust_ephem

        config.star_trackers[0].hard_constraint.ephem = rust_ephem.Ephemeris.earth()

        current_time = time.time()
        # Should be valid even if constraint is violated
        # (because we require minimum 0 functional)
        assert config.is_pointing_valid(0.0, 0.0, current_time)
