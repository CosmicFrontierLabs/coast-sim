"""Pytest fixtures for star tracker tests."""

import pytest

from conops.config import (
    Constraint,
    StarTracker,
    StarTrackerConfiguration,
    StarTrackerOrientation,
)


@pytest.fixture
def basic_orientation():
    """Basic star tracker orientation (boresight along +X)."""
    return StarTrackerOrientation(boresight=(1.0, 0.0, 0.0))


@pytest.fixture
def rotated_orientation():
    """Star tracker rotated to point toward +Y direction."""
    # 90 degree pitch (rotation about Y) gives: (0, 1, 0)
    return StarTrackerOrientation(boresight=(0.0, 1.0, 0.0))


@pytest.fixture
def basic_constraint():
    """A basic constraint for testing."""
    return Constraint()


@pytest.fixture
def basic_star_tracker(basic_orientation):
    """Basic star tracker with default orientation."""
    return StarTracker(
        name="ST1",
        orientation=basic_orientation,
    )


@pytest.fixture
def star_tracker_with_hard_constraint(basic_orientation, basic_constraint):
    """Star tracker with hard constraint."""
    return StarTracker(
        name="ST_Hard",
        orientation=basic_orientation,
        hard_constraint=basic_constraint,
    )


@pytest.fixture
def star_tracker_with_soft_constraint(basic_orientation, basic_constraint):
    """Star tracker with soft constraint."""
    return StarTracker(
        name="ST_Soft",
        orientation=basic_orientation,
        soft_constraint=basic_constraint,
    )


@pytest.fixture
def star_tracker_mode_dependent(basic_orientation):
    """Star tracker with mode-dependent lock requirements."""
    return StarTracker(
        name="ST_ModeDependent",
        orientation=basic_orientation,
        modes_require_lock=[0, 2],  # Lock required in modes 0 and 2
    )


@pytest.fixture
def basic_config():
    """Basic star tracker configuration with no trackers."""
    return StarTrackerConfiguration(star_trackers=[])


@pytest.fixture
def config_single_tracker(basic_star_tracker):
    """Configuration with single star tracker."""
    return StarTrackerConfiguration(
        star_trackers=[basic_star_tracker],
        min_functional_trackers=1,
    )


@pytest.fixture
def config_dual_trackers(basic_star_tracker, rotated_orientation):
    """Configuration with two star trackers."""
    st2 = StarTracker(
        name="ST2",
        orientation=rotated_orientation,
    )
    return StarTrackerConfiguration(
        star_trackers=[basic_star_tracker, st2],
        min_functional_trackers=1,
    )


@pytest.fixture
def config_triple_trackers_high_requirement(basic_star_tracker, rotated_orientation):
    """Configuration with three star trackers requiring minimum of 2 functional."""
    st2 = StarTracker(
        name="ST2",
        orientation=rotated_orientation,
    )
    # 90 degree roll about X (rotation around boresight axis) - boresight stays (1,0,0)
    st3 = StarTracker(
        name="ST3",
        orientation=StarTrackerOrientation(boresight=(1.0, 0.0, 0.0)),
    )
    return StarTrackerConfiguration(
        star_trackers=[basic_star_tracker, st2, st3],
        min_functional_trackers=2,
    )
