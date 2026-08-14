"""Tests for preserving compound logic in roll-dependent constraints."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import rust_ephem
from rust_ephem.constraints import (
    AndConstraint,
    AtLeastConstraint,
    BoresightOffsetConstraint,
    NotConstraint,
    OrConstraint,
    XorConstraint,
)

from conops.common.enums import ACSMode
from conops.config import (
    Constraint,
    StarTracker,
    StarTrackerConfiguration,
    StarTrackerOrientation,
)
from conops.simulation.roll import _roll_valid_mask


def _offset(yaw_deg: float) -> BoresightOffsetConstraint:
    return rust_ephem.SunConstraint(min_angle=45.0).boresight_offset(yaw_deg=yaw_deg)


class TestRollDependentProjection:
    def test_retains_complete_mixed_compound_trees(self) -> None:
        first = _offset(45.0)
        second = _offset(-45.0)
        independent = rust_ephem.MoonConstraint(min_angle=10.0)
        cases = [
            AndConstraint(constraints=[first, independent]),
            OrConstraint(constraints=[first, independent]),
            NotConstraint(constraint=OrConstraint(constraints=[first, independent])),
            XorConstraint(constraints=[first, independent]),
            AtLeastConstraint(
                min_violated=2,
                constraints=[first, independent, second],
            ),
        ]

        for tree in cases:
            assert Constraint._roll_dependent_subtree(tree) is tree

    def test_omits_pure_roll_independent_tree(self) -> None:
        tree = NotConstraint(constraint=rust_ephem.MoonConstraint(min_angle=10.0))

        assert Constraint._roll_dependent_subtree(tree) is None


def _fixed_roll_validity(
    tree: AtLeastConstraint,
    ephem: rust_ephem.TLEEphemeris,
) -> np.ndarray:
    sample_time = ephem.timestamp[0]
    return np.array(
        [
            not bool(
                tree.in_constraint(
                    ephemeris=ephem,
                    target_ra=0.0,
                    target_dec=-60.0,
                    time=sample_time,
                    target_roll=float(roll),
                )
            )
            for roll in range(360)
        ],
        dtype=bool,
    )


def _assert_optimized_mask_matches_fixed_rolls(
    tree: AtLeastConstraint,
    ephem: rust_ephem.TLEEphemeris,
) -> np.ndarray:
    constraint = Constraint(
        star_tracker_soft_constraint=tree,
        star_tracker_enforce_modes=[ACSMode.SCIENCE],
        ignore_roll=True,
        ephem=ephem,
    )
    sample_utime = ephem.timestamp[0].timestamp()
    actual = _roll_valid_mask(0.0, -60.0, sample_utime, ephem, constraint)
    fixed_valid = _fixed_roll_validity(tree, ephem)

    if fixed_valid.all() or not fixed_valid.any():
        assert actual is None
    else:
        assert actual is not None
        np.testing.assert_array_equal(actual, fixed_valid)
    return fixed_valid


def _test_ephemeris() -> rust_ephem.TLEEphemeris:
    tle_path = Path(__file__).parents[2] / "examples" / "example.tle"
    return rust_ephem.TLEEphemeris(
        begin=datetime(2025, 12, 1, 0, 0, 0),
        end=datetime(2025, 12, 1, 1, 0, 0),
        step_size=60,
        tle=str(tle_path),
    )


@pytest.mark.parametrize(
    ("min_functional_trackers", "expected_threshold"),
    [(1, 2), (2, 1)],
    ids=("one-functional-tracker", "both-functional-trackers"),
)
def test_two_tracker_roll_mask_matches_fixed_roll_evaluation(
    min_functional_trackers: int,
    expected_threshold: int,
) -> None:
    ephem = _test_ephemeris()
    tracker_constraint = Constraint(
        sun_constraint=rust_ephem.SunConstraint(min_angle=45.0)
    )
    diagonal = 2**-0.5
    tracker_config = StarTrackerConfiguration(
        star_trackers=[
            StarTracker(
                name="STR_pY",
                orientation=StarTrackerOrientation(boresight=(diagonal, diagonal, 0.0)),
                soft_constraint=tracker_constraint,
            ),
            StarTracker(
                name="STR_nY",
                orientation=StarTrackerOrientation(
                    boresight=(diagonal, -diagonal, 0.0)
                ),
                soft_constraint=tracker_constraint,
            ),
        ],
        min_functional_trackers=min_functional_trackers,
        modes_require_lock=[ACSMode.SCIENCE],
    )
    tree = tracker_config.startracker_constraint
    assert isinstance(tree, AtLeastConstraint)
    assert tree.min_violated == expected_threshold

    _assert_optimized_mask_matches_fixed_rolls(tree, ephem)


@pytest.mark.parametrize(
    ("independent_constraint", "expected_violated", "expected_all_valid"),
    [
        (rust_ephem.SunConstraint(min_angle=0.0), False, True),
        (rust_ephem.SunConstraint(min_angle=179.0), True, False),
    ],
    ids=("independent-clear", "independent-violated"),
)
def test_mixed_at_least_roll_mask_matches_fixed_roll_evaluation(
    independent_constraint: rust_ephem.SunConstraint,
    expected_violated: bool,
    expected_all_valid: bool,
) -> None:
    ephem = _test_ephemeris()
    sample_time = ephem.timestamp[0]
    independent_violated = bool(
        independent_constraint.in_constraint(
            ephemeris=ephem,
            target_ra=0.0,
            target_dec=-60.0,
            time=sample_time,
            target_roll=0.0,
        )
    )
    assert independent_violated is expected_violated

    tree = AtLeastConstraint(
        min_violated=2,
        constraints=[
            _offset(45.0),
            independent_constraint,
            _offset(-45.0),
        ],
    )
    fixed_valid = _assert_optimized_mask_matches_fixed_rolls(tree, ephem)

    assert bool(fixed_valid.all()) is expected_all_valid
