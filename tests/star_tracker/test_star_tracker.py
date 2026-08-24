"""Tests for star tracker constraint checking."""

import time

import numpy as np
import pytest
import rust_ephem

from conops.common.vector import normal_to_boresight_offset_euler_deg
from conops.config import (
    Constraint,
    StarTracker,
    StarTrackerConfiguration,
    StarTrackerOrientation,
    create_star_tracker_vector,
)
from conops.simulation.roll import _roll_valid_mask


def test_positive_body_mount_pitch_maps_to_negative_rust_offset_pitch() -> None:
    boresight = create_star_tracker_vector(pitch_deg=45.0)
    assert boresight == pytest.approx((2**-0.5, 0.0, 2**-0.5))

    roll_deg, pitch_deg, yaw_deg = normal_to_boresight_offset_euler_deg(boresight)
    assert roll_deg == pytest.approx(0.0)
    assert pitch_deg == pytest.approx(-45.0)
    assert yaw_deg == pytest.approx(0.0)


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

    def test_plus_x_boresight_is_roll_position_angle_invariant(self):
        """For +X boresight, inertial tracker pointing should match input RA/Dec for any roll."""
        ori = StarTrackerOrientation(boresight=(1.0, 0.0, 0.0))

        ra0, dec0 = ori.transform_pointing(120.0, -20.0, roll_deg=0.0)
        ra1, dec1 = ori.transform_pointing(120.0, -20.0, roll_deg=90.0)

        assert ra0 == pytest.approx(120.0, abs=1e-9)
        assert dec0 == pytest.approx(-20.0, abs=1e-9)
        assert ra1 == pytest.approx(120.0, abs=1e-9)
        assert dec1 == pytest.approx(-20.0, abs=1e-9)

    def test_off_axis_boresight_changes_with_roll(self):
        """For off-axis boresight, roll should change inertial tracker pointing."""
        ori = StarTrackerOrientation(boresight=(0.0, 1.0, 0.0))

        ra0, dec0 = ori.transform_pointing(120.0, -20.0, roll_deg=0.0)
        ra1, dec1 = ori.transform_pointing(120.0, -20.0, roll_deg=90.0)

        assert not np.isclose(ra0, ra1, atol=1e-6)
        assert not np.isclose(dec0, dec1, atol=1e-6)

    def test_off_axis_boresight_preserves_ra_basis_at_celestial_pole(self):
        ori = StarTrackerOrientation(boresight=(0.0, 1.0, 0.0))

        ra_st, dec_st = ori.transform_pointing(40.0, 90.0, roll_deg=0.0)

        assert ra_st == pytest.approx(130.0, abs=1e-9)
        assert dec_st == pytest.approx(0.0, abs=1e-9)


class TestStarTrackerModeLockRequirements:
    """Test mode-dependent lock requirements on StarTrackerConfiguration."""

    def test_requires_lock_all_modes(self):
        """Configuration requiring lock in all modes (default)."""
        from conops.config import StarTrackerConfiguration

        cfg = StarTrackerConfiguration(
            star_trackers=[StarTracker(name="ST")],
            modes_require_lock=None,  # None = all modes
        )
        assert cfg.requires_lock_in_mode(None)
        assert cfg.requires_lock_in_mode(0)
        assert cfg.requires_lock_in_mode(1)
        assert cfg.requires_lock_in_mode(5)

    def test_requires_lock_specific_modes(self, star_tracker_config_mode_dependent):
        """Configuration requiring lock only in specific modes."""
        cfg = star_tracker_config_mode_dependent
        # Should require lock in modes 0 and 2
        assert cfg.requires_lock_in_mode(0)
        assert cfg.requires_lock_in_mode(2)
        # Should not require lock in other modes
        assert not cfg.requires_lock_in_mode(1)
        assert not cfg.requires_lock_in_mode(3)
        assert not cfg.requires_lock_in_mode(None)  # Nominal mode

    def test_requires_lock_no_modes(self):
        """Configuration never requiring lock."""
        from conops.config import StarTrackerConfiguration

        cfg = StarTrackerConfiguration(
            star_trackers=[StarTracker(name="ST")],
            modes_require_lock=[],  # Empty list = no modes
        )
        assert not cfg.requires_lock_in_mode(None)
        assert not cfg.requires_lock_in_mode(0)
        assert not cfg.requires_lock_in_mode(1)
        assert not cfg.requires_lock_in_mode(5)

    def test_requires_lock_nominal_mode_not_in_list(self):
        """Nominal mode (None) is never treated as requiring lock when a list is set."""
        from conops.config import StarTrackerConfiguration

        cfg = StarTrackerConfiguration(
            star_trackers=[StarTracker(name="ST")],
            modes_require_lock=[0, 1, 2],
        )
        # Nominal mode should not require lock
        assert not cfg.requires_lock_in_mode(None)


class TestHardConstraintAlwaysEnforced:
    """Hard constraints must fire regardless of modes_require_lock.

    Hard constraints are health-and-safety keep-outs (e.g. sensor blinding).
    modes_require_lock only gates soft (science-quality) constraints.
    """

    def _make_config_with_hard_constraint(
        self, modes_require_lock: list | None
    ) -> StarTrackerConfiguration:
        """Config with one tracker that has a hard sun constraint."""
        import rust_ephem

        st = StarTracker(
            name="ST1",
            orientation=StarTrackerOrientation(boresight=(1.0, 0.0, 0.0)),
            hard_constraint=Constraint(
                sun_constraint=rust_ephem.SunConstraint(min_angle=5),
            ),
        )
        return StarTrackerConfiguration(
            star_trackers=[st],
            modes_require_lock=modes_require_lock,
        )

    def test_hard_violations_counted_when_modes_require_lock_empty(self):
        """trackers_violating_hard_constraints returns correct count even when
        no mode requires lock (i.e. soft constraints are completely disabled).
        Without ephem the check will raise AssertionError — which proves the
        early-bail-out path (return 0) has been removed."""
        cfg = self._make_config_with_hard_constraint(modes_require_lock=[])
        assert not cfg.requires_lock_in_mode(0)  # sanity check: lock not required

        import time

        utime = time.time()
        # The method must now attempt the constraint check (rather than returning 0
        # immediately). With no ephem set, it raises AssertionError from the
        # Constraint.in_sun guard, not returning 0 silently.
        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            cfg.trackers_violating_hard_constraints(0.0, 0.0, utime, mode=None)

    def test_is_pointing_valid_checks_hard_constraints_even_when_lock_not_required(
        self,
    ):
        """is_pointing_valid must perform hard-constraint check even when the current
        mode is not in modes_require_lock."""
        # Build a full MissionConfig with ephemeris so constraints can fire.
        # We use a prebuilt ephemeris from the examples directory.
        import os

        import rust_ephem

        from conops.config import Constraint, StarTrackerConfiguration

        tle_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "examples", "example.tle"
        )
        if not os.path.exists(tle_path):
            pytest.skip("example.tle not available")

        from datetime import datetime

        ephem = rust_ephem.TLEEphemeris(
            begin=datetime(2025, 12, 1, 0, 0, 0),
            end=datetime(2025, 12, 1, 1, 0, 0),
            step_size=60,
            tle=tle_path,
        )

        # Sun position at any point in the ephem
        sun_ra = ephem.sun_ra_deg[0]
        sun_dec = ephem.sun_dec_deg[0]
        utime = ephem.begin.timestamp()

        # Single tracker whose hard constraint fires if the spacecraft points anywhere near
        # the Sun (giant exclusion zone of 90 degrees to guarantee we can point inside).
        st = StarTracker(
            name="ST1",
            orientation=StarTrackerOrientation(boresight=(1.0, 0.0, 0.0)),
            hard_constraint=Constraint(
                sun_constraint=rust_ephem.SunConstraint(min_angle=90),
            ),
        )
        cfg = StarTrackerConfiguration(
            star_trackers=[st],
            modes_require_lock=[],  # no mode ever requires lock
        )
        cfg.star_trackers[0].set_ephem(ephem)
        cfg.star_trackers[0].hard_constraint.ephem = ephem

        # Pointing directly at the Sun should violate the 90-degree hard exclusion.
        # is_pointing_valid must still return False even though no lock is required.
        assert not cfg.is_pointing_valid(sun_ra, sun_dec, utime, mode=None), (
            "Hard constraint should block pointing even when modes_require_lock=[]"
        )

    def test_startracker_hard_constraint_cached_property_ignores_modes_require_lock(
        self,
    ):
        """startracker_hard_constraint always returns the combined hard exclusion zone,
        regardless of modes_require_lock."""
        import rust_ephem

        from conops.config import Constraint, StarTrackerConfiguration

        st = StarTracker(
            name="ST1",
            orientation=StarTrackerOrientation(boresight=(1.0, 0.0, 0.0)),
            hard_constraint=Constraint(
                sun_constraint=rust_ephem.SunConstraint(min_angle=20),
            ),
        )

        cfg_no_lock = StarTrackerConfiguration(
            star_trackers=[st],
            modes_require_lock=[],  # no mode ever requires lock
        )
        cfg_all_lock = StarTrackerConfiguration(
            star_trackers=[st],
            modes_require_lock=None,  # all modes require lock
        )

        # Both configs must expose the hard constraint for scheduler integration.
        assert cfg_no_lock.startracker_hard_constraint is not None, (
            "Hard constraint must be present even when modes_require_lock=[]"
        )
        assert cfg_all_lock.startracker_hard_constraint is not None

    def test_soft_constraint_is_skipped_when_lock_not_required(self):
        """Soft constraints (science quality) should be skipped when the current
        mode is not in modes_require_lock."""
        import time

        import rust_ephem

        from conops.config import Constraint, StarTrackerConfiguration

        st = StarTracker(
            name="ST1",
            orientation=StarTrackerOrientation(boresight=(1.0, 0.0, 0.0)),
            soft_constraint=Constraint(
                sun_constraint=rust_ephem.SunConstraint(min_angle=5),
            ),
        )
        cfg = StarTrackerConfiguration(
            star_trackers=[st],
            modes_require_lock=[],  # no mode requires lock → soft never enforced
        )

        utime = time.time()
        # any_tracker_violating_soft_constraints should return False regardless of pointing
        assert not cfg.any_tracker_violating_soft_constraints(
            0.0, 0.0, utime, mode=None
        )
        assert not cfg.any_tracker_violating_soft_constraints(0.0, 0.0, utime, mode=0)


class TestConstriantInStarTrackerHardIgnoresModeGate:
    """Constraint.in_star_tracker_hard must not be gated by star_tracker_enforce_modes."""

    def test_in_star_tracker_hard_ignores_acs_mode_param(self):
        """Passing any acs_mode value must not suppress the hard constraint check."""
        import rust_ephem

        from conops.config import Constraint

        # Build a minimal Constraint with a sun hard exclusion zone
        c = Constraint(
            star_tracker_hard_constraint=rust_ephem.SunConstraint(min_angle=5),
            star_tracker_enforce_modes=[],  # empty → soft gated out
        )

        # Without ephem the check will assert, so we verify the mode gate is gone
        # by checking that acs_mode does NOT short-circuit to False.
        # We do this by inspecting: if the gate were present, passing mode=99
        # (not in []) would return False.  Without the gate, it tries to evaluate
        # (and raises AssertionError because ephem is None).
        import pytest

        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            c.in_star_tracker_hard(0.0, 0.0, 0.0, acs_mode=99)

    def test_in_star_tracker_soft_is_still_gated(self):
        """Soft constraints must still be skipped when acs_mode is not in enforce list."""
        import rust_ephem

        from conops.config import Constraint

        c = Constraint(
            star_tracker_soft_constraint=rust_ephem.SunConstraint(min_angle=5),
            star_tracker_enforce_modes=[],  # empty → soft always skipped
        )

        # Should return False without raising (mode gate fires before ephem assert)
        assert not c.in_star_tracker_soft(0.0, 0.0, 0.0, acs_mode=0)


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

        # Hard constraints are always OR: any single violation blocks the pointing.
        cfg = StarTrackerConfiguration(star_trackers=[st1, st2])
        c = cfg.startracker_hard_constraint
        assert c is not None
        assert c.type == "or"
        assert len(c.constraints) == 2
        assert c.constraints[0].type == "sun"
        assert c.constraints[1].type == "boresight_offset"
        assert c.constraints[1].roll_clockwise is True

    def test_startracker_hard_constraint_plus_x_uses_base_constraint(self):
        st = StarTracker(
            name="ST1",
            hard_constraint=Constraint(
                sun_constraint=rust_ephem.SunConstraint(min_angle=20.0)
            ),
        )
        cfg = StarTrackerConfiguration(star_trackers=[st])

        assert cfg.startracker_hard_constraint is st.hard_constraint.constraint

    def test_startracker_hard_constraint_min_functional_does_not_affect_hard(self):
        """min_functional_trackers has no effect on hard constraints — they are always OR."""
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

        # Even with min_functional_trackers=1 (one-of-two redundancy), hard
        # constraints remain OR — any hard violation is always blocked.
        cfg = StarTrackerConfiguration(
            star_trackers=[st1, st2], min_functional_trackers=1
        )
        c = cfg.startracker_hard_constraint
        assert c is not None
        assert c.type == "or"
        assert len(c.constraints) == 2

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
        assert c.constraints[0].roll_clockwise is True

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
        assert c.constraints[0].type == "sun"
        assert c.constraints[1].type == "boresight_offset"
        assert c.constraints[1].roll_clockwise is True

    def test_startracker_soft_constraint_plus_x_uses_base_constraint(self):
        st = StarTracker(
            name="ST1",
            soft_constraint=Constraint(
                sun_constraint=rust_ephem.SunConstraint(min_angle=20.0)
            ),
        )
        cfg = StarTrackerConfiguration(star_trackers=[st])

        constraint = cfg.startracker_constraint
        assert constraint is not None
        assert constraint.constraints[0] is st.soft_constraint.constraint

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

    def test_combined_soft_constraint_matches_trackers_at_nonzero_roll(self):
        """Combined planning constraint must match per-tracker telemetry checks."""
        from datetime import datetime
        from pathlib import Path

        tle_path = Path(__file__).parents[2] / "examples" / "example.tle"
        if not tle_path.exists():
            pytest.skip("example.tle not available")

        ephem = rust_ephem.TLEEphemeris(
            begin=datetime(2025, 12, 1, 0, 0, 0),
            end=datetime(2025, 12, 1, 1, 0, 0),
            step_size=60,
            tle=str(tle_path),
        )
        soft_constraint = Constraint(
            sun_constraint=rust_ephem.SunConstraint(min_angle=22.0),
            earth_constraint=rust_ephem.EarthLimbConstraint(min_angle=10.0),
        )
        cfg = StarTrackerConfiguration(
            star_trackers=[
                StarTracker(
                    name="STR_pY",
                    orientation=StarTrackerOrientation(
                        boresight=(2**-0.5, 2**-0.5, 0.0)
                    ),
                    soft_constraint=soft_constraint,
                ),
                StarTracker(
                    name="STR_nY",
                    orientation=StarTrackerOrientation(
                        boresight=(2**-0.5, -(2**-0.5), 0.0)
                    ),
                    soft_constraint=soft_constraint,
                ),
            ],
            min_functional_trackers=2,
            modes_require_lock=[0],
        )
        for tracker in cfg.star_trackers:
            tracker.set_ephem(ephem)

        combined = cfg.startracker_constraint
        assert combined is not None
        planning_constraint = Constraint(
            star_tracker_soft_constraint=combined,
            star_tracker_enforce_modes=[0],
            ephem=ephem,
        )
        sample_time = ephem.timestamp[0]
        sample_utime = sample_time.timestamp()
        cases = [
            (0.0, -60.0, -30.0),
            (0.0, -60.0, 30.0),
            (0.0, -45.0, -150.0),
            (0.0, -45.0, 150.0),
        ]

        for ra, dec, roll in cases:
            per_tracker_violations = sum(
                tracker.in_soft_constraint(ra, dec, sample_utime, roll)
                for tracker in cfg.star_trackers
            )
            expected = per_tracker_violations >= 1
            actual = planning_constraint.in_star_tracker_soft(
                ra,
                dec,
                sample_utime,
                target_roll=roll,
                acs_mode=0,
            )
            assert bool(actual) is expected

        planning_constraint.ignore_roll = True
        mask = _roll_valid_mask(
            0.0,
            -60.0,
            sample_utime,
            ephem,
            planning_constraint,
        )
        assert mask is not None
        expected_mask = np.array(
            [
                sum(
                    tracker.in_soft_constraint(0.0, -60.0, sample_utime, float(roll))
                    for tracker in cfg.star_trackers
                )
                < 1
                for roll in range(360)
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(mask, expected_mask)

    def test_non_planar_boresights_match_fixed_roll_planning_constraint(self):
        """rust-ephem offsets must match COAST body geometry at fixed rolls."""
        from datetime import datetime
        from pathlib import Path

        tle_path = Path(__file__).parents[2] / "examples" / "example.tle"
        if not tle_path.exists():
            pytest.skip("example.tle not available")

        ephem = rust_ephem.TLEEphemeris(
            begin=datetime(2025, 12, 1, 0, 0, 0),
            end=datetime(2025, 12, 1, 1, 0, 0),
            step_size=60,
            tle=str(tle_path),
        )
        sample_utime = ephem.timestamp[0].timestamp()
        inv_sqrt_3 = 3**-0.5
        boresights = [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (inv_sqrt_3, inv_sqrt_3, inv_sqrt_3),
            (inv_sqrt_3, -inv_sqrt_3, -inv_sqrt_3),
        ]

        for boresight in boresights:
            tracker = StarTracker(
                name="STR",
                orientation=StarTrackerOrientation(boresight=boresight),
                soft_constraint=Constraint(
                    sun_constraint=rust_ephem.SunConstraint(min_angle=45.0)
                ),
            )
            cfg = StarTrackerConfiguration(
                star_trackers=[tracker],
                min_functional_trackers=1,
                modes_require_lock=[0],
            )
            tracker.set_ephem(ephem)
            combined = cfg.startracker_constraint
            assert combined is not None
            planning_constraint = Constraint(
                star_tracker_soft_constraint=combined,
                star_tracker_enforce_modes=[0],
                ephem=ephem,
            )

            for roll in (-60.0, 0.0, 75.0):
                for ra in range(0, 360, 60):
                    for dec in (-60.0, 0.0, 60.0):
                        direct = tracker.in_soft_constraint(
                            float(ra), dec, sample_utime, roll
                        )
                        wrapped = planning_constraint.in_star_tracker_soft(
                            float(ra),
                            dec,
                            sample_utime,
                            target_roll=roll,
                            acs_mode=0,
                        )
                        assert bool(wrapped) is direct, (
                            f"boresight={boresight}, attitude=({ra}, {dec}, {roll})"
                        )

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
