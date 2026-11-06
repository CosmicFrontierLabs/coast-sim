"""A roll-dependent explicit science constraint, such as a stray-light patch
written as a rust_ephem field-of-view polygon, steers the roll optimiser."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import rust_ephem
from rust_ephem import TLEEphemeris
from rust_ephem.constraints import BodyConstraint, BrightStarConstraint

from conops.config import Constraint, SolarPanel, SolarPanelSet, Telescope
from conops.config.constraint import imaging_quality_attitude_constraint_names
from conops.simulation.roll import optimum_body_roll, optimum_instrument_roll

TLE = "examples/example.tle"
BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
UTIME = BEGIN.timestamp()

# A patch 20 degrees from the boresight towards -u, 10 degrees square, in
# the gnomonic degrees rust_ephem's polygons use (tan 20 degrees is 20.85).
WEST_PATCH = [(-26.0, -5.0), (-16.0, -5.0), (-16.0, 5.0), (-26.0, 5.0)]
# A patch that holds the Sun 20 degrees off at every roll.
EVERYWHERE = [(-30.0, -30.0), (30.0, -30.0), (30.0, 30.0), (-30.0, 30.0)]


@pytest.fixture(scope="module")
def ephem() -> TLEEphemeris:
    return TLEEphemeris(
        tle=TLE, begin=BEGIN, end=BEGIN + timedelta(hours=2), step_size=60
    )


def _sun_north(ephem: TLEEphemeris) -> tuple[float, float]:
    """A boresight with the Sun 20 degrees due north of it."""
    return float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0]) - 20.0


def _side_panel() -> SolarPanelSet:
    return SolarPanelSet(
        panels=[
            SolarPanel(
                normal=(0.0, 1.0, 0.0), conversion_efficiency=0.3, max_power=800.0
            )
        ],
        conversion_efficiency=0.3,
    )


def _constraint(ephem: TLEEphemeris, science: object = None) -> Constraint:
    constraint = Constraint(science_constraint=science)
    constraint.ephem = ephem
    return constraint


class TestRollDependence:
    def test_polygon_and_offset_depend_on_roll(self) -> None:
        polygon = BodyConstraint(body="Sun", fov_polygon=WEST_PATCH)
        assert Constraint(science_constraint=polygon).science_constraint_roll_dependent
        stars = BrightStarConstraint(stars=[(10.0, 10.0)], fov_polygon=WEST_PATCH)
        assert Constraint(
            science_constraint=rust_ephem.SunConstraint(min_angle=30.0) | stars
        ).science_constraint_roll_dependent
        offset = rust_ephem.SunConstraint(min_angle=45.0).boresight_offset(yaw_deg=30.0)
        assert Constraint(science_constraint=~offset).science_constraint_roll_dependent

    def test_plain_constraint_does_not(self) -> None:
        assert not Constraint().science_constraint_roll_dependent
        plain = rust_ephem.SunConstraint(min_angle=30.0) & rust_ephem.MoonConstraint(
            min_angle=10.0
        )
        assert not Constraint(
            science_constraint=plain
        ).science_constraint_roll_dependent
        circle = BrightStarConstraint(stars=[(10.0, 10.0)], fov_radius=2.0)
        assert not Constraint(
            science_constraint=circle
        ).science_constraint_roll_dependent

    def test_invalidated_with_the_combined_trees(self) -> None:
        constraint = Constraint()
        assert not constraint.science_constraint_roll_dependent
        constraint.science_constraint = BodyConstraint(
            body="Sun", fov_polygon=WEST_PATCH
        )
        constraint.invalidate_combined_constraint_cache()
        assert constraint.science_constraint_roll_dependent


class TestBodyRoll:
    """With the Sun due north and a +Y panel the power-optimal body roll is
    90 degrees, which puts the Sun 20 degrees towards -u in the instrument
    frame: on the west patch."""

    def test_first_clear_roll_in_power_order(self, ephem: TLEEphemeris) -> None:
        ra, dec = _sun_north(ephem)
        assert optimum_body_roll(
            ra, dec, UTIME, ephem, _side_panel(), _constraint(ephem)
        ) == pytest.approx(90.0)
        constraint = _constraint(
            ephem, BodyConstraint(body="Sun", fov_polygon=WEST_PATCH)
        )
        assert constraint.in_explicit_science(ra, dec, UTIME, target_roll=90.0)
        roll = optimum_body_roll(ra, dec, UTIME, ephem, _side_panel(), constraint)
        assert not constraint.in_explicit_science(ra, dec, UTIME, target_roll=roll)
        # The nearest clear roll on either side of the optimum.
        assert abs(roll - 90.0) == pytest.approx(15.0, abs=1.0)
        assert imaging_quality_attitude_constraint_names(
            constraint, ra, dec, UTIME, target_roll=90.0
        ) == ["Science"]
        assert (
            imaging_quality_attitude_constraint_names(
                constraint, ra, dec, UTIME, target_roll=roll
            )
            == []
        )

    def test_without_panels(self, ephem: TLEEphemeris) -> None:
        ra, dec = _sun_north(ephem)
        constraint = _constraint(
            ephem, BodyConstraint(body="Sun", fov_polygon=WEST_PATCH)
        )
        roll = optimum_body_roll(ra, dec, UTIME, ephem, None, constraint)
        assert not constraint.in_explicit_science(ra, dec, UTIME, target_roll=roll)

    def test_other_attitudes_keep_the_power_optimum(self, ephem: TLEEphemeris) -> None:
        ra, dec = _sun_north(ephem)
        constraint = _constraint(
            ephem, BodyConstraint(body="Sun", fov_polygon=WEST_PATCH)
        )
        roll = optimum_body_roll(
            ra, dec, UTIME, ephem, _side_panel(), constraint, science_attitude=False
        )
        assert roll == pytest.approx(90.0)

    def test_no_clear_roll_falls_back_to_the_optimum(self, ephem: TLEEphemeris) -> None:
        ra, dec = _sun_north(ephem)
        constraint = _constraint(
            ephem, BodyConstraint(body="Sun", fov_polygon=EVERYWHERE)
        )
        roll = optimum_body_roll(ra, dec, UTIME, ephem, _side_panel(), constraint)
        assert roll == pytest.approx(90.0)

    def test_reachable_rolls_bound_the_search(self, ephem: TLEEphemeris) -> None:
        ra, dec = _sun_north(ephem)
        constraint = _constraint(
            ephem, BodyConstraint(body="Sun", fov_polygon=WEST_PATCH)
        )
        roll = optimum_body_roll(
            ra,
            dec,
            UTIME,
            ephem,
            _side_panel(),
            constraint,
            reference_roll=100.0,
            max_roll_delta=10.0,
        )
        assert 90.0 <= roll <= 110.0
        assert not constraint.in_explicit_science(ra, dec, UTIME, target_roll=roll)

    def test_bright_star_patch(self, ephem: TLEEphemeris) -> None:
        # A star 20 degrees due north of the boresight, and the west patch:
        # the star is on it when the +v axis points east, at roll 90.
        constraint = _constraint(
            ephem, BrightStarConstraint(stars=[(40.0, 30.0)], fov_polygon=WEST_PATCH)
        )
        assert constraint.in_explicit_science(40.0, 10.0, UTIME, target_roll=90.0)
        roll = optimum_body_roll(40.0, 10.0, UTIME, ephem, _side_panel(), constraint)
        assert not constraint.in_explicit_science(40.0, 10.0, UTIME, target_roll=roll)


def _turned(
    polygon: list[tuple[float, float]], angle_deg: float
) -> list[tuple[float, float]]:
    """The polygon turned about the boresight by ``angle_deg``."""
    angle = np.radians(angle_deg)
    return [
        (
            float(u * np.cos(angle) + v * np.sin(angle)),
            float(-u * np.sin(angle) + v * np.cos(angle)),
        )
        for u, v in polygon
    ]


class TestMountedRoll:
    def test_mounted_telescope_clears_the_patch(self, ephem: TLEEphemeris) -> None:
        ra, dec = _sun_north(ephem)
        telescope = Telescope(boresight=(0.0, 1.0, 0.0))
        unguarded = optimum_instrument_roll(
            ra, dec, UTIME, ephem, telescope, _side_panel(), _constraint(ephem)
        )
        # A patch at +v, turned to wherever the unguarded roll puts the Sun.
        north_patch = [(-5.0, 16.0), (5.0, 16.0), (5.0, 26.0), (-5.0, 26.0)]
        for angle in range(0, 360, 5):
            constraint = _constraint(
                ephem,
                BodyConstraint(body="Sun", fov_polygon=_turned(north_patch, angle)),
            )
            if constraint.in_explicit_science(ra, dec, UTIME, target_roll=unguarded):
                break
        else:
            pytest.fail("no turning of the patch holds the Sun at the unguarded roll")
        roll = optimum_instrument_roll(
            ra, dec, UTIME, ephem, telescope, _side_panel(), constraint
        )
        assert roll != unguarded
        assert not constraint.in_explicit_science(ra, dec, UTIME, target_roll=roll)
