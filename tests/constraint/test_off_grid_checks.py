"""Constraint checks between ephemeris samples are made at the sample at or
before the time, since rust_ephem evaluates only at its samples."""

from datetime import datetime, timedelta, timezone

import pytest
import rust_ephem
from rust_ephem import TLEEphemeris
from rust_ephem.constraints import BodyConstraint

from conops.config import Constraint, SolarPanel, SolarPanelSet
from conops.simulation.roll import optimum_body_roll

TLE = "examples/example.tle"
BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
STEP = 60
WEST_PATCH = [(-26.0, -5.0), (-16.0, -5.0), (-16.0, 5.0), (-26.0, 5.0)]


@pytest.fixture(scope="module")
def ephem() -> TLEEphemeris:
    return TLEEphemeris(
        tle=TLE, begin=BEGIN, end=BEGIN + timedelta(hours=2), step_size=STEP
    )


def _constraint(ephem: TLEEphemeris, **fields: object) -> Constraint:
    constraint = Constraint(**fields)  # type: ignore[arg-type]
    constraint.ephem = ephem
    return constraint


class TestSampleTime:
    @pytest.mark.parametrize("offset", [0.0, 1.0, 29.0, 31.0, 45.0, 59.0])
    def test_sample_at_or_before(self, ephem: TLEEphemeris, offset: float) -> None:
        constraint = _constraint(ephem)
        sample_utime = BEGIN.timestamp() + 10 * STEP
        assert constraint._sample_time(sample_utime + offset) == ephem.timestamp[10]

    def test_before_the_first_sample(self, ephem: TLEEphemeris) -> None:
        constraint = _constraint(ephem)
        assert constraint._sample_time(BEGIN.timestamp() - 5.0) == ephem.timestamp[0]

    def test_a_double_is_asked_at_the_time(self) -> None:
        from unittest.mock import Mock

        constraint = Constraint()
        constraint.ephem = Mock()
        assert constraint._sample_time(1700000000.0) == datetime.fromtimestamp(
            1700000000.0, tz=timezone.utc
        )


class TestChecksBetweenSamples:
    def test_circular_check_between_samples(self, ephem: TLEEphemeris) -> None:
        constraint = _constraint(
            ephem, sun_constraint=rust_ephem.SunConstraint(min_angle=45.0)
        )
        sun_ra, sun_dec = float(ephem.sun_ra_deg[5]), float(ephem.sun_dec_deg[5])
        on_grid = BEGIN.timestamp() + 5 * STEP
        assert constraint.in_sun(sun_ra, sun_dec, on_grid)
        assert constraint.in_sun(sun_ra, sun_dec, on_grid + 17.0)
        assert not constraint.in_sun(sun_ra + 180.0, -sun_dec, on_grid + 17.0)

    def test_polygon_check_between_samples(self, ephem: TLEEphemeris) -> None:
        constraint = _constraint(
            ephem, science_constraint=BodyConstraint(body="Sun", fov_polygon=WEST_PATCH)
        )
        sun_ra, sun_dec = float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0])
        ra, dec = sun_ra, sun_dec - 20.0
        off_grid = BEGIN.timestamp() + 16.0
        assert constraint.in_explicit_science(ra, dec, off_grid, target_roll=90.0)
        assert not constraint.in_explicit_science(ra, dec, off_grid, target_roll=0.0)

    def test_batch_check_between_samples(self, ephem: TLEEphemeris) -> None:
        constraint = _constraint(
            ephem, sun_constraint=rust_ephem.SunConstraint(min_angle=45.0)
        )
        sun_ra, sun_dec = float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0])
        violations = constraint.in_imaging_quality_constraint_batch(
            [sun_ra, sun_ra + 180.0], [sun_dec, -sun_dec], BEGIN.timestamp() + 16.0
        )
        assert list(violations) == [True, False]

    def test_roll_choice_at_the_end_of_a_slew(self, ephem: TLEEphemeris) -> None:
        """Dispatch chooses a candidate's roll at the end of its slew, which
        rarely falls on an ephemeris sample."""
        constraint = _constraint(
            ephem, science_constraint=BodyConstraint(body="Sun", fov_polygon=WEST_PATCH)
        )
        sun_ra, sun_dec = float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0])
        ra, dec = sun_ra, sun_dec - 20.0
        panels = SolarPanelSet(
            panels=[
                SolarPanel(
                    normal=(0.0, 1.0, 0.0), conversion_efficiency=0.3, max_power=800.0
                )
            ],
            conversion_efficiency=0.3,
        )
        off_grid = BEGIN.timestamp() + 16.0
        roll = optimum_body_roll(ra, dec, off_grid, ephem, panels, constraint)
        assert not constraint.in_explicit_science(ra, dec, off_grid, target_roll=roll)
        assert abs(roll - 90.0) == pytest.approx(15.0, abs=1.0)
