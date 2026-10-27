"""The thermal settling table: the settle after a slew by how far the slew
moves the Sun's direction in the body frame."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError
from rust_ephem import TLEEphemeris

from conops.config import (
    AttitudeControlSystem,
    Constraint,
    MissionConfig,
    ThermalSettlingStep,
)
from conops.ditl.queue_ditl import QueueDITL
from conops.schedulers.context import SchedulingContext
from conops.simulation.attitude import AttitudeExecutionError, AttitudeTrajectory
from conops.simulation.slew import Slew
from conops.targets import PlanEntry

TLE = "examples/example.tle"
BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
UTIME = BEGIN.timestamp()
TABLE = [
    ThermalSettlingStep(max_sun_vector_change_deg=10.0, settle_seconds=20.0),
    ThermalSettlingStep(max_sun_vector_change_deg=60.0, settle_seconds=200.0),
]


@pytest.fixture(scope="module")
def ephem() -> TLEEphemeris:
    return TLEEphemeris(
        tle=TLE, begin=BEGIN, end=BEGIN + timedelta(hours=2), step_size=60
    )


def _acs(table: list[ThermalSettlingStep] | None = None) -> AttitudeControlSystem:
    return AttitudeControlSystem(
        max_slew_rate=1.0,
        slew_acceleration=0.5,
        settle_time=120.0,
        thermal_settling=table or [],
    )


def _config(ephem: TLEEphemeris, acs: AttitudeControlSystem) -> MissionConfig:
    config = MissionConfig()
    config.spacecraft_bus.attitude_control = acs
    config.constraint = Constraint()
    config.constraint.ephem = ephem
    return config


def _slew(
    config: MissionConfig,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> Slew:
    slew = Slew(config=config)
    slew.slewrequest = slew.slewstart = UTIME
    slew.startra, slew.startdec, slew.startroll = start
    slew.endra, slew.enddec, slew.endroll = end
    slew.calc_slewtime()
    return slew


class TestTable:
    def test_steps_sorted_and_distinct(self) -> None:
        acs = _acs(list(reversed(TABLE)))
        assert [s.max_sun_vector_change_deg for s in acs.thermal_settling] == [
            10.0,
            60.0,
        ]
        with pytest.raises(ValidationError, match="distinct angles"):
            _acs([TABLE[0], TABLE[0]])

    def test_settle_by_change(self) -> None:
        acs = _acs(TABLE)
        assert acs.settle_seconds(0.0) == 20.0
        assert acs.settle_seconds(10.0) == 20.0
        assert acs.settle_seconds(10.5) == 200.0
        assert acs.settle_seconds(150.0) == 200.0
        assert acs.max_settle_seconds == 200.0
        assert _acs().settle_seconds(150.0) == 120.0
        assert _acs().max_settle_seconds == 120.0

    def test_slew_time_takes_the_change(self) -> None:
        acs = _acs(TABLE)
        motion = acs.motion_time(30.0)
        assert acs.slew_time(30.0) == pytest.approx(motion + 120.0)
        assert acs.slew_time(30.0, None, 5.0) == pytest.approx(motion + 20.0)
        assert acs.slew_time(30.0, None, 90.0) == pytest.approx(motion + 200.0)
        assert acs.slew_time(0.0, None, 90.0) == 0.0


class TestSlew:
    def test_sun_kept_on_the_boresight_settles_little(
        self, ephem: TLEEphemeris
    ) -> None:
        config = _config(ephem, _acs(TABLE))
        sun = (float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0]))
        slew = _slew(config, (*sun, 0.0), (*sun, 90.0))
        assert slew.sun_vector_change_deg == pytest.approx(0.0, abs=1e-6)
        assert slew.settle_seconds == 20.0
        assert slew.slewtime == int(
            np.ceil(
                config.spacecraft_bus.attitude_control.motion_time(slew.slewdist) + 20.0
            )
        )

    def test_sun_turned_through_ninety_degrees_settles_long(
        self, ephem: TLEEphemeris
    ) -> None:
        config = _config(ephem, _acs(TABLE))
        sun_ra, sun_dec = float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0])
        slew = _slew(config, (sun_ra, sun_dec, 0.0), (sun_ra, sun_dec - 90.0, 0.0))
        assert slew.sun_vector_change_deg == pytest.approx(90.0, abs=1e-6)
        assert slew.settle_seconds == 200.0
        motion = config.spacecraft_bus.attitude_control.motion_time(slew.slewdist)
        assert slew.slewtime == int(np.ceil(motion + 200.0))

    def test_without_a_table(self, ephem: TLEEphemeris) -> None:
        config = _config(ephem, _acs())
        sun_ra, sun_dec = float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0])
        slew = _slew(config, (sun_ra, sun_dec, 0.0), (sun_ra, sun_dec - 90.0, 0.0))
        assert slew.sun_vector_change_deg == 0.0
        assert slew.settle_seconds == 120.0
        motion = config.spacecraft_bus.attitude_control.motion_time(slew.slewdist)
        assert slew.slewtime == int(np.ceil(motion + 120.0))

    def test_no_distance_no_settle(self, ephem: TLEEphemeris) -> None:
        config = _config(ephem, _acs(TABLE))
        slew = _slew(config, (10.0, 20.0, 30.0), (10.0, 20.0, 30.0))
        assert slew.settle_seconds == 0.0
        assert slew.slewtime == 0

    def test_duration_bound_covers_the_longest_settle(
        self, ephem: TLEEphemeris
    ) -> None:
        acs = _acs(TABLE)
        config = _config(ephem, acs)
        sun_ra, sun_dec = float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0])
        slew = _slew(config, (sun_ra, sun_dec, 0.0), (sun_ra + 180.0, -sun_dec, 0.0))
        assert slew.slewtime <= Slew.duration_upper_bound(acs)
        assert Slew.duration_upper_bound(acs) == Slew.duration_upper_bound(_acs()) + 80


class TestExecutor:
    def test_short_settle_from_the_table_is_accepted(self, ephem: TLEEphemeris) -> None:
        acs = _acs(TABLE)
        config = _config(ephem, acs)
        sun = (float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0]))
        slew = _slew(config, (*sun, 0.0), (*sun, 90.0))
        assert slew.settle_seconds == 20.0 < acs.settle_time
        AttitudeTrajectory.from_slew(slew, acs)
        # Cut short of its own settle, the slew is rejected.
        slew.slewend = slew.slewstart + slew.slewtime - 15.0
        with pytest.raises(AttitudeExecutionError, match="settling"):
            AttitudeTrajectory.from_slew(slew, acs)


class TestRecording:
    def test_plan_entry_takes_both_fields(self, ephem: TLEEphemeris) -> None:
        config = _config(ephem, _acs(TABLE))
        sun_ra, sun_dec = float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0])
        slew = _slew(config, (sun_ra, sun_dec, 0.0), (sun_ra, sun_dec - 90.0, 0.0))
        entry = PlanEntry(
            ra=sun_ra, dec=sun_dec - 90.0, roll=0.0, begin=UTIME, end=UTIME + 1000
        )
        QueueDITL._apply_slew_metadata(entry, slew)
        assert entry.sun_vector_change_deg == pytest.approx(90.0, abs=1e-6)
        assert entry.settle_seconds == 200.0
        assert "settle_seconds" in entry.model_dump()
        assert "sun_vector_change_deg" in entry.model_dump()

    def test_a_slew_sampled_together_matches_each_sample(
        self, ephem: TLEEphemeris
    ) -> None:
        """A battery ledger samples a slew's path at every step: the
        attitudes sampled together are the attitudes sampled one by one,
        before the slew starts, along it and after it ends."""
        config = _config(ephem, _acs(TABLE))
        slew = _slew(config, (10.0, 20.0, 30.0), (80.0, -10.0, 120.0))
        slew.slewstart = UTIME
        slew.calc_slewtime()
        times = [UTIME - 60.0, *np.arange(UTIME, UTIME + slew.slewtime + 120.0, 60.0)]
        together = slew.attitudes(times)
        assert len(together) == len(times)
        for utime, sampled in zip(times, together, strict=True):
            assert sampled == pytest.approx(slew.attitude(float(utime)), abs=1e-9)
        assert together[0] == (10.0, 20.0, 30.0)
        assert together[-1] == pytest.approx((80.0, -10.0, 120.0), abs=1e-9)

    def test_context_retimes_a_cached_path_under_a_table(
        self, ephem: TLEEphemeris
    ) -> None:
        """With a table the settle follows the Sun, so a cached slew is
        retimed along its path for each start: the slew returned is the one
        a fresh computation gives at that start, and the path is predicted
        once per pair of attitudes."""
        end = BEGIN + timedelta(hours=2)
        config = _config(ephem, _acs(TABLE))
        ctx = SchedulingContext(config, BEGIN, end)
        start, finish = (10.0, 20.0, 30.0), (80.0, -10.0, 120.0)
        predicted = 0
        original = Slew.predict_slew

        def counting(self: Slew) -> None:
            nonlocal predicted
            predicted += 1
            original(self)

        with patch.object(Slew, "predict_slew", counting):
            slews = [ctx.slew(start, finish, UTIME + k * 1800) for k in range(4)]
        assert predicted == 1
        for k, slew in enumerate(slews):
            fresh = _slew(config, start, finish)
            fresh.slewstart = UTIME + k * 1800
            fresh.calc_slewtime()
            assert slew.slewstart == fresh.slewstart
            assert slew.slewtime == fresh.slewtime
            assert slew.slewend == fresh.slewend
            assert slew.settle_seconds == fresh.settle_seconds
            assert slew.sun_vector_change_deg == pytest.approx(
                fresh.sun_vector_change_deg
            )
            assert slew.slewdist == pytest.approx(fresh.slewdist)
        assert slews[0].sun_vector_change_deg > 0.0
