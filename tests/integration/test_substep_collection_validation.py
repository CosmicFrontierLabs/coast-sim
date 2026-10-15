"""Audit short replay entries against collection from the real ACS executor."""

from datetime import timedelta
from unittest.mock import Mock

import pytest
import rust_ephem

from conops import DITL
from conops.common import ACSMode, ObsType
from conops.config import (
    AttitudeControlSystem,
    GroundStationRegistry,
    MissionConfig,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
)
from conops.targets import Plan, PlanEntry
from scripts.check_default_plan_output import (
    SCENARIO_BEGIN,
    DeterministicConstraint,
    DeterministicEphemeris,
)


@pytest.fixture
def replay_factory(monkeypatch):
    monkeypatch.setattr(
        rust_ephem,
        "get_eop_provenance",
        lambda: {"ut1": {"available": False}, "polar_motion": {"available": False}},
    )
    monkeypatch.setattr(
        "conops.config.solar_panel._get_eclipse_constraint",
        lambda: Mock(in_constraint=Mock(return_value=False)),
    )

    def make(entries, *, slew_rate=10, acceleration=10, settle_time=0):
        begin = SCENARIO_BEGIN
        end = begin + timedelta(seconds=240)
        ephem = DeterministicEphemeris(begin, end, step_size_seconds=60)
        config = MissionConfig(
            constraint=DeterministicConstraint(),
            ground_stations=GroundStationRegistry(stations=[]),
            solar_panel=SolarPanelSet(panels=[]),
            spacecraft_bus=SpacecraftBus(
                attitude_control=AttitudeControlSystem(
                    max_slew_rate=slew_rate,
                    slew_acceleration=acceleration,
                    settle_time=settle_time,
                ),
                star_trackers=StarTrackerConfiguration(
                    star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
                ),
                radiators=RadiatorConfiguration(radiators=[]),
            ),
        )
        config.constraint.ephem = ephem
        replay = DITL(
            config=config, ephem=ephem, plan=Plan(entries=entries), begin=begin, end=end
        )
        assert replay.calc()
        return replay

    return make


@pytest.mark.parametrize("obstype", [ObsType.PPT, ObsType.AT, ObsType.TOO])
@pytest.mark.parametrize("slew_rate,valid", [(1, False), (10, True)])
def test_short_replay_requires_actual_collection(
    replay_factory, slew_rate, valid, obstype
):
    begin = SCENARIO_BEGIN
    entry = PlanEntry(
        obsid=7,
        ra=180,
        dec=0,
        roll=0,
        begin=begin.timestamp(),
        end=begin.timestamp() + 59,
        slewtime=20,
        obstype=obstype,
    )
    replay = replay_factory([entry], slew_rate=slew_rate)
    collected = sum(hk.collection_seconds for hk in replay.telemetry.housekeeping)
    assert (collected > 0) == valid

    mismatches = replay.validate_plan_matches_execution()
    if valid:
        assert mismatches == []
    else:
        assert any("no_science_execution" in str(m) for m in mismatches)


def test_science_mode_during_unfinished_setup_does_not_prove_collection(replay_factory):
    begin = SCENARIO_BEGIN.timestamp()
    entry = PlanEntry(
        obsid=7,
        ra=20,
        dec=0,
        roll=0,
        begin=begin,
        end=begin + 180,
        slewtime=0,
        collection_begin=begin + 120,
        collection_end=begin + 150,
        obstype=ObsType.AT,
    )
    replay = replay_factory([entry], slew_rate=1, acceleration=0.5, settle_time=10)
    # The real slew takes 79 seconds. The plan implies another 120 seconds of
    # setup, so readiness is after the collection window despite SCIENCE at +120.
    sample = replay.telemetry.housekeeping[2]
    assert sample.timestamp.timestamp() == begin + 120
    assert sample.acs_mode == ACSMode.SCIENCE
    assert sample.obsid == entry.obsid
    assert all(hk.collection_seconds == 0 for hk in replay.telemetry.housekeeping)
    mismatches = replay.validate_plan_matches_execution()
    assert len(mismatches) == 1
    assert "no_science_execution" in str(mismatches[0])


def test_back_to_back_observations_at_same_pointing_record_collection(replay_factory):
    begin = SCENARIO_BEGIN.timestamp()
    entries = [
        PlanEntry(
            obsid=obsid,
            ra=20,
            dec=0,
            roll=0,
            begin=begin + offset,
            end=begin + offset + 120,
            slewtime=20 if offset == 0 else 0,
            obstype=ObsType.AT,
        )
        for obsid, offset in [(7, 0), (8, 120)]
    ]
    replay = replay_factory(entries)
    for entry in entries:
        assert any(
            hk.obsid == entry.obsid and hk.collection_seconds > 0
            for hk in replay.telemetry.housekeeping
        )
    assert replay.validate_plan_matches_execution() == []
