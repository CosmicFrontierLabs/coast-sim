"""Audit short replay entries against collection from the real ACS executor."""

from datetime import timedelta
from unittest.mock import Mock

import pytest
import rust_ephem

from conops import DITL
from conops.common import ObsType
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


@pytest.mark.parametrize("obstype", [ObsType.PPT, ObsType.AT, ObsType.TOO])
@pytest.mark.parametrize("slew_rate,valid", [(1, False), (10, True)])
def test_short_replay_requires_actual_collection(
    monkeypatch, slew_rate, valid, obstype
):
    begin = SCENARIO_BEGIN
    end = begin + timedelta(seconds=240)
    ephem = DeterministicEphemeris(begin, end, step_size_seconds=60)
    config = MissionConfig(
        constraint=DeterministicConstraint(),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        spacecraft_bus=SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                max_slew_rate=slew_rate, slew_acceleration=10, settle_time=0
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
    )
    config.constraint.ephem = ephem
    monkeypatch.setattr(
        rust_ephem,
        "get_eop_provenance",
        lambda: {"ut1": {"available": False}, "polar_motion": {"available": False}},
    )
    monkeypatch.setattr(
        "conops.config.solar_panel._get_eclipse_constraint",
        lambda: Mock(in_constraint=Mock(return_value=False)),
    )
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
    replay = DITL(
        config=config, ephem=ephem, plan=Plan(entries=[entry]), begin=begin, end=end
    )
    assert replay.calc()
    collected = sum(hk.collection_seconds for hk in replay.telemetry.housekeeping)
    assert (collected > 0) == valid

    mismatches = replay.validate_plan_matches_execution()
    if valid:
        assert mismatches == []
    else:
        assert any("no_science_execution" in str(m) for m in mismatches)
