"""TOO interruptions preserve collected science and reserve payload teardown."""

from datetime import timedelta
from math import ceil
from unittest.mock import Mock

import pytest
import rust_ephem

from conops import QueueDITL
from conops.config import (
    AttitudeControlSystem,
    Battery,
    DataGeneration,
    GroundStationRegistry,
    Instrument,
    MissionConfig,
    ObservationTiming,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
)
from scripts.check_default_plan_output import (
    SCENARIO_BEGIN,
    DeterministicConstraint,
    DeterministicEphemeris,
)


@pytest.mark.parametrize("cleanup,handoff", [(0, 0), (20, 5), (85, 40)])
@pytest.mark.parametrize("submit_offset", [60, 900, 1920])
def test_too_waits_for_cleanup_without_losing_collected_science(
    monkeypatch, cleanup, handoff, submit_offset
):
    begin = SCENARIO_BEGIN
    end = begin + timedelta(seconds=3600)
    ephem = DeterministicEphemeris(begin, end, step_size_seconds=60)
    config = MissionConfig(
        constraint=DeterministicConstraint(),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        battery=Battery(watthour=100000),
        spacecraft_bus=SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                max_slew_rate=1, slew_acceleration=0.5, settle_time=10
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
    )
    config.constraint.ephem = ephem
    config.payload.observation_timing = ObservationTiming(
        setup_seconds=10, cleanup_seconds=cleanup, handoff_seconds=handoff
    )
    config.payload.instruments = [
        Instrument(data_generation=DataGeneration(rate_gbps=0.001))
    ]
    monkeypatch.setattr(
        rust_ephem,
        "get_eop_provenance",
        lambda: {"ut1": {"available": False}, "polar_motion": {"available": False}},
    )
    monkeypatch.setattr(
        "conops.config.solar_panel._get_eclipse_constraint",
        lambda: Mock(in_constraint=Mock(return_value=False)),
    )
    ditl = QueueDITL(config=config, ephem=ephem, begin=begin, end=end)
    ditl.queue.add(obsid=1, ra=20, dec=0, merit=100, exptime=1800, ss_min=300)
    target = ditl.queue.targets[0]
    request = ditl.submit_too(
        obsid=30001,
        ra=60,
        dec=10,
        merit=200,
        exptime=600,
        name="Urgent",
        submit_time=begin.timestamp() + submit_offset,
    )

    assert ditl.calc()
    assert request.executed
    assert not any(
        event.event_type == "operational_fault"
        for event in ditl.fault_management.events
    )
    assert ditl.validate_plan_matches_execution() == []
    too = next(entry for entry in ditl.plan if entry.obsid == 30001)
    interrupted = [
        entry for entry in ditl.plan if entry.obsid == 1 and entry.begin < too.begin
    ]
    if submit_offset == 60:
        # No collection happened before this slew was interrupted.
        assert not interrupted
    else:
        first = interrupted[0]
        assert first.collection_begin == begin.timestamp() + 115
        cutoff = min(begin.timestamp() + submit_offset, first.collection_begin + 1800)
        assert first.collection_end == cutoff
        assert first.end == cutoff + cleanup + handoff
        ready_offset = max(submit_offset, first.end - begin.timestamp())
        assert too.begin == begin.timestamp() + ceil(ready_offset / 60) * 60
        collected = sum(
            hk.collection_seconds
            for hk in ditl.telemetry.housekeeping
            if hk.obsid == 1 and hk.timestamp.timestamp() < too.begin
        )
        assert collected == pytest.approx(first.collection_end - first.collection_begin)

    # The interrupted target resumes with only its remaining science budget.
    target_collection = sum(
        hk.collection_seconds for hk in ditl.telemetry.housekeeping if hk.obsid == 1
    )
    assert target_collection == pytest.approx(1800)
    assert target.exptime == pytest.approx(0)
    total_collection = sum(hk.collection_seconds for hk in ditl.telemetry.housekeeping)
    assert ditl.data_generated_gb[-1] == pytest.approx(total_collection * 0.001)
