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
    TargetConfig,
)
from conops.targets import Pointing
from scripts.check_default_plan_output import (
    SCENARIO_BEGIN,
    DeterministicConstraint,
    DeterministicEphemeris,
)


def _simulation(
    monkeypatch,
    cleanup=20,
    handoff=5,
    submit_offset=900,
    *,
    merit=200,
    exptime=600,
    deadline=None,
    urgency=0,
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
    config.targets = TargetConfig(urgency_weight=urgency)
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
    request = ditl.submit_too(
        obsid=30001,
        ra=60,
        dec=10,
        merit=merit,
        exptime=exptime,
        deadline=deadline,
        name="Urgent",
        submit_time=begin.timestamp() + submit_offset,
    )

    return ditl, request


@pytest.mark.parametrize("cleanup,handoff", [(0, 0), (20, 5), (85, 40)])
@pytest.mark.parametrize("submit_offset", [60, 900, 1920])
def test_too_waits_for_cleanup_without_losing_collected_science(
    monkeypatch, cleanup, handoff, submit_offset
):
    begin = SCENARIO_BEGIN
    ditl, request = _simulation(monkeypatch, cleanup, handoff, submit_offset)
    target = ditl.queue.targets[0]
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


def test_deadline_that_only_fits_without_cleanup_does_not_interrupt(monkeypatch):
    control, _ = _simulation(monkeypatch, cleanup=0, handoff=0)
    assert control.calc()
    control_too = next(entry for entry in control.plan if entry.obsid == 30001)
    deadline = control_too.collection_begin + 1
    ditl, request = _simulation(monkeypatch, deadline=deadline)
    assert ditl.calc()
    assert not request.executed
    assert [entry.obsid for entry in ditl.plan] == [1]
    assert ditl.plan[0].collection_end == SCENARIO_BEGIN.timestamp() + 1915
    assert not any(
        "Stopping collection" in event.description for event in ditl.log.events
    )
    assert not any(
        command.slew is not None and command.slew.obsid == 30001
        for command in ditl.acs.executed_commands
    )
    assert ditl.validate_plan_matches_execution() == []


@pytest.mark.parametrize("rejection", ["short_exposure", "visibility"])
def test_inadmissible_too_leaves_science_running(monkeypatch, rejection):
    ditl, request = _simulation(
        monkeypatch, exptime=120 if rejection == "short_exposure" else 600
    )
    if rejection == "visibility":
        visibility = Pointing.visibility

        def short_window(target, *args, **kwargs):
            result = visibility(target, *args, **kwargs)
            if target.obsid == 30001:
                target.windows = [
                    [SCENARIO_BEGIN.timestamp(), SCENARIO_BEGIN.timestamp() + 930]
                ]
            return result

        monkeypatch.setattr(Pointing, "visibility", short_window)
    assert ditl.calc()
    assert not request.executed
    assert [entry.obsid for entry in ditl.plan] == [1]
    assert ditl.plan[0].collection_end == SCENARIO_BEGIN.timestamp() + 1915
    assert not any(
        "Stopping collection" in event.description for event in ditl.log.events
    )
    assert ditl.validate_plan_matches_execution() == []


def test_urgency_preemption_waits_without_commanding_or_marking_executed(monkeypatch):
    begin = SCENARIO_BEGIN.timestamp()
    ditl, request = _simulation(
        monkeypatch,
        cleanup=85,
        handoff=40,
        merit=70,
        urgency=50,
        deadline=begin + 1800,
    )
    check_interrupt = ditl._check_too_interrupt
    pending_checks = []

    def check(utime, ra, dec):
        queued = list(ditl.acs.command_queue)
        executed = list(ditl.acs.executed_commands)
        distances = list(ditl.acs.slew_dists)
        result = check_interrupt(utime, ra, dec)
        if begin + 900 <= utime < begin + 1080:
            pending_checks.append(utime)
            assert not result
            assert not request.executed
            assert ditl.ppt.obsid == 1
            assert ditl.ppt.collection_end == begin + 900
            assert ditl.ppt.end == begin + 1025
            assert ditl.acs.command_queue == queued
            assert ditl.acs.executed_commands == executed
            assert ditl.acs.slew_dists == distances
        return result

    monkeypatch.setattr(ditl, "_check_too_interrupt", check)
    assert ditl.calc()
    assert pending_checks == [begin + 900, begin + 960, begin + 1020]
    too = next(entry for entry in ditl.plan if entry.obsid == 30001)
    assert too.begin == begin + 1080
    assert too.collection_begin <= request.deadline
    assert request.executed
    assert ditl.validate_plan_matches_execution() == []
