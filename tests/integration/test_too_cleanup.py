"""TOO interruptions preserve collected science and reserve payload teardown."""

from datetime import timedelta
from math import ceil
from unittest.mock import Mock

import pytest
import rust_ephem

from conops import QueueDITL
from conops.common import ACSCommandType
from conops.config import (
    AttitudeControlSystem,
    Battery,
    DataGeneration,
    GroundStationRegistry,
    Instrument,
    MissionConfig,
    ObservationTiming,
    RadiatorConfiguration,
    ScienceData,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
    TargetConfig,
)
from conops.simulation.acs_command import ACSCommand
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
        Instrument(data_generation=DataGeneration(science=ScienceData(rate_gbps=0.001)))
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


@pytest.mark.parametrize(
    "cleanup,handoff,collection_stop_offset",
    [(0, 0, 900), (20, 5, 935), (40, 20, 900), (85, 40, 955)],
)
@pytest.mark.parametrize("submit_offset", [60, 900, 1920])
def test_too_waits_for_cleanup_without_losing_collected_science(
    monkeypatch, cleanup, handoff, collection_stop_offset, submit_offset
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
        # Keep collecting through the unused fraction of the dispatch tick;
        # an observation already in teardown retains its original window.
        cutoff = (
            begin.timestamp() + collection_stop_offset
            if submit_offset == 900
            else first.collection_begin + 1800
        )
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


def test_alignment_does_not_extend_an_observation_that_will_finish(monkeypatch):
    begin = SCENARIO_BEGIN.timestamp()
    ditl, request = _simulation(monkeypatch, cleanup=85, handoff=40, submit_offset=1860)
    target = ditl.queue.targets[0]

    assert ditl.calc()
    assert request.executed
    original = [entry for entry in ditl.plan if entry.obsid == 1]
    assert len(original) == 1
    assert original[0].collection_end == begin + 1915
    assert original[0].end == begin + 2040
    too = next(entry for entry in ditl.plan if entry.obsid == 30001)
    assert too.begin == begin + 2040
    assert target.collected_seconds == pytest.approx(1800)
    assert target.exptime == pytest.approx(0)
    assert ditl.validate_plan_matches_execution() == []


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
        "Reserved TOO cleanup" in event.description for event in ditl.log.events
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
        "Reserved TOO cleanup" in event.description for event in ditl.log.events
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
            assert ditl.ppt.collection_end == begin + 955
            assert ditl.ppt.end == begin + 1080
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


@pytest.mark.parametrize("rejection", ["rank_drop", "queue_rejection"])
def test_reserved_cleanup_commits_to_admitted_too(monkeypatch, rejection):
    """A later selection failure cannot abandon an already admitted handoff."""
    begin = SCENARIO_BEGIN.timestamp()
    ditl, request = _simulation(monkeypatch, cleanup=85, handoff=40)
    check_interrupt = ditl._check_too_interrupt
    real_fetch = ditl.queue.get
    dispatch_fetches = []

    def reject_dispatch(ra, dec, utime, **kwargs):
        if utime == begin + 1080:
            dispatch_fetches.append(utime)
            return None
        return real_fetch(ra, dec, utime, **kwargs)

    def check(utime, ra, dec):
        if utime == begin + 960:
            assert ditl._committed_too is not None
            assert not request.executed
            if rejection == "rank_drop":
                request.merit = 1
                ditl._too_targets[id(request)].fom = 1
            else:
                monkeypatch.setattr(ditl.queue, "get", reject_dispatch)
        return check_interrupt(utime, ra, dec)

    monkeypatch.setattr(ditl, "_check_too_interrupt", check)
    assert ditl.calc()
    assert request.executed
    assert dispatch_fetches == []
    too = next(entry for entry in ditl.plan if entry.obsid == 30001)
    assert too.begin == begin + 1080
    assert ditl.plan[0].collection_end == begin + 955
    assert ditl.plan[0].end == too.begin
    reservations = [
        event
        for event in ditl.log.events
        if "Reserved TOO cleanup" in event.description
    ]
    assert len(reservations) == 1
    assert reservations[0].time == begin + 900
    assert not any(
        event.time == begin + 1080 and "Time window elapsed" in event.description
        for event in ditl.log.events
    )
    assert ditl.validate_plan_matches_execution() == []


def _queue_state(ditl):
    return (
        ditl.queue.utime,
        [id(target) for target in ditl.queue.targets],
        [
            (target.model_dump(), target.merit_breakdown)
            for target in ditl.queue.targets
        ],
        list(ditl.log.events),
    )


@pytest.mark.parametrize("admissible", [True, False])
def test_admission_preview_does_not_change_queue_targets_or_log(
    monkeypatch, admissible
):
    ditl, request = _simulation(
        monkeypatch, cleanup=85, handoff=40, exptime=600 if admissible else 120
    )
    preview_admission = ditl._preview_too_admission
    previews = []

    def preview(dispatch_time, candidates):
        before = _queue_state(ditl)
        result = preview_admission(dispatch_time, candidates)
        assert _queue_state(ditl) == before
        previews.append(result)
        return result

    monkeypatch.setattr(ditl, "_preview_too_admission", preview)
    assert ditl.calc()
    assert previews
    assert request.executed == admissible
    if admissible:
        assert len(previews) == 1
    else:
        target = ditl._too_targets[id(request)]
        assert target.begin == 0
        assert target.end == 0
        assert target.collection_begin is None
        assert target.collection_end is None
        assert ditl.plan[0].collection_end == SCENARIO_BEGIN.timestamp() + 1915
    assert ditl.validate_plan_matches_execution() == []


def test_admission_exception_does_not_leak_target_state_or_logs(monkeypatch):
    ditl, _ = _simulation(monkeypatch, cleanup=85, handoff=40)
    fetch_inner = QueueDITL._fetch_new_ppt_inner
    preview_admission = ditl._preview_too_admission

    def fail_after_admission(self, *args, **kwargs):
        result = fetch_inner(self, *args, **kwargs)
        if kwargs.get("enqueue_slew") is False:
            assert result is not None
            raise RuntimeError("preview failed")
        return result

    def preview(dispatch_time, candidates):
        before = _queue_state(ditl)
        try:
            return preview_admission(dispatch_time, candidates)
        finally:
            assert _queue_state(ditl) == before

    monkeypatch.setattr(QueueDITL, "_fetch_new_ppt_inner", fail_after_admission)
    monkeypatch.setattr(ditl, "_preview_too_admission", preview)
    with pytest.raises(RuntimeError, match="preview failed"):
        ditl.calc()
    assert ditl._committed_too is None
    assert ditl.ppt.collection_end == SCENARIO_BEGIN.timestamp() + 1915


@pytest.mark.parametrize(
    "command_type",
    [ACSCommandType.START_PASS, ACSCommandType.START_BATTERY_CHARGE],
)
@pytest.mark.parametrize("command_offset", [960, 1080])
def test_queued_command_prevents_speculative_cleanup(
    monkeypatch, command_type, command_offset
):
    begin = SCENARIO_BEGIN.timestamp()
    ditl, request = _simulation(monkeypatch, cleanup=85, handoff=40)
    check_interrupt = ditl._check_too_interrupt

    class CheckedReservationError(Exception):
        pass

    def check(utime, ra, dec):
        if utime != begin + 900:
            return check_interrupt(utime, ra, dec)
        command = ACSCommand(
            command_type=command_type, execution_time=begin + command_offset
        )
        ditl.acs.command_queue.append(command)
        assert not check_interrupt(utime, ra, dec)
        assert not request.executed
        assert ditl._committed_too is None
        assert ditl.ppt.collection_end == begin + 1915
        assert ditl.ppt.end == begin + 2040
        assert ditl.plan[0].collection_end == begin + 1915
        assert ditl.acs.command_queue == [command]
        assert not any(
            "Reserved TOO cleanup" in event.description for event in ditl.log.events
        )
        raise CheckedReservationError

    monkeypatch.setattr(ditl, "_check_too_interrupt", check)
    with pytest.raises(CheckedReservationError):
        ditl.calc()


def test_preview_shares_read_only_runtime_inputs(monkeypatch):
    """Native ephemeris/constraint objects need not support deep copies."""
    ditl, request = _simulation(monkeypatch, cleanup=85, handoff=40)

    def cannot_copy(_self, _memo):
        raise TypeError("native runtime inputs cannot be copied")

    monkeypatch.setattr(
        type(ditl.constraint), "__deepcopy__", cannot_copy, raising=False
    )
    monkeypatch.setattr(type(ditl.ephem), "__deepcopy__", cannot_copy, raising=False)
    assert ditl.calc()
    assert request.executed
    assert ditl.validate_plan_matches_execution() == []


def test_external_termination_clears_committed_handoff(monkeypatch):
    """A superseding operation cannot leave a stale TOO to dispatch later."""
    begin = SCENARIO_BEGIN.timestamp()
    ditl, request = _simulation(monkeypatch, cleanup=85, handoff=40)
    check_interrupt = ditl._check_too_interrupt

    class CheckedCancellationError(Exception):
        pass

    def check(utime, ra, dec):
        if utime != begin + 1080:
            return check_interrupt(utime, ra, dec)
        assert ditl._committed_too is not None
        ditl._terminate_ppt(utime, reason="Superseded by protective operation")
        assert ditl._committed_too is None
        assert not check_interrupt(utime, ra, dec)
        assert not request.executed
        assert not ditl.acs.command_queue
        target = ditl._too_targets[id(request)]
        assert target.begin == 0
        assert target.end == 0
        assert target.collection_begin is None
        assert target.collection_end is None
        raise CheckedCancellationError

    monkeypatch.setattr(ditl, "_check_too_interrupt", check)
    with pytest.raises(CheckedCancellationError):
        ditl.calc()
