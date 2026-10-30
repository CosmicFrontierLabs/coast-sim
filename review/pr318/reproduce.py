"""Independent review probes; run from the PR worktree with PYTHONPATH=."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from conops.common import ACSMode, AntennaType
from conops.config import HousekeepingData, MissionConfig, OnboardRecorder, ScienceData
from conops.ditl.ditl_mixin import DITLMixin
from conops.schedulers import PriorityScheduler
from conops.schedulers.priority_scheduler import StartState
from conops.simulation import ContactSpan, StationCoverage
from tests.scheduler.planning_scenario import BEGIN, T0, make_config, make_target
from tests.simulation.test_coverage import _comms, _walk


def outage_probe():
    coverage = _walk([[1]], [100.0], table=[(0.0, 20.0)], repoint_seconds=10.0)
    config = MissionConfig(recorder=OnboardRecorder(capacity_gb=0.05))
    config.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0.01)
    config.spacecraft_bus.communications = _comms(
        AntennaType.GIMBALED, gimbal_range_deg=180.0
    )
    coverage.config = config
    coverage._to_station[:] = [1.0, 0.0, 0.0]
    coverage.range_km = np.full((1, 1), 700.0)
    sim = SimpleNamespace(
        recorder=config.recorder,
        coverage=coverage,
        uend=60.0,
        acs=SimpleNamespace(current_pass=None, ra=0.0, dec=0.0, roll=0.0),
    )
    generated, sent = DITLMixin._process_store_data(
        sim, config, 0.0, ACSMode.SCIENCE, 60, collection_seconds=0.0
    )
    print("OUTAGE contacts:", coverage.contacts)
    print(
        "OUTAGE simulated generated/sent/remaining/lost:",
        generated,
        sent,
        config.recorder.current_volume_gb,
        generated - sent - config.recorder.current_volume_gb,
    )
    exact = OnboardRecorder(capacity_gb=0.05)
    cursor = sent_exact = 0.0
    for span in coverage.contacts:
        exact.add_data(0.01 * (span.begin - cursor))
        sent_exact += exact.transfer(
            0.01 * (span.end - span.begin),
            span.rate_mbps * (span.end - span.begin) / 1000,
        )
        cursor = span.end
    exact.add_data(0.01 * (60 - cursor))
    print(
        "OUTAGE chronological sent/remaining/lost:",
        sent_exact,
        exact.current_volume_gb,
        generated - sent_exact - exact.current_volume_gb,
    )
    return sent, sent_exact


def bisection_probe():
    config = make_config(1)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.constraint.invalidate_combined_constraint_cache()
    config.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0.0)
    config.recorder = OnboardRecorder(capacity_gb=20.0, current_volume_gb=17.0)
    target = make_target(config, 1, 105.0, 10.0, minutes=5, snapshot=5)
    target.science_data = {"Default Instrument": ScienceData(rate_gbps=0.08)}
    coverage = Mock(spec=StationCoverage)
    coverage.repoints = []
    coverage.step_size = 60.0
    raw = [
        ContactSpan(T0, T0 + 120, "TEST", 0.01),
        ContactSpan(T0 + 120, T0 + 420, "TEST", 100.0),
        ContactSpan(T0 + 420, T0 + 900, "TEST", 0.01),
    ]

    def rate_spans(lo, hi, recorder=None):
        return [
            s._replace(begin=max(s.begin, lo), end=min(s.end, hi))
            for s in raw
            if max(s.begin, lo) < min(s.end, hi)
        ]

    def spans(lo, hi):
        pieces = rate_spans(lo, hi)
        if not pieces:
            return []
        begin, end = pieces[0].begin, pieces[-1].end
        rate = sum((s.end - s.begin) * s.rate_mbps for s in pieces) / (end - begin)
        return [ContactSpan(begin, end, "TEST", rate)]

    coverage.spans.side_effect = spans
    coverage.rate_spans.side_effect = rate_spans
    coverage.scheduled_passes = spans(T0, T0 + 900)
    scheduler = PriorityScheduler(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(seconds=900),
        include_passes=False,
        coverage=coverage,
        start_state=StartState(time=T0, attitude=(105.0, 10.0, 0.0)),
    )
    plan = scheduler.schedule()
    print(
        "BISECTION plan:",
        [(e.begin - T0, e.end - T0) for e in plan],
        "unplaced:",
        [t.obsid for t in scheduler.unplaced],
    )
    request = scheduler._requests()[0]
    valid = []
    for start in range(0, 601, 60):
        fit = scheduler._fit_in_gap(request, scheduler._origin, None, T0 + start)
        if fit is not None:
            block = fit[0]
            ok = scheduler._ledger.fits_all(
                scheduler._volumes, scheduler._block_additions(block)
            )
            if ok:
                valid.append(block.entry.begin - T0)
                print(
                    "BISECTION valid candidate:",
                    block.entry.begin - T0,
                    block.entry.collection_begin - T0,
                    block.entry.collection_end - T0,
                )
    target.earliest_start = T0 + 164.0
    guided = PriorityScheduler(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(seconds=900),
        include_passes=False,
        coverage=coverage,
        start_state=StartState(time=T0, attitude=(105.0, 10.0, 0.0)),
    ).schedule()
    print(
        "BISECTION same request with explicit earliest_start:",
        [
            (e.begin - T0, e.collection_begin - T0, e.collection_end - T0)
            for e in guided
        ],
    )
    assert guided, "The same public scheduler must accept the known feasible start"
    return len(plan), valid


def booking_probe():
    config = make_config(14, stations=True)
    config.spacecraft_bus.communications = _comms(AntennaType.OMNI)
    original = StationCoverage(config)
    span = max(original.scheduled_passes, key=lambda s: s.end - s.begin)
    begin, end = span.begin + 15.0, span.end - 15.0
    replay = StationCoverage(config, scheduled_passes=[(begin, end, span.station)])
    print("BOOKING reserved:", begin - T0, end - T0)
    print("BOOKING actual:", [(s.begin - T0, s.end - T0) for s in replay.contacts])
    print(
        "BOOKING downlink after reservation ends:", replay.downlink_gb(end, end + 60.0)
    )
    return replay.downlink_gb(end, end + 60.0)


if __name__ == "__main__":
    outage_probe()
    bisection_probe()
    booking_probe()
