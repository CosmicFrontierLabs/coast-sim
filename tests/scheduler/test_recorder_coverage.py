"""Time ordering of concurrent collection and downlink."""

from unittest.mock import Mock

import pytest

from conops.config import HousekeepingData, MissionConfig, OnboardRecorder, ScienceData
from conops.schedulers.recorder import RecorderLedger, amount_total
from conops.simulation import ContactSpan, StationCoverage
from conops.targets import PlanEntry


def _ledger(begin, end, rate, *, start=0.0, horizon=600):
    coverage = Mock(spec=StationCoverage)
    coverage.step_size = 60.0

    def spans(lo, hi):
        left, right = max(begin, lo), min(end, hi)
        return [ContactSpan(left, right, "TEST", rate)] if right > left else []

    coverage.spans.side_effect = spans
    coverage.rate_spans.side_effect = lambda lo, hi, recorder=None: spans(lo, hi)
    config = MissionConfig(
        recorder=OnboardRecorder(capacity_gb=20, current_volume_gb=start)
    )
    return RecorderLedger(
        config, 0, horizon, [], start_volume_gb=start, step_size=60, coverage=coverage
    )


def _science(rate, *, fixed=0.0, begin=0.0, end=600.0):
    return PlanEntry(
        begin=begin,
        end=end,
        collection_begin=begin,
        collection_end=end,
        science_data={
            "Default Instrument": ScienceData(rate_gbps=rate, per_observation_gb=fixed)
        },
    )


def test_a_burst_must_fit_before_the_rest_of_the_contact_downlinks():
    ledger = _ledger(0, 600, 10, start=17)
    entry = _science(0.1, end=60)
    volumes = ledger.initial_volumes([entry])
    assert max(ledger.peaks(volumes)) >= 22.4
    assert not ledger.fits_all(volumes, [])


def test_unused_early_link_capacity_cannot_downlink_later_collection():
    ledger = _ledger(0, 60, 10)
    entry = _science(0.001)
    volumes = ledger.initial_volumes([entry])
    # Only the 0.06 Gb collected in the first minute can use this link;
    # its other 0.54 Gb arrive after the station has gone away.
    assert ledger.levels(volumes)[-1] == pytest.approx(0.54)


def test_a_contact_ending_within_a_step_leaves_the_steps_later_data():
    ledger = _ledger(0, 30, 100)
    entry = _science(0.001)
    # DITL's first step generates 0.06 Gb uniformly: the first half sends
    # 0.03 Gb; the second half and the later steps retain 0.57 Gb.
    assert ledger.levels(ledger.initial_volumes([entry]))[-1] == pytest.approx(0.57)


@pytest.mark.parametrize("include_fixed", [True, False])
def test_fixed_science_is_counted_once_across_downlink_boundaries(include_fixed):
    ledger = _ledger(0, 600, 1)
    entry = _science(0.001, fixed=2)
    additions = ledger.science_additions(entry, 600, 0, include_fixed=include_fixed)
    assert sum(amount_total(volume) for _, volume in additions) == pytest.approx(
        0.6 + (2 if include_fixed else 0)
    )


def test_weeklong_continuous_contact_coalesces_without_losing_production_timing():
    week = 7 * 86400
    ledger = _ledger(0, week, 10, start=17, horizon=week)
    assert ledger.segments <= 3  # Independent of the 10,080 telemetry steps.
    early = _science(0.1, end=60)
    assert not ledger.fits_all(ledger.initial_volumes([early]), [])
    late = _science(0.001, begin=week - 600, end=week)
    assert ledger.fits_all(ledger.initial_volumes([late]), [])
    assert ledger.levels(ledger.initial_volumes([late]))[-1] == pytest.approx(0.0)


@pytest.mark.parametrize("production", ["housekeeping", "planned-science", "both"])
def test_retry_bound_deducts_production_that_consumes_the_link(production):
    week = 7 * 86400
    ledger = _ledger(0, week, 0.01, start=17, horizon=week)
    housekeeping = {"housekeeping": 0.00001, "planned-science": 0.0, "both": 0.000005}[
        production
    ]
    ledger.config.spacecraft_bus.data_generation = HousekeepingData(
        rate_gbps=housekeeping
    )
    entries = (
        []
        if production == "housekeeping"
        else [_science(0.00001 - housekeeping, end=week)]
    )
    planned = ledger.initial_volumes(entries)
    assert ledger.science_capacity_time(planned, 4.8, 60, week) is None


def test_retry_bound_discards_unused_capacity_before_the_candidate():
    ledger = _ledger(0, 600, 100, start=17, horizon=3600)
    planned = ledger.initial_volumes([])
    # Early downlink empties the recorder. It cannot make a later 24 Gb
    # snapshot fit in the 18 Gb planning limit after the link has ended.
    assert ledger.science_capacity_time(planned, 24, 600, 3600) is None


def test_retry_bound_reserves_room_for_future_planned_production():
    ledger = _ledger(0, 600, 10, start=17)
    planned = ledger.initial_volumes([_science(0.0, fixed=5, begin=540, end=600)])
    assert ledger.fits_all(planned, [])
    # By 380 seconds there is room for 4.8 Gb, but the later fixed production
    # still needs that capacity. This must also bound retries in an earlier gap.
    assert ledger.science_capacity_time(planned, 4.8, 0, 300) is None
    assert ledger.science_capacity_time(planned, 4.8, 0, 600) is None


def test_retry_bound_finds_capacity_within_a_contact_and_respects_gap_end():
    ledger = _ledger(0, 600, 10, start=17)
    planned = ledger.initial_volumes([])
    assert ledger.science_capacity_time(planned, 4.8, 0, 600) == pytest.approx(380)
    assert ledger.science_capacity_time(planned, 4.8, 0, 300) is None
