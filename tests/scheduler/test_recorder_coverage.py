"""Time ordering of concurrent collection and downlink."""

from unittest.mock import Mock

import pytest

from conops.config import MissionConfig, OnboardRecorder, ScienceData
from conops.schedulers.recorder import RecorderLedger
from conops.simulation import ContactSpan, StationCoverage
from conops.targets import PlanEntry


def _ledger(begin, end, rate, *, start=0.0):
    coverage = Mock(spec=StationCoverage)
    coverage.step_size = 60.0

    def spans(lo, hi):
        left, right = max(begin, lo), min(end, hi)
        return [ContactSpan(left, right, "TEST", rate)] if right > left else []

    coverage.spans.side_effect = spans
    config = MissionConfig(
        recorder=OnboardRecorder(capacity_gb=20, current_volume_gb=start)
    )
    return RecorderLedger(
        config, 0, 600, [], start_volume_gb=start, step_size=60, coverage=coverage
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
    assert sum(volume for _, volume in additions) == pytest.approx(
        0.6 + (2 if include_fixed else 0)
    )
