import pytest

from conops.config import DataGeneration, Instrument, ObservationTiming
from scripts import check_default_plan_output as scenario
from scripts.check_default_plan_output import (
    DEFAULT_BASELINE,
    DEFAULT_NUMERIC_ABS_TOL,
    build_default_plan_payload,
    compare_to_baseline,
)


def test_default_plan_output_matches_baseline() -> None:
    actual = build_default_plan_payload()
    diffs = compare_to_baseline(
        actual, DEFAULT_BASELINE, abs_tol=DEFAULT_NUMERIC_ABS_TOL
    )
    assert not diffs, "\n".join(diffs[:25])


@pytest.mark.parametrize("budgets", [(0, 0, 0), (8.25, 1.5, 2.5)])
def test_budgeted_plan_matches_fractional_collection_and_data(monkeypatch, budgets):
    constructor = scenario.QueueDITL
    simulations = []

    def timed_simulation(**kwargs):
        payload = kwargs["config"].payload
        payload.observation_timing = ObservationTiming(
            setup_seconds=budgets[0],
            cleanup_seconds=budgets[1],
            handoff_seconds=budgets[2],
        )
        payload.instruments = [
            Instrument(data_generation=DataGeneration(rate_gbps=0.001))
        ]
        ditl = constructor(**kwargs)
        simulations.append(ditl)
        return ditl

    monkeypatch.setattr(scenario, "QueueDITL", timed_simulation)
    scenario.build_default_plan_payload()
    ditl = simulations[0]
    science = [entry for entry in ditl.plan if entry.collection_begin is not None]
    planned_collection = sum(
        entry.collection_end - entry.collection_begin for entry in science
    )
    assert science
    assert not ditl.validate_plan_matches_execution()
    assert all(
        entry.collection_begin == entry.begin + entry.slewtime + budgets[0]
        for entry in science
    )
    assert all(
        entry.collection_end == entry.end - sum(budgets[1:]) for entry in science
    )
    assert sum(
        hk.collection_seconds for hk in ditl.telemetry.housekeeping
    ) == pytest.approx(planned_collection)
    assert ditl.data_generated_gb[-1] == pytest.approx(planned_collection * 0.001)
