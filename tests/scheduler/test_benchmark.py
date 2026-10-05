"""The benchmark harness runs every scheduling mode and measures them alike."""

from datetime import timedelta

import pytest

from conops.benchmark import (
    BenchmarkScenario,
    Contender,
    TOOSpec,
    dispatch,
    format_results,
    planned,
    rolling,
    run_benchmark,
)
from conops.config import MissionConfig
from conops.schedulers import LocalSearchPlanner, PriorityPlanner
from conops.targets import Pointing

from .planning_scenario import BEGIN, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 2


def _scenario() -> BenchmarkScenario:
    def make_targets(config: MissionConfig) -> list[Pointing]:
        return [
            _target(config, 100 + i, ra, dec, merit=90 - 10 * i, minutes=40)
            for i, (ra, dec) in enumerate(PATCH)
        ]

    return BenchmarkScenario(
        name="patch",
        begin=BEGIN,
        end=BEGIN + timedelta(hours=HOURS),
        make_config=lambda: _config(HOURS),
        make_targets=make_targets,
        toos=[
            TOOSpec(
                obsid=1_000_001,
                ra=115.0,
                dec=0.0,
                merit=500.0,
                exptime=600,
                name="ToO",
                submit_time=T0 + 30 * MIN,
                deadline=T0 + 50 * MIN,
            )
        ],
    )


@pytest.fixture(scope="module")
def results() -> dict[str, object]:
    search = {"time_limit": 600, "max_iterations": 50, "seed": 1}
    contenders = [
        dispatch(),
        planned(PriorityPlanner),
        planned(LocalSearchPlanner, **search),
        rolling(PriorityPlanner, replan_interval=timedelta(hours=1)),
        rolling(
            LocalSearchPlanner,
            replan_interval=timedelta(hours=1),
            planner_options=search,
        ),
    ]
    return {r.contender: r for r in run_benchmark(_scenario(), contenders)}


def test_every_contender_runs(results: dict[str, object]) -> None:
    assert list(results) == [
        "dispatch",
        "planned:priority",
        "planned:local_search",
        "rolling:priority",
        "rolling:local_search",
    ]
    for result in results.values():
        assert result.error is None  # type: ignore[attr-defined]
        assert result.science_hours > 0  # type: ignore[attr-defined]


def test_plans_execute_exactly(results: dict[str, object]) -> None:
    for name, result in results.items():
        assert result.mismatches == 0, name  # type: ignore[attr-defined]


def test_only_closed_loop_modes_respond_to_the_too(
    results: dict[str, object],
) -> None:
    responses = {
        name: result.too_response_seconds[1_000_001]  # type: ignore[attr-defined]
        for name, result in results.items()
    }

    assert responses["planned:priority"] is None
    assert responses["planned:local_search"] is None
    for name in ("dispatch", "rolling:priority", "rolling:local_search"):
        response = responses[name]
        assert response is not None and response <= 20 * MIN, name


def test_planning_time_is_reported_for_planners(results: dict[str, object]) -> None:
    assert results["dispatch"].planning_seconds is None  # type: ignore[attr-defined]
    assert results["planned:priority"].planning_seconds is not None  # type: ignore[attr-defined]
    assert results["rolling:priority"].planning_seconds is not None  # type: ignore[attr-defined]


def test_time_accounting_fits_in_the_run(results: dict[str, object]) -> None:
    for result in results.values():
        assert result.science_hours <= HOURS  # type: ignore[attr-defined]
        sampled = (
            result.slewing_hours  # type: ignore[attr-defined]
            + result.idle_hours  # type: ignore[attr-defined]
            + result.pass_hours  # type: ignore[attr-defined]
        )
        assert sampled <= HOURS + 1e-9


def test_failing_contender_is_reported_not_raised() -> None:
    def explode(scenario: BenchmarkScenario) -> object:
        raise RuntimeError("boom")

    failing = Contender(name="broken", simulate=explode)  # type: ignore[arg-type]

    (result,) = run_benchmark(_scenario(), [failing])

    assert result.error == "RuntimeError: boom"
    lines = format_results([result]).splitlines()
    assert lines[2].split() == ["broken", "error:", "RuntimeError"]
    assert lines[-1] == "broken: RuntimeError: boom"


def test_a_long_error_does_not_widen_the_table() -> None:
    def explode(scenario: BenchmarkScenario) -> object:
        raise RuntimeError("x" * 500)

    failing = Contender(name="broken", simulate=explode)  # type: ignore[arg-type]

    (result,) = run_benchmark(_scenario(), [failing])
    header, rule, row, *_ = format_results([result]).splitlines()

    assert len(rule) < 200
    assert "x" * 500 in format_results([result])


def test_table_lists_each_contender(results: dict[str, object]) -> None:
    table = format_results(list(results.values()))  # type: ignore[arg-type]

    lines = table.splitlines()
    assert lines[0].startswith("contender")
    assert len(lines) == 2 + len(results)
    assert all(name in table for name in results)
