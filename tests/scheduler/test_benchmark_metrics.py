"""Benchmark metrics: visits, cadence error, program shares and ToO deadlines."""

from datetime import timedelta

import pytest

from conops.benchmark import (
    BenchmarkResult,
    BenchmarkScenario,
    TOOSpec,
    dispatch,
    format_results,
    planned,
    rolling,
    run_benchmark,
)
from conops.benchmark.metrics import cadence_error, program_shares, visit_starts
from conops.benchmark.scenarios import SCENARIOS, random_targets
from conops.config import MissionConfig
from conops.config.observation_categories import ObservationCategory
from conops.schedulers import PriorityPlanner
from conops.targets import Pointing

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

STEP = 60.0


class TestVisits:
    def test_consecutive_steps_are_one_visit(self) -> None:
        records = [(1, T0 + k * STEP, 60.0) for k in range(5)]

        assert visit_starts(records, STEP) == {1: [T0]}

    def test_a_gap_starts_a_new_visit(self) -> None:
        records = [(1, T0, 60.0), (1, T0 + STEP, 60.0), (1, T0 + 10 * STEP, 30.0)]

        assert visit_starts(records, STEP) == {1: [T0, T0 + 10 * STEP]}

    def test_steps_without_collection_are_ignored(self) -> None:
        records = [(1, T0, 0.0), (2, T0 + STEP, 60.0)]

        assert visit_starts(records, STEP) == {2: [T0 + STEP]}


class TestCadence:
    def test_exact_cadence_has_no_error(self) -> None:
        starts = {1: [T0, T0 + 4 * HOUR, T0 + 8 * HOUR]}

        assert cadence_error(starts, {1: 4 * HOUR}) == (0.0, 1)

    def test_error_is_relative_to_the_interval(self) -> None:
        starts = {1: [T0, T0 + 6 * HOUR], 2: [T0, T0 + 2 * HOUR]}

        error, revisited = cadence_error(starts, {1: 4 * HOUR, 2: 4 * HOUR})

        assert revisited == 2
        assert error == pytest.approx(0.5)

    def test_targets_seen_once_are_not_revisited(self) -> None:
        assert cadence_error({1: [T0]}, {1: 4 * HOUR}) == (None, 0)


class TestProgramShares:
    def test_shares_sum_to_one_by_program(self) -> None:
        shares = program_shares({1: 300.0, 2: 100.0, 3: 0.0}, {1: "A", 2: "B", 3: "B"})

        assert shares == {"A": 0.75, "B": 0.25}

    def test_nothing_collected(self) -> None:
        assert program_shares({1: 0.0}, {1: "A"}) == {}


def _scenario(toos: list[TOOSpec]) -> BenchmarkScenario:
    categories = [
        ObservationCategory(
            name="Monitor", obsid_min=100, obsid_max=102, cadence_seconds=HOUR
        ),
        ObservationCategory(
            name="Survey", obsid_min=102, obsid_max=200, time_share=0.5
        ),
    ]

    def make_config() -> MissionConfig:
        config = _config(2)
        config.observation_categories.categories = categories
        return config

    return BenchmarkScenario(
        name="metrics",
        begin=BEGIN,
        end=BEGIN + timedelta(hours=2),
        make_config=make_config,
        make_targets=lambda config: [
            _target(config, 100 + i, ra, dec, minutes=40, snapshot=10)
            for i, (ra, dec) in enumerate(PATCH)
        ],
        toos=toos,
    )


@pytest.fixture(scope="module")
def results() -> list[BenchmarkResult]:
    toos = [
        TOOSpec(
            obsid=1_000_001,
            ra=115.0,
            dec=0.0,
            merit=500.0,
            exptime=600,
            name="on time",
            submit_time=T0 + 30 * MIN,
            deadline=T0 + 50 * MIN,
        ),
        TOOSpec(
            obsid=1_000_002,
            ra=110.0,
            dec=20.0,
            merit=500.0,
            exptime=600,
            name="too late",
            submit_time=T0 + 30 * MIN,
            deadline=T0 + 30 * MIN + 1,
        ),
    ]
    return run_benchmark(_scenario(toos), [dispatch(), planned(PriorityPlanner)])


class TestMeasuredResults:
    def test_deadline_hits_are_counted(self, results: list[BenchmarkResult]) -> None:
        dispatched, plan = results

        assert dispatched.too_on_time == 1
        assert plan.too_on_time == 0

    def test_programs_and_cadence_are_measured(
        self, results: list[BenchmarkResult]
    ) -> None:
        for result in results:
            # ToOs fall outside the categories, under the default name.
            assert set(result.program_share) <= {"Monitor", "Survey", "Observation"}
            assert sum(result.program_share.values()) == pytest.approx(1.0)
            assert result.cadence_targets == 2

    def test_table_shows_the_extra_columns(
        self, results: list[BenchmarkResult]
    ) -> None:
        header = format_results(results).splitlines()[0]

        assert "ToO on time" in header
        assert "programs" in header
        assert "cadence" in header

    def test_extra_columns_hidden_when_unused(self) -> None:
        plain = BenchmarkResult(contender="plain", scenario="s")

        header = format_results([plain]).splitlines()[0]

        assert "programs" not in header and "cadence" not in header


class TestStandardScenarios:
    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_builds_fresh_configs_and_targets(self, name: str) -> None:
        scenario = SCENARIOS[name]("examples/example.tle")

        first = scenario.make_config()
        second = scenario.make_config()
        targets = scenario.make_targets(first)

        assert first is not second
        assert targets
        assert scenario.end > scenario.begin
        assert all(t.config is first for t in targets)

    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_battery_never_raises_an_alert(self, name: str) -> None:
        """The scenarios have no solar panels; charging must never take over."""
        config = SCENARIOS[name]("examples/example.tle").make_config()
        config.battery.charge_level = 0.0

        assert not config.battery.battery_alert

    def test_random_targets_are_reproducible(self) -> None:
        config = _config(2)

        first = random_targets(config, 5, seed=3, deadline_fraction=1.0)
        second = random_targets(config, 5, seed=3, deadline_fraction=1.0)

        assert [(t.ra, t.dec, t.deadline) for t in first] == [
            (t.ra, t.dec, t.deadline) for t in second
        ]
        assert all(t.deadline is not None for t in first)


class TestDeadlines:
    def test_counts_requests_whose_exposure_was_collected(self) -> None:
        def make_config() -> MissionConfig:
            return _config(2)

        def make_targets(config: MissionConfig) -> list[Pointing]:
            return [
                _target(config, 1, *PATCH[0], minutes=10, deadline=T0 + HOUR),
                # Due before it could ever be reached.
                _target(config, 2, *PATCH[1], minutes=10, deadline=T0 - 1),
                _target(config, 3, *PATCH[2], minutes=10),
            ]

        scenario = BenchmarkScenario(
            name="deadlines",
            begin=BEGIN,
            end=BEGIN + timedelta(hours=2),
            make_config=make_config,
            make_targets=make_targets,
        )

        (result,) = run_benchmark(scenario, [dispatch()])

        assert (result.deadlines_met, result.deadline_requests) == (1, 2)
        assert "deadlines met" in format_results([result]).splitlines()[0]

    def test_column_hidden_without_deadlines(self) -> None:
        plain = BenchmarkResult(contender="plain", scenario="s")

        assert "deadlines" not in format_results([plain]).splitlines()[0]


class TestContenderNames:
    def test_names_show_the_allocator_and_its_reserve(self) -> None:
        day = timedelta(days=1)

        assert dispatch().name == "dispatch"
        assert dispatch(allocation=day).name == "dispatch+alloc"
        assert (
            dispatch(allocation=day, allocation_reserve=0.0).name
            == "dispatch+alloc(reserve 0)"
        )
        assert (
            rolling(PriorityPlanner, allocation=day, allocation_reserve=0.2).name
            == "rolling:priority+alloc(reserve 0.2)"
        )


def test_long_range_toos_alternate_urgent_and_routine() -> None:
    scenario = SCENARIOS["long-range"]("examples/example.tle")

    windows = [(too.deadline or 0) - too.submit_time for too in scenario.toos]

    assert len(windows) == 10
    assert all(2 * HOUR <= w <= 6 * HOUR for w in windows[0::2])
    assert all(24 * HOUR <= w <= 72 * HOUR for w in windows[1::2])
