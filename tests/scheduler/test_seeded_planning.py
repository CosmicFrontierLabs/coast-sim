"""A plan seeded by a dispatch run: the local-search planners decode the plan
a queue dispatch executed as a candidate start and improve on the better of
it and the priority-first plan, so the plan they return scores at least as
well as the dispatch run; the benchmark reports both."""

import pytest

from conops.benchmark import dispatch, format_results, planned, run_benchmark
from conops.benchmark.contenders import _dispatch, too_targets
from conops.benchmark.scenarios import baseline, program
from conops.common import ObsType
from conops.ditl.plan_validator import PLAN_SCIENCE_OBSTYPES, entry_obstype
from conops.schedulers import LocalSearchScheduler, PriorityScheduler

TLE = "examples/example.tle"


def _weighted(results: list, name: str) -> float:
    result = next(r for r in results if r.contender == name)
    assert result.error is None, result.error
    return result.weighted_science


class TestStartPlan:
    def test_a_dispatch_plan_decodes_to_its_snapshots_at_their_times(self) -> None:
        """Each science entry of the dispatch plan, for a request the planner
        has, becomes a snapshot at its own start; the ToO the queue observed
        is a request too, so it decodes with the rest."""
        scenario = baseline(TLE, hours=4)
        run = _dispatch(scenario)
        config = scenario.make_config()
        targets = [*scenario.make_targets(config), *too_targets(config, scenario.toos)]
        scheduler = LocalSearchScheduler(
            config,
            targets,
            scenario.begin,
            scenario.end,
            start_plan=run.plan,
            time_limit=0.0,
        )
        scheduler._reserve_fixed()
        start = scheduler._start_from_priority_plan()
        decoded = scheduler._decode_plan(run.plan)
        dispatched = [e for e in run.plan if entry_obstype(e) in PLAN_SCIENCE_OBSTYPES]
        assert any(int(e.obsid) == scenario.toos[0].obsid for e in dispatched)
        releases = {float(e.begin) for e in dispatched}
        assert decoded.sequence and all(s.release in releases for s in decoded.sequence)
        assert any(s.obsid == scenario.toos[0].obsid for s in decoded.sequence)
        placed = [
            b
            for b in decoded.final.timeline
            if entry_obstype(b.entry) in PLAN_SCIENCE_OBSTYPES
        ]
        # Each where the plan had it: at its release or a little after, and
        # before the plan's next snapshot begins.
        for k, (block, snapshot) in enumerate(
            zip(placed, decoded.sequence, strict=True)
        ):
            assert snapshot.release is not None
            assert float(block.entry.begin) >= snapshot.release
            following = (
                decoded.sequence[k + 1] if k + 1 < len(decoded.sequence) else None
            )
            if following is not None and following.release is not None:
                assert float(block.entry.begin) < following.release
        assert scheduler.seed_score is not None and scheduler.seed_score > 0.0
        assert scheduler._objective(start.final) == scheduler.start_score
        unseeded = LocalSearchScheduler(
            config, targets, scenario.begin, scenario.end, time_limit=0.0
        )
        unseeded._reserve_fixed()
        unseeded._start_from_priority_plan()
        assert scheduler.start_score == max(unseeded.start_score, scheduler.seed_score)

    def test_seeding_needs_a_local_search_scheduler(self) -> None:
        with pytest.raises(TypeError, match="start plan"):
            planned(PriorityScheduler, seeded_by_dispatch=True)


class TestAcceptance:
    """The seeded plan's merit-weighted science is at least the dispatch
    run's, measured the same way from telemetry, within one percent: a
    snapshot the queue began and a ToO cut short before the request's
    minimum is science no plan can place, and the planner's slews and
    rounding shift what it decodes by a minute or two."""

    BOUND = 0.99

    @pytest.mark.parametrize("make", [baseline, program])
    def test_the_seeded_local_search_plan_scores_at_least_dispatch(self, make) -> None:  # type: ignore[no-untyped-def]
        scenario = make(TLE, hours=6)
        results = run_benchmark(
            scenario,
            [
                dispatch(),
                planned(LocalSearchScheduler, seeded_by_dispatch=True, time_limit=2.0),
            ],
        )
        table = format_results(results)
        assert "planned:local_search seeded by dispatch" in table
        assert _weighted(
            results, "planned:local_search seeded by dispatch"
        ) >= self.BOUND * _weighted(results, "dispatch")

    def test_the_seeded_cp_sat_plan_scores_at_least_dispatch(self) -> None:
        pytest.importorskip("ortools")
        from conops.schedulers import CpSatScheduler

        scenario = program(TLE, hours=6)
        results = run_benchmark(
            scenario,
            [
                dispatch(),
                planned(CpSatScheduler, seeded_by_dispatch=True, solver_time_limit=2.0),
            ],
        )
        assert _weighted(
            results, "planned:cp_sat seeded by dispatch"
        ) >= self.BOUND * _weighted(results, "dispatch")


class TestProgramScenario:
    def test_the_program_has_its_structure(self) -> None:
        scenario = program(TLE, hours=48)
        config = scenario.make_config()
        targets = scenario.make_targets(config)
        categories = {c.name for c in config.observation_categories.categories}
        assert categories == {"Deep fields", "Campaign", "Monitoring", "Pool"}
        campaign = [t for t in targets if t.sequence is not None]
        assert len(campaign) == 6 and all(len(t.visit_windows) == 2 for t in campaign)
        assert not config.observation_categories.get_category(20000).interruptible
        assert (
            config.observation_categories.get_category(10000).observation_timing
            is not None
        )
        assert len(scenario.toos) == 2
        names = [a.name for a in scenario.activities]
        assert names.count("Calibration") == 4 and "Maintenance" in names
        assert any(a.slides for a in scenario.activities)
        assert any(a.fixed for a in scenario.activities)

    def test_every_contender_executes_the_activities(self) -> None:
        scenario = program(TLE, hours=6)
        results = run_benchmark(scenario, [dispatch(), planned(PriorityScheduler)])
        for result in results:
            assert result.error is None, result.error
        run = _dispatch(scenario)
        assert any(e.obstype == ObsType.ACTIVITY for e in run.plan)
