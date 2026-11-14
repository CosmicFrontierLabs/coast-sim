"""A cadence visit placed at the opening of its window moves later, within
the spacing to the target's next visit, to let a snapshot be placed before
it; the visits stay a cadence apart."""

from unittest.mock import patch

from conops.benchmark.scenarios import program
from conops.common import ObsType
from conops.ditl import DITL
from conops.ditl.plan_validator import PLAN_SCIENCE_OBSTYPES, entry_obstype
from conops.schedulers import PriorityScheduler

TLE = "examples/example.tle"


def _science_hours(plan) -> float:  # type: ignore[no-untyped-def]
    return (
        sum(
            e.collection_seconds_between(float(e.begin), float(e.end))
            for e in plan
            if entry_obstype(e) in PLAN_SCIENCE_OBSTYPES
        )
        / 3600.0
    )


def _plan(disable_moves: bool = False):  # type: ignore[no-untyped-def]
    scenario = program(TLE, hours=24)
    config = scenario.make_config()
    targets = scenario.make_targets(config)
    maintenance = [a for a in scenario.activities if a.name == "Maintenance"]
    scheduler = PriorityScheduler(
        config, targets, scenario.begin, scenario.end, activities=maintenance
    )
    if disable_moves:
        with patch.object(
            PriorityScheduler, "_fit_delaying_successors", return_value=None
        ):
            plan = scheduler.schedule()
    else:
        plan = scheduler.schedule()
    return config, scenario, scheduler, plan


class TestSuccessorShift:
    def test_cadence_visits_move_later_and_the_day_fills(self) -> None:
        """With monitoring visits anchored at their windows' openings and a
        fixed maintenance window, the plan holds a day's science only when
        the visits may move; every target's visits stay a cadence apart."""
        config, scenario, scheduler, plan = _plan()
        assert _science_hours(plan) > 8.0
        moved = [e for e in scheduler.log.events if e.description.startswith("Moved")]
        assert any(30000 <= int(e.obsid) < 30010 for e in moved if e.obsid is not None)
        cadence = config.observation_categories.get_category(30000).cadence_seconds
        assert cadence is not None
        visits: dict[int, list[tuple[float, float]]] = {}
        for e in plan:
            if 30000 <= int(e.obsid) < 30010 and e.collection_end is not None:
                visits.setdefault(int(e.obsid), []).append(
                    (float(e.begin), float(e.collection_end))
                )
        for spans in visits.values():
            spans.sort()
            for (_, end), (begin, _) in zip(spans, spans[1:]):
                assert begin >= end + cadence - 1e-6
        _, _, _, fixed = _plan(disable_moves=True)
        assert _science_hours(fixed) < _science_hours(plan) / 2

    def test_the_plan_executes_as_planned(self) -> None:
        config, scenario, _, plan = _plan()
        ditl = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=plan,
            begin=scenario.begin,
            end=scenario.end,
        )
        ditl.step_size = 60
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        assert any(e.obstype == ObsType.ACTIVITY for e in ditl.plan)
