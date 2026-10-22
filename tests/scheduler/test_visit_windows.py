"""Visits at set times: a target with visit windows is one request, and
every planner and both DITLs take each visit within its window."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from conops import DITL, QueueDITL, create_ditl
from conops.config import MissionConfig
from conops.ditl import RollingHorizonDITL
from conops.schedulers import LocalSearchScheduler, PriorityScheduler
from conops.targets import (
    Plan,
    Pointing,
    TargetQueue,
    VisitWindow,
    cadence_windows,
    visit_windows,
)

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 12
STEP = 60
END = BEGIN + timedelta(hours=HOURS)
VISIT = 10
"""Minutes per visit."""
SLACK = 30 * MIN
STARTS = [T0 + 1 * HOUR, T0 + 4 * HOUR, T0 + 7 * HOUR]
WINDOWS = visit_windows(STARTS, SLACK)


def _timed_target(
    config: MissionConfig, obsid: int = 900, merit: float = 90.0
) -> Pointing:
    return Pointing(
        config=config,
        obsid=obsid,
        name=f"V{obsid}",
        ra=PATCH[0][0],
        dec=PATCH[0][1],
        merit=merit,
        ss_min=VISIT * MIN,
        ss_max=VISIT * MIN,
        visit_windows=list(WINDOWS),
    )


def _fillers(config: MissionConfig) -> list[Pointing]:
    """Long, lower-merit targets that would otherwise fill the horizon."""
    return [
        _target(config, 1 + k, *PATCH[1 + k], merit=50.0, minutes=180, snapshot=60)
        for k in range(3)
    ]


def _visits(plan: Plan, obsid: int = 900) -> list:
    return [e for e in plan if int(e.obsid) == obsid and e.collection_begin is not None]


def _in_windows(entries: list) -> None:
    assert [e.visit for e in entries] == list(range(len(entries)))
    for entry in entries:
        window = WINDOWS[entry.visit]
        assert window.start <= float(entry.collection_begin) <= window.deadline
        assert float(entry.collection_end) - float(entry.collection_begin) == (
            pytest.approx(VISIT * MIN)
        )


class TestModel:
    def test_windows_accept_datetimes_and_strings_and_round_trip(self) -> None:
        when = datetime(2025, 11, 1, 1, tzinfo=timezone.utc)
        window = VisitWindow(start=when, deadline=when.isoformat())
        assert window.start == when.timestamp() == window.deadline
        assert window.contains(when.timestamp())
        loaded = VisitWindow.model_validate_json(window.model_dump_json())
        assert loaded == window
        with pytest.raises(ValidationError, match="deadline"):
            VisitWindow(start=10.0, deadline=5.0)

    def test_helpers_build_windows(self) -> None:
        assert [w.start for w in cadence_windows(T0, 6 * HOUR, 3, 600.0)] == [
            T0,
            T0 + 6 * HOUR,
            T0 + 12 * HOUR,
        ]
        assert all(w.deadline - w.start == SLACK for w in WINDOWS)
        with pytest.raises(ValueError):
            cadence_windows(T0, 0.0, 3, 600.0)
        with pytest.raises(ValueError):
            visit_windows([T0], -1.0)

    def test_a_target_is_visited_once_per_window_whole(self) -> None:
        config = _config(HOURS)
        target = _timed_target(config)
        assert target.visits == 3
        assert target.ss_min == target.ss_max == VISIT * MIN
        assert target.exptime == 3 * VISIT * MIN
        assert target.per_visit_seconds == VISIT * MIN
        shuffled = Pointing(
            config=config,
            obsid=1,
            ra=0.0,
            dec=0.0,
            ss_max=600,
            visit_windows=[WINDOWS[2], WINDOWS[0], WINDOWS[1]],
        )
        assert [w.start for w in shuffled.visit_windows] == STARTS
        with pytest.raises(ValidationError, match="overlap"):
            Pointing(
                config=config,
                obsid=1,
                ra=0.0,
                dec=0.0,
                ss_max=600,
                visit_windows=visit_windows([T0, T0 + 10 * MIN], 20 * MIN),
            )
        dumped = target.model_dump(mode="json")
        assert len(dumped["visit_windows"]) == 3
        assert "visit_windows" not in _target(config, 2, 0.0, 0.0).model_dump(
            mode="json"
        )

    def test_visits_are_found_by_time_and_marked_when_collected(self) -> None:
        target = _timed_target(_config(HOURS))
        assert target.visit_index_at(T0) is None
        assert target.visit_index_at(T0 + 2 * HOUR) == 0
        assert target.visit_index_at(T0 + 9 * HOUR) == 2
        assert target.next_visit(T0) == 0
        assert target.next_visit(T0 + 1 * HOUR + SLACK + 1) == 1
        assert target.next_visit(T0 + 1 * HOUR, after_end=T0 + 1 * HOUR + 20 * MIN) == 1
        assert target.window_bounds(T0) == (WINDOWS[0].start, WINDOWS[0].deadline)
        target.record_collection(T0 + 1 * HOUR + 5 * MIN, 5 * MIN)
        assert target.visits_done == {0}
        assert target.next_visit(T0) == 1
        assert target.window_bounds(T0 + 7 * HOUR + 10 * MIN) == (
            WINDOWS[2].start,
            WINDOWS[2].deadline,
        )
        target.visit = 2
        assert target.current_deadline == WINDOWS[2].deadline
        target.reset()
        assert target.visits_done == frozenset() and target.visit is None
        assert target.next_visit(T0 + 11 * HOUR) is None
        assert target.window_bounds(T0 + 11 * HOUR) == (None, None)


class TestQueue:
    def _queue(self) -> tuple[TargetQueue, Pointing]:
        config = _config(HOURS)
        queue = TargetQueue(config=config, ephem=config.constraint.ephem)
        target = queue.add(
            ra=PATCH[0][0],
            dec=PATCH[0][1],
            obsid=900,
            merit=90.0,
            ss_min=VISIT * MIN,
            ss_max=VISIT * MIN,
            visit_windows=WINDOWS,
        )
        queue.add(ra=PATCH[1][0], dec=PATCH[1][1], obsid=1, merit=50.0, exptime=HOUR)
        return queue, target

    def test_selected_only_while_an_unvisited_window_is_open(self) -> None:
        queue, target = self._queue()
        assert target.exptime == 3 * VISIT * MIN
        assert queue.get(ra=0, dec=0, utime=T0).obsid == 1
        chosen = queue.get(ra=0, dec=0, utime=STARTS[0])
        assert chosen is target and chosen.visit == 0
        assert chosen.end - chosen.begin == pytest.approx(
            chosen.slewtime + VISIT * MIN + queue.observation_timing.total_seconds
        )
        assert queue.get(ra=0, dec=0, utime=STARTS[0] + SLACK + 60).obsid == 1
        target.record_collection(STARTS[0] + 3 * MIN, 3 * MIN)
        assert queue.get(ra=0, dec=0, utime=STARTS[0] + 5 * MIN).obsid == 1
        assert queue.get(ra=0, dec=0, utime=STARTS[1]).visit == 1

    def test_the_next_window_is_a_request_opening(self) -> None:
        queue, target = self._queue()
        assert queue.request_openings(T0) == [(STARTS[0], 90.0, 900)]
        target.record_collection(STARTS[0] + 3 * MIN, 3 * MIN)
        assert queue.request_openings(STARTS[0] + 5 * MIN) == [(STARTS[1], 90.0, 900)]


def _execute(config: MissionConfig, plan: Plan) -> DITL:
    replay = Plan.model_validate_json(plan.model_dump_json())
    ditl = DITL(
        config=config, ephem=config.constraint.ephem, plan=replay, begin=BEGIN, end=END
    )
    ditl.step_size = STEP
    assert ditl.calc()
    return ditl


class TestPlanners:
    def test_the_priority_plan_takes_each_visit_in_its_window(self) -> None:
        config = _config(HOURS)
        targets = [_timed_target(config), *_fillers(config)]
        plan = PriorityScheduler(config, targets, BEGIN, END, step_size=STEP).schedule()
        visits = _visits(plan)
        assert len(visits) == 3
        _in_windows(visits)
        assert [int(e.obsid) for e in plan if e.collection_begin is not None].count(
            900
        ) == 3
        ditl = _execute(config, plan)
        assert ditl.validate_plan_matches_execution() == []

    def test_a_window_that_cannot_be_met_is_skipped(self) -> None:
        """The first window has closed before the plan starts; the other
        two visits are still taken."""
        config = _config(HOURS)
        late = BEGIN + timedelta(hours=2)
        targets = [_timed_target(config), *_fillers(config)]
        plan = PriorityScheduler(config, targets, late, END, step_size=STEP).schedule()
        visits = _visits(plan)
        assert [e.visit for e in visits] == [1, 2]
        for entry in visits:
            assert WINDOWS[entry.visit].contains(float(entry.collection_begin))

    def test_local_search_keeps_the_visits_in_their_windows(self) -> None:
        config = _config(HOURS)
        targets = [_timed_target(config), *_fillers(config)]
        scheduler = LocalSearchScheduler(
            config,
            targets,
            BEGIN,
            END,
            step_size=STEP,
            max_iterations=200,
            time_limit=30.0,
        )
        plan = scheduler.schedule()
        assert scheduler.score >= scheduler.initial_score
        visits = _visits(plan)
        assert len(visits) == 3
        _in_windows(visits)
        assert _execute(config, plan).validate_plan_matches_execution() == []

    def test_cpsat_keeps_the_visits_in_their_windows(self) -> None:
        pytest.importorskip("ortools")
        from conops.schedulers import CpSatScheduler

        config = _config(HOURS)
        targets = [_timed_target(config), *_fillers(config)]
        scheduler = CpSatScheduler(
            config,
            targets,
            BEGIN,
            END,
            step_size=STEP,
            solver_time_limit=10.0,
            workers=1,
            seed=1,
            max_iterations=0,
        )
        plan = scheduler.schedule()
        assert scheduler.solver_score >= scheduler.start_score
        visits = _visits(plan)
        assert len(visits) == 3
        _in_windows(visits)
        assert _execute(config, plan).validate_plan_matches_execution() == []


class TestDispatch:
    def test_dispatch_observes_each_visit_in_its_window(self) -> None:
        config = _config(HOURS)
        targets = [_timed_target(config), *_fillers(config)]
        ditl = create_ditl(config, targets, BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        visits = [
            e
            for e in ditl.plan
            if int(e.obsid) == 900 and e.collection_begin is not None
        ]
        assert len(visits) == 3
        _in_windows(visits)
        assert targets[0].visits_done == {0, 1, 2}
        assert targets[0].done

    def test_a_rolling_horizon_run_keeps_the_visits_in_their_windows(self) -> None:
        config = _config(HOURS)
        targets = [_timed_target(config), *_fillers(config)]
        ditl = RollingHorizonDITL(
            config,
            targets,
            begin=BEGIN,
            end=END,
            horizon=timedelta(hours=5),
            replan_interval=timedelta(hours=2),
            commit_lead_time=timedelta(minutes=30),
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert len(ditl.replans) >= 4
        visits = [
            e
            for e in ditl.plan
            if int(e.obsid) == 900 and e.collection_begin is not None
        ]
        assert len(visits) == 3
        _in_windows(visits)
        assert ditl.validate_plan_matches_execution() == []
