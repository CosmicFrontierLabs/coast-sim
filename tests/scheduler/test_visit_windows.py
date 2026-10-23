"""Visits at set times: a target with visit windows is one request, and
every planner and both DITLs take each visit within its window."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from conops import DITL, QueueDITL, create_ditl
from conops.config import MissionConfig, TargetConfig
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
from conops.targets.merit import MeritModel

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
        with pytest.raises(ValidationError, match="earliest_start and deadline"):
            Pointing(
                config=config,
                obsid=1,
                ra=0.0,
                dec=0.0,
                ss_max=600,
                deadline=T0 + HOUR,
                visit_windows=list(WINDOWS),
            )
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

    def test_urgency_rises_towards_the_next_windows_deadline(self) -> None:
        """A windowed target's urgency is its next window's, as a deadline's
        would be: none well before it, full within the timescale of it."""
        config = _config(HOURS, weights=TargetConfig(urgency_weight=50.0))
        model = MeritModel(config)
        target = _timed_target(config)
        target.visibility()
        early = model.value_terms(target, STARTS[0] - 6 * HOUR).urgency
        near = model.value_terms(target, WINDOWS[0].deadline - 10 * MIN).urgency
        assert early < 50.0
        assert near == pytest.approx(50.0)
        target.record_collection(STARTS[0] + 3 * MIN, 3 * MIN)
        later = model.value_terms(target, WINDOWS[0].deadline - 10 * MIN).urgency
        assert later < near

    def test_the_next_window_is_a_request_opening(self) -> None:
        """The look-ahead keeps time for the next unvisited window from its
        latest start: its deadline less the slew allowance and the setup."""
        queue, target = self._queue()
        allowance = queue._slew_allowance()
        assert allowance is not None
        setup = queue.observation_timing.setup_seconds
        latest = [w.deadline - allowance - setup for w in WINDOWS]
        assert queue.request_openings(T0) == [(latest[0], 90.0, 900)]
        target.record_collection(STARTS[0] + 3 * MIN, 3 * MIN)
        assert queue.request_openings(STARTS[0] + 5 * MIN) == [(latest[1], 90.0, 900)]
        # Asked to, the target is kept from each window's opening.
        target.protect_opening = True
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

    def test_a_window_points_and_rolls_on_its_own(self) -> None:
        """Windows with their own pointing are observed there, in dispatch
        and in the priority plan, and a window's roll offset turns the
        visit's roll by that much from the roll it would otherwise have."""
        positions = [PATCH[0], PATCH[4], (112.0, 15.0)]
        windows = [
            VisitWindow(
                start=w.start,
                deadline=w.deadline,
                ra=ra,
                dec=dec,
                roll_offset_deg=5.0 if k == 1 else None,
            )
            for k, (w, (ra, dec)) in enumerate(zip(WINDOWS, positions, strict=True))
        ]
        plain = [
            VisitWindow(start=w.start, deadline=w.deadline, ra=ra, dec=dec)
            for w, (ra, dec) in zip(WINDOWS, positions, strict=True)
        ]

        def entries(plan) -> list:
            return sorted(
                (
                    e
                    for e in plan
                    if int(e.obsid) == 900 and e.collection_begin is not None
                ),
                key=lambda e: float(e.begin),
            )

        rolls: dict[str, list[float]] = {}
        for label, given in (("offset", windows), ("plain", plain)):
            config = _config(HOURS)
            target = _timed_target(config)
            target.visit_windows = list(given)
            ditl = create_ditl(config, [target, *_fillers(config)], BEGIN, END)
            assert isinstance(ditl, QueueDITL)
            ditl.step_size = STEP
            assert ditl.calc()
            visits = entries(ditl.plan)
            assert len(visits) == 3
            _in_windows(visits)
            for entry, (ra, dec) in zip(visits, positions, strict=True):
                assert (float(entry.ra), float(entry.dec)) == pytest.approx((ra, dec))
            assert [e.visit for e in visits] == [0, 1, 2]
            queued = next(t for t in ditl.queue.targets if int(t.obsid) == 900)
            assert queued.visits_done == {0, 1, 2}
            rolls[label] = [float(e.roll) for e in visits]
        turned = (rolls["offset"][1] - rolls["plain"][1]) % 360.0
        assert min(turned, 360.0 - turned) == pytest.approx(5.0, abs=0.5)
        assert rolls["offset"][0] == pytest.approx(rolls["plain"][0], abs=0.5)

        config = _config(HOURS)
        target = _timed_target(config)
        target.visit_windows = list(windows)
        scheduler = PriorityScheduler(
            config, [target, *_fillers(config)], BEGIN, END, step_size=STEP
        )
        planned = entries(scheduler.schedule())
        assert len(planned) == 3
        _in_windows(planned)
        for entry, (ra, dec) in zip(planned, positions, strict=True):
            assert (float(entry.ra), float(entry.dec)) == pytest.approx((ra, dec))
        replay = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=Plan.model_validate_json(scheduler.plan.model_dump_json()),
            begin=BEGIN,
            end=END,
        )
        replay.step_size = STEP
        assert replay.calc()
        assert replay.validate_plan_matches_execution() == []

    def test_a_window_validates_its_own_request(self) -> None:
        with pytest.raises(ValidationError, match="both ra and dec"):
            VisitWindow(start=T0, deadline=T0 + SLACK, ra=10.0)
        with pytest.raises(ValidationError, match="not both"):
            VisitWindow(
                start=T0, deadline=T0 + SLACK, roll_deg=10.0, roll_offset_deg=5.0
            )
        assert not VisitWindow(start=T0, deadline=T0 + SLACK).own_request
        assert VisitWindow(start=T0, deadline=T0 + SLACK, roll_deg=10.0).own_request

    def test_a_visit_cut_short_is_taken_again_in_its_window(self) -> None:
        """A target of opportunity cuts the first visit: the window counts
        as unvisited, the visit is taken again whole while the window is
        open, the cut science still counts, and the retries do not end the
        target before its last window."""
        config = _config(HOURS)
        target = _timed_target(config)
        ditl = create_ditl(config, [target, *_fillers(config)], BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        cut_at = STARTS[0] + 6 * MIN
        ditl.submit_too(
            obsid=30001,
            ra=PATCH[0][0] + 2.0,
            dec=PATCH[0][1],
            merit=1000.0,
            exptime=5 * MIN,
            deadline=cut_at + 30 * MIN,
            name="Urgent",
            submit_time=cut_at,
        )
        assert ditl.calc()
        visits = sorted(
            (
                e
                for e in ditl.plan
                if int(e.obsid) == 900 and e.collection_begin is not None
            ),
            key=lambda e: float(e.begin),
        )
        assert len(visits) == 4
        first, again = visits[0], visits[1]
        assert first.visit == 0 and again.visit == 0
        assert 0 < first.exposure < VISIT * MIN
        assert again.exposure == VISIT * MIN
        assert float(again.collection_begin) <= WINDOWS[0].deadline
        assert [e.visit for e in visits[2:]] == [1, 2]
        queued = next(t for t in ditl.queue.targets if int(t.obsid) == 900)
        assert queued.visits_done == {0, 1, 2}
        assert queued.done
        assert queued.collected_seconds > 3 * VISIT * MIN
        assert any(
            "Visit 1 of V900 cut short: taken again" in e.description
            for e in ditl.log.events
        )

    def test_the_planners_take_a_cut_visit_again_in_its_window(self) -> None:
        """A target whose first visit was cut short before the plan keeps
        that window open: the plan places the visit again within it."""
        config = _config(HOURS)
        target = _timed_target(config)
        cut = STARTS[0] + 5 * MIN
        target.record_collection(cut, 3 * MIN)
        target.record_interruption(cut)
        assert target.visits_done == set()
        assert target.exptime == 3 * VISIT * MIN
        scheduler = PriorityScheduler(
            config,
            [target, *_fillers(config)],
            BEGIN + timedelta(minutes=70),
            END,
            step_size=STEP,
        )
        plan = scheduler.schedule()
        visits = sorted(
            (e for e in plan if int(e.obsid) == 900 and e.collection_begin is not None),
            key=lambda e: float(e.begin),
        )
        assert [e.visit for e in visits] == [0, 1, 2]
        _in_windows(visits)
        assert all(e.exposure == VISIT * MIN for e in visits)

    def test_an_interruption_reopens_only_an_incomplete_window(self) -> None:
        config = _config(HOURS)
        target = _timed_target(config)
        begin = STARTS[0] + 60.0
        target.record_collection(begin + 3 * MIN, 3 * MIN)
        assert target.visits_done == {0}
        target.record_interruption(begin + 3 * MIN)
        assert target.visits_done == set()
        target.record_collection(begin + 10 * MIN + 3 * MIN, VISIT * MIN)
        target.record_interruption(begin + 10 * MIN + 3 * MIN + 30.0)
        assert target.visits_done == {0}
        assert not target.done
        assert target.collected_seconds == 13 * MIN
        target.exptime = 0
        assert not target.done

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
