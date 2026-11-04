"""CpSatScheduler: solver plans must execute exactly and never lose to priority-first."""

import sys
from datetime import timedelta

import numpy as np
import pytest

pytest.importorskip("ortools")

from ortools.sat.python import cp_model, cp_model_helper  # noqa: E402

from conops import DITL  # noqa: E402
from conops.common import ObsType  # noqa: E402
from conops.config import MissionConfig, TargetConfig  # noqa: E402
from conops.config.observation_categories import ObservationCategory  # noqa: E402
from conops.ditl import RollingHorizonDITL  # noqa: E402
from conops.schedulers import CpSatScheduler  # noqa: E402
from conops.schedulers.cpsat_scheduler import _ChunkModel, _Task  # noqa: E402
from conops.schedulers.local_search_scheduler import _Snapshot  # noqa: E402
from conops.schedulers.priority_scheduler import program_shares  # noqa: E402
from conops.targets import Plan, PlanEntry, Pointing  # noqa: E402

from .planning_scenario import BEGIN, CONSTRAINED, HOUR, MIN, PATCH, T0  # noqa: E402
from .planning_scenario import make_config as _config  # noqa: E402
from .planning_scenario import make_target as _target  # noqa: E402

MONITOR = ObservationCategory(
    name="Monitor", obsid_min=1, obsid_max=2, cadence_seconds=HOUR
)
PROGRAMS = [
    ObservationCategory(name="A", obsid_min=100, obsid_max=200, time_share=0.5),
    ObservationCategory(name="B", obsid_min=200, obsid_max=300, time_share=0.5),
]


def _targets(
    hours: int, *, stations: bool = False, config: MissionConfig | None = None
) -> list[Pointing]:
    config = config if config is not None else _config(hours, stations=stations)
    return [
        _target(config, 100 + i, ra, dec, merit=90 - 7 * i, minutes=60, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH + CONSTRAINED)
    ]


def _random_targets(config: MissionConfig) -> list[Pointing]:
    """Targets whose snapshots straddle Earth occultations, so the priority-first
    scheduler moves snapshots later to fill the time a target is hidden."""
    rng = np.random.default_rng(5)
    return [
        _target(
            config,
            100 + i,
            rng.uniform(0, 360),
            rng.uniform(-90, 90),
            merit=rng.uniform(10, 100),
            minutes=2000 / 60,
            snapshot=1000 / 60,
            ss_min=5,
        )
        for i in range(40)
    ]


def _monitored(config: MissionConfig) -> list[Pointing]:
    """A cadence target wanting three ten-minute visits, and filler."""
    monitor = _target(config, 1, *PATCH[0], merit=60, minutes=30, snapshot=10)
    filler = [
        _target(config, 10 + i, ra, dec, merit=40, minutes=60, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH[1:])
    ]
    return [monitor, *filler]


def _programs(config: MissionConfig) -> list[Pointing]:
    """Program A outranks B, and either could fill the horizon alone."""
    a = [
        _target(config, 100 + i, ra, dec, merit=100, minutes=120, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH[:3])
    ]
    b = [
        _target(config, 200 + i, ra, dec, merit=60, minutes=120, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH[2:])
    ]
    return [*a, *b]


Option = float | int | timedelta | dict[int, float] | list[PlanEntry]


def _scheduler(
    targets: list[Pointing], hours: int, **options: Option
) -> CpSatScheduler:
    config = targets[0].config
    assert config is not None
    settings: dict[str, Option] = {
        "solver_time_limit": 0.5,
        "workers": 1,
        "seed": 1,
    }
    settings.update(options)
    return CpSatScheduler(
        config,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=hours),
        **settings,  # type: ignore[arg-type]
    )


def _execute(targets: list[Pointing], plan: Plan, hours: int) -> DITL:
    config = targets[0].config
    assert config is not None
    ditl = DITL(
        config=config,
        ephem=config.constraint.ephem,
        plan=Plan.model_validate_json(plan.model_dump_json()),
        begin=BEGIN,
        end=BEGIN + timedelta(hours=hours),
    )
    ditl.step_size = 60
    assert ditl.calc()
    return ditl


def _science(plan: Plan) -> list[int]:
    return [int(e.obsid) for e in plan if e.obstype != ObsType.GSP]


def _visits(plan: Plan, obsid: int) -> list[tuple[float, float]]:
    return sorted(
        (float(e.collection_begin), float(e.collection_end))
        for e in plan
        if e.obsid == obsid
        and e.collection_begin is not None
        and e.collection_end is not None
    )


def _share(plan: Plan, low: int, high: int) -> float:
    """Fraction of the plan's science time spent on obsids in ``[low, high)``."""
    seconds = {int(e.obsid): 0.0 for e in plan if e.collection_begin is not None}
    for e in plan:
        if e.collection_begin is not None and e.collection_end is not None:
            seconds[int(e.obsid)] += float(e.collection_end) - float(e.collection_begin)
    total = sum(seconds.values())
    return sum(s for obsid, s in seconds.items() if low <= obsid < high) / total


def _hinted_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hold the solver to its hint: the priority-first plan's snapshots."""
    solve = cp_model.CpSolver.solve

    def hinted_only(
        solver: cp_model.CpSolver, model: cp_model.CpModel
    ) -> cp_model_helper.CpSolverStatus:
        solver.parameters.fix_variables_to_their_hinted_value = True
        # Check hint feasibility regardless of CI load; the scheduler's short
        # per-chunk search budget can expire before the fixed hint is checked.
        solver.parameters.max_time_in_seconds = float("inf")
        return solve(solver, model)

    monkeypatch.setattr(cp_model.CpSolver, "solve", hinted_only)


def _collected(plan: Plan) -> float:
    return sum(
        float(e.collection_end) - float(e.collection_begin)
        for e in plan
        if e.collection_begin is not None and e.collection_end is not None
    )


class TestPlans:
    def test_never_worse_than_priority_first_and_executes_exactly(self) -> None:
        targets = _targets(4)
        scheduler = _scheduler(targets, 4)

        plan = scheduler.schedule()
        ditl = _execute(targets, plan, 4)

        assert scheduler.score >= scheduler.initial_score
        assert scheduler.solver_statuses
        assert set(scheduler.solver_statuses) <= {"OPTIMAL", "FEASIBLE"}
        assert ditl.validate_plan_matches_execution() == []
        executed = sum(hk.collection_seconds for hk in ditl.telemetry.housekeeping)
        assert executed == pytest.approx(_collected(plan))

    def test_solves_in_several_chunks(self) -> None:
        targets = _targets(3)
        scheduler = _scheduler(targets, 3, chunk=timedelta(hours=1))

        plan = scheduler.schedule()

        assert len(scheduler.solver_statuses) == 3
        assert len(scheduler.solver_chunks_used) == 3
        assert scheduler.solver_score is not None
        assert scheduler.solver_score >= scheduler.start_score
        assert _execute(targets, plan, 3).validate_plan_matches_execution() == []

    def test_plan_with_ground_passes_executes_exactly(self) -> None:
        targets = _targets(12, stations=True)
        scheduler = _scheduler(targets, 12)

        plan = scheduler.schedule()

        assert any(entry.obstype == ObsType.GSP for entry in plan)
        assert _execute(targets, plan, 12).validate_plan_matches_execution() == []

    def test_inputs_are_not_modified(self) -> None:
        targets = _targets(3)

        _scheduler(targets, 3).schedule()

        assert all(t.exptime == 60 * MIN for t in targets)

    def test_with_slews_before_targets_rise_the_plan_executes_exactly(self) -> None:
        config = _config(4)
        config.spacecraft_bus.attitude_control.slew_before_visible = True
        targets = _targets(4, config=config)
        scheduler = _scheduler(targets, 4)

        plan = scheduler.schedule()

        assert scheduler.score >= scheduler.initial_score
        assert _execute(targets, plan, 4).validate_plan_matches_execution() == []

    def test_local_search_after_the_solver_keeps_its_gains(self) -> None:
        targets = _targets(4)
        scheduler = _scheduler(targets, 4, time_limit=600, max_iterations=200)

        plan = scheduler.schedule()

        assert scheduler.iterations == 200
        assert scheduler.solver_score is not None
        assert scheduler.score >= scheduler.solver_score
        assert _execute(targets, plan, 4).validate_plan_matches_execution() == []


def _solve_tasks(tasks: list[_Task], budget: dict[int, float]) -> list[bool]:
    """Solve a chunk of the given tasks, with no time to slew between them,
    and return which are present."""
    chunk = _ChunkModel(tasks, np.zeros((len(tasks), len(tasks))), 60, 10_000, budget)
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.max_time_in_seconds = 10.0
    assert solver.solve(chunk.model) == cp_model.OPTIMAL
    return [bool(solver.value(present)) for present in chunk.present]


ORIGIN = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


class TestCadenceOrder:
    def test_visits_listed_later_window_first_are_spaced_in_time_order(
        self,
    ) -> None:
        """Candidates are ranked, so a later window can be listed before an
        earlier one. Both visits fit an hour apart and both are taken."""

        def visit(window_start: int) -> _Task:
            window = (window_start, window_start + 3000)
            return _Task(
                (1, float(window_start)),
                ORIGIN,
                (window[0], window[1] - 300),
                spacing=HOUR,
                collection=(300, 300),
                window=window,
                rate=1.0,
            )

        tasks = [_Task(None, ORIGIN, (0, 0)), visit(6000), visit(0)]

        assert _solve_tasks(tasks, {1: 600}) == [True, True, True]


class TestCadenceLateness:
    def test_every_visit_on_cadence_is_proposed(self) -> None:
        """Each visit's lateness counts from when it is due, a cadence after
        the visit before it. Counted from the first visit's due time, later
        visits were worth less than nothing and the solver proposed two of
        six; the chunk then lost to the priority-first plan."""
        proposed: list[list[_Snapshot] | None] = []

        class Capture(CpSatScheduler):
            def _solve_chunk(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                proposed.append(super()._solve_chunk(*args, **kwargs))
                return proposed[-1]

        hours = 3
        config = _config(
            hours,
            weights=TargetConfig(cadence_weight=50),
            categories=[
                ObservationCategory(
                    name="Monitor", obsid_min=1, obsid_max=2, cadence_seconds=600
                )
            ],
        )
        target = _target(config, 1, *PATCH[0], merit=60, minutes=30, snapshot=5)
        scheduler = Capture(
            config,
            [target],
            BEGIN,
            BEGIN + timedelta(hours=hours),
            solver_time_limit=3.0,
            workers=1,
            seed=1,
            include_passes=False,
        )

        scheduler.schedule()

        assert scheduler.solver_statuses == ["OPTIMAL"]
        assert scheduler.solver_chunks_used == [True]
        assert proposed[0] is not None and len(proposed[0]) == 6

    def test_a_visit_long_overdue_is_still_worth_taking(self) -> None:
        """A visit loses at most earliness_weight of its value, however late,
        as a deadline request does at its deadline."""
        cadence = 600
        weight = 0.5
        window = (6000, 9000)
        visit = _Task(
            (1, float(window[0])),
            ORIGIN,
            (window[0], window[1] - 300),
            spacing=cadence,
            collection=(300, 300),
            window=window,
            rate=1.0,
            late=weight / cadence,
            cadence=cadence,
        )

        present = _solve_tasks([_Task(None, ORIGIN, (0, 0)), visit], {1: 300})

        assert present == [True, True]


class TestLatenessDiscount:
    HOURS = 3
    WEIGHT = 1.0

    def _scheduler_and_request(
        self,
    ) -> tuple[CpSatScheduler, int, float, float]:
        config = _config(self.HOURS)
        deadline = T0 + 2 * HOUR
        target = _target(
            config, 100, *PATCH[0], minutes=60, snapshot=60, ss_min=1, deadline=deadline
        )
        scheduler = _scheduler(
            [target], self.HOURS, earliness_weight=self.WEIGHT, time_limit=0.0
        )
        scheduler._start_from_priority_plan()
        return scheduler, 100, deadline - T0, scheduler.ctx.setup_seconds

    def test_the_discount_is_local_search_s_for_any_collection(self) -> None:
        """Per second of collection, the solver's value at an arrival is the
        value local search gives the snapshot, so a shortened snapshot loses
        in proportion to its collection, not to the longest it could have."""
        scheduler, obsid, available, setup = self._scheduler_and_request()
        request = scheduler._by_obsid[obsid]
        rate, earliness, offset, _ = scheduler._candidate_value(request, {}, T0)

        for arrival in (0, 1800, 5400, 7000):
            for collection in (60, 600, 3600):
                solver_value = (rate + offset - earliness * arrival) * collection
                delay = arrival + setup
                local_search = rate * collection * (1 - self.WEIGHT * delay / available)
                assert solver_value == pytest.approx(local_search)
                assert solver_value > 0

    def test_a_short_late_snapshot_is_worth_taking(self) -> None:
        """Only 100 s fit, near the deadline: worth a little, but more than
        nothing. Valued at its longest collection it was worth less than none."""
        scheduler, obsid, available, setup = self._scheduler_and_request()
        rate, earliness, offset, _ = scheduler._candidate_value(
            scheduler._by_obsid[obsid], {}, T0
        )
        # On a step, so the slew can start as it arrives, a few minutes
        # before the deadline.
        start = (int(available - setup) - 200) // 60 * 60
        window = (start, start + 100)
        late = _Task(
            (obsid, float(start)),
            ORIGIN,
            (start, start),
            collection=(60, 3600),
            window=window,
            rate=rate,
            earliness=earliness,
            offset=offset,
        )

        present = _solve_tasks([_Task(None, ORIGIN, (0, 0)), late], {obsid: 3600})

        assert present == [True, True]


class TestNoMinimumSnapshot:
    def test_a_request_with_no_minimum_length_is_planned(self) -> None:
        """With ss_min of zero and no observation overhead, a snapshot is
        still at least a second long, as in the priority-first scheduler."""
        config = _config(1)
        target = _target(config, 100, *PATCH[0], minutes=1, snapshot=1, ss_min=0)

        plan = _scheduler([target], 1).schedule()

        assert _science(plan) == [100]
        ditl = _execute([target], plan, 1)
        assert ditl.validate_plan_matches_execution() == []


class TestHint:
    def test_hint_is_feasible_and_reproduces_the_priority_first_plan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Held to its hint, the solver must return the priority-first plan.

        The hint covers each slew starting on a step, passes, and several
        snapshots of one request in a window, so any mismatch between the
        model and the decoder makes a chunk infeasible or lose value.
        """
        _hinted_only(monkeypatch)
        targets = _targets(12, stations=True)
        scheduler = _scheduler(targets, 12)

        scheduler.schedule()

        assert set(scheduler.solver_statuses) == {"OPTIMAL"}
        assert all(scheduler.solver_chunks_used)
        assert scheduler.solver_score == scheduler.start_score

    def test_hint_is_feasible_where_snapshots_were_moved_into_gaps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The priority-first scheduler moved and cut short snapshots to fill the
        time targets were hidden; the hint must still reproduce its plan."""
        _hinted_only(monkeypatch)
        hours = 12
        config = _config(hours)
        targets = _random_targets(config)
        scheduler = _scheduler(targets, hours)

        plan = scheduler.schedule()

        assert set(scheduler.solver_statuses) == {"OPTIMAL"}
        assert scheduler.solver_score == scheduler.start_score
        assert _execute(targets, plan, hours).validate_plan_matches_execution() == []

    def test_hint_is_feasible_with_slews_before_targets_rise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _hinted_only(monkeypatch)
        config = _config(4)
        config.spacecraft_bus.attitude_control.slew_before_visible = True
        scheduler = _scheduler(_targets(4, config=config), 4)

        scheduler.schedule()

        assert set(scheduler.solver_statuses) == {"OPTIMAL"}
        assert scheduler.solver_score == scheduler.start_score

    def test_no_solution_in_time_keeps_the_priority_first_plan(self) -> None:
        targets = _targets(6)
        scheduler = _scheduler(targets, 6, solver_time_limit=1e-6)

        plan = scheduler.schedule()

        assert "UNKNOWN" in scheduler.solver_statuses
        assert not any(scheduler.solver_chunks_used)
        assert scheduler.solver_score == scheduler.start_score
        assert _execute(targets, plan, 6).validate_plan_matches_execution() == []


class TestSeveralSnapshotsInAWindow:
    def test_solver_collects_an_exposure_longer_than_one_snapshot(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, minutes=60, snapshot=20)
        scheduler = CpSatScheduler(
            config,
            [target],
            BEGIN,
            BEGIN + timedelta(hours=2),
            solver_time_limit=5.0,
            workers=1,
            seed=1,
        )

        plan = scheduler.schedule()

        assert scheduler.solver_chunks_used == [True]
        assert _science(plan) == [1, 1, 1]
        assert _collected(plan) == pytest.approx(60 * MIN)

    def test_snapshots_split_by_a_fixed_task_have_a_candidate_each(self) -> None:
        """One full-length snapshot would cover the exposure, but a locked
        observation splits the window, so the priority-first plan takes two
        shorter ones. With one candidate the solver could propose only one,
        and the chunk lost to the priority-first plan."""
        candidates: list[list[_Task]] = []
        proposed: list[list[_Snapshot] | None] = []

        class Capture(CpSatScheduler):
            def _candidates(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                candidates.append(super()._candidates(*args, **kwargs))
                return candidates[-1]

            def _solve_chunk(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                proposed.append(super()._solve_chunk(*args, **kwargs))
                return proposed[-1]

        config = _config(1)
        target = _target(
            config, 1, 105.0, 10.0, merit=100, minutes=20, snapshot=20, ss_min=5
        )
        locked = PlanEntry(
            config=config,
            name="Fixed",
            obsid=900,
            ra=115.0,
            dec=10.0,
            roll=0.0,
            obstype=ObsType.AT,
            begin=T0 + 720,
            end=T0 + 1320,
            collection_begin=T0 + 720,
            collection_end=T0 + 1320,
            ss_min=300,
        )
        scheduler = Capture(
            config,
            [target],
            BEGIN,
            BEGIN + timedelta(minutes=35),
            solver_time_limit=3.0,
            workers=1,
            seed=1,
            include_passes=False,
            locked=[locked],
        )

        plan = scheduler.schedule()

        assert len(_visits(plan, 1)) == 2
        assert len(candidates[0]) == 2
        assert proposed[0] is not None
        assert sum(snapshot.seconds for snapshot in proposed[0]) == pytest.approx(
            20 * MIN
        )
        assert scheduler.solver_statuses == ["OPTIMAL"]
        assert scheduler.solver_chunks_used == [True]


class TestShortWindowRequest:
    def test_solver_fits_the_request_priority_order_lost(self) -> None:
        """Priority order fills B's only slot with A; the solver fits both."""
        config = _config(3)
        flexible = _target(config, 1, 105.0, 10.0, merit=100, minutes=60)
        urgent = _target(
            config, 2, 110.0, 10.0, merit=70, minutes=30, deadline=T0 + 30 * MIN
        )

        scheduler = CpSatScheduler(
            config,
            [flexible, urgent],
            BEGIN,
            BEGIN + timedelta(hours=3),
            solver_time_limit=5.0,
            workers=1,
            seed=1,
        )
        plan = scheduler.schedule()

        assert _science(plan) == [2, 1]
        assert scheduler.solver_score is not None
        assert scheduler.solver_score > scheduler.initial_score
        urgent_entry = next(e for e in plan if e.obsid == 2)
        assert urgent_entry.collection_begin is not None
        assert urgent_entry.collection_begin <= T0 + 30 * MIN


class TestEarliestStart:
    def test_solver_keeps_the_earliest_start(self) -> None:
        config = _config(3)
        flexible = _target(config, 1, 105.0, 10.0, merit=100, minutes=60)
        later = _target(
            config,
            2,
            110.0,
            10.0,
            merit=70,
            minutes=30,
            earliest_start=T0 + HOUR,
            deadline=T0 + 90 * MIN,
        )

        scheduler = CpSatScheduler(
            config,
            [flexible, later],
            BEGIN,
            BEGIN + timedelta(hours=3),
            solver_time_limit=5.0,
            workers=1,
            seed=1,
        )
        plan = scheduler.schedule()

        entry = next(e for e in plan if e.obsid == 2)
        assert entry.collection_begin is not None
        assert T0 + HOUR <= entry.collection_begin <= T0 + 90 * MIN


class TestSlewBeforeEarliestStart:
    def test_slew_may_start_before_the_earliest_start(self) -> None:
        """The earliest start bounds the collection, not the slew. The request's
        one-minute collection window is shorter than its slew, so the slew
        must start before the window opens."""
        config = _config(3)
        flexible = _target(config, 1, 105.0, 10.0, merit=100, minutes=60)
        urgent = _target(
            config,
            2,
            110.0,
            10.0,
            merit=70,
            minutes=30,
            earliest_start=T0 + 1200,
            deadline=T0 + 1260,
        )
        targets = [flexible, urgent]
        scheduler = _scheduler(targets, 3, solver_time_limit=3.0, include_passes=False)

        plan = scheduler.schedule()

        assert {entry.obsid for entry in plan} == {1, 2}
        entry = next(e for e in plan if e.obsid == 2)
        assert entry.collection_begin is not None
        assert T0 + 1200 <= entry.collection_begin <= T0 + 1260
        assert float(entry.begin) < T0 + 1200
        assert _execute(targets, plan, 3).validate_plan_matches_execution() == []


class TestDynamicMerit:
    """The solver keeps the cadence and program-share rules of the
    priority-first scheduler, and its hint stays feasible with them."""

    HOURS = 4

    def _monitor_config(self) -> MissionConfig:
        return _config(
            self.HOURS, weights=TargetConfig(cadence_weight=50), categories=[MONITOR]
        )

    def _programs_config(self) -> MissionConfig:
        return _config(
            self.HOURS,
            weights=TargetConfig(completion_deficit_weight=100),
            categories=PROGRAMS,
        )

    def test_cadence_visits_stay_apart(self) -> None:
        config = self._monitor_config()
        targets = _monitored(config)

        plan = _scheduler(targets, self.HOURS).schedule()

        visits = _visits(plan, 1)
        assert len(visits) == 3
        for (_, end), (begin, _) in zip(visits, visits[1:]):
            assert begin >= end + HOUR
        assert (
            _execute(targets, plan, self.HOURS).validate_plan_matches_execution() == []
        )

    def test_committed_visits_delay_the_next(self) -> None:
        config = self._monitor_config()

        plan = _scheduler(
            _monitored(config),
            self.HOURS,
            reserved_visits={1: T0 + 20 * MIN},
        ).schedule()

        first, _ = _visits(plan, 1)[0]
        assert first >= T0 + 80 * MIN

    def test_cadence_visits_stay_clear_of_a_locked_visit(self) -> None:
        config = self._monitor_config()
        lock_start = T0 + 90 * MIN
        locked = PlanEntry(
            config=config,
            name="locked",
            ra=PATCH[0][0],
            dec=PATCH[0][1],
            roll=0.0,
            obsid=1,
            obstype=ObsType.AT,
            begin=lock_start,
            end=lock_start + 10 * MIN,
            collection_begin=lock_start,
            collection_end=lock_start + 10 * MIN,
            ss_min=60,
        )

        plan = _scheduler(_monitored(config), self.HOURS, locked=[locked]).schedule()

        planned = [
            v for v in _visits(plan, 1) if v != (lock_start, lock_start + 10 * MIN)
        ]
        assert planned
        for begin, end in planned:
            assert end + HOUR <= lock_start or begin >= lock_start + 10 * MIN + HOUR

    def test_shares_balance_programs(self) -> None:
        config = self._programs_config()
        targets = _programs(config)

        plan = _scheduler(targets, self.HOURS).schedule()

        assert _share(plan, 200, 300) > 0.2
        assert (
            _execute(targets, plan, self.HOURS).validate_plan_matches_execution() == []
        )

    def test_entries_carry_their_value_where_they_are_placed(self) -> None:
        """Each entry's merit counts the program shares the plan delivers before
        it, so a Target of Opportunity must beat that value to interrupt it."""
        config = self._programs_config()
        scheduler = _scheduler(_programs(config), self.HOURS)

        plan = scheduler.schedule()

        programs: dict[str, float] = {}
        merits = []
        for entry in sorted(
            (e for e in plan if e.obstype != ObsType.GSP), key=lambda e: e.begin
        ):
            request = scheduler._by_obsid[int(entry.obsid)]
            expected = scheduler._value(request, program_shares(programs))
            merits.append(float(entry.merit))
            assert float(entry.merit) == pytest.approx(expected)
            program = request.category.program_name
            programs[program] = programs.get(
                program, 0.0
            ) + entry.collection_seconds_between(entry.begin, entry.end)
        # Not all at the value they had at the start of the horizon.
        assert len(set(merits)) > 2

    def test_hint_is_feasible_with_dynamic_terms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _hinted_only(monkeypatch)
        config = _config(
            self.HOURS,
            weights=TargetConfig(cadence_weight=50, completion_deficit_weight=100),
            categories=[MONITOR, *PROGRAMS],
        )
        scheduler = _scheduler(
            [*_monitored(config), *_programs(config)],
            self.HOURS,
            chunk=timedelta(hours=1),
        )

        scheduler.schedule()

        assert set(scheduler.solver_statuses) == {"OPTIMAL"}
        assert scheduler.solver_score == scheduler.start_score


class TestRollingHorizon:
    def test_rolling_replanning_uses_the_cp_sat_scheduler(self) -> None:
        targets = _targets(3)
        ditl = RollingHorizonDITL(
            targets[0].config,  # type: ignore[arg-type]
            targets,
            begin=BEGIN,
            end=BEGIN + timedelta(hours=3),
            replan_interval=timedelta(hours=1),
            scheduler=CpSatScheduler,
            scheduler_options={"solver_time_limit": 0.5, "workers": 1, "seed": 1},
        )
        ditl.step_size = 60

        assert ditl.calc()

        assert len(ditl.replans) == 3
        assert ditl.validate_plan_matches_execution() == []


class TestArguments:
    @pytest.mark.parametrize(
        "options",
        [
            {"solver_time_limit": 0.0},
            {"chunk": timedelta(0)},
            {"workers": 0},
            {"max_candidates": 0},
        ],
    )
    def test_rejects_invalid_settings(
        self, options: dict[str, float | int | timedelta]
    ) -> None:
        config = _config(1)

        with pytest.raises(ValueError):
            CpSatScheduler(
                config,
                [],
                BEGIN,
                BEGIN + timedelta(hours=1),
                **options,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("seed", [2**32 - 1, 2**31, -1])
    def test_seeds_outside_cp_sat_range_are_accepted(self, seed: int) -> None:
        """CP-SAT's seed is a signed 32-bit integer; local search takes any."""
        targets = _targets(1)

        plan = _scheduler(targets, 1, seed=seed).schedule()

        assert _science(plan)

    def test_a_large_configured_seed_is_accepted(self) -> None:
        config = _config(1)
        config.random_seed = 2**32 - 1
        targets = _targets(1, config=config)

        scheduler = CpSatScheduler(
            config,
            targets,
            BEGIN,
            BEGIN + timedelta(hours=1),
            solver_time_limit=5.0,
            workers=1,
        )

        assert _science(scheduler.schedule())

    def test_missing_ortools_names_the_package(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        targets = _targets(1)
        monkeypatch.setitem(sys.modules, "ortools.sat.python", None)

        with pytest.raises(ImportError, match="ortools"):
            _scheduler(targets, 1).schedule()
