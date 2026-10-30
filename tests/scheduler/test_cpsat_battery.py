"""CpSatScheduler plans within the battery, in the model itself."""

from datetime import timedelta

import numpy as np
import pytest

pytest.importorskip("ortools")

from ortools.sat.python import cp_model  # noqa: E402

from conops.schedulers import CpSatScheduler  # noqa: E402
from conops.schedulers.cpsat_scheduler import (  # noqa: E402
    _ChunkBattery,
    _ChunkModel,
    _Task,
)

from .planning_scenario import BEGIN  # noqa: E402
from .test_battery_constraint import (  # noqa: E402
    BORESIGHT,
    HOURS,
    LEVEL,
    STEP,
    _battery_config,
    _charges,
    _execute,
    _targets,
)
from .test_cpsat_scheduler import _hinted_only  # noqa: E402
from .test_priority_scheduler import _science  # noqa: E402

ORIGIN = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
WH = 3600
"""Watt-seconds."""


def _solve(
    tasks: list[_Task], battery: _ChunkBattery | None
) -> tuple[list[bool], list[int]]:
    """Solve a chunk with no time to slew: which tasks are present, and their arrivals."""
    chunk = _ChunkModel(
        tasks,
        np.zeros((len(tasks), len(tasks))),
        60,
        20_000,
        {1: 1e9, 2: 1e9},
        None,
        battery,
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.max_time_in_seconds = 10.0
    assert solver.solve(chunk.model) == cp_model.OPTIMAL
    return (
        [bool(solver.value(p)) for p in chunk.present],
        [int(solver.value(a)) for a in chunk.arrival],
    )


def _snapshot(obsid: int, arrival: tuple[int, int], watts: int) -> _Task:
    """A 600 s snapshot drawing ``watts`` net, in sunlight or eclipse alike."""
    return _Task(
        (obsid, 0.0),
        ORIGIN,
        arrival,
        collection=(600, 600),
        window=(0, 20_000),
        rate=1.0,
        power=(watts, watts),
    )


def _battery(
    edges: list[int],
    sunlit: list[bool],
    *,
    limit_wh: float,
    idle: list[int],
    floor_wh: float = 0.0,
) -> _ChunkBattery:
    runs = len(edges) + 1
    return _ChunkBattery(
        edges=edges,
        sunlit=sunlit,
        entering=0,
        idle=idle,
        slew=[0] * runs,
        fixed=[0] * runs,
        limit=int(limit_wh * WH),
        floor=int(floor_wh * WH),
        horizon=20_000,
    )


class TestChunkModel:
    def test_an_eclipse_holds_only_what_the_battery_carries(self) -> None:
        """Two 600 s snapshots at 300 W draw 50 Wh each; an eclipse run with a
        60 Wh limit and no idle draw holds one of them, and the other goes
        after the run ends, or spills past it, where the sunlit time charges."""
        tasks = [
            _Task(None, ORIGIN, (0, 0)),
            _snapshot(1, (0, 15_000), 300),
            _snapshot(2, (0, 15_000), 300),
        ]
        battery = _battery([5000], [False, True], limit_wh=60.0, idle=[0, -400])

        present, arrival = _solve(tasks, battery)

        assert present == [True, True, True]
        assert min(arrival[1:]) < 5000
        assert max(arrival[1:]) >= 5000 - 600

    def test_a_snapshot_that_fits_nowhere_is_left_out(self) -> None:
        tasks = [
            _Task(None, ORIGIN, (0, 0)),
            _snapshot(1, (0, 3000), 300),
            _snapshot(2, (0, 3000), 300),
        ]
        battery = _battery([5000], [False, True], limit_wh=60.0, idle=[0, -400])

        present, _ = _solve(tasks, battery)

        assert present[0] and present[1:].count(True) == 1

    def test_idle_draw_in_eclipse_counts(self) -> None:
        """At 100 W idle, a 5000 s eclipse run draws 139 Wh on its own, and a
        600 s snapshot at 300 W adds 33 Wh for the idle time it replaces: it
        fits a 180 Wh limit but not a 170 Wh one."""
        tasks = [_Task(None, ORIGIN, (0, 0)), _snapshot(1, (0, 3000), 300)]
        fits = _battery([5000], [False, True], limit_wh=180.0, idle=[100, -400])
        assert _solve(tasks, fits)[0] == [True, True]
        tight = _battery([5000], [False, True], limit_wh=170.0, idle=[100, -400])
        assert _solve(tasks, tight)[0] == [True, False]

    def test_charging_is_not_counted_below_the_recharge_floor(self) -> None:
        """An eclipse run at 50 W idle draws 69 Wh; the sunlit run after it
        charges far more back, so a 50 Wh snapshot in the eclipse after that
        fits a 100 Wh limit. With a 60 Wh floor the sunlit run leaves 60 Wh
        of deficit, and the snapshot does not fit."""
        tasks = [_Task(None, ORIGIN, (0, 0)), _snapshot(1, (10_000, 15_000), 300)]
        edges, sunlit, idle = [5000, 10_000], [False, True, False], [50, -400, 0]
        full = _battery(edges, sunlit, limit_wh=100.0, idle=idle)
        assert _solve(tasks, full)[0] == [True, True]
        floored = _battery(edges, sunlit, limit_wh=100.0, idle=idle, floor_wh=60.0)
        assert _solve(tasks, floored)[0] == [True, False]

    def test_a_snapshot_spilling_through_a_short_run_is_counted_in_each(
        self,
    ) -> None:
        """A 600 s snapshot at 300 W from 700 s: 300 s in an eclipse run at
        0 W idle, 100 s in a 100 s sunlit run charging at 400 W, and 200 s in
        the eclipse after. That is 25 + 19.4 + 16.7 Wh, less the 11.1 Wh the
        sunlit run's idle charges: 50 Wh, within a 60 Wh limit. Spilled
        wholly into the sunlit run it counted 72 Wh and was left out."""
        tasks = [_Task(None, ORIGIN, (0, 0)), _snapshot(1, (700, 700), 300)]
        battery = _battery(
            [1000, 1100], [False, True, False], limit_wh=60.0, idle=[0, -400, 0]
        )
        assert _solve(tasks, battery)[0] == [True, True]

    def test_without_a_battery_everything_fits(self) -> None:
        tasks = [
            _Task(None, ORIGIN, (0, 0)),
            _snapshot(1, (0, 3000), 300),
            _snapshot(2, (0, 3000), 300),
        ]
        assert _solve(tasks, None)[0] == [True, True, True]


def _scheduler(config, targets, **options) -> CpSatScheduler:
    settings = {"solver_time_limit": 20.0, "workers": 1, "seed": 1, "max_iterations": 0}
    settings.update(options)
    return CpSatScheduler(
        config,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=HOURS),
        step_size=STEP,
        **settings,  # type: ignore[arg-type]
    )


class TestScheduler:
    def test_the_solvers_plan_stays_within_the_level_and_executes(self) -> None:
        config = _battery_config(BORESIGHT, stations=True)
        # Exercise a real solve and execution; optimizing the plan to exhaustion
        # adds no coverage to the battery and execution assertions below.
        scheduler = _scheduler(config, _targets(config), solver_time_limit=2.0)

        plan = scheduler.schedule()

        assert scheduler.solver_score >= scheduler.start_score
        assert "INFEASIBLE" not in scheduler.solver_statuses
        assert set(scheduler.solver_statuses) & {"OPTIMAL", "FEASIBLE"}
        assert _science(plan) and _charges(plan)
        ditl = _execute(config, plan)
        assert ditl.validate_plan_matches_execution() == []
        assert min(ditl.batterylevel) >= LEVEL - 1e-6

    def test_hint_with_the_battery_reproduces_the_priority_first_plan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Held to its hint, the model's battery constraints must accept the
        priority-first plan, which the ledger built within the level."""
        _hinted_only(monkeypatch)
        config = _battery_config(BORESIGHT, stations=True)
        scheduler = _scheduler(config, _targets(config))

        scheduler.schedule()

        assert set(scheduler.solver_statuses) <= {"OPTIMAL", "FEASIBLE"}
        assert scheduler.solver_score == scheduler.start_score

    def test_hint_with_a_recharge_level_reproduces_the_priority_first_plan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With charging credited only to 90%, the carry-over between runs is
        bounded below by the recharge floor too, and the priority-first plan,
        built within the same floor, must still be accepted."""
        _hinted_only(monkeypatch)
        config = _battery_config(BORESIGHT, stations=True)
        config.battery.planning_recharge_level = 0.9
        scheduler = _scheduler(config, _targets(config))

        scheduler.schedule()

        assert scheduler._battery.recharge_floor == pytest.approx(100.0)
        assert set(scheduler.solver_statuses) <= {"OPTIMAL", "FEASIBLE"}
        assert scheduler.solver_score == scheduler.start_score

    def test_without_a_level_the_model_has_no_battery(self) -> None:
        config = _battery_config(BORESIGHT, level=None)
        scheduler = _scheduler(config, _targets(config), solver_time_limit=5.0)
        scheduler.schedule()
        assert not scheduler._battery.active
        assert scheduler.solver_score >= scheduler.start_score
