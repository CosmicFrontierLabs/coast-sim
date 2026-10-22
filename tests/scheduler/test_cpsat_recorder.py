"""CpSatScheduler plans within the onboard recorder, in the model itself."""

from datetime import timedelta

import numpy as np
import pytest

pytest.importorskip("ortools")

from ortools.sat.python import cp_model  # noqa: E402

from conops.config import HousekeepingData  # noqa: E402
from conops.schedulers import CpSatScheduler  # noqa: E402
from conops.schedulers.cpsat_scheduler import (  # noqa: E402
    _ChunkModel,
    _ChunkRecorder,
    _Task,
)

from .planning_scenario import BEGIN  # noqa: E402
from .test_cpsat_scheduler import _hinted_only  # noqa: E402
from .test_priority_scheduler import _execute, _science  # noqa: E402
from .test_recorder_constraint import (  # noqa: E402
    HIGH_RATE,
    HOURS,
    LIMIT,
    LOW_RATE,
    STEP,
    _data_target,
    _recorder_config,
)

ORIGIN = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
GB = 1_000_000
"""Kilobits."""


def _solve(
    tasks: list[_Task], recorder: _ChunkRecorder | None
) -> tuple[list[bool], list[int]]:
    """Solve a chunk with no time to slew: which tasks are present, and their arrivals."""
    chunk = _ChunkModel(
        tasks,
        np.zeros((len(tasks), len(tasks))),
        60,
        20_000,
        {1: 1e9, 2: 1e9},
        recorder,
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.max_time_in_seconds = 10.0
    assert solver.solve(chunk.model) == cp_model.OPTIMAL
    return (
        [bool(solver.value(p)) for p in chunk.present],
        [int(solver.value(a)) for a in chunk.arrival],
    )


def _snapshot(obsid: int, arrival: tuple[int, int], gb: float) -> _Task:
    """A 600 s snapshot generating ``gb`` whenever it is taken."""
    return _Task(
        (obsid, 0.0),
        ORIGIN,
        arrival,
        collection=(600, 600),
        window=(0, 20_000),
        rate=1.0,
        data=(0, int(gb * GB)),
    )


def _recorder(
    edges: list[int], removal: list[int], *, limit: float, entering: float = 0.0
) -> _ChunkRecorder:
    segments = len(edges) + 1
    return _ChunkRecorder(
        edges=edges,
        entering=int(entering * GB),
        accrual=[0] * segments,
        fixed=[0] * segments,
        removal=[0, *(int(r * GB) for r in removal)],
        removal_accrual=[0] * segments,
        limit=int(limit * GB),
        setup=0,
    )


class TestChunkModel:
    def test_two_snapshots_that_do_not_fit_together_straddle_the_contact(self) -> None:
        """Each snapshot is 12 Gb and the limit 18: both are taken, one before
        the contact at 5000 s and one after it, where the contact has made room."""
        tasks = [
            _Task(None, ORIGIN, (0, 0)),
            _snapshot(1, (0, 15_000), 12.0),
            _snapshot(2, (0, 15_000), 12.0),
        ]
        recorder = _recorder([5000], [12.0], limit=18.0)

        present, arrival = _solve(tasks, recorder)

        assert present == [True, True, True]
        assert min(arrival[1:]) < 5000 <= max(arrival[1:])

    def test_a_snapshot_that_fits_nowhere_is_left_out(self) -> None:
        """Both snapshots can only start before the contact, so one is left out."""
        tasks = [
            _Task(None, ORIGIN, (0, 0)),
            _snapshot(1, (0, 3000), 12.0),
            _snapshot(2, (0, 3000), 12.0),
        ]
        recorder = _recorder([5000], [12.0], limit=18.0)

        present, _ = _solve(tasks, recorder)

        assert present[0] and present[1:].count(True) == 1

    def test_what_a_contact_cannot_remove_carries_over(self) -> None:
        """The contact removes 4 Gb: 12 before it leaves 8, so a second 12 Gb
        snapshot after it would reach 20, over the limit, and one is left out
        even though each segment alone would hold it."""
        tasks = [
            _Task(None, ORIGIN, (0, 0)),
            _snapshot(1, (0, 15_000), 12.0),
            _snapshot(2, (0, 15_000), 12.0),
        ]
        recorder = _recorder([5000], [4.0], limit=18.0)

        present, _ = _solve(tasks, recorder)

        assert present[1:].count(True) == 1

    def test_the_volume_entering_the_chunk_counts(self) -> None:
        tasks = [_Task(None, ORIGIN, (0, 0)), _snapshot(1, (0, 15_000), 12.0)]

        present, _ = _solve(tasks, _recorder([], [], limit=18.0, entering=10.0))
        assert present == [True, False]

        present, _ = _solve(tasks, _recorder([], [], limit=18.0, entering=5.0))
        assert present == [True, True]

    def test_without_a_recorder_everything_fits(self) -> None:
        tasks = [
            _Task(None, ORIGIN, (0, 0)),
            _snapshot(1, (0, 3000), 12.0),
            _snapshot(2, (0, 3000), 12.0),
        ]
        present, _ = _solve(tasks, None)
        assert present == [True, True, True]


def _targets(config):
    return [
        _data_target(
            config, 1, 105.0, 10.0, HIGH_RATE, merit=90, minutes=200, snapshot=20
        ),
        _data_target(
            config, 2, 110.0, 10.0, LOW_RATE, merit=50, minutes=200, snapshot=20
        ),
        _data_target(
            config, 3, 115.0, 5.0, LOW_RATE, merit=60, minutes=100, snapshot=20
        ),
        _data_target(
            config, 4, 120.0, 0.0, LOW_RATE, merit=40, minutes=200, snapshot=20
        ),
        _data_target(
            config, 5, 100.0, 15.0, HIGH_RATE, merit=80, minutes=100, snapshot=20
        ),
    ]


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
    def test_the_solvers_plan_stays_within_the_limit_and_executes(self) -> None:
        config = _recorder_config()
        targets = _targets(config)
        scheduler = _scheduler(config, targets)

        plan = scheduler.schedule()

        assert any(scheduler.solver_chunks_used)
        assert scheduler.solver_score >= scheduler.start_score
        peaks = scheduler._ledger.peaks(scheduler._best.final.volumes)
        assert all(peak <= LIMIT + 1e-6 for peak in peaks)
        ditl = _execute(config, plan, HOURS, STEP)
        assert ditl.validate_plan_matches_execution() == []
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6
        assert _science(plan)

    def test_a_distinct_pass_rate_executes_within_the_limit(self) -> None:
        """The model counts each contact's ingress as the priority-first plan
        has it; a solution that lengthens one is checked by the decoder, so
        the plan it keeps executes within the limit."""
        config = _recorder_config()
        config.spacecraft_bus.data_generation = HousekeepingData(
            rate_gbps=0.0001, rate_mode={"PASS": 0.001}
        )
        targets = _targets(config)
        scheduler = _scheduler(config, targets)

        plan = scheduler.schedule()

        assert scheduler.solver_score >= scheduler.start_score
        entries = [block.entry for block in scheduler._best.final.timeline]
        assert scheduler._best.final.volumes == pytest.approx(
            scheduler._ledger.initial_volumes(entries)
        )
        ditl = _execute(config, plan, HOURS, STEP)
        assert ditl.validate_plan_matches_execution() == []
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6

    def test_hint_with_the_recorder_reproduces_the_priority_first_plan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Held to its hint, the model's recorder constraints must accept the
        priority-first plan, which the ledger built within the limit."""
        _hinted_only(monkeypatch)
        config = _recorder_config()
        scheduler = _scheduler(config, _targets(config))

        scheduler.schedule()

        assert set(scheduler.solver_statuses) == {"OPTIMAL"}
        assert all(scheduler.solver_chunks_used)
        assert scheduler.solver_score == scheduler.start_score

    def test_rates_round_up_so_the_model_never_undercounts(self) -> None:
        """Two targets at 0.005000245 Gbps: a 20-minute snapshot is 6.0003 Gb
        and two are 12.0006, over a 12.0004 Gb limit. Rounded to the nearest
        kilobit per second the model counted 12.0000 and accepted both, and
        the decoder then rejected every chunk; rounded up it counts 12.0024
        and places one after the contact, as the ledger would."""
        config = _recorder_config(12.0004 / 20.0)
        config.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0.0)
        rate = 0.005000245
        targets = [
            _data_target(config, 1, 105.0, 10.0, rate, merit=90, minutes=20),
            _data_target(config, 2, 110.0, 10.0, rate, merit=80, minutes=20),
        ]
        scheduler = _scheduler(config, targets)

        plan = scheduler.schedule()

        assert scheduler._candidate_data(targets[0]) == (5001, 0)
        assert "INFEASIBLE" not in scheduler.solver_statuses
        assert all(scheduler.solver_chunks_used)
        assert scheduler.solver_score >= scheduler.start_score
        assert len(_science(plan)) == 2
        ditl = _execute(config, plan, HOURS, STEP)
        assert ditl.validate_plan_matches_execution() == []
        assert max(ditl.recorder_volume_gb) < 12.0004

    def test_without_a_limit_the_model_has_no_recorder(self) -> None:
        config = _recorder_config(None)
        scheduler = _scheduler(config, _targets(config), solver_time_limit=5.0)
        scheduler.schedule()
        assert not scheduler._ledger.active
        assert scheduler.solver_score >= scheduler.start_score
