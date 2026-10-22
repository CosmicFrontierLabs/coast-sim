"""Sequence visits placed by the planners as one block of steps."""

import importlib.util
from datetime import timedelta

import pytest

from conops import DITL
from conops.common import ObsType
from conops.config import MissionConfig, ObservationTiming
from conops.ditl import RollingHorizonDITL
from conops.schedulers import LocalSearchScheduler, PriorityScheduler
from conops.targets import Plan, PlanEntry, Pointing

from .planning_scenario import BEGIN, MIN, PATCH
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target
from .test_observation_sequences import REFERENCE, SCIENCE, SEQUENCE, TARGET, TIMING

HOURS = 6
STEP = 60
END = BEGIN + timedelta(hours=HOURS)
HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None


def _sequence_target(config: MissionConfig, visits: int = 1) -> Pointing:
    return Pointing(
        config=config,
        obsid=500,
        name="Campaign",
        ra=TARGET[0],
        dec=TARGET[1],
        merit=95.0,
        sequence=SEQUENCE,
        visits=visits,
    )


def _targets(config: MissionConfig, visits: int = 1) -> list[Pointing]:
    others = [
        _target(config, 100 + i, ra, dec, merit=60 - i, minutes=200, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH[3:])
    ]
    return [_sequence_target(config, visits), *others]


def _timed_config(hours: int = HOURS) -> MissionConfig:
    config = _config(hours)
    config.payload.observation_timing = TIMING
    return config


def _chains(plan: Plan | list) -> list[list]:
    """The visits in a plan: runs of consecutive step entries of the target."""
    chains: list[list] = []
    for entry in plan:
        if entry.obsid == 500 and entry.step is not None:
            if entry.step == SEQUENCE.steps[0].name:
                chains.append([entry])
            else:
                chains[-1].append(entry)
    return chains


def _check_chain(chain: list, timing: ObservationTiming = TIMING) -> None:
    assert [e.step for e in chain] == [s.name for s in SEQUENCE.steps]
    assert [e.calibration for e in chain] == [not s.science for s in SEQUENCE.steps]
    # Contiguous: each step's slew starts where the last step's collection ends.
    for earlier, later in zip(chain, chain[1:]):
        assert float(later.begin) == pytest.approx(float(earlier.collection_end))
        assert float(later.collection_begin) == pytest.approx(
            float(later.begin) + later.slewtime
        )
    assert float(chain[0].collection_begin) == pytest.approx(
        float(chain[0].begin) + chain[0].slewtime + timing.setup_seconds
    )
    assert float(chain[-1].end) - float(chain[-1].collection_end) == pytest.approx(
        timing.post_collection_seconds
    )
    for entry, step in zip(chain, SEQUENCE.steps):
        assert entry.exposure == pytest.approx(step.duration_seconds)
    assert (chain[1].ra, chain[1].dec) == REFERENCE
    assert (chain[2].ra, chain[2].dec) == TARGET
    # Reference to target and back: real slews; same pointing: none.
    assert chain[1].slewdist == 0 and chain[2].slewdist > 0 and chain[3].slewdist > 0


def _execute(config: MissionConfig, plan: Plan, hours: int = HOURS) -> DITL:
    replay = Plan.model_validate_json(plan.model_dump_json())
    ditl = DITL(
        config=config,
        ephem=config.constraint.ephem,
        plan=replay,
        begin=BEGIN,
        end=BEGIN + timedelta(hours=hours),
    )
    ditl.step_size = STEP
    assert ditl.calc()
    return ditl


class TestPriorityScheduler:
    def test_a_visit_is_one_block_of_steps_that_executes(self) -> None:
        config = _timed_config()
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=STEP
        )
        plan = scheduler.schedule()
        chains = _chains(plan)
        assert len(chains) == 1
        _check_chain(chains[0])
        # Nothing between the steps: the entries are consecutive in the plan.
        entries = list(plan)
        first = entries.index(chains[0][0])
        assert entries[first : first + 4] == chains[0]
        # One block on the timeline, with the steps inside it.
        visits = [b for b in scheduler.timeline if b.is_visit]
        assert len(visits) == 1 and len(visits[0].steps) == 4
        assert visits[0].attitude_out != visits[0].attitude_in
        # The ledgers count the science steps' data and every step's power.
        ledger = scheduler._ledger
        expected = sum(
            ledger.science_volume(
                step, step.collection_seconds_between(step.begin, step.end)
            )
            for step in visits[0].steps
            if not step.calibration
        )
        assert scheduler._block_volume(visits[0]) == pytest.approx(expected)
        addition = scheduler._block_addition(visits[0])
        assert addition.begin == float(chains[0][0].begin)
        assert addition.end == float(chains[0][-1].end)
        ditl = _execute(config, plan)
        assert ditl.validate_plan_matches_execution() == []

    def test_two_visits_are_two_whole_chains(self) -> None:
        config = _timed_config()
        plan = PriorityScheduler(
            config, _targets(config, visits=2), BEGIN, END, step_size=STEP
        ).schedule()
        chains = _chains(plan)
        assert len(chains) == 2
        for chain in chains:
            _check_chain(chain)
        assert float(chains[1][0].begin) >= float(chains[0][-1].end)

    def test_a_visit_that_does_not_fit_before_the_next_activity_waits(self) -> None:
        """A visit is never shortened: with less room than a visit before a
        locked entry, it goes after it."""
        config = _timed_config()
        lock = PlanEntry(
            config=config,
            name="Locked",
            ra=PATCH[4][0],
            dec=PATCH[4][1],
            obsid=777,
            obstype=ObsType.PPT,
            begin=BEGIN.timestamp() + 30 * MIN,
            slewtime=0,
            end=BEGIN.timestamp() + 50 * MIN,
            collection_begin=BEGIN.timestamp() + 30 * MIN + 120,
            collection_end=BEGIN.timestamp() + 50 * MIN - 60,
        )
        plan = PriorityScheduler(
            config,
            _targets(config),
            BEGIN,
            END,
            step_size=STEP,
            locked=[lock],
        ).schedule()
        chains = _chains(plan)
        assert len(chains) == 1
        _check_chain(chains[0])
        assert float(chains[0][0].begin) >= float(lock.end)


class TestLocalSearch:
    def test_the_search_keeps_a_visit_whole(self) -> None:
        config = _timed_config()
        plan = LocalSearchScheduler(
            config,
            _targets(config),
            BEGIN,
            END,
            step_size=STEP,
            max_iterations=40,
            seed=3,
        ).schedule()
        chains = _chains(plan)
        assert len(chains) == 1
        _check_chain(chains[0])
        ditl = _execute(config, plan)
        assert ditl.validate_plan_matches_execution() == []


@pytest.mark.skipif(not HAS_ORTOOLS, reason="needs OR-Tools")
class TestCpSat:
    def test_the_solver_places_a_visit_whole(self) -> None:
        from conops.schedulers import CpSatScheduler

        config = _timed_config()
        plan = CpSatScheduler(
            config, _targets(config), BEGIN, END, step_size=STEP, solver_time_limit=5.0
        ).schedule()
        chains = _chains(plan)
        assert len(chains) == 1
        _check_chain(chains[0])
        ditl = _execute(config, plan)
        assert ditl.validate_plan_matches_execution() == []


class TestRolling:
    def test_each_step_is_credited_to_the_target(self) -> None:
        config = _timed_config()
        targets = _targets(config)
        ditl = RollingHorizonDITL(
            config,
            targets,
            begin=BEGIN,
            end=END,
            horizon=timedelta(hours=3),
            replan_interval=timedelta(hours=2),
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        chains = _chains(ditl.plan)
        assert len(chains) >= 1
        _check_chain(chains[0])
        campaign = targets[0]
        assert campaign.collected_seconds == pytest.approx(
            SCIENCE * len(chains), abs=2 * STEP
        )
        assert campaign.calibration_seconds == pytest.approx(
            10 * MIN * len(chains), abs=2 * STEP
        )
