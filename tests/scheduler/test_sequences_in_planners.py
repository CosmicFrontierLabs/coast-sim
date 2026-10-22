"""Sequence visits placed by the planners as one block of steps."""

import importlib.util
from datetime import timedelta

import pytest

from conops import DITL
from conops.common import ObsType
from conops.config import BudgetStep, HousekeepingData, MissionConfig, ObservationTiming
from conops.ditl import ReplanReason, RollingHorizonDITL
from conops.schedulers import LocalSearchScheduler, PriorityScheduler
from conops.targets import ObservationSequence, Plan, PlanEntry, Pointing, SequenceStep

from .planning_scenario import BEGIN, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target
from .test_instrument_modes import _moded_config
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
    def test_step_entries_carry_their_modes_and_the_target_timing(self) -> None:
        """A planned step is observed in the step's modes, with the
        calibration step collecting no science data, and the visit's budgets
        are the target's own where it has them."""
        config = _moded_config(HOURS)
        dark = BudgetStep(
            name="dark",
            duration_seconds=5 * MIN,
            instrument_modes={"spectrograph": "SCIENCE"},
            data_gb=0.2,
        )
        own = ObservationTiming(setup_seconds=60, cleanup_steps=[dark])
        sequence = ObservationSequence(
            steps=[
                SequenceStep(
                    name="flat",
                    duration_seconds=5 * MIN,
                    science=False,
                    instrument_modes={"spectrograph": "SCIENCE"},
                ),
                SequenceStep(
                    name="target",
                    duration_seconds=10 * MIN,
                    instrument_modes={"camera": "SCIENCE"},
                ),
                SequenceStep(name="again", duration_seconds=5 * MIN),
            ]
        )
        target = Pointing(
            config=config,
            obsid=500,
            name="Moded campaign",
            ra=TARGET[0],
            dec=TARGET[1],
            merit=95.0,
            sequence=sequence,
            instrument_modes={"camera": "SCIENCE", "spectrograph": "SCIENCE"},
            own_timing=own,
        )
        scheduler = PriorityScheduler(config, [target], BEGIN, END, step_size=STEP)
        plan = scheduler.schedule()
        steps = {e.step: e for e in plan if e.obsid == 500}
        assert set(steps) == {"flat", "target", "again"}
        assert steps["flat"].instrument_modes == {"spectrograph": "SCIENCE"}
        assert steps["target"].instrument_modes == {"camera": "SCIENCE"}
        # A step naming no modes observes in the target's.
        assert steps["again"].instrument_modes == {
            "camera": "SCIENCE",
            "spectrograph": "SCIENCE",
        }
        assert steps["flat"].calibration and steps["flat"].science_data == {}
        assert all(e.own_timing == own for e in steps.values())
        # The visit's block on the timeline carries them too, so the
        # ledgers see the target's budgets.
        visit = next(b for b in scheduler.timeline if b.is_visit)
        assert visit.entry.own_timing == own
        assert visit.entry.instrument_modes == target.instrument_modes
        # The dark follows the last step, as the target's own cleanup.
        last = steps["again"]
        assert float(last.end) - float(last.collection_end) == pytest.approx(
            5 * MIN, abs=STEP
        )
        assert last.extra_data_gb() == pytest.approx(0.2)
        # The executed plan observes each step in its modes and ends the
        # visit with the dark's frames on the recorder.
        config.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0.0)
        ditl = _execute(config, plan)
        assert ditl.validate_plan_matches_execution() == []
        samples = ditl.telemetry.housekeeping

        def modes_during(entry: PlanEntry) -> set[tuple[tuple[str, str], ...]]:
            begin, end = float(entry.collection_begin), float(entry.collection_end)
            return {
                tuple(sorted(hk.instrument_modes.items()))
                for hk in samples
                if begin + STEP <= hk.timestamp.timestamp() < end
            }

        assert modes_during(steps["flat"]) == {
            (("camera", "IDLE"), ("spectrograph", "SCIENCE"))
        }
        assert modes_during(steps["target"]) == {
            (("camera", "SCIENCE"), ("spectrograph", "IDLE"))
        }
        assert ditl.recorder_volume_gb[-1] == pytest.approx(
            sum(e.extra_data_gb() for e in steps.values())
            + sum(
                config.science_generated(
                    float(e.collection_end) - float(e.collection_begin),
                    science=e.science_data,
                    instrument_modes=e.instrument_modes,
                )
                for e in steps.values()
                if not e.calibration
            )
        )

    @pytest.mark.parametrize(
        "scheduler_class", [PriorityScheduler, LocalSearchScheduler]
    )
    def test_a_visit_at_the_end_of_the_horizon_stays_within_the_ephemeris(
        self, scheduler_class: type[PriorityScheduler]
    ) -> None:
        """A visit whose last step would end at the horizon, with cleanup
        beyond it, is dropped before its hold is sampled past the ephemeris,
        which ends with the horizon here, so the plan is built whole."""
        config = _config(2)
        config.payload.observation_timing = ObservationTiming(
            setup_seconds=0, cleanup_seconds=3 * STEP, handoff_seconds=0
        )
        sequence = ObservationSequence(
            steps=[
                SequenceStep(name="a", duration_seconds=STEP),
                SequenceStep(name="b", duration_seconds=STEP, roll_offset_deg=30.0),
                SequenceStep(
                    name="c", duration_seconds=STEP, ra=PATCH[1][0], dec=PATCH[1][1]
                ),
            ]
        )
        target = Pointing(
            config=config,
            obsid=500,
            name="Dense",
            ra=TARGET[0],
            dec=TARGET[1],
            merit=95.0,
            sequence=sequence,
            visits=200,
        )
        end = BEGIN + timedelta(hours=2)
        scheduler = scheduler_class(config, [target], BEGIN, end, step_size=STEP)
        plan = scheduler.schedule()
        assert len(plan) > 0
        assert max(float(e.end) for e in plan) <= end.timestamp()
        assert max(float(e.end) for e in plan) > T0 + 1.5 * 3600

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


class TestReplanCutoff:
    def test_a_rapid_replan_does_not_cut_a_visit_for_a_too(self) -> None:
        """A ToO that outranks the visit under way, submitted during its
        middle step with a deadline inside the visit, waits for the visit
        as it does in dispatch: the visit completes whole and the ToO is
        placed after it, or not at all."""
        config = _timed_config()
        targets = _targets(config)
        ditl = RollingHorizonDITL(
            config,
            targets,
            begin=BEGIN,
            end=END,
            horizon=timedelta(hours=3),
            replan_interval=timedelta(hours=2),
            commit_lead_time=timedelta(minutes=5),
            rapid_replans=True,
            allow_interrupts=True,
        )
        ditl.step_size = STEP
        # The visit is the first observation: its target step begins after
        # the two reference steps and the setup.
        submit = T0 + 20 * MIN
        ditl.submit_too(
            obsid=900,
            ra=PATCH[4][0],
            dec=PATCH[4][1],
            merit=500.0,
            exptime=10 * MIN,
            name="ToO",
            submit_time=submit,
            deadline=submit + 90 * MIN,
            interrupt=True,
        )
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        rapid = [r for r in ditl.replans if r.reason is ReplanReason.RAPID]
        assert len(rapid) == 1 and rapid[0].interrupted_obsid is None
        chains = _chains(ditl.plan)
        assert chains, "no visit was planned"
        for chain in chains:
            _check_chain(chain)
        first = chains[0]
        assert float(first[0].begin) < submit < float(first[-1].end)
        too = [e for e in ditl.plan if e.obsid == 900]
        assert too and all(float(e.begin) >= float(first[-1].end) for e in too)
        descriptions = [e.description for e in ditl.log.events]
        assert not any("interrupts observation 500" in d for d in descriptions)
        assert any("waits for the visit of 500" in d for d in descriptions)
        assert targets[0].collected_seconds == pytest.approx(
            SCIENCE * len(chains), abs=2 * STEP
        )

    def test_a_visit_under_way_at_a_replan_cutoff_is_finished(self) -> None:
        """Replanning every half hour with a five-minute commit lead puts a
        cutoff inside the first visit. The visit is kept whole across it:
        every chain in the executed plan is a complete visit, and the target
        is credited one visit per chain."""
        config = _timed_config()
        targets = _targets(config, visits=2)
        interval, lead = timedelta(minutes=30), timedelta(minutes=5)
        ditl = RollingHorizonDITL(
            config,
            targets,
            begin=BEGIN,
            end=END,
            horizon=timedelta(hours=3),
            replan_interval=interval,
            commit_lead_time=lead,
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        chains = _chains(ditl.plan)
        assert len(chains) >= 2
        for chain in chains:
            _check_chain(chain)
        begin = BEGIN.timestamp()
        cutoffs = [
            begin + k * interval.total_seconds() + lead.total_seconds()
            for k in range(1, int(HOURS * 3600 / interval.total_seconds()))
        ]
        straddled = [
            chain
            for chain in chains
            if any(float(chain[0].begin) < c < float(chain[-1].end) for c in cutoffs)
        ]
        assert straddled, "no visit crossed a replan cutoff"
        campaign = targets[0]
        assert campaign.collected_seconds == pytest.approx(
            SCIENCE * len(chains), abs=2 * STEP
        )
