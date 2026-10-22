"""Observation sequences: ordered steps observed as one visit by dispatch."""

from datetime import timedelta

import pytest

from conops import create_ditl
from conops.common import ObsType
from conops.config import MissionConfig, ObservationTiming
from conops.ditl import QueueDITL
from conops.targets import ObservationSequence, Pointing, SequenceStep

from .planning_scenario import BEGIN, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 6
STEP = 60
END = BEGIN + timedelta(hours=HOURS)
TARGET = PATCH[0]
REFERENCE = PATCH[2]
SEQUENCE = ObservationSequence(
    steps=[
        SequenceStep(
            name="reference calibration",
            duration_seconds=10 * MIN,
            ra=REFERENCE[0],
            dec=REFERENCE[1],
            science=False,
        ),
        SequenceStep(
            name="reference",
            duration_seconds=5 * MIN,
            ra=REFERENCE[0],
            dec=REFERENCE[1],
        ),
        SequenceStep(name="target", duration_seconds=20 * MIN),
        SequenceStep(
            name="reference again",
            duration_seconds=5 * MIN,
            ra=REFERENCE[0],
            dec=REFERENCE[1],
        ),
    ]
)
SCIENCE = 30 * MIN
TIMING = ObservationTiming(setup_seconds=120, cleanup_seconds=60, handoff_seconds=0)


def _sequence_target(config: MissionConfig, visits: int = 1) -> Pointing:
    target = Pointing(
        config=config,
        obsid=500,
        name="Campaign",
        ra=TARGET[0],
        dec=TARGET[1],
        merit=95.0,
        sequence=SEQUENCE,
        visits=visits,
    )
    return target


def _targets(config: MissionConfig, visits: int = 1) -> list[Pointing]:
    others = [
        _target(config, 100 + i, ra, dec, merit=60 - i, minutes=200, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH[3:])
    ]
    return [_sequence_target(config, visits), *others]


def _steps(plan) -> list:
    return [e for e in plan if e.obsid == 500]


class TestModel:
    def test_science_and_calibration_seconds(self) -> None:
        assert SEQUENCE.science_seconds == SCIENCE
        assert SEQUENCE.calibration_seconds == 10 * MIN
        assert SEQUENCE.pointing_of(2, 1.0, 2.0) == (1.0, 2.0)
        assert SEQUENCE.pointing_of(1, 1.0, 2.0) == REFERENCE

    def test_a_sequence_sets_the_targets_snapshot_and_exposure(self) -> None:
        config = _config(2)
        target = _sequence_target(config, visits=3)
        assert (target.ss_min, target.ss_max) == (SCIENCE, SCIENCE)
        assert target.exptime == 3 * SCIENCE
        plain = _target(config, 1, 10.0, 10.0, minutes=40)
        assert plain.sequence is None and plain.visits == 1

    def test_step_timing(self) -> None:
        first = SEQUENCE.step_timing(TIMING, 0)
        middle = SEQUENCE.step_timing(TIMING, 1)
        last = SEQUENCE.step_timing(TIMING, 3)
        assert (first.setup_seconds, first.cleanup_seconds) == (120, 0)
        assert (middle.setup_seconds, middle.post_collection_seconds) == (0, 0)
        assert (last.setup_seconds, last.cleanup_seconds) == (0, 60)
        with_setup = ObservationSequence(
            steps=[
                SequenceStep(name="a", duration_seconds=60),
                SequenceStep(name="b", duration_seconds=60, setup=True),
            ]
        )
        assert with_setup.step_timing(TIMING, 1).setup_seconds == 120

    def test_overhead_counts_calibration_and_the_slews_between_pointings(self) -> None:
        config = _config(2)
        acs = config.spacecraft_bus.attitude_control
        target = _sequence_target(config)
        overhead = target.sequence_overhead_seconds(acs, TIMING)
        # Two slews between the reference and the target, each a few
        # degrees plus settling, on top of the calibration step.
        assert overhead > 10 * MIN + 2 * acs.settle_time
        assert overhead < 10 * MIN + 2 * acs.slew_time(10.0)
        assert (
            _target(config, 1, 10.0, 10.0).sequence_overhead_seconds(acs, TIMING) == 0
        )

    def test_step_pointings(self) -> None:
        config = _config(2)
        target = _sequence_target(config)
        calibration = target.step_pointing(0)
        science = target.step_pointing(2)
        assert (calibration.ra, calibration.dec) == REFERENCE
        assert calibration.calibration and calibration.step == "reference calibration"
        assert calibration.science_data == {}
        assert (science.ra, science.dec) == TARGET and not science.calibration
        assert science.obsid == 500 and science.name == "Campaign: target"
        # A step has no minimum of its own: the visit is checked whole.
        assert (science.ss_min, science.ss_max, science.exptime) == (
            0,
            20 * MIN,
            20 * MIN,
        )
        assert science.windows

    def test_round_trips_through_json(self) -> None:
        config = _config(2)
        target = _sequence_target(config, visits=2)
        loaded = Pointing.model_validate_json(target.model_dump_json())
        assert loaded.sequence == SEQUENCE and loaded.visits == 2
        plain = _target(config, 1, 10.0, 10.0)
        assert "visits" not in plain.model_dump()


class TestDispatch:
    def _run(self, visits: int = 1, **kwargs) -> QueueDITL:
        config = _config(HOURS)
        config.payload.observation_timing = TIMING
        ditl = create_ditl(config, _targets(config, visits), BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        for key, value in kwargs.items():
            setattr(ditl, key, value)
        ditl.step_size = STEP
        assert ditl.calc()
        return ditl

    def test_a_visit_runs_its_steps_in_order_with_nothing_between(self) -> None:
        ditl = self._run()
        steps = _steps(ditl.plan)
        assert [e.step for e in steps] == [
            "reference calibration",
            "reference",
            "target",
            "reference again",
        ]
        assert [e.calibration for e in steps] == [True, False, False, False]
        for earlier, later in zip(steps, steps[1:]):
            # The next step starts the tick the last ends, slew included.
            assert float(later.begin) - float(earlier.end) <= STEP
            assert later.obstype == ObsType.AT
        # The first step pays the setup, the last the cleanup, none between.
        assert (
            float(steps[0].collection_begin) - float(steps[0].begin) - steps[0].slewtime
            == 120
        )
        assert (
            float(steps[1].collection_begin) - float(steps[1].begin) - steps[1].slewtime
            == 0
        )
        assert float(steps[3].end) - float(steps[3].collection_end) == 60
        assert float(steps[2].end) - float(steps[2].collection_end) == 0
        # Each step collects its duration; the reference steps point elsewhere.
        for entry, step in zip(steps, SEQUENCE.steps):
            assert entry.exposure == pytest.approx(step.duration_seconds, abs=STEP)
        assert (steps[1].ra, steps[1].dec) == REFERENCE
        assert (steps[2].ra, steps[2].dec) == TARGET
        # The target is credited the science and the calibration separately.
        target = next(t for t in ditl.queue.targets if t.obsid == 500)
        assert target.collected_seconds == pytest.approx(SCIENCE, abs=STEP)
        assert target.calibration_seconds == pytest.approx(10 * MIN, abs=STEP)
        assert target.exptime == pytest.approx(0, abs=STEP) and target.done
        descriptions = [e.description for e in ditl.log.events]
        assert "Sequence visit of Campaign begins" in descriptions
        assert "Sequence visit of Campaign complete" in descriptions
        assert not any("cut short" in d for d in descriptions)

    def test_two_visits_are_two_whole_runs_of_the_steps(self) -> None:
        ditl = self._run(visits=2)
        steps = _steps(ditl.plan)
        assert [e.step for e in steps] == 2 * [
            "reference calibration",
            "reference",
            "target",
            "reference again",
        ]
        target = next(t for t in ditl.queue.targets if t.obsid == 500)
        assert target.collected_seconds == pytest.approx(2 * SCIENCE, abs=2 * STEP)
        # Something else was observed between the visits or after them.
        assert any(e.obsid != 500 for e in ditl.plan)

    def test_a_too_waits_for_the_visit_to_end(self) -> None:
        config = _config(HOURS)
        config.payload.observation_timing = TIMING
        ditl = create_ditl(config, _targets(config), BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        # Submitted during the visit's second step, outranking everything.
        too_time = T0 + 20 * MIN
        ditl.submit_too(
            9000,
            PATCH[4][0],
            PATCH[4][1],
            1000.0,
            10 * MIN,
            "Burst",
            submit_time=too_time,
            interrupt=True,
        )
        assert ditl.calc()
        steps = _steps(ditl.plan)
        assert [e.step for e in steps] == [
            "reference calibration",
            "reference",
            "target",
            "reference again",
        ]
        burst = [e for e in ditl.plan if e.obsid == 9000]
        assert burst and float(burst[0].begin) >= float(steps[-1].end)
        assert not any("cut short" in e.description for e in ditl.log.events)
