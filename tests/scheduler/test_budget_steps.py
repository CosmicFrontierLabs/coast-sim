"""Calibration steps inside the setup and cleanup budgets, with their own
instrument modes and data, and the data a scheduled activity produces."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from conops import DITL, QueueDITL, create_ditl
from conops.common import ACSMode
from conops.config import (
    BudgetStep,
    HousekeepingData,
    MissionConfig,
    ObservationTiming,
)
from conops.ditl import RollingHorizonDITL
from conops.schedulers import PriorityScheduler
from conops.targets import (
    ObservationSequence,
    Plan,
    Pointing,
    ScheduledActivity,
    SequenceStep,
)

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_target as _target
from .test_instrument_modes import _moded_config

HOURS = 6
STEP = 60
END = BEGIN + timedelta(hours=HOURS)
ACQUIRE = BudgetStep(
    name="acquire", duration_seconds=120, instrument_modes={"camera": "SCIENCE"}
)
CHARACTERISE = BudgetStep(
    name="characterise",
    duration_seconds=240,
    instrument_modes={"spectrograph": "SCIENCE"},
    data_gb=0.5,
)
DARK = BudgetStep(name="dark", duration_seconds=120, data_gb=0.25)
TIMING = ObservationTiming(
    setup_steps=[ACQUIRE, CHARACTERISE], cleanup_steps=[DARK], handoff_seconds=30
)


def _stepped_config() -> MissionConfig:
    config = _moded_config(HOURS)
    config.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0.0)
    config.payload.observation_timing = TIMING
    return config


class TestTiming:
    def test_steps_set_their_budget(self) -> None:
        assert TIMING.setup_seconds == 360.0
        assert TIMING.cleanup_seconds == 120.0
        assert TIMING.post_collection_seconds == 150.0
        assert TIMING.budget_data_gb == pytest.approx(0.75)
        assert TIMING.setup_step_at(0.0) is ACQUIRE
        assert TIMING.setup_step_at(119.9) is ACQUIRE
        assert TIMING.setup_step_at(120.0) is CHARACTERISE
        assert TIMING.setup_step_at(360.0) is None
        assert TIMING.cleanup_step_at(10.0) is DARK
        assert TIMING.cleanup_step_at(-1.0) is None
        flat = ObservationTiming(setup_seconds=120)
        assert flat.setup_step_at(10.0) is None and flat.budget_data_gb == 0.0
        # A sequence step pays the budgets' steps with the budgets.
        bound = ObservationTiming.bound([TIMING, flat])
        assert bound.setup_seconds == 360.0 and not bound.setup_steps

    def test_budget_step_modes_are_checked_with_the_configuration(self) -> None:
        config = _stepped_config()
        with pytest.raises(ValidationError, match="no instrument"):
            MissionConfig.model_validate(
                {
                    **config.model_dump(),
                    "payload": {
                        **config.payload.model_dump(),
                        "observation_timing": {
                            "setup_steps": [
                                {
                                    "name": "bad",
                                    "duration_seconds": 10,
                                    "instrument_modes": {"radar": "SCIENCE"},
                                }
                            ]
                        },
                    },
                }
            )


class TestEntry:
    def test_modes_follow_the_step_under_way(self) -> None:
        config = _stepped_config()
        target = _target(config, 1, *PATCH[0], minutes=20, snapshot=20)
        target.instrument_modes = {"camera": "SCIENCE", "spectrograph": "SCIENCE"}
        plan = PriorityScheduler(
            config, [target], BEGIN, END, step_size=STEP
        ).schedule()
        entry = next(e for e in plan if e.obsid == 1)
        begin, end = float(entry.collection_begin), float(entry.collection_end)
        assert begin - (float(entry.begin) + entry.slewtime) == pytest.approx(360.0)
        assert entry.instrument_modes_at(begin - 360.0) == {"camera": "SCIENCE"}
        assert entry.instrument_modes_at(begin - 240.0) == {"spectrograph": "SCIENCE"}
        assert entry.instrument_modes_at(begin - 1.0) == {"spectrograph": "SCIENCE"}
        assert entry.instrument_modes_at(begin) == {
            "camera": "SCIENCE",
            "spectrograph": "SCIENCE",
        }
        # A cleanup step naming no modes keeps the observation's.
        assert entry.instrument_modes_at(end + 10.0) == {
            "camera": "SCIENCE",
            "spectrograph": "SCIENCE",
        }
        assert entry.extra_data_gb() == pytest.approx(0.75)

    def test_a_sequence_step_carries_the_budgets_steps(self) -> None:
        config = _stepped_config()
        target = Pointing(
            config=config,
            obsid=2,
            ra=PATCH[0][0],
            dec=PATCH[0][1],
            sequence=ObservationSequence(
                steps=[
                    SequenceStep(name="a", duration_seconds=300),
                    SequenceStep(name="b", duration_seconds=300),
                ]
            ),
        )
        first, last = target.step_pointing(0), target.step_pointing(1)
        assert first.step_timing is not None and last.step_timing is not None
        assert [s.name for s in first.step_timing.setup_steps] == [
            "acquire",
            "characterise",
        ]
        assert first.step_timing.cleanup_steps == []
        assert (
            last.step_timing.setup_steps == [] and last.step_timing.setup_seconds == 0.0
        )
        assert [s.name for s in last.step_timing.cleanup_steps] == ["dark"]
        assert first.extra_data_gb() == pytest.approx(0.5)
        assert last.extra_data_gb() == pytest.approx(0.25)


class TestExecution:
    def test_setup_steps_draw_their_modes_and_put_their_data_on_the_recorder(
        self,
    ) -> None:
        """During the acquisition step only the camera observes; during the
        characterisation only the spectrograph; the 0.75 Gb of the steps'
        frames is on the recorder once the observation is over, in the
        executed run and in the ledger's volume for the entry."""
        config = _stepped_config()
        target = _target(config, 1, *PATCH[0], minutes=20, snapshot=20)
        target.instrument_modes = {"camera": "SCIENCE", "spectrograph": "SCIENCE"}
        scheduler = PriorityScheduler(config, [target], BEGIN, END, step_size=STEP)
        plan = scheduler.schedule()
        entry = next(e for e in plan if e.obsid == 1)
        assert scheduler._ledger.science_volume(entry, 1200.0) == pytest.approx(0.75)
        replay = Plan.model_validate_json(plan.model_dump_json())
        ditl = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=replay,
            begin=BEGIN,
            end=END,
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        begin = float(entry.collection_begin)
        samples = sorted(
            (hk.timestamp.timestamp(), hk.instrument_modes, float(hk.power_payload))
            for hk in ditl.telemetry.housekeeping
        )

        def sample_at(when: float) -> tuple:
            """The sample whose step contains ``when``."""
            return next(s[1:] for s in reversed(samples) if s[0] <= when)

        # A whole step inside each span: 60 s into the 120 s acquisition,
        # 60 s into the 240 s characterisation, 60 s into collection.
        acquire = sample_at(begin - 360.0 + 60.0)
        characterise = sample_at(begin - 240.0 + 60.0)
        collecting = sample_at(begin + 60.0)
        assert acquire[0] == {"camera": "SCIENCE", "spectrograph": "IDLE"}
        assert characterise[0] == {"camera": "IDLE", "spectrograph": "SCIENCE"}
        assert collecting[0] == {"camera": "SCIENCE", "spectrograph": "SCIENCE"}
        assert acquire[1] == pytest.approx(80.0 + 20.0)
        assert characterise[1] == pytest.approx(10.0 + 150.0)
        assert max(ditl.recorder_volume_gb) == pytest.approx(0.75, abs=1e-6)

    def test_dispatch_does_the_same(self) -> None:
        config = _stepped_config()
        target = _target(config, 1, *PATCH[0], minutes=20, snapshot=20)
        target.instrument_modes = {"camera": "SCIENCE", "spectrograph": "SCIENCE"}
        ditl = create_ditl(config, [target], BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        modes = {
            tuple(sorted(hk.instrument_modes.items()))
            for hk in ditl.telemetry.housekeeping
            if int(hk.acs_mode) == int(ACSMode.SCIENCE) and hk.instrument_modes
        }
        assert (("camera", "SCIENCE"), ("spectrograph", "IDLE")) in modes
        assert (("camera", "IDLE"), ("spectrograph", "SCIENCE")) in modes
        assert max(ditl.recorder_volume_gb) == pytest.approx(0.75, abs=1e-6)


class TestActivityData:
    def test_an_activity_puts_its_frames_on_the_recorder(self) -> None:
        """A DITL shares its configuration's recorder, so each run here gets
        its own configuration."""

        def fresh() -> MissionConfig:
            config = _moded_config(HOURS)
            config.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0.0)
            return config

        lamp = ScheduledActivity(
            name="lamp",
            begin=T0 + 2 * HOUR,
            duration_seconds=20 * MIN,
            instrument_modes={"spectrograph": "SCIENCE"},
            data_gb=1.5,
        )
        # Dispatch.
        config = fresh()
        ditl = create_ditl(config, [], BEGIN, END, activities=[lamp])
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        assert max(ditl.recorder_volume_gb) == pytest.approx(1.5)
        # Counted as generated, at the step the activity begins.
        assert ditl.data_generated_gb[-1] == pytest.approx(1.5)
        began = ditl.utime.index(T0 + 2 * HOUR)
        assert ditl.data_generated_gb[began - 1] == pytest.approx(0.0)
        assert ditl.data_generated_gb[began] == pytest.approx(1.5)
        entry = next(e for e in ditl.plan if e.name == "lamp")
        assert entry.fixed_data_gb == 1.5
        # The dispatched plan executed.
        config = fresh()
        replay = Plan.model_validate_json(ditl.plan.model_dump_json())
        again = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=replay,
            begin=BEGIN,
            end=END,
        )
        again.step_size = STEP
        assert again.calc()
        assert max(again.recorder_volume_gb) == pytest.approx(1.5)
        assert again.data_generated_gb[-1] == pytest.approx(1.5)
        # The ledger's view of it.
        config = fresh()
        scheduler = PriorityScheduler(
            config, [], BEGIN, END, step_size=STEP, activities=[lamp]
        )
        scheduler.schedule()
        assert sum(scheduler._volumes) == pytest.approx(1.5)
        # A rolling run's forecast of a committed activity.
        config = fresh()
        rolling = RollingHorizonDITL(
            config,
            [_target(config, 1, *PATCH[0], minutes=60, snapshot=20)],
            begin=BEGIN,
            end=END,
            horizon=timedelta(hours=3),
            replan_interval=timedelta(hours=1),
            commit_lead_time=timedelta(minutes=30),
            activities=[lamp],
        )
        rolling.step_size = STEP
        assert rolling.calc()
        assert rolling.validate_plan_matches_execution() == []
        assert max(rolling.recorder_volume_gb) >= 1.5 - 1e-6
