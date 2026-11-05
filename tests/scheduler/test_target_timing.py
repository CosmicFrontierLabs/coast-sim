"""A target's own timing budgets, with their calibration steps, in place of
its category's: a long dark after one deep field."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from conops import DITL, QueueDITL, create_ditl
from conops.common import ACSMode
from conops.config import BudgetStep, HousekeepingData, MissionConfig, ObservationTiming
from conops.schedulers import PriorityScheduler, SchedulingContext
from conops.targets import Plan, Pointing

from .planning_scenario import BEGIN, MIN, PATCH
from .planning_scenario import make_target as _target
from .test_instrument_modes import _moded_config

HOURS = 6
STEP = 60
END = BEGIN + timedelta(hours=HOURS)
CATEGORY = ObservationTiming(setup_seconds=60, cleanup_seconds=30)
DARK = BudgetStep(
    name="long dark",
    duration_seconds=20 * MIN,
    instrument_modes={"spectrograph": "SCIENCE"},
    data_gb=0.4,
)
DEEP = ObservationTiming(setup_seconds=60, cleanup_steps=[DARK], handoff_seconds=30)


def _config() -> MissionConfig:
    config = _moded_config(HOURS)
    config.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0.0)
    config.payload.observation_timing = CATEGORY
    return config


def _targets(config: MissionConfig) -> list[Pointing]:
    deep = _target(config, 1, *PATCH[0], merit=90.0, minutes=30, snapshot=30)
    deep.instrument_modes = {"camera": "SCIENCE"}
    deep.own_timing = DEEP
    plain = _target(config, 2, *PATCH[1], merit=50.0, minutes=30, snapshot=30)
    plain.instrument_modes = {"camera": "SCIENCE"}
    return [deep, plain]


class TestModel:
    def test_a_target_resolves_its_own_timing_first(self) -> None:
        config = _config()
        deep, plain = _targets(config)
        assert deep.observation_timing is DEEP
        assert plain.observation_timing is CATEGORY
        assert deep.extra_data_gb() == pytest.approx(0.4)
        assert plain.extra_data_gb() == 0.0
        dumped = deep.model_dump(mode="json")
        assert dumped["own_timing"]["cleanup_steps"][0]["name"] == "long dark"
        assert "own_timing" not in plain.model_dump(mode="json")
        with pytest.raises(ValidationError, match="no instrument"):
            Pointing(
                config=config,
                obsid=3,
                ra=0.0,
                dec=0.0,
                own_timing=ObservationTiming(
                    cleanup_steps=[
                        BudgetStep(
                            name="bad",
                            duration_seconds=10,
                            instrument_modes={"radar": "SCIENCE"},
                        )
                    ]
                ),
            )

    def test_the_context_bounds_cover_registered_timings(self) -> None:
        config = _config()
        ctx = SchedulingContext(config, BEGIN, END, step_size=STEP)
        assert ctx.post_collection_seconds == 30.0
        ctx.register_timing(1, DEEP)
        assert ctx.timing_for(1) is DEEP
        assert ctx.timing_for(2) is CATEGORY
        assert ctx.post_collection_seconds == 20 * MIN + 30.0
        assert ctx.setup_seconds == 60.0


class TestPlanning:
    def test_the_deep_field_gets_its_dark_and_the_other_target_does_not(self) -> None:
        config = _config()
        targets = _targets(config)
        scheduler = PriorityScheduler(config, targets, BEGIN, END, step_size=STEP)
        plan = scheduler.schedule()
        entries = {e.obsid: e for e in plan if e.obsid in (1, 2)}
        assert set(entries) == {1, 2}
        deep, plain = entries[1], entries[2]
        assert float(deep.end) - float(deep.collection_end) == pytest.approx(
            20 * MIN + 30
        )
        assert float(plain.end) - float(plain.collection_end) == pytest.approx(30.0)
        assert deep.own_timing is not None and plain.own_timing is None
        assert deep.instrument_modes_at(float(deep.collection_end) + 60.0) == {
            "spectrograph": "SCIENCE"
        }
        assert scheduler._ledger.science_volume(deep, 1800.0) == pytest.approx(0.4)
        # The serialized plan carries the dark and executes it.
        replay = Plan.model_validate_json(plan.model_dump_json())
        assert next(e for e in replay if e.obsid == 1).own_timing is not None
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
        samples = sorted(
            (hk.timestamp.timestamp(), hk.instrument_modes)
            for hk in ditl.telemetry.housekeeping
        )
        during_dark = next(
            s[1] for s in reversed(samples) if s[0] <= float(deep.collection_end) + 60.0
        )
        assert during_dark == {"camera": "IDLE", "spectrograph": "SCIENCE"}
        assert max(ditl.recorder_volume_gb) == pytest.approx(0.4, abs=1e-6)

    def test_dispatch_gives_the_deep_field_its_dark(self) -> None:
        config = _config()
        targets = _targets(config)
        ditl = create_ditl(config, targets, BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        entries = {
            e.obsid: e for e in ditl.plan if e.obsid in (1, 2) and e.collection_end
        }
        assert set(entries) == {1, 2}
        deep, plain = entries[1], entries[2]
        assert float(deep.end) - float(deep.collection_end) == pytest.approx(
            20 * MIN + 30
        )
        assert float(plain.end) - float(plain.collection_end) == pytest.approx(30.0)
        modes = {
            tuple(sorted(hk.instrument_modes.items()))
            for hk in ditl.telemetry.housekeeping
            if int(hk.acs_mode) == int(ACSMode.SCIENCE) and hk.instrument_modes
        }
        assert (("camera", "IDLE"), ("spectrograph", "SCIENCE")) in modes
        assert max(ditl.recorder_volume_gb) == pytest.approx(0.4, abs=1e-6)
