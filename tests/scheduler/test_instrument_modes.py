"""A request's instrument modes set the payload's draw during its
observation, in both DITLs and the battery ledger."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from conops import DITL, QueueDITL, create_ditl
from conops.common import ACSMode
from conops.config import (
    ConcurrencyRules,
    DataGeneration,
    Instrument,
    InstrumentMode,
    MissionConfig,
    Payload,
    PowerDraw,
    ScienceData,
)
from conops.schedulers import PriorityScheduler
from conops.targets import (
    ObservationSequence,
    Plan,
    Pointing,
    ScheduledActivity,
    SequenceStep,
)

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target
from .test_battery_constraint import LEVEL, _battery_config, _ledger

HOURS = 6
STEP = 60
END = BEGIN + timedelta(hours=HOURS)
IDLE = {"camera": 10.0, "spectrograph": 20.0}
SCIENCE = {"camera": 80.0, "spectrograph": 150.0}
BUS = 100.0


def _moded_config(hours: int = HOURS) -> MissionConfig:
    """The planning scenario with a 100 W bus in every mode and two
    instruments with IDLE and SCIENCE modes."""
    config = _config(hours)
    config.spacecraft_bus.power_draw = PowerDraw(nominal_power=BUS, power_mode={})
    config.payload = Payload(
        instruments=[
            Instrument(
                name=name,
                modes=[
                    InstrumentMode(name="IDLE", power_w=IDLE[name]),
                    InstrumentMode(name="SCIENCE", power_w=SCIENCE[name]),
                ],
            )
            for name in ("camera", "spectrograph")
        ]
    )
    return config


def _payload_power(ditl: DITL | QueueDITL) -> dict[int, set[float]]:
    """The payload power recorded in each ACS mode."""
    seen: dict[int, set[float]] = {}
    for mode, watts in zip(ditl.mode, ditl.power_payload, strict=True):
        seen.setdefault(int(mode), set()).add(round(float(watts), 6))
    return seen


class TestRequests:
    def test_a_request_names_the_instruments_it_uses(self) -> None:
        config = _moded_config()
        target = _target(config, 1, *PATCH[0])
        target.instrument_modes = {"camera": "SCIENCE"}
        assert Pointing.model_validate(target).effective_instrument_modes() == {
            "camera": "SCIENCE"
        }
        with pytest.raises(ValidationError, match="no instrument"):
            Pointing(
                config=config,
                obsid=2,
                ra=0.0,
                dec=0.0,
                instrument_modes={"radar": "SCIENCE"},
            )
        with pytest.raises(ValidationError, match="no mode"):
            Pointing(
                config=config,
                obsid=2,
                ra=0.0,
                dec=0.0,
                instrument_modes={"camera": "WARM"},
            )

    def test_a_request_naming_no_modes_uses_every_instrument(self) -> None:
        """The instrument a request names for its pointing is no restriction:
        with no modes named, every instrument with modes observes."""
        config = _moded_config()
        plain = _target(config, 1, *PATCH[0])
        assert plain.effective_instrument_modes() == {
            "camera": "SCIENCE",
            "spectrograph": "SCIENCE",
        }
        named = Pointing(
            config=config, obsid=2, ra=0.0, dec=0.0, instrument_name="spectrograph"
        )
        assert named.effective_instrument_modes() == {
            "camera": "SCIENCE",
            "spectrograph": "SCIENCE",
        }
        # A scheduled activity names its modes or uses nothing.
        quiet = ScheduledActivity(name="quiet", begin=T0, duration_seconds=600, obsid=5)
        assert (
            quiet.plan_entry(config, (0.0, 0.0, 0.0)).effective_instrument_modes() == {}
        )
        lamp = ScheduledActivity(
            name="lamp",
            begin=T0,
            duration_seconds=600,
            obsid=6,
            instrument_modes={"spectrograph": "SCIENCE"},
        )
        assert lamp.plan_entry(config, (0.0, 0.0, 0.0)).instrument_modes == {
            "spectrograph": "SCIENCE"
        }
        with pytest.raises(ValueError, match="no instrument"):
            ScheduledActivity(
                name="bad",
                begin=T0,
                duration_seconds=600,
                obsid=7,
                instrument_modes={"radar": "SCIENCE"},
            ).plan_entry(config, (0.0, 0.0, 0.0))

    def test_a_sequence_step_carries_its_own_modes(self) -> None:
        config = _moded_config()
        target = Pointing(
            config=config,
            obsid=3,
            ra=PATCH[0][0],
            dec=PATCH[0][1],
            instrument_modes={"camera": "SCIENCE"},
            sequence=ObservationSequence(
                steps=[
                    SequenceStep(name="image", duration_seconds=300),
                    SequenceStep(
                        name="spectrum",
                        duration_seconds=300,
                        instrument_modes={"spectrograph": "SCIENCE"},
                    ),
                ]
            ),
        )
        assert target.step_pointing(0).effective_instrument_modes() == {
            "camera": "SCIENCE"
        }
        assert target.step_pointing(1).effective_instrument_modes() == {
            "spectrograph": "SCIENCE"
        }
        dumped = target.model_dump(mode="json")
        assert dumped["instrument_modes"] == {"camera": "SCIENCE"}
        assert "instrument_modes" not in _target(config, 4, 0.0, 0.0).model_dump(
            mode="json"
        )


class TestExecution:
    def test_the_payload_draws_the_observations_modes_while_observing(self) -> None:
        """Observing with the camera only: 80 W plus the spectrograph's
        20 W idle during science, 30 W idle in every other mode."""
        config = _moded_config()
        target = _target(config, 1, *PATCH[0], minutes=30, snapshot=30)
        target.instrument_modes = {"camera": "SCIENCE"}
        plan = PriorityScheduler(
            config, [target], BEGIN, END, step_size=STEP
        ).schedule()
        entry = next(e for e in plan if e.obsid == 1)
        assert entry.instrument_modes == {"camera": "SCIENCE"}
        replay = Plan.model_validate_json(plan.model_dump_json())
        assert next(e for e in replay if e.obsid == 1).instrument_modes == {
            "camera": "SCIENCE"
        }
        ditl = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=replay,
            begin=BEGIN,
            end=END,
        )
        ditl.step_size = STEP
        assert ditl.calc()
        seen = _payload_power(ditl)
        assert seen[int(ACSMode.SCIENCE)] == {80.0 + 20.0}
        for mode, watts in seen.items():
            if mode != int(ACSMode.SCIENCE):
                assert watts == {30.0}, ACSMode(mode)

    def test_dispatch_draws_the_modes_of_the_target_it_observes(self) -> None:
        config = _moded_config()
        camera = _target(config, 1, *PATCH[0], merit=90.0, minutes=30, snapshot=30)
        camera.instrument_modes = {"camera": "SCIENCE"}
        both = _target(config, 2, *PATCH[1], merit=50.0, minutes=30, snapshot=30)
        ditl = create_ditl(config, [camera, both], BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        seen = _payload_power(ditl)
        assert seen[int(ACSMode.SCIENCE)] == {80.0 + 20.0, 80.0 + 150.0}
        assert seen[int(ACSMode.IDLE)] == {30.0}
        science = [
            e for e in ditl.plan if e.obsid in (1, 2) and e.collection_begin is not None
        ]
        assert {e.obsid: e.instrument_modes for e in science} == {
            1: {"camera": "SCIENCE"},
            2: {},
        }

    def test_a_too_and_an_activity_draw_their_modes(self) -> None:
        """A ToO submitted with modes draws them; a calibration activity with
        a lamp draws the lamp's instrument during its hold, in dispatch and
        when its plan is executed."""
        config = _moded_config()
        filler = _target(config, 2, *PATCH[1], merit=50.0, minutes=60, snapshot=20)
        filler.instrument_modes = {"camera": "SCIENCE"}
        lamp = ScheduledActivity(
            name="lamp",
            begin=T0 + 3 * HOUR,
            duration_seconds=20 * MIN,
            instrument_modes={"spectrograph": "SCIENCE"},
        )
        ditl = create_ditl(config, [filler], BEGIN, END, activities=[lamp])
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        ditl.submit_too(
            obsid=30001,
            ra=PATCH[0][0],
            dec=PATCH[0][1],
            merit=500.0,
            exptime=20 * MIN,
            name="burst",
            submit_time=T0 + HOUR,
            instrument_modes={"spectrograph": "SCIENCE"},
        )
        assert ditl.calc()
        by_obsid: dict[int, set[float]] = {}
        for utime, mode, watts in zip(
            ditl.utime, ditl.mode, ditl.power_payload, strict=True
        ):
            entry = next(
                (e for e in ditl.plan if float(e.begin) <= utime < float(e.end)), None
            )
            if entry is not None and int(mode) in (
                int(ACSMode.SCIENCE),
                int(ACSMode.IDLE),
            ):
                by_obsid.setdefault(int(entry.obsid), set()).add(round(float(watts), 6))
        assert 150.0 + 10.0 in by_obsid[30001]
        lamp_obsid = next(e.obsid for e in ditl.plan if e.name == "lamp")
        assert 150.0 + 10.0 in by_obsid[int(lamp_obsid)]
        # The same when the dispatched plan is executed.
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
        seen = _payload_power(again)
        assert 150.0 + 10.0 in seen[int(ACSMode.IDLE)]
        with pytest.raises(ValueError, match="no instrument"):
            ditl.submit_too(
                obsid=30002,
                ra=0.0,
                dec=0.0,
                merit=1.0,
                exptime=60,
                name="bad",
                instrument_modes={"radar": "SCIENCE"},
            )


def _moded_battery_config() -> MissionConfig:
    """The battery scenario's side panel, with the two moded instruments."""
    config = _battery_config((0.0, 1.0, 0.0))
    config.payload = _moded_config().payload
    return config


class TestLedger:
    def test_an_activity_with_modes_draws_them_in_the_ledger(self) -> None:
        config = _moded_battery_config()
        ledger = _ledger(config)
        quiet = ScheduledActivity(
            name="quiet", begin=T0 + HOUR, duration_seconds=1200, obsid=5
        )
        lamp = ScheduledActivity(
            name="lamp",
            begin=T0 + HOUR,
            duration_seconds=1200,
            obsid=6,
            instrument_modes={"spectrograph": "SCIENCE"},
        )
        attitude = (PATCH[0][0], PATCH[0][1], 0.0)
        idle = ledger.addition(quiet.plan_entry(config, attitude))
        lit = ledger.addition(lamp.plan_entry(config, attitude))
        assert idle is not None and lit is not None
        # 130 W more for 20 minutes.
        assert lit.amount - idle.amount == pytest.approx(130.0 * 1200 / 3600.0)

    def test_the_ledger_counts_the_observations_modes(self) -> None:
        """A snapshot with the camera alone draws 100 W less than one with
        both instruments, and the plan executes to the ledger's prediction."""
        config = _moded_battery_config()
        camera = _target(config, 1, *PATCH[0], merit=90.0, minutes=40, snapshot=40)
        camera.instrument_modes = {"camera": "SCIENCE"}
        both = _target(config, 2, *PATCH[1], merit=50.0, minutes=40, snapshot=40)
        scheduler = PriorityScheduler(
            config, [camera, both], BEGIN, END, step_size=STEP
        )
        plan = scheduler.schedule()
        ledger = scheduler._battery
        entries = {e.obsid: e for e in plan if e.obsid in (1, 2)}
        assert set(entries) == {1, 2}
        first = ledger.addition(entries[1])
        second = ledger.addition(entries[2])
        assert first is not None and second is not None
        # Per science step, the camera-only snapshot draws 130 W less...
        bus = config.spacecraft_bus.power(ACSMode.SCIENCE.value)
        assert ledger._watts(ACSMode.SCIENCE, False, {"camera": "SCIENCE"}) == (
            bus + 80.0 + 20.0
        )
        assert ledger._watts(
            ACSMode.SCIENCE, False, entries[2].effective_instrument_modes()
        ) == (bus + 80.0 + 150.0)
        # ...and the ledger's prediction is what executes.
        predicted = min(scheduler._battery.charge_levels(scheduler._energy))
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
        assert min(ditl.batterylevel) == pytest.approx(predicted, abs=0.01)
        assert min(ditl.batterylevel) >= LEVEL - 1e-6


class TestScienceData:
    def test_an_observation_generates_only_what_it_uses(self) -> None:
        """Both instruments generate 1 Mbps in SCIENCE; a camera-only
        observation puts half as much on the recorder as one using both,
        in the ledger and when the plan is executed."""
        config = _moded_config()
        for instrument in config.payload.instruments:
            instrument.data_generation = DataGeneration(
                science=ScienceData(rate_gbps=0.001)
            )
        camera = _target(config, 1, *PATCH[0], merit=90.0, minutes=20, snapshot=20)
        camera.instrument_modes = {"camera": "SCIENCE"}
        both = _target(config, 2, *PATCH[1], merit=50.0, minutes=20, snapshot=20)
        assert config.science_generated(
            1200.0, instrument_modes=camera.effective_instrument_modes()
        ) == pytest.approx(1.2)
        assert config.science_generated(
            1200.0, instrument_modes=both.effective_instrument_modes()
        ) == pytest.approx(2.4)
        scheduler = PriorityScheduler(
            config, [camera, both], BEGIN, END, step_size=STEP
        )
        plan = scheduler.schedule()
        entries = {e.obsid: e for e in plan if e.obsid in (1, 2)}
        assert scheduler._ledger.science_volume(entries[1], 1200.0) == pytest.approx(
            1.2
        )
        assert scheduler._ledger.science_volume(entries[2], 1200.0) == pytest.approx(
            2.4
        )
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
        assert max(ditl.recorder_volume_gb) == pytest.approx(1.2 + 2.4, abs=0.01)


def _ruled_config() -> MissionConfig:
    config = _moded_config()
    config.payload.concurrency = ConcurrencyRules(
        max_observing=1, exclusive=[("camera", "spectrograph")]
    )
    return config


class TestConcurrency:
    def test_a_request_breaking_the_rules_is_rejected(self) -> None:
        config = _ruled_config()
        Pointing(
            config=config,
            obsid=1,
            ra=0.0,
            dec=0.0,
            instrument_modes={"camera": "SCIENCE"},
        )
        with pytest.raises(ValidationError, match="allows 1"):
            Pointing(
                config=config,
                obsid=2,
                ra=0.0,
                dec=0.0,
                instrument_modes={"camera": "SCIENCE", "spectrograph": "SCIENCE"},
            )
        # Naming no modes means every instrument with modes, which breaks them.
        with pytest.raises(ValidationError, match="allows 1"):
            Pointing(config=config, obsid=3, ra=0.0, dec=0.0)

    def test_a_sequence_step_breaking_the_rules_is_rejected(self) -> None:
        config = _ruled_config()
        steps = ObservationSequence(
            steps=[
                SequenceStep(name="image", duration_seconds=300),
                SequenceStep(
                    name="both",
                    duration_seconds=300,
                    instrument_modes={"camera": "SCIENCE", "spectrograph": "SCIENCE"},
                ),
            ]
        )
        with pytest.raises(ValidationError, match="allows 1"):
            Pointing(
                config=config,
                obsid=4,
                ra=0.0,
                dec=0.0,
                instrument_modes={"camera": "SCIENCE"},
                sequence=steps,
            )

    def test_toos_and_activities_are_checked_too(self) -> None:
        config = _ruled_config()
        ditl = create_ditl(config, [], BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        with pytest.raises(ValueError, match="allows 1"):
            ditl.submit_too(
                obsid=30001,
                ra=0.0,
                dec=0.0,
                merit=1.0,
                exptime=60,
                name="both",
                instrument_modes={"camera": "SCIENCE", "spectrograph": "SCIENCE"},
            )
        with pytest.raises(ValueError, match="allows 1"):
            ditl.submit_too(
                obsid=30002, ra=0.0, dec=0.0, merit=1.0, exptime=60, name="all"
            )
        lamp = ScheduledActivity(
            name="lamp",
            begin=T0,
            duration_seconds=600,
            obsid=5,
            instrument_modes={"camera": "SCIENCE", "spectrograph": "SCIENCE"},
        )
        with pytest.raises(ValueError, match="allows 1"):
            lamp.plan_entry(config, (0.0, 0.0, 0.0))
        quiet = ScheduledActivity(name="quiet", begin=T0, duration_seconds=600, obsid=6)
        assert quiet.plan_entry(config, (0.0, 0.0, 0.0)).instrument_modes == {}
