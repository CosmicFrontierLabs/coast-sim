"""Emergency charging keeps collected science and pays the observation's teardown."""

from datetime import timedelta
from math import ceil
from unittest.mock import Mock

import pytest
import rust_ephem

from conops import QueueDITL
from conops.common import ObsType
from conops.config import (
    AttitudeControlSystem,
    Battery,
    GroundStationRegistry,
    MissionConfig,
    ObservationTiming,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
)
from conops.targets import ObservationSequence, Pointing, SequenceStep
from conops.targets.plan_entry import PlanEntry
from scripts.check_default_plan_output import (
    SCENARIO_BEGIN,
    DeterministicConstraint,
    DeterministicEphemeris,
)

BEGIN = SCENARIO_BEGIN.timestamp()
STEP = 60


def _simulation(
    monkeypatch: pytest.MonkeyPatch,
    cleanup: float,
    handoff: float,
    alert: tuple[float, float],
) -> tuple[QueueDITL, Pointing]:
    """A two-hour run whose battery alert is raised between two offsets."""
    end = SCENARIO_BEGIN + timedelta(seconds=7200)
    ephem = DeterministicEphemeris(SCENARIO_BEGIN, end, step_size_seconds=STEP)
    config = MissionConfig(
        constraint=DeterministicConstraint(),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        battery=Battery(watthour=100000),
        spacecraft_bus=SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                max_slew_rate=1, slew_acceleration=0.5, settle_time=10
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
    )
    config.constraint.ephem = ephem
    config.payload.observation_timing = ObservationTiming(
        setup_seconds=10, cleanup_seconds=cleanup, handoff_seconds=handoff
    )
    monkeypatch.setattr(
        rust_ephem,
        "get_eop_provenance",
        lambda: {"ut1": {"available": False}, "polar_motion": {"available": False}},
    )
    monkeypatch.setattr(
        "conops.config.solar_panel._get_eclipse_constraint",
        lambda: Mock(in_constraint=Mock(return_value=False)),
    )
    ditl = QueueDITL(config=config, ephem=ephem, begin=SCENARIO_BEGIN, end=end)
    ditl.queue.add(obsid=1, ra=20, dec=0, merit=100, exptime=1800, ss_min=300)
    monkeypatch.setattr(
        ditl.emergency_charging,
        "should_initiate_charging",
        lambda utime, *_: alert[0] <= utime - BEGIN < alert[1],
    )
    return ditl, ditl.queue.targets[0]


def _entries(ditl: QueueDITL, obstype: ObsType) -> list[PlanEntry]:
    return [entry for entry in ditl.plan if entry.obstype == obstype]


def _collected(ditl: QueueDITL) -> float:
    return sum(
        entry.collection_end - entry.collection_begin
        for entry in _entries(ditl, ObsType.AT)
    )


@pytest.mark.parametrize("offset", [420, 900])
@pytest.mark.parametrize("cleanup,handoff", [(0, 0), (20, 5), (40, 20), (85, 40)])
def test_charging_waits_for_cleanup_without_losing_science(
    monkeypatch, cleanup, handoff, offset
) -> None:
    post = cleanup + handoff
    ditl, target = _simulation(monkeypatch, cleanup, handoff, (offset, offset + 240))

    assert ditl.calc()

    starts = offset + ceil(post / STEP) * STEP
    cut = _entries(ditl, ObsType.AT)[0]
    charge = _entries(ditl, ObsType.CHARGE)[0]
    assert cut.end - BEGIN == starts
    assert cut.collection_end - BEGIN == starts - post
    assert charge.begin - BEGIN == starts
    # What was collected is kept and the rest is collected later.
    assert _collected(ditl) == pytest.approx(target.collected_seconds)
    assert target.collected_seconds == pytest.approx(1800)
    assert ditl.validate_plan_matches_execution() == []


def test_charging_cuts_an_observation_still_slewing_at_once(monkeypatch) -> None:
    ditl, target = _simulation(monkeypatch, 20, 5, (60, 300))

    assert ditl.calc()

    assert _entries(ditl, ObsType.CHARGE)[0].begin - BEGIN == 60
    assert _collected(ditl) == pytest.approx(target.collected_seconds)
    assert ditl.validate_plan_matches_execution() == []


def test_alert_during_the_observations_own_cleanup_waits_for_its_end(
    monkeypatch,
) -> None:
    undisturbed, _ = _simulation(monkeypatch, 85, 40, (1e9, 1e9))
    assert undisturbed.calc()
    entry = _entries(undisturbed, ObsType.AT)[0]
    in_cleanup = ceil((entry.collection_end - BEGIN) / STEP) * STEP
    assert in_cleanup < entry.end - BEGIN

    ditl, target = _simulation(monkeypatch, 85, 40, (in_cleanup, in_cleanup + 600))
    assert ditl.calc()

    flown = _entries(ditl, ObsType.AT)[0]
    assert flown.collection_end == entry.collection_end
    assert flown.end == entry.end
    assert _entries(ditl, ObsType.CHARGE)[0].begin >= entry.end
    assert target.collected_seconds == pytest.approx(1800)
    assert ditl.validate_plan_matches_execution() == []


def test_no_charging_attitude_leaves_the_observation_alone(monkeypatch) -> None:
    ditl, target = _simulation(monkeypatch, 20, 5, (420, 900))
    monkeypatch.setattr(
        ditl.emergency_charging, "create_charging_pointing", lambda *_, **__: None
    )

    assert ditl.calc()

    assert len(_entries(ditl, ObsType.AT)) == 1
    assert not _entries(ditl, ObsType.CHARGE)
    assert target.collected_seconds == pytest.approx(1800)


def test_alert_that_clears_during_cleanup_resumes_the_target(monkeypatch) -> None:
    ditl, target = _simulation(monkeypatch, 85, 40, (420, 421))

    assert ditl.calc()

    assert not _entries(ditl, ObsType.CHARGE)
    assert ditl.emergency_charging.current_charging_ppt is None
    assert len(_entries(ditl, ObsType.AT)) == 2
    assert _collected(ditl) == pytest.approx(target.collected_seconds)
    assert target.collected_seconds == pytest.approx(1800)
    assert ditl.validate_plan_matches_execution() == []


def test_waiting_does_not_use_up_charging_obsids(monkeypatch) -> None:
    ditl, _ = _simulation(monkeypatch, 20, 5, (420, 481))
    first = ditl.emergency_charging.next_charging_obsid

    assert ditl.calc()

    assert [entry.obsid for entry in _entries(ditl, ObsType.CHARGE)] == [first]


def _visit(
    monkeypatch: pytest.MonkeyPatch, alert: tuple[float, float]
) -> tuple[QueueDITL, Pointing]:
    """A run of one two-step sequence target, with 125 s of teardown."""
    ditl, _ = _simulation(monkeypatch, 85, 40, alert)
    ditl.queue.targets.clear()
    target = Pointing(
        config=ditl.config,
        ephem=ditl.ephem,
        obsid=1,
        ra=20,
        dec=0,
        merit=100,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(name="reference", duration_seconds=300, ra=30, dec=0),
                SequenceStep(name="target", duration_seconds=900),
            ]
        ),
    )
    target.visibility()
    ditl.queue.targets.append(target)
    return ditl, target


@pytest.mark.parametrize("during_cleanup", [False, True])
def test_charging_during_a_visits_last_step_pays_its_teardown(
    monkeypatch, during_cleanup
) -> None:
    """Teardown follows a visit's last step; a visit fully collected is not retaken."""

    def simulate(alert: tuple[float, float]) -> tuple[QueueDITL, Pointing]:
        return _visit(monkeypatch, alert)

    undisturbed, _ = simulate((1e9, 1e9))
    assert undisturbed.calc()
    last = _entries(undisturbed, ObsType.AT)[-1]
    moment = last.collection_end if during_cleanup else last.collection_begin + 120
    alert = ceil((moment - BEGIN) / STEP) * STEP

    ditl, target = simulate((alert, alert + 240))
    assert ditl.calc()

    charge = _entries(ditl, ObsType.CHARGE)[0]
    cut = [e for e in _entries(ditl, ObsType.AT) if e.end <= charge.begin][-1]
    assert cut.end - cut.collection_end == 125
    retaken = [e for e in _entries(ditl, ObsType.AT) if e.begin >= charge.end]
    assert len(retaken) == (0 if during_cleanup else 2)
    assert target.done
    assert ditl.validate_plan_matches_execution() == []


def _starts(ditl: QueueDITL) -> list[str]:
    return [
        event.description
        for event in ditl.log.events
        if "Starting EMERGENCY CHARGING" in event.description
    ]


def test_charging_during_a_visits_first_step_pays_the_visits_teardown(
    monkeypatch,
) -> None:
    undisturbed, _ = _visit(monkeypatch, (1e9, 1e9))
    assert undisturbed.calc()
    first = _entries(undisturbed, ObsType.AT)[0]
    alert = ceil((first.collection_begin + 60 - BEGIN) / STEP) * STEP

    ditl, target = _visit(monkeypatch, (alert, alert + 181))
    assert ditl.calc()

    cut = _entries(ditl, ObsType.AT)[0]
    charge = _entries(ditl, ObsType.CHARGE)[0]
    assert cut.end - cut.collection_end == 125
    assert charge.begin - BEGIN == alert + ceil(125 / STEP) * STEP
    assert charge.begin >= cut.end
    # The visit was cut, and is retaken whole.
    assert len(target.interruptions) == 1
    assert len([e for e in _entries(ditl, ObsType.AT) if e.begin >= charge.end]) == 2
    assert target.done
    assert ditl.validate_plan_matches_execution() == []


@pytest.mark.parametrize("remaining", [30, 150])
def test_a_visit_whose_alert_clears_does_not_go_on_to_its_next_step(
    monkeypatch, remaining
) -> None:
    """Its teardown is paid, so the visit is over, however little was cut."""
    undisturbed, _ = _visit(monkeypatch, (1e9, 1e9))
    assert undisturbed.calc()
    first = _entries(undisturbed, ObsType.AT)[0]
    alert = ceil((first.collection_end - remaining - BEGIN) / STEP) * STEP

    ditl, target = _visit(monkeypatch, (alert, alert + 1))
    assert ditl.calc()

    assert not _entries(ditl, ObsType.CHARGE)
    entries = _entries(ditl, ObsType.AT)
    assert entries[0].end - entries[0].collection_end == 125
    assert len(entries) == 3
    assert len(target.interruptions) == 1
    assert target.done
    assert ditl.validate_plan_matches_execution() == []


def _below_floor_from(ditl: QueueDITL, offset: float, monkeypatch) -> None:
    monkeypatch.setattr(
        type(ditl.battery),
        "below_minimum_charge_level",
        property(lambda _: len(ditl.batterylevel) * STEP >= offset),
    )


@pytest.mark.parametrize("cleanup,handoff", [(85, 40), (400, 100)])
def test_below_the_battery_floor_charging_cuts_collection_at_once(
    monkeypatch, cleanup, handoff
) -> None:
    ditl, target = _simulation(monkeypatch, cleanup, handoff, (900, 1800))
    _below_floor_from(ditl, 900, monkeypatch)

    assert ditl.calc()

    cut = _entries(ditl, ObsType.AT)[0]
    assert cut.end == cut.collection_end == BEGIN + 900
    assert _entries(ditl, ObsType.CHARGE)[0].begin == BEGIN + 900
    assert _collected(ditl) == pytest.approx(target.collected_seconds)
    assert ditl.validate_plan_matches_execution() == []


def test_falling_below_the_floor_cuts_the_cleanup_short(monkeypatch) -> None:
    ditl, target = _simulation(monkeypatch, 400, 100, (900, 1800))
    _below_floor_from(ditl, 1080, monkeypatch)

    assert ditl.calc()

    cut = _entries(ditl, ObsType.AT)[0]
    # Deferred to 1440 s; collection stopped at 940 s.
    assert cut.collection_end == BEGIN + 940
    assert cut.end == BEGIN + 1080
    assert _entries(ditl, ObsType.CHARGE)[0].begin == BEGIN + 1080
    assert _collected(ditl) == pytest.approx(target.collected_seconds)
    assert ditl.validate_plan_matches_execution() == []


def test_no_science_is_given_up_for_charging_that_would_meet_eclipse(
    monkeypatch,
) -> None:
    ditl, target = _simulation(monkeypatch, 85, 40, (420, 540))
    monkeypatch.setattr(
        ditl.emergency_charging,
        "_is_in_sunlight",
        lambda utime, _: utime < BEGIN + 540,
    )

    assert ditl.calc()

    assert len(_entries(ditl, ObsType.AT)) == 1
    assert not _entries(ditl, ObsType.CHARGE)
    assert target.collected_seconds == pytest.approx(1800)
    assert not _starts(ditl)
    assert not [e for e in ditl.log.events if e.event_type == "ERROR"]


def test_no_science_is_given_up_for_charging_after_the_run(monkeypatch) -> None:
    undisturbed, target = _simulation(monkeypatch, 85, 40, (1e9, 1e9))
    target.exptime = 86400
    target.ss_max = 86400
    assert undisturbed.calc()
    expected = _entries(undisturbed, ObsType.AT)[-1].collection_end

    alert = 7200 - 120
    ditl, target = _simulation(monkeypatch, 85, 40, (alert, 7200))
    target.exptime = 86400
    target.ss_max = 86400
    assert ditl.calc()
    assert _entries(ditl, ObsType.AT)[-1].collection_end == expected
    assert ditl.validate_plan_matches_execution() == []


def test_a_deferred_charge_is_logged_as_started_once(monkeypatch) -> None:
    ditl, _ = _simulation(monkeypatch, 85, 40, (420, 601))

    assert ditl.calc()

    assert len(_starts(ditl)) == len(_entries(ditl, ObsType.CHARGE)) == 1


def test_a_finished_observation_is_not_logged_as_interrupted(monkeypatch) -> None:
    ditl, target = _simulation(monkeypatch, 85, 40, (420, 601))

    assert ditl.calc()

    log = [event.description for event in ditl.log.events]
    assert "Time window elapsed, ending observation" in log
    assert not [line for line in log if "interrupting science observation" in line]
    assert _entries(ditl, ObsType.AT)[0].end <= _entries(ditl, ObsType.CHARGE)[0].begin
    assert target.collected_seconds == pytest.approx(1800)


def test_collection_already_flown_is_never_relabelled(monkeypatch) -> None:
    """The stop is not before now, however short the observation's own teardown."""
    ditl, _ = _simulation(monkeypatch, 85, 40, (1e9, 1e9))
    assert ditl.calc()
    entry = _entries(ditl, ObsType.AT)[0]
    ppt = entry.model_copy()
    utime = float(entry.collection_begin) + 120
    ppt.collection_end = utime + 100
    ppt.end = utime + 110
    ditl.ppt = ppt
    ditl._sequence = None
    ditl._defer_charging(utime, ditl._charging_begin(utime))
    assert ppt.collection_end >= utime


def _draining(
    monkeypatch: pytest.MonkeyPatch,
    cleanup: float,
    handoff: float,
    offset: float,
    *,
    floor: bool = False,
) -> tuple[QueueDITL, Pointing]:
    """A run whose battery itself falls below its threshold at ``offset``."""
    ditl, target = _simulation(monkeypatch, cleanup, handoff, (0, 0))
    monkeypatch.undo()
    monkeypatch.setattr(
        rust_ephem,
        "get_eop_provenance",
        lambda: {"ut1": {"available": False}, "polar_motion": {"available": False}},
    )
    monkeypatch.setattr(
        "conops.config.solar_panel._get_eclipse_constraint",
        lambda: Mock(in_constraint=Mock(return_value=False)),
    )
    battery = ditl.battery
    level = battery.minimum_charge_level + (-0.01 if floor else 0.01)
    battery.recharge_threshold = max(battery.recharge_threshold, level + 0.01)
    drained = False
    drain = type(battery).drain

    def fall(self: Battery, power: float, period: float) -> bool:
        nonlocal drained
        result = drain(self, power, period)
        if not drained and len(ditl.batterylevel) * STEP >= offset:
            drained = True
            self.charge_level = level * self.watthour
        return result

    monkeypatch.setattr(type(battery), "drain", fall)
    return ditl, target


@pytest.mark.parametrize("offset", [420, 900])
@pytest.mark.parametrize("cleanup,handoff", [(0, 0), (20, 5), (85, 40), (400, 100)])
def test_the_batterys_own_alert_pays_the_teardown(
    monkeypatch, cleanup, handoff, offset
) -> None:
    """The alert the power update raises is seen before the step's collection."""
    post = cleanup + handoff
    ditl, _ = _draining(monkeypatch, cleanup, handoff, offset)

    assert ditl.calc()

    cut = _entries(ditl, ObsType.AT)[0]
    charge = _entries(ditl, ObsType.CHARGE)[0]
    assert cut.end - cut.collection_end == post
    assert cut.collection_end >= BEGIN + offset
    assert charge.begin == max(cut.end, BEGIN + offset + STEP)
    assert charge.begin - BEGIN == offset + max(STEP, ceil(post / STEP) * STEP)
    assert ditl.validate_plan_matches_execution() == []


def test_the_batterys_own_fall_below_its_floor_cuts_at_once(monkeypatch) -> None:
    ditl, target = _draining(monkeypatch, 400, 100, 900, floor=True)

    assert ditl.calc()

    charge = _entries(ditl, ObsType.CHARGE)[0]
    assert charge.begin == BEGIN + 960
    cut = _entries(ditl, ObsType.AT)[0]
    assert cut.end == BEGIN + 960
    assert _collected(ditl) == pytest.approx(target.collected_seconds)
    assert ditl.validate_plan_matches_execution() == []


def test_the_batterys_own_alert_gives_up_nothing_without_a_charging_attitude(
    monkeypatch,
) -> None:
    ditl, target = _draining(monkeypatch, 85, 40, 420)
    monkeypatch.setattr(
        ditl.emergency_charging, "create_charging_pointing", lambda *_, **__: None
    )

    assert ditl.calc()

    assert not _entries(ditl, ObsType.CHARGE)
    assert target.collected_seconds == pytest.approx(1800)
