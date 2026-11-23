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


@pytest.mark.parametrize("during_cleanup", [False, True])
def test_charging_during_a_visits_last_step_pays_its_teardown(
    monkeypatch, during_cleanup
) -> None:
    """Teardown follows a visit's last step; a visit fully collected is not retaken."""
    sequence = ObservationSequence(
        steps=[
            SequenceStep(name="reference", duration_seconds=300, ra=30, dec=0),
            SequenceStep(name="target", duration_seconds=900),
        ]
    )

    def simulate(alert: tuple[float, float]) -> tuple[QueueDITL, Pointing]:
        ditl, _ = _simulation(monkeypatch, 85, 40, alert)
        ditl.queue.targets.clear()
        target = Pointing(
            config=ditl.config,
            ephem=ditl.ephem,
            obsid=1,
            ra=20,
            dec=0,
            merit=100,
            sequence=sequence,
        )
        target.visibility()
        ditl.queue.targets.append(target)
        return ditl, target

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
