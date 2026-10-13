"""Replaying a plan must keep its scheduled contacts, whatever the RNG draws."""

from datetime import timedelta
from unittest.mock import Mock

import numpy as np
import pytest
import rust_ephem

from conops import DITL, QueueDITL
from conops.common import ACSMode, ObsType
from conops.config import (
    AttitudeControlSystem,
    Battery,
    GroundStation,
    GroundStationRegistry,
    MissionConfig,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
)
from conops.simulation.passes import PassTimes
from conops.targets import Plan
from scripts.check_default_plan_output import (
    SCENARIO_BEGIN,
    DeterministicConstraint,
    DeterministicEphemeris,
)

STEP = 60
CONTACT = slice(5, 21)  # Visible from +300 s until +1200 s.


def _config(ephem: DeterministicEphemeris) -> MissionConfig:
    config = MissionConfig(
        constraint=DeterministicConstraint(),
        ground_stations=GroundStationRegistry(
            stations=[
                GroundStation(
                    code="GS",
                    name="Coin-flip station",
                    latitude_deg=0,
                    longitude_deg=0,
                    schedule_probability=0.5,
                )
            ]
        ),
        solar_panel=SolarPanelSet(panels=[]),
        battery=Battery(watthour=100_000),
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
    return config


def _visibility(
    self: PassTimes, station: GroundStation, startindex: int, endindex: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic ground geometry: one contact window, fixed station direction."""
    count = endindex - startindex
    elevation = np.full(count, -10.0)
    elevation[CONTACT] = 45.0
    gs_to_sat = np.tile(np.array([0.0, 0.0, 1.0]), (count, 1))
    return elevation, gs_to_sat


def test_replay_keeps_a_planned_contact_the_replay_draw_would_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PassTimes, "_station_visibility", _visibility)
    monkeypatch.setattr(
        rust_ephem,
        "get_eop_provenance",
        lambda: {"ut1": {"available": False}, "polar_motion": {"available": False}},
    )
    begin = SCENARIO_BEGIN
    end = begin + timedelta(seconds=1500)
    ephem = DeterministicEphemeris(begin, end, step_size_seconds=STEP)

    # Planning wins the 50% scheduling draw for the contact.
    planner = QueueDITL(config=_config(ephem), ephem=ephem, begin=begin, end=end)
    planner.acs.passrequests.rng = Mock(random=Mock(return_value=0.1))
    planner.acs.passrequests.get(begin.year, begin.timetuple().tm_yday, 1)
    assert planner.calc()
    plan = Plan.model_validate_json(planner.plan.model_dump_json())
    assert [entry.obstype for entry in plan].count(ObsType.GSP) == 1

    # A fresh draw would now reject the same opportunity.
    replay = DITL(config=_config(ephem), ephem=ephem, plan=plan, begin=begin, end=end)
    replay.step_size = STEP
    replay.acs.passrequests.rng = Mock(random=Mock(return_value=0.9))
    assert replay.calc()

    replay.acs.passrequests.rng.random.assert_not_called()
    assert ACSMode.PASS in replay.mode
    assert replay.validate_plan_matches_execution() == []


@pytest.mark.parametrize(("apply_scheduling", "expected"), [(True, 1), (False, 2)])
def test_reconstruction_keeps_overlapping_contacts_for_plan_matching(
    monkeypatch: pytest.MonkeyPatch, apply_scheduling: bool, expected: int
) -> None:
    """Overlap deconfliction picks new opportunities; replay matches by station."""
    monkeypatch.setattr(PassTimes, "_station_visibility", _visibility)
    begin = SCENARIO_BEGIN
    ephem = DeterministicEphemeris(
        begin, begin + timedelta(seconds=1500), step_size_seconds=STEP
    )
    config = _config(ephem)
    config.ground_stations.add(
        GroundStation(code="GS2", name="Second", latitude_deg=0, longitude_deg=0)
    )
    passes = PassTimes(config=config, rng=Mock(random=Mock(return_value=0.1)))

    passes.get(
        begin.year, begin.timetuple().tm_yday, 1, apply_scheduling=apply_scheduling
    )

    assert len(passes.passes) == expected
