"""A contact reached off a parked attitude is planned, flown and replayed alike."""

from datetime import timedelta

import numpy as np
import pytest
import rust_ephem

from conops import DITL, QueueDITL
from conops.common import ACSMode, ObsType
from conops.common.vector import angular_separation
from conops.config import (
    AttitudeConstraintScope,
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
CONTACT = slice(25, 41)  # Visible from +1500 s.
TARGET = (60.0, 20.0)
FORBIDDEN_FROM = SCENARIO_BEGIN.timestamp() + 1200


class ParkedConstraint(DeterministicConstraint):
    """The panels must not face away near the target, from 1200 s on."""

    def in_panel(
        self, ra: float, dec: float, time: float, target_roll: float | None = None
    ) -> bool:
        return time >= FORBIDDEN_FROM and angular_separation(ra, dec, *TARGET) < 5.0


def _config(ephem: DeterministicEphemeris) -> MissionConfig:
    config = MissionConfig(
        constraint=ParkedConstraint(),
        ground_stations=GroundStationRegistry(
            stations=[
                GroundStation(code="GS", name="GS", latitude_deg=0, longitude_deg=0)
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
        attitude_constraint_scopes={
            "SCIENCE": [AttitudeConstraintScope.POWER_GENERATION],
            "PASS": [AttitudeConstraintScope.POWER_GENERATION],
            "SLEWING": [AttitudeConstraintScope.HARDWARE_SAFETY],
            "IDLE": [AttitudeConstraintScope.HARDWARE_SAFETY],
        },
    )
    config.constraint.ephem = ephem
    return config


def _visibility(
    self: PassTimes, station: GroundStation, startindex: int, endindex: int
) -> tuple[np.ndarray, np.ndarray]:
    count = endindex - startindex
    elevation = np.full(count, -10.0)
    elevation[CONTACT] = 45.0
    return elevation, np.tile(np.array([0.0, 0.0, 1.0]), (count, 1))


def test_flown_and_replayed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PassTimes, "_station_visibility", _visibility)
    monkeypatch.setattr(
        rust_ephem,
        "get_eop_provenance",
        lambda: {"ut1": {"available": False}, "polar_motion": {"available": False}},
    )
    begin = SCENARIO_BEGIN
    end = begin + timedelta(seconds=3000)
    ephem = DeterministicEphemeris(begin, end, step_size_seconds=STEP)
    ditl = QueueDITL(config=_config(ephem), ephem=ephem, begin=begin, end=end)
    ditl.queue.add(
        obsid=1, ra=TARGET[0], dec=TARGET[1], merit=100, exptime=600, ss_min=60
    )
    ditl.acs.passrequests.get(begin.year, begin.timetuple().tm_yday, 1)
    assert ditl.calc()

    (contact,) = [e for e in ditl.plan if e.obstype == ObsType.GSP]
    assert contact.contact_ingress == "reposition"
    assert ACSMode.PASS in ditl.mode
    assert ditl.validate_plan_matches_execution() == []

    plan = Plan.model_validate_json(ditl.plan.model_dump_json())
    assert [e.contact_ingress for e in plan if e.obstype == ObsType.GSP] == [
        "reposition"
    ]
    replay = DITL(config=_config(ephem), ephem=ephem, plan=plan, begin=begin, end=end)
    replay.step_size = STEP
    assert replay.calc()
    assert replay.validate_plan_matches_execution() == []
    assert list(replay.mode) == list(ditl.mode)
