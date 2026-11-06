"""A partial run samples orbital state at its UTC, rather than its loop index."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from rust_ephem import TLEEphemeris

from conops import DITL
from conops.common import ObsType, radec2vec, scbodyvector
from conops.config import (
    AttitudeControlSystem,
    Battery,
    Constraint,
    GroundStationRegistry,
    MissionConfig,
    Payload,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
    StoredMomentumConfig,
)
from conops.targets import Plan, PlanEntry


@pytest.mark.parametrize("step_seconds", [60, 120])
@pytest.mark.parametrize("offset_seconds", [0, 3600])
def test_partial_execution_matches_the_same_orbit_trimmed_to_its_window(
    step_seconds, offset_seconds
):
    origin = datetime(2025, 11, 1, tzinfo=timezone.utc)
    begin = origin + timedelta(seconds=offset_seconds)
    end = begin + timedelta(minutes=30)
    runs = []
    for ephem_begin in [origin, begin]:
        ephem = TLEEphemeris(
            tle="examples/example.tle", begin=ephem_begin, end=end, step_size=60
        )
        config = MissionConfig(
            constraint=Constraint(),
            battery=Battery(watthour=100000),
            solar_panel=SolarPanelSet(panels=[]),
            payload=Payload(instruments=[]),
            ground_stations=GroundStationRegistry(stations=[]),
            spacecraft_bus=SpacecraftBus(
                inertia_tensor_body_kg_m2=np.diag([1000, 2000, 3000]),
                attitude_control=AttitudeControlSystem(
                    stored_momentum=StoredMomentumConfig(gravity_gradient_enabled=True)
                ),
                star_trackers=StarTrackerConfiguration(
                    star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
                ),
                radiators=RadiatorConfiguration(radiators=[]),
            ),
        )
        config.constraint.ephem = ephem
        t = begin.timestamp()
        plan = Plan(
            entries=[
                PlanEntry(
                    config=config,
                    obsid=100,
                    obstype=ObsType.PPT,
                    ra=105,
                    dec=10,
                    roll=0,
                    begin=t,
                    end=t + 600,
                    slewtime=0,
                    collection_begin=t,
                    collection_end=t + 600,
                    ss_min=60,
                    ss_max=600,
                    exptime=600,
                )
            ]
        )
        run = DITL(config, ephem=ephem, plan=plan, begin=begin, end=end)
        run.step_size = step_seconds
        run.acs.ra, run.acs.dec, run.acs.roll = 105, 10, 0
        assert run.calc()
        assert not run.validate_plan_matches_execution()
        runs.append(run)
    for original, trimmed in zip(
        runs[0].telemetry.housekeeping, runs[1].telemetry.housekeeping, strict=True
    ):
        assert original.timestamp == trimmed.timestamp
        index = runs[0].ephem.index(original.timestamp)
        expected_sun = scbodyvector(
            *np.radians([original.ra, original.dec, original.roll]),
            radec2vec(
                np.radians(float(runs[0].ephem.sun_ra_deg[index])),
                np.radians(float(runs[0].ephem.sun_dec_deg[index])),
            ),
        )
        assert original.sun_body_vector == pytest.approx(expected_sun, abs=1e-12)
        assert original.sun_body_vector == pytest.approx(
            trimmed.sun_body_vector, abs=1e-12
        )
        assert original.earth_body_vector == pytest.approx(
            trimmed.earth_body_vector, abs=1e-12
        )
        assert original.stored_momentum_body_n_m_s == pytest.approx(
            trimmed.stored_momentum_body_n_m_s, abs=1e-10
        )
