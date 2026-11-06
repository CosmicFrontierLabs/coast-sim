"""Subsecond safety checks must not depend on which cache entry was filled first."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pytest
import rust_ephem

from conops import (
    ACSMode,
    AttitudeControlSystem,
    Constraint,
    MissionConfig,
    QueueDITL,
    SpacecraftBus,
    StoredMomentumConfig,
)
from conops.config import RadiatorConfiguration, StarTrackerConfiguration
from conops.simulation.momentum_planning import MomentumObservationPlanner


@pytest.mark.parametrize("offsets", [(0.0, 0.25), (0.25, 0.0)])
def test_fractional_times_have_independent_cached_results(
    constraint_with_ephem, offsets
):
    start = 1700000000.0

    def check(*, time, **kwargs):
        return time.microsecond > 0

    with patch("rust_ephem.SunConstraint.in_constraint", side_effect=check) as backend:
        for offset in offsets:
            result = constraint_with_ephem.in_sun(
                45, 30, start + offset, target_roll=15
            )
            assert result is (offset > 0)
        for offset in offsets:
            assert constraint_with_ephem.in_sun(
                45, 30, start + offset, target_roll=15
            ) is (offset > 0)
        assert backend.call_count == 2
        assert constraint_with_ephem.cache_stats() == (2, 2)


@pytest.mark.parametrize("pitch", [0, 90])
def test_momentum_slew_checks_fractional_samples_with_real_ephemeris(tmp_path, pitch):
    begin = datetime(2027, 12, 1, 19, tzinfo=timezone.utc)
    path = tmp_path / "orbit.txt"
    path.write_text(
        f"ScenarioEpoch {begin.isoformat()}\nCoordinateSystem GCRS\n"
        "0 7000 0 0 0 7.5 0\n120 7000 900 0 0 7.5 0\n"
    )
    ephem = rust_ephem.FileEphemeris(
        str(path), begin=begin, end=begin + timedelta(seconds=120), step_size=2
    )
    # Exercise the same hardware-safety route with axial and transverse mounts.
    # This permissive keepout allows every trajectory sample to be checked.
    config = MissionConfig(
        constraint=Constraint(
            ephem=ephem,
            safety_constraint=rust_ephem.SunConstraint(min_angle=0).boresight_offset(
                pitch_deg=pitch
            ),
        ),
        spacecraft_bus=SpacecraftBus(
            inertia_tensor_body_kg_m2=np.eye(3) * 1000,
            attitude_control=AttitudeControlSystem(
                max_slew_rate=2,
                slew_acceleration=1,
                settle_time=3,
                stored_momentum=StoredMomentumConfig(capacity_n_m_s=100),
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
    )
    sim = QueueDITL(config=config, ephem=ephem, calculate_field_of_regard=False)
    sim.ustart, sim.uend = begin.timestamp(), ephem.timestamp[-1].timestamp()
    sim.acs.ra, sim.acs.dec, sim.acs.roll = 0, 0, 0
    sim.acs._advance_attitude(sim.ustart)
    planner = MomentumObservationPlanner(sim)
    slew, trajectory = planner._slew(planner.executor.state, (17.3, 4.1, 9.7), 1)
    with patch.object(
        sim,
        "_attitude_constraint_name_for_attitude",
        wraps=sim._attitude_constraint_name_for_attitude,
    ) as check:
        assert sim._slew_attitude_constraint_violation(slew, ACSMode.SLEWING) is None
        assert (
            planner._first_violation(trajectory, trajectory.end, ACSMode.SLEWING)
            is None
        )
    checked = [call.args[3] for call in check.call_args_list]
    assert any(abs(t - round(t)) > 1e-3 for t in checked)
    assert set(trajectory.boundaries[:-1]).issubset(checked)
    assert config.constraint.cache_stats()[1] > 2
