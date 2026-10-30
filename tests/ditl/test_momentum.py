from datetime import timedelta
from unittest.mock import Mock

import numpy as np
import pytest

from conops import (
    DITL,
    AttitudeControlSystem,
    MissionConfig,
    QueueDITL,
    SpacecraftBus,
    StoredMomentumConfig,
)
from conops.common.vector import attitude_to_quat

from .conftest import DummyEphemeris


@pytest.fixture(params=[DITL, QueueDITL])
def simulation(request):
    ephem = DummyEphemeris()
    ephem.step_size = 1
    ephem.timestamp = [ephem.timestamp[0] + timedelta(seconds=i) for i in range(5)]
    bus = SpacecraftBus(
        inertia_tensor_body_kg_m2=np.diag([10.0, 8.0, 6.0]),
        attitude_control=AttitudeControlSystem(
            max_slew_rate_body=(0.2, 0.2, 2.0),
            stored_momentum=StoredMomentumConfig(gravity_gradient_enabled=True),
        ),
    )
    sim = request.param(config=MissionConfig(spacecraft_bus=bus), ephem=ephem)
    sim.step_size = 1
    return sim


def _sample(sim):
    time = sim.begin.timestamp()
    sim._reset_stored_momentum_tracker()
    position = sim.ephem.gcrs_pv.position[0]
    quaternion = attitude_to_quat(0.0, 45.0, 0.0)
    sim._update_stored_momentum(time, position, quaternion)
    return sim._update_stored_momentum(time + 1.0, position, quaternion)


@pytest.mark.parametrize(
    "edit", ["inertia", "initial", "disable", "enable", "rate", "angle"]
)
def test_run_reload_matches_fresh_configuration(simulation, edit):
    bus = simulation.config.spacecraft_bus
    control = bus.attitude_control
    if edit == "enable":
        control.stored_momentum.gravity_gradient_enabled = False
    before = _sample(simulation)
    if edit == "inertia":
        bus.inertia_tensor_body_kg_m2 = np.diag([20.0, 16.0, 12.0])
    elif edit == "initial":
        control.stored_momentum.initial_momentum_body_n_m_s = (1.0, 2.0, 3.0)
    elif edit in ("enable", "disable"):
        control.stored_momentum.gravity_gradient_enabled = edit == "enable"
    elif edit == "rate":
        control.max_slew_rate_body = (0.2, 0.2, 4.0)
    else:
        control.stored_momentum.max_attitude_step_deg = 2.0

    fresh_bus = SpacecraftBus.model_validate(bus.model_dump())
    fresh = type(simulation)(
        config=MissionConfig(spacecraft_bus=fresh_bus), ephem=simulation.ephem
    )
    fresh.step_size = 1
    actual, expected = _sample(simulation), _sample(fresh)
    assert actual == expected
    if edit == "inertia":
        assert actual.stored_momentum_norm_n_m_s == pytest.approx(
            2.0 * before.stored_momentum_norm_n_m_s
        )
    elif edit == "rate":
        assert simulation._momentum_slew_sample_s == 1.25
    elif edit == "angle":
        assert simulation._momentum_slew_sample_s == 1.0


def test_update_reuses_supplied_position_and_quaternion(simulation):
    position = np.array([100.0, 200.0, 7000.0])
    quaternion = attitude_to_quat(10.0, 20.0, 30.0)
    tracker = Mock()
    simulation._stored_momentum_tracker = tracker
    # No ephemeris access or angle conversion should be needed here.
    simulation.ephem = None
    sample = simulation._update_stored_momentum(1000.0, position, quaternion)
    tracker.update.assert_called_once()
    args = tracker.update.call_args.kwargs
    assert args["utime"] == 1000.0
    assert args["position_eci_km"] is position
    assert args["attitude_quaternion_eci_to_body"] is quaternion
    assert sample is tracker.update.return_value


def test_a_slew_within_a_coarse_step_is_sampled_from_the_acs(simulation, monkeypatch):
    """At a 120 s step with a 2.5 s attitude step, a slew flown during the
    step is integrated at 48 samples taken from the ACS, each at the
    attitude it had then, with the orbit interpolated between the step's
    ends; a step in which nothing moved is one sample."""
    simulation._reset_stored_momentum_tracker()
    tracker = Mock()
    simulation._stored_momentum_tracker = tracker
    time = simulation.begin.timestamp()
    first = np.array([7000.0, 0.0, 0.0])
    second = np.array([6000.0, 3000.0, 0.0])
    quaternion = attitude_to_quat(0.0, 45.0, 0.0)
    monkeypatch.setattr(simulation.acs, "attitude_moved", lambda begin, end: False)
    simulation._update_stored_momentum(time, first, quaternion)
    assert tracker.update.call_count == 1
    monkeypatch.setattr(simulation.acs, "attitude_moved", lambda begin, end: True)
    monkeypatch.setattr(
        simulation.acs,
        "executed_attitude",
        lambda utime: (0.0, 45.0 + 0.2 * (utime - time), 0.0),
    )
    simulation._update_stored_momentum(time + 120.0, second, quaternion)
    calls = tracker.update.call_args_list[1:]
    assert len(calls) == 48
    times = [call.kwargs["utime"] for call in calls]
    assert times == pytest.approx([time + 2.5 * k for k in range(1, 49)])
    halfway = calls[23].kwargs
    assert halfway["position_eci_km"] == pytest.approx((first + second) / 2.0)
    assert halfway["attitude_quaternion_eci_to_body"] == pytest.approx(
        attitude_to_quat(0.0, 45.0 + 0.2 * 60.0, 0.0)
    )
    assert calls[-1].kwargs["attitude_quaternion_eci_to_body"] is quaternion
    monkeypatch.setattr(simulation.acs, "attitude_moved", lambda begin, end: False)
    simulation._update_stored_momentum(time + 240.0, first, quaternion)
    assert tracker.update.call_count == 50


def test_disabled_tracking_does_not_restrict_sampling(simulation):
    simulation.config.spacecraft_bus.attitude_control.stored_momentum.gravity_gradient_enabled = False
    simulation.step_size = simulation.ephem.step_size = 60
    assert _sample(simulation) is None


def test_enabling_without_inertia_is_rejected_at_run_start(simulation):
    simulation.config.spacecraft_bus = SpacecraftBus()
    simulation.config.spacecraft_bus.attitude_control.stored_momentum.gravity_gradient_enabled = True
    with pytest.raises(ValueError, match="inertia_tensor_body_kg_m2 is required"):
        simulation._reset_stored_momentum_tracker()


def test_scalar_rate_is_used_without_body_limits(simulation):
    control = simulation.config.spacecraft_bus.attitude_control
    control.max_slew_rate_body = None
    control.max_slew_rate = 2.0
    simulation._reset_stored_momentum_tracker()
    assert simulation._momentum_slew_sample_s == 2.5


def test_queue_ditl_takes_its_step_from_the_ephemeris(simulation, monkeypatch):
    if not isinstance(simulation, QueueDITL):
        pytest.skip("the plan-driven DITL keeps the step it is given")
    simulation.step_size = 60
    monkeypatch.setattr(simulation, "_setup_simulation_timing", lambda: False)
    assert simulation.calc() is False
    assert simulation.step_size == 1


@pytest.mark.parametrize("kind", [DITL, QueueDITL])
def test_solar_radiation_pressure_is_tracked_and_reported(kind):
    """With sunlight on the surfaces tracked, a run reports both sources'
    torques by name and their total, the surfaces coming from the placeholder
    model when none are configured."""
    ephem = DummyEphemeris()
    ephem.step_size = 1
    ephem.timestamp = [ephem.timestamp[0] + timedelta(seconds=i) for i in range(5)]
    bus = SpacecraftBus(
        inertia_tensor_body_kg_m2=np.diag([10.0, 8.0, 6.0]),
        attitude_control=AttitudeControlSystem(
            max_slew_rate_body=(0.2, 0.2, 2.0),
            stored_momentum=StoredMomentumConfig(
                gravity_gradient_enabled=True, solar_radiation_pressure_enabled=True
            ),
        ),
    )
    sim = kind(config=MissionConfig(spacecraft_bus=bus), ephem=ephem)
    sim.step_size = 1
    sim._reset_stored_momentum_tracker()
    tracker = sim._stored_momentum_tracker
    assert [source.name for source in tracker.sources] == [
        "gravity_gradient",
        "solar_radiation_pressure",
    ]
    time = sim.begin.timestamp()
    position = sim.ephem.gcrs_pv.position[0]
    quaternion = attitude_to_quat(0.0, 45.0, 0.0)
    sim._update_stored_momentum(time, position, quaternion, index=0)
    sample = sim._update_stored_momentum(time + 1.0, position, quaternion, index=0)
    assert set(sample.torques_body_n_m) == {
        "gravity_gradient",
        "solar_radiation_pressure",
    }
    assert np.linalg.norm(sample.torques_body_n_m["solar_radiation_pressure"]) > 0
    assert sample.torque_body_n_m == pytest.approx(
        np.add(*sample.torques_body_n_m.values())
    )
    # Sunlight alone, with no inertia asymmetry to speak of, still fills the wheels.
    only_sun = StoredMomentumConfig(solar_radiation_pressure_enabled=True)
    bus.attitude_control.stored_momentum = only_sun
    sim._reset_stored_momentum_tracker()
    assert [s.name for s in sim._stored_momentum_tracker.sources] == [
        "solar_radiation_pressure"
    ]


@pytest.mark.parametrize("kind", [DITL, QueueDITL])
def test_magnetorquers_and_a_residual_dipole_are_sources_too(kind):
    """Magnetorquers configured on the stored-momentum tracking are an
    unloading source, the residual dipole a torque, both from the dipole
    field at the ephemeris's Earth-fixed position, which the telemetry reports."""
    ephem = DummyEphemeris()
    ephem.step_size = 1
    ephem.timestamp = [ephem.timestamp[0] + timedelta(seconds=i) for i in range(5)]
    ephem.itrs_pv = Mock(position=np.array([[6971.0, 0.0, 0.0]] * 5))
    bus = SpacecraftBus(
        inertia_tensor_body_kg_m2=np.diag([10.0, 8.0, 6.0]),
        attitude_control=AttitudeControlSystem(
            max_slew_rate_body=(0.2, 0.2, 2.0),
            stored_momentum=StoredMomentumConfig(
                gravity_gradient_enabled=True,
                magnetorquer_max_dipole_a_m2=10.0,
                residual_dipole_body_a_m2=(0.5, 0.0, 0.0),
                initial_momentum_body_n_m_s=(1.0, 0.0, 0.0),
            ),
        ),
    )
    sim = kind(config=MissionConfig(spacecraft_bus=bus), ephem=ephem)
    sim.step_size = 1
    sim._reset_stored_momentum_tracker()
    assert [s.name for s in sim._stored_momentum_tracker.sources] == [
        "gravity_gradient",
        "residual_dipole",
        "magnetorquer_unloading",
    ]
    time = sim.begin.timestamp()
    position = sim.ephem.gcrs_pv.position[0]
    quaternion = attitude_to_quat(0.0, 45.0, 0.0)
    sim._update_stored_momentum(time, position, quaternion, index=0)
    sample = sim._update_stored_momentum(time + 1.0, position, quaternion, index=0)
    assert sample.geomagnetic_field_body_t is not None
    assert np.linalg.norm(sample.geomagnetic_field_body_t) > 1e-5
    unloading = np.asarray(sample.torques_body_n_m["magnetorquer_unloading"])
    # The unloading torque works against the stored momentum.
    assert unloading @ np.asarray(sample.stored_momentum_body_n_m_s) < 0
    assert np.linalg.norm(sample.torques_body_n_m["residual_dipole"]) > 0
