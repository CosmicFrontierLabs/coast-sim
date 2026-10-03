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
    sim._update_stored_momentum(time, 0.0, 45.0, 0.0)
    return sim._update_stored_momentum(time + 1.0, 0.0, 45.0, 0.0)


@pytest.mark.parametrize(
    "edit", ["inertia", "initial", "disable", "enable", "rate", "interval"]
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
        control.stored_momentum.max_sample_interval_s = 1.0

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
        assert simulation._stored_momentum_tracker.max_sample_interval_s == 1.25
    elif edit == "interval":
        assert simulation._stored_momentum_tracker.max_sample_interval_s == 1.0


def test_calc_rejects_coarse_cadence_before_execution(simulation, monkeypatch):
    simulation.step_size = simulation.ephem.step_size = 60
    pointing = Mock()
    monkeypatch.setattr(simulation.acs, "pointing", pointing)
    battery_before = simulation.battery.battery_level
    with pytest.raises(ValueError, match="sample interval 60 s exceeds 2.5 s"):
        simulation.calc()
    pointing.assert_not_called()
    assert simulation.battery.battery_level == battery_before
    assert not simulation.telemetry.housekeeping


def test_coarse_ephemeris_is_not_hidden_by_fine_simulation_step(simulation):
    simulation.ephem.step_size = 60
    with pytest.raises(ValueError, match="sample interval 60 s exceeds 2.5 s"):
        simulation._reset_stored_momentum_tracker()


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
    assert simulation._stored_momentum_tracker.max_sample_interval_s == 2.5


def test_calc_checks_its_effective_execution_step(simulation, monkeypatch):
    simulation.step_size = 60  # overwritten from the fine ephemeris before checking
    if isinstance(simulation, QueueDITL):
        monkeypatch.setattr(simulation, "_setup_simulation_timing", lambda: False)
        assert simulation.calc() is False
        assert simulation.step_size == 1
    else:
        with pytest.raises(ValueError, match="sample interval 60 s exceeds 2.5 s"):
            simulation.calc()
