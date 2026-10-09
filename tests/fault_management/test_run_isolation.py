"""Fault reports are scoped to executions, not shared configuration or dates."""

from datetime import timedelta
from types import SimpleNamespace

import pytest

from conops import DITL, ACSMode, FaultManagement, MissionConfig, QueueDITL
from conops.config import (
    AttitudeControlSystem,
    Battery,
    GroundStationRegistry,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
)
from conops.ditl.telemetry import Housekeeping
from conops.simulation.attitude import AttitudeExecutionError
from scripts.check_default_plan_output import (
    SCENARIO_BEGIN,
    DeterministicConstraint,
    DeterministicEphemeris,
)


def test_policy_and_reports_are_independent(acs_stub):
    policy = FaultManagement()
    policy.add_threshold("battery_level", yellow=0.5, red=0.4)
    before = policy.model_dump()
    first = policy.new_run()
    hk = Housekeeping(timestamp=SCENARIO_BEGIN, battery_level=0.1)
    first.check(hk, acs_stub)
    recorded = first.model_dump()
    second = policy.new_run()
    assert first.run_id != second.run_id
    assert first.safe_mode_requested
    assert not second.safe_mode_requested
    assert second.events == []
    assert second.states == {}
    second.check(hk, acs_stub)
    assert second.events[0].utime == first.events[0].utime
    policy.thresholds[0].red = 0.2
    assert first.thresholds[0].red == second.thresholds[0].red == 0.4
    assert first.model_dump() == recorded
    assert set(before) == {"thresholds", "red_limit_constraints", "safe_mode_on_red"}


@pytest.fixture(params=[DITL, QueueDITL])
def simulation(request, monkeypatch):
    monkeypatch.setattr(
        "conops.config.solar_panel._get_eclipse_constraint",
        lambda: SimpleNamespace(in_constraint=lambda *args, **kwargs: False),
    )
    end = SCENARIO_BEGIN + timedelta(seconds=60)
    ephem = DeterministicEphemeris(SCENARIO_BEGIN, end, step_size_seconds=2)
    config = MissionConfig(
        constraint=DeterministicConstraint(),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        battery=Battery(watthour=100_000),
        spacecraft_bus=SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                max_slew_rate=2, slew_acceleration=0.5, settle_time=0
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
    )
    sim = request.param(config=config, ephem=ephem, begin=SCENARIO_BEGIN, end=end)
    sim.step_size = 2
    sim.acs.ra, sim.acs.dec, sim.acs.roll = 0, 0, 0
    sim.acs._hold_idle_attitude(0, 0, 0, SCENARIO_BEGIN.timestamp())
    return sim


def test_shared_config_does_not_share_faults(simulation):
    first = simulation
    second = type(first)(
        config=first.config, ephem=first.ephem, begin=first.begin, end=first.end
    )
    second.step_size = 2
    first.fault_management.check(
        Housekeeping(timestamp=SCENARIO_BEGIN, battery_level=0), first.acs
    )
    recorded = first.fault_management.model_dump()
    assert second.calc()
    assert second.config is first.config
    assert not second.acs.in_safe_mode
    assert not second.fault_management.safe_mode_requested
    assert first.fault_management.model_dump() == recorded


def test_repeated_calc_preserves_reports_without_reusing_faults(simulation):
    sim = simulation
    assert sim.calc()
    first = sim.fault_management
    first.check(Housekeeping(timestamp=SCENARIO_BEGIN, battery_level=0), sim.acs)
    sim.acs.in_safe_mode = True
    sim.acs.acsmode = ACSMode.SAFE
    recorded = first.model_dump()
    assert sim.calc()
    assert not sim.acs.in_safe_mode
    assert sim.fault_runs == [first, sim.fault_management]
    assert sim.fault_management.run_id != first.run_id
    assert not sim.fault_management.safe_mode_requested
    assert first.model_dump() == recorded
    assert len(sim.telemetry.housekeeping) == len(sim.utime) == len(sim.ra) == 30
    assert sim.fault_management is sim.acs.fault_management


def test_archiving_report_elsewhere_does_not_disable_run_boundary(simulation):
    sim = simulation
    assert sim.calc()
    archived = sim.fault_runs.pop()
    sim.acs.in_safe_mode = True
    assert sim.calc()
    assert not sim.acs.in_safe_mode
    assert sim.fault_management.run_id != archived.run_id
    assert len(sim.telemetry.housekeeping) == 30


def test_exception_keeps_report_and_next_calc_gets_fresh_state(simulation, monkeypatch):
    sim = simulation

    def fail(utime):
        sim.fault_management.check(
            Housekeeping(timestamp=SCENARIO_BEGIN, battery_level=0), sim.acs
        )
        raise RuntimeError("Synthetic execution failure")

    monkeypatch.setattr(sim.acs, "pointing", fail)
    with pytest.raises(RuntimeError, match="Synthetic execution failure"):
        sim.calc()
    first = sim.fault_management
    assert first.events
    recorded = first.model_dump()
    assert sim.calc()
    assert len(sim.fault_runs) == 2
    assert first.model_dump() == recorded
    assert not sim.fault_management.safe_mode_requested


def test_new_run_restarts_delayed_threshold_timer(acs_stub):
    policy = FaultManagement()
    policy.add_threshold(
        "battery_level", yellow=0.5, red=0.4, safe_mode_delay_seconds=1
    )
    first = policy.new_run()
    hk = Housekeeping(timestamp=SCENARIO_BEGIN, battery_level=0)
    first.check(hk, acs_stub)
    first.check(
        hk.model_copy(update={"timestamp": SCENARIO_BEGIN + timedelta(seconds=1)}),
        acs_stub,
    )
    assert first.safe_mode_requested
    second = policy.new_run()
    second.check(hk, acs_stub)
    assert not second.safe_mode_requested
    assert second.states["battery_level"].continuous_red_seconds == 0


def test_execution_fault_finishes_run_and_does_not_leak(simulation, monkeypatch):
    sim = simulation
    original = sim.acs._update_dwell_guidance
    failed = False

    def fail_once(utime):
        nonlocal failed
        if not failed:
            failed = True
            raise AttitudeExecutionError("Rejected discontinuous guidance")
        return original(utime)

    monkeypatch.setattr(sim.acs, "_update_dwell_guidance", fail_once)
    assert sim.calc() is False
    assert sim.acs.in_safe_mode
    assert len(sim.telemetry.housekeeping) == 30
    assert sim.plan.attitude_timeseries.num_samples == 30
    assert not sim._attitude_rate_violations()
    assert all(hk.collection_seconds == 0 for hk in sim.telemetry.housekeeping)
    first = sim.fault_management
    recorded = first.model_dump()
    assert first.states["attitude_execution"].current == "red"
    assert sim.calc() is True
    assert sim.fault_management is not first
    assert first.model_dump() == recorded
    assert "attitude_execution" not in sim.fault_management.states
    assert not sim.acs.in_safe_mode


def test_real_safehold_does_not_latch_the_next_run(simulation, monkeypatch):
    sim = simulation
    # Keep this regression about lifecycle, not solar-pointing geometry.
    monkeypatch.setattr(
        SolarPanelSet, "optimal_charging_pointing", lambda *args: (0.0, 0.0)
    )
    threshold = next(
        t for t in sim.config.fault_management.thresholds if t.name == "battery_level"
    )
    threshold.red = threshold.yellow = 1.0
    assert sim.calc()
    assert sim.acs.in_safe_mode
    first = sim.fault_management
    recorded = first.model_dump()
    threshold.red = threshold.yellow = 0.0
    assert sim.calc()
    assert not sim.acs.in_safe_mode
    assert all(hk.acs_mode != ACSMode.SAFE for hk in sim.telemetry.housekeeping)
    assert first.model_dump() == recorded
    assert first.thresholds[0].red != sim.fault_management.thresholds[0].red


def test_legacy_run_state_in_config_is_ignored_with_warning():
    legacy = {
        "thresholds": [{"name": "battery_level", "yellow": 0.5, "red": 0.4}],
        "states": {},
        "safe_mode_requested": False,
        "events": [],
    }
    with pytest.warns(DeprecationWarning, match="run-state fields"):
        policy = FaultManagement.model_validate(legacy)
    assert [t.name for t in policy.thresholds] == ["battery_level"]
    assert "events" not in policy.model_dump()


def test_run_keeps_its_own_state_fields(recwarn):
    run = FaultManagement().new_run()
    restored = type(run).model_validate(run.model_dump())
    assert restored.events == [] and restored.states == {}
    assert not any(issubclass(w.category, DeprecationWarning) for w in recwarn)


def test_example_yaml_config_loads():
    MissionConfig.from_yaml_file("examples/example_config.yaml")
