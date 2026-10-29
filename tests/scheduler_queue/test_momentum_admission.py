from datetime import timedelta
from unittest.mock import Mock

import numpy as np
import pytest

from conops import (
    ACSMode,
    AttitudeControlSystem,
    Battery,
    GroundStationRegistry,
    MissionConfig,
    QueueDITL,
    SolarPanelSet,
    SpacecraftBus,
    StoredMomentumConfig,
)
from conops.common import ObsType
from conops.config import (
    ObservationTiming,
    RadiatorConfiguration,
    StarTrackerConfiguration,
    Telescope,
)
from conops.simulation.attitude import AttitudeExecutor
from conops.simulation.idle_safety import IdleSafetyPlanner
from conops.simulation.momentum_budget import MomentumBudget, MomentumLimitError
from conops.simulation.momentum_planning import MomentumObservationPlanner
from conops.simulation.passes import Pass
from conops.simulation.slew import Slew
from conops.targets import PlanEntry
from scripts.check_default_plan_output import (
    SCENARIO_BEGIN,
    DeterministicConstraint,
    DeterministicEphemeris,
)


@pytest.fixture
def scenario(monkeypatch):
    ephem = DeterministicEphemeris(
        SCENARIO_BEGIN, SCENARIO_BEGIN + timedelta(hours=1), 60
    )
    config = MissionConfig(
        constraint=DeterministicConstraint(),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        battery=Battery(watthour=100_000),
        spacecraft_bus=SpacecraftBus(
            inertia_tensor_body_kg_m2=np.eye(3) * 1000,
            attitude_control=AttitudeControlSystem(
                max_slew_rate=2,
                slew_acceleration=1,
                settle_time=90,
                stored_momentum=StoredMomentumConfig(
                    capacity_n_m_s=20, roll_step_deg=90
                ),
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
    )
    config.payload.observation_timing = ObservationTiming(
        setup_seconds=43, cleanup_seconds=2, handoff_seconds=10
    )
    config.constraint.ephem = ephem
    # The synthetic orbit adapter cannot be passed to compiled eclipse code.
    # No panels are present; keep power outside this attitude/momentum test.
    monkeypatch.setattr(
        SolarPanelSet,
        "evaluate_executed_attitude",
        Mock(return_value=(0.0, 0.0, config.solar_panel.initial_drive_state())),
    )
    sim = QueueDITL(config=config, ephem=ephem, calculate_field_of_regard=False)
    sim.step_size = 60
    sim.ustart, sim.uend = (
        ephem.timestamp[0].timestamp(),
        ephem.timestamp[-1].timestamp(),
    )
    sim.acs.ra, sim.acs.dec, sim.acs.roll = 0.0, 0.0, 0.0
    sim.acs._advance_attitude(sim.ustart)
    sim._idle_safety = IdleSafetyPlanner(config, sim.uend)
    sim._ppt_optimum_roll = Mock(return_value=0.0)
    sim._next_charge_science_deadline = Mock(return_value=None)
    sim.queue.add(ra=40, dec=15, obsid=100, exptime=1500, ss_min=300)
    target = sim.queue.targets[0]
    target.visibility()
    sim.ppt = target
    return sim, target


def test_admission_slows_only_when_needed_and_keeps_settling(scenario):
    sim, target = scenario
    before = sim.acs.momentum_budget.current
    plan = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    assert plan is not None
    assert plan.slew.rate_fraction == 0.5
    assert plan.slew.motion_limits.slew_acceleration == 1
    assert plan.slew.motion_limits.settle_time == 90
    assert sim.config.spacecraft_bus.attitude_control.max_slew_rate == 2
    assert sim.acs.momentum_budget.current is before
    assert plan.end % sim.step_size == 0
    assert plan.end - plan.slew.slewend - 55 >= 300
    assert plan.recovery.slewstart == plan.end
    sim.acs._start_slew(plan.slew, sim.ustart)
    assert sim.acs._executor._trajectory is plan.slew._momentum_trajectory
    sim.acs._advance_attitude(plan.end)
    assert sim.acs.momentum_budget.current.sample.actuator_momentum_norm_n_m_s <= 20
    sim.acs._start_slew(plan.recovery, plan.end)
    sim.acs._advance_attitude(plan.recovery._momentum_hold_end)


def test_delayed_admission_is_replanned_from_actual_state_and_time(scenario):
    sim, target = scenario
    plan = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    assert plan is not None
    sim.acs._start_slew(plan.slew, sim.ustart + 60)
    actual = sim.acs.current_slew
    assert actual is not None and actual is not plan.slew
    assert actual.slewstart == sim.ustart + 60
    assert target.begin == actual.slewstart
    assert target.collection_begin == actual._momentum_collection_begin
    assert sim._momentum_recovery.slewstart == target.end
    assert sim.acs._executor._trajectory is actual._momentum_trajectory


def test_shorter_legal_snapshot_when_full_hold_is_unavailable(scenario, monkeypatch):
    sim, target = scenario
    original = sim._attitude_constraint_name_for_attitude

    def constrained(ra, dec, roll, utime, mode, **kwargs):
        if mode == ACSMode.SCIENCE and utime > sim.ustart + 900:
            return ("test_tracker", "science")
        return original(ra, dec, roll, utime, mode, **kwargs)

    monkeypatch.setattr(sim, "_attitude_constraint_name_for_attitude", constrained)
    plan = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    assert plan is not None and plan.end <= sim.ustart + 900
    assert plan.end - plan.slew.slewend - 55 >= target.ss_min


def test_no_science_without_a_valid_recovery(scenario, monkeypatch):
    sim, target = scenario
    monkeypatch.setattr(MomentumObservationPlanner, "recovery", lambda *args: None)
    # One minimum-length attempt per roll keeps this rejection test small.
    target.exptime = target.ss_min + 60
    assert MomentumObservationPlanner(sim).observation(target, sim.ustart) is None
    assert sim.acs.momentum_budget.current.utime == sim.ustart


def test_other_roll_can_make_the_whole_observation_feasible(scenario, monkeypatch):
    sim, target = scenario
    target.exptime = target.ss_min + 60

    def constrained(ra, dec, roll, utime, mode, **kwargs):
        return (
            ("tracker", "science")
            if mode == ACSMode.SCIENCE and abs(roll - 90) > 1e-5
            else None
        )

    monkeypatch.setattr(sim, "_attitude_constraint_name_for_attitude", constrained)
    plan = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    assert plan is not None and plan.slew.instrument_roll == 90


@pytest.mark.parametrize("enqueue_slew", [False, True])
def test_queue_admission_returns_slew_without_dispatching_a_preview(
    scenario, enqueue_slew
):
    sim, target = scenario
    sim.ppt = None
    commands = list(sim.acs.command_queue)
    distances = list(sim.acs.slew_dists)
    slew = sim._fetch_new_ppt(sim.ustart, 0.0, 0.0, enqueue_slew=enqueue_slew)
    assert slew is not None
    assert sim.ppt is target
    assert sim._momentum_recovery is not None
    if enqueue_slew:
        assert sim.acs.command_queue[-1].slew is slew
        assert sim.acs.slew_dists == distances + [slew.slewdist]
    else:
        assert sim.acs.command_queue == commands
        assert sim.acs.slew_dists == distances


def test_queue_executes_reserved_recovery_only_when_no_target_follows(scenario):
    sim, target = scenario
    sim.ppt = None
    sim._fetch_new_ppt(sim.ustart, 0.0, 0.0)
    assert sim.ppt is target
    reserved = sim._momentum_recovery
    assert reserved is not None
    assert all(c.slew is not reserved for c in sim.acs.command_queue)
    sim.acs._start_slew(sim.acs.command_queue.pop().slew, sim.ustart)
    sim.acs._advance_attitude(target.end)
    sim.acs.end_science_observation()
    sim.ppt = None
    sim._schedule_idle_recovery(target.end)
    assert sim.acs.command_queue[-1].slew is reserved


def test_delayed_too_keeps_its_admitted_momentum_recovery(scenario, monkeypatch):
    sim, _ = scenario
    sim.ppt = None
    sim.config.payload.observation_timing = ObservationTiming(
        setup_seconds=43, cleanup_seconds=85, handoff_seconds=40
    )
    request = sim.submit_too(
        obsid=30001,
        ra=100,
        dec=15,
        merit=1000,
        exptime=600,
        name="Urgent",
        submit_time=sim.ustart + 600,
    )
    preview = sim._preview_too_admission
    dispatch = sim._dispatch_committed_too
    dispatched = []

    def preview_without_side_effects(*args, **kwargs):
        commands = list(sim.acs.command_queue)
        distances = list(sim.acs.slew_dists)
        recovery = sim._momentum_recovery
        result = preview(*args, **kwargs)
        assert sim.acs.command_queue == commands
        assert sim.acs.slew_dists == distances
        assert sim._momentum_recovery is recovery
        return result

    def dispatch_with_recovery(utime):
        commitment = sim._committed_too
        result = dispatch(utime)
        if result:
            assert commitment.momentum_recovery is not None
            assert sim._momentum_recovery is commitment.momentum_recovery
            assert sim.ppt.slew_rate_fraction == commitment.slew.rate_fraction
            dispatched.append(commitment)
        return result

    monkeypatch.setattr(sim, "_preview_too_admission", preview_without_side_effects)
    monkeypatch.setattr(sim, "_dispatch_committed_too", dispatch_with_recovery)
    assert sim.calc()
    assert request.executed and len(dispatched) == 1
    assert not sim.validate_plan_matches_execution()
    assert all(
        hk.actuator_momentum_norm_n_m_s <= 20 for hk in sim.telemetry.housekeeping
    )


def test_real_queue_run_with_capacity_and_coarse_scheduler(scenario):
    sim, _ = scenario
    # Fresh ACS is initialized by calc as in a normal run. No external torque
    # here: a 60-second orbit suffices, while motion is integrated internally.
    sim.ppt = None
    assert sim.calc()
    assert sim.plan and sim.telemetry.housekeeping
    assert all(
        hk.actuator_momentum_norm_n_m_s is not None
        and hk.actuator_momentum_norm_n_m_s <= 20
        for hk in sim.telemetry.housekeeping
    )
    assert sim.acs.momentum_budget.current.utime == sim.uend
    assert not sim.validate_plan_matches_execution()


def test_reduced_speed_survives_plan_entry_roundtrip_and_dispatch(scenario):
    sim, target = scenario
    plan = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    assert plan is not None
    target.end = plan.end
    sim._apply_slew_metadata(target, plan.slew)
    restored = PlanEntry.model_validate_json(target.model_dump_json())
    assert restored.slew_rate_fraction == 0.5
    assert "slew_rate_fraction" not in PlanEntry().model_dump()
    restored.config = sim.config
    restored.constraint = sim.config.constraint
    restored.ephem = sim.ephem
    restored.acs_config = sim.config.spacecraft_bus.attitude_control
    assert sim.acs._enqueue_slew(
        *restored.target_body_attitude()[:2],
        restored.obsid,
        sim.ustart,
        roll=restored.target_body_attitude()[2],
        target_request=restored,
    )
    commanded = sim.acs.command_queue[-1].slew
    assert commanded.rate_fraction == 0.5
    assert commanded.motion_limits.settle_time == 90


def test_invisible_target_is_not_admitted(scenario, monkeypatch):
    sim, target = scenario
    monkeypatch.setattr(type(target), "next_vis", lambda *args: False)
    assert MomentumObservationPlanner(sim).observation(target, sim.ustart) is None


def test_hold_is_shortened_for_accumulating_gravity_gradient(scenario):
    sim, target = scenario
    ephem = DeterministicEphemeris(
        SCENARIO_BEGIN, SCENARIO_BEGIN + timedelta(hours=1), 2
    )
    ephem.gcrs_pv.position[:] = [7000.0, 0.0, 0.0]
    sim.ephem = sim.acs.ephem = sim.config.constraint.ephem = ephem
    target.ephem = ephem
    bus = sim.config.spacecraft_bus
    bus.inertia_tensor_body_kg_m2 = np.diag([1000.0, 800.0, 600.0])
    bus.attitude_control.max_slew_rate = 0.02
    bus.attitude_control.stored_momentum.capacity_n_m_s = 1.0
    bus.attitude_control.stored_momentum.gravity_gradient_enabled = True
    bus.attitude_control.stored_momentum.roll_step_deg = 360
    executor = AttitudeExecutor(sim.ustart, (0.0, 45.0, 0.0))
    executor.momentum = MomentumBudget(sim.config, ephem, executor.state)
    sim.acs._executor = executor
    target.ra, target.dec = 0.0, 46.0
    plan = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    assert plan is not None
    collection = (
        plan.end
        - plan.slew.slewend
        - sim.config.payload.observation_timing.total_seconds
    )
    assert target.ss_min <= collection < target.exptime
    sim.acs._start_slew(plan.slew, sim.ustart)
    sim.acs._advance_attitude(plan.end)
    assert sim.acs.momentum_budget.current.sample.stored_momentum_norm_n_m_s > 0.1
    sim.acs._start_slew(plan.recovery, plan.end)
    sim.acs._advance_attitude(plan.recovery._momentum_hold_end)


def test_second_target_can_follow_without_executing_reserved_return(scenario):
    sim, first = scenario
    sim.ppt = None
    sim._fetch_new_ppt(sim.ustart, 0.0, 0.0)
    recovery = sim._momentum_recovery
    sim.acs._start_slew(sim.acs.command_queue.pop().slew, sim.ustart)
    sim.acs._advance_attitude(first.end)
    first.done = True
    sim.ppt = None
    sim.queue.add(ra=45.0, dec=20.0, obsid=101, exptime=900, ss_min=300)
    sim._fetch_new_ppt(first.end, sim.acs.ra, sim.acs.dec)
    assert sim.ppt is not None and sim.ppt.obsid == 101
    assert all(command.slew is not recovery for command in sim.acs.command_queue)
    following = sim.acs.command_queue[-1].slew
    assert following.slewstart == first.end
    assert following.startra == pytest.approx(sim.acs.ra)
    assert following.startdec == pytest.approx(sim.acs.dec)


@pytest.mark.parametrize(
    "kind", [ObsType.PPT, ObsType.CHARGE, ObsType.SAFE, ObsType.IDLE]
)
def test_no_command_mode_bypasses_execution_capacity(scenario, kind):
    sim, _ = scenario
    command = Slew(config=sim.config, obstype=kind, endra=40, enddec=15)
    with pytest.raises(MomentumLimitError):
        sim.acs._start_slew(command, sim.ustart)
    assert sim.acs.current_slew is None
    assert sim.acs._executor._trajectory is None


def test_failed_initialization_cannot_leave_an_unguarded_executor(scenario):
    sim, _ = scenario
    sim.config.spacecraft_bus.attitude_control.stored_momentum.initial_momentum_body_n_m_s = (
        30,
        0,
        0,
    )
    acs = type(sim.acs)(config=sim.config)
    acs.ra, acs.dec, acs.roll = 0.0, 0.0, 0.0
    for _ in range(2):
        with pytest.raises(MomentumLimitError):
            acs._advance_attitude(sim.ustart)
        assert acs._executor is None


@pytest.mark.parametrize("fraction", [0, -1, 1.1, float("inf"), float("nan")])
def test_mutated_speed_cap_cannot_bypass_configured_limits(scenario, fraction):
    sim, _ = scenario
    command = Slew(config=sim.config)
    command.rate_fraction = fraction
    with pytest.raises(ValueError, match="rate fraction"):
        _ = command.motion_limits


def test_recovery_checks_slewing_constraints_not_only_idle(scenario, monkeypatch):
    sim, _ = scenario
    monkeypatch.setattr(
        sim,
        "_attitude_constraint_name_for_attitude",
        lambda ra, dec, roll, utime, mode, **kwargs: (
            ("tracker", "hardware") if mode == ACSMode.SLEWING else None
        ),
    )
    planner = MomentumObservationPlanner(sim)
    assert (
        planner.recovery(
            sim.acs._executor.state, sim.acs.momentum_budget.current, (40, 15, 0)
        )
        is None
    )


def test_short_pass_cannot_hide_inside_acquisition(scenario):
    sim, target = scenario
    # Acquisition plus settling lasts more than 100 s; neither endpoint lies
    # inside this pass. Check the entire interval, not just the two endpoints.
    sim.acs.passrequests.passes.append(
        Pass(station="test", begin=sim.ustart + 30, length=30)
    )
    assert MomentumObservationPlanner(sim).observation(target, sim.ustart) is None


def test_full_hold_at_another_roll_beats_shortened_preferred_roll(
    scenario, monkeypatch
):
    sim, target = scenario

    def constrained(ra, dec, roll, utime, mode, **kwargs):
        return (
            ("tracker", "science")
            if mode == ACSMode.SCIENCE and abs(roll) < 1e-5 and utime > sim.ustart + 900
            else None
        )

    monkeypatch.setattr(sim, "_attitude_constraint_name_for_attitude", constrained)
    plan = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    assert plan is not None and plan.slew.instrument_roll == 90
    assert (
        plan.end
        - plan.slew.slewend
        - sim.config.payload.observation_timing.total_seconds
        > 1400
    )


@pytest.mark.parametrize(
    "capacity, unsafe_after, fraction", [(10, 500, 0.25), (5, 900, 0.125)]
)
def test_idle_leaves_before_momentum_limited_escape_is_lost(
    scenario, monkeypatch, capacity, unsafe_after, fraction
):
    sim, _ = scenario
    sim.ppt = None
    limits = sim.config.spacecraft_bus.attitude_control
    limits.stored_momentum.capacity_n_m_s = capacity
    limits.idle_min_hold_s = 0
    executor = sim.acs._executor
    executor.momentum = MomentumBudget(sim.config, sim.ephem, executor.state)
    cutoff = sim.ustart + unsafe_after

    def constraint(ra, dec, roll, utime, mode, **kwargs):
        return ("timed keepout", "hardware") if utime >= cutoff and ra < 179.9 else None

    monkeypatch.setattr(sim, "_attitude_constraint_name_for_attitude", constraint)
    monkeypatch.setattr(
        sim._idle_safety,
        "first_violation",
        lambda attitude, start: cutoff if attitude[0] < 179.9 else float("inf"),
    )
    # The current attitude must not count as an escape merely because a short
    # zero-motion hold fits. The only actual exit requires a slower turn.
    monkeypatch.setattr(
        sim.acs,
        "_idle_safe_attitude_candidates",
        lambda now: [(0.0, 0.0), (180.0, 0.0)],
    )
    monkeypatch.setattr(sim.acs, "_idle_safe_roll_candidates", lambda roll: [0.0])
    planner = MomentumObservationPlanner(sim)
    immediate = planner.recovery(
        executor.state, executor.momentum.current, (180.0, 0.0, 0.0)
    )
    assert immediate is not None
    assert immediate.rate_fraction == fraction
    assert immediate.slewtime > sim._idle_safety.reserve

    recovery = None
    for offset in range(0, unsafe_after, 60):
        now = sim.ustart + offset
        sim.acs._advance_attitude(now)
        before = executor.momentum.current
        recovery = planner.idle_recovery(now)
        assert executor.momentum.current is before
        if recovery is not None:
            break
    assert recovery is not None
    assert recovery.rate_fraction == fraction
    assert recovery.slewstart < cutoff - sim._idle_safety.reserve
    # The selected command itself, not just its nominal duration, stays safe.
    sim.acs._start_slew(recovery, recovery.slewstart)
    for time in np.arange(
        recovery.slewstart, recovery._momentum_hold_end, planner.budget.step
    ):
        assert (
            constraint(*executor.predict(float(time)).attitude, time, ACSMode.IDLE)
            is None
        )
    sim.acs._advance_attitude(recovery._momentum_hold_end)
    assert executor.momentum.current.peak_n_m_s <= capacity


def test_idle_can_hold_to_end_without_an_escape_when_torque_is_enabled(
    scenario, monkeypatch
):
    sim, _ = scenario
    sim.ppt = None
    ephem = DeterministicEphemeris(
        SCENARIO_BEGIN, SCENARIO_BEGIN + timedelta(hours=1), 2
    )
    sim.ephem = sim.acs.ephem = sim.config.constraint.ephem = ephem
    settings = sim.config.spacecraft_bus.attitude_control.stored_momentum
    settings.gravity_gradient_enabled = True
    settings.capacity_n_m_s = 0.1
    executor = sim.acs._executor
    # The isotropic inertia produces no gravity-gradient torque. No nonzero
    # slew fits this small capacity, but the existing rest hold is safe.
    executor.momentum = MomentumBudget(sim.config, ephem, executor.state)
    monkeypatch.setattr(
        sim.acs,
        "_idle_safe_attitude_candidates",
        lambda now: [(0.0, 0.0), (180.0, 0.0)],
    )
    monkeypatch.setattr(sim.acs, "_idle_safe_roll_candidates", lambda roll: [0.0])
    before = executor.momentum.current
    assert MomentumObservationPlanner(sim).idle_recovery(sim.ustart) is None
    assert executor.momentum.current is before


@pytest.mark.parametrize("exposure", [300, 355.5])
def test_exact_exposure_survives_padded_hold_execution_and_export(scenario, exposure):
    sim, target = scenario
    target.exptime = exposure
    target.ss_min = 300
    sim.ppt = None
    assert sim.calc()
    entries = [entry for entry in sim.plan if entry.obsid == target.obsid]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.collection_end - entry.collection_begin == pytest.approx(exposure)
    assert entry.end % sim.step_size == 0
    assert entry.end > entry.collection_end + 12
    assert target.exptime == 0
    assert sum(
        hk.collection_seconds for hk in sim.telemetry.housekeeping
    ) == pytest.approx(exposure)
    assert not sim.validate_plan_matches_execution()
    restored = PlanEntry.model_validate_json(entry.model_dump_json())
    assert restored.collection_begin == entry.collection_begin
    assert restored.collection_end == entry.collection_end


def test_minimum_exposure_does_not_round_a_hard_deadline_up(scenario, monkeypatch):
    sim, target = scenario
    target.exptime = target.ss_min = 300
    sim.config.spacecraft_bus.attitude_control.stored_momentum.roll_step_deg = 360
    monkeypatch.setattr(
        sim,
        "_next_science_deadline",
        lambda *args, **kwargs: (sim.ustart + 500, "test"),
    )
    assert MomentumObservationPlanner(sim).observation(target, sim.ustart) is None


def test_recharge_and_metadata_sync_do_not_extend_exact_collection(scenario):
    sim, target = scenario
    target.exptime = target.ss_min = 300
    sim.ppt = None
    sim._fetch_new_ppt(sim.ustart, 0.0, 0.0)
    assert sim.ppt is target
    slew = sim.acs.command_queue[-1].slew
    assert target.end == sim.ustart + 540
    assert target.collection_begin == sim.ustart + 177
    assert target.collection_end == sim.ustart + 477
    sim.plan.append(target.model_copy())
    sim._next_charge_science_deadline = Mock(return_value=sim.ustart + 520)
    sim._reserve_pending_charge_cleanup(sim.ustart + 420)
    sim._sync_slew_metadata(slew)
    assert target.end == sim.plan[-1].end == sim.ustart + 520
    assert target.collection_end == sim.plan[-1].collection_end == sim.ustart + 477


@pytest.mark.parametrize("preferred, allowed", [(0.0, 90.0), (90.0, 0.0)])
def test_mounted_candidate_checks_trial_instrument_roll(
    scenario, monkeypatch, preferred, allowed
):
    sim, target = scenario
    telescope = Telescope(name="test", boresight=(0.0, 1.0, 0.0))
    sim.config.payload.instruments = [telescope]
    target.instrument_name = telescope.name
    target.roll = 0.0
    target.exptime = target.ss_min = 300
    sim._ppt_optimum_roll = Mock(return_value=preferred)
    checked = []

    def science_constraint(self, ra, dec, time, target_roll=None):
        checked.append((ra, dec, target_roll))
        return abs(target_roll - allowed) > 1e-5

    monkeypatch.setattr(
        DeterministicConstraint, "in_explicit_science", science_constraint
    )
    before = sim.acs.momentum_budget.current
    plan = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    assert plan is not None
    assert plan.slew.instrument_roll == allowed
    assert target.roll == 0.0  # Speculative checks must not change the live target.
    assert sim.acs.momentum_budget.current is before
    assert any(roll == preferred for _, _, roll in checked)
    assert any(roll == allowed for _, _, roll in checked)
    assert all((ra, dec) == (target.ra, target.dec) for ra, dec, _ in checked)
    target.end = plan.end
    sim._apply_slew_metadata(target, plan.slew)
    assert (
        sim._attitude_constraint_name_for_attitude(
            *target.spacecraft_attitude, plan.slew.slewend, ACSMode.SCIENCE
        )
        is None
    )


def test_momentum_admission_preserves_queue_collection_cap(scenario):
    sim, target = scenario
    target.ss_min = 60
    target.collection_limit = 200
    admitted = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    assert admitted is not None
    target.end = admitted.end
    sim._apply_slew_metadata(target, admitted.slew)
    assert target.exposure == 200
    assert admitted.slew.at is target
    assert admitted.slew.ephem is sim.ephem


def test_recorder_rechecks_the_actual_slower_slew(scenario, monkeypatch):
    sim, target = scenario
    target.ss_min = 60
    target.collection_limit = 200
    original_slewtime = target.slewtime
    monkeypatch.setattr(sim, "_recorder_room_gb", lambda *args: 1.0)
    durations = []

    def within(trial, room):
        durations.append(trial.slewtime)
        assert trial is not target and room == 1.0
        return 123.8 if trial.slewtime >= 130 else 200

    monkeypatch.setattr(sim.queue, "collection_within", within)
    admitted = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    assert admitted is not None
    assert durations == [admitted.slew.slewtime]
    assert target.slewtime == original_slewtime
    target.end = admitted.end
    sim._apply_slew_metadata(target, admitted.slew)
    assert target.exposure == 123


def test_slower_acquisition_must_meet_target_start_deadline(scenario):
    sim, target = scenario
    target.deadline = sim.ustart + 170
    assert MomentumObservationPlanner(sim).observation(target, sim.ustart) is None
    assert sim.acs.current_slew is None


def test_saa_delay_moves_collection_window_and_preserves_minimum(scenario):
    sim, target = scenario
    sim.config.payload.observation_timing.setup_seconds = 60
    sim.acs.saa = Mock(insaa=lambda time: time < sim.ustart + 480)
    target.exptime = target.ss_min
    admitted = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    assert admitted is not None
    target.end = admitted.end
    sim._apply_slew_metadata(target, admitted.slew)
    assert target.collection_begin == sim.ustart + 480
    assert target.exposure == target.ss_min
    sim._apply_slew_metadata(target, admitted.slew)
    assert target.collection_begin == sim.ustart + 480
    target.deadline = sim.ustart + 479
    assert MomentumObservationPlanner(sim).observation(target, sim.ustart) is None


def test_saa_delay_cannot_consume_minimum_collect_time(scenario, monkeypatch):
    sim, target = scenario
    sim.config.payload.observation_timing.setup_seconds = 60
    sim.acs.saa = Mock(insaa=lambda time: time < sim.ustart + 480)
    monkeypatch.setattr(
        sim,
        "_next_science_deadline",
        lambda *args, **kwargs: (sim.ustart + 720, "pass"),
    )
    assert MomentumObservationPlanner(sim).observation(target, sim.ustart) is None


def test_science_constraint_window_is_scanned_once_per_roll(scenario, monkeypatch):
    sim, target = scenario
    checks = []

    def constraint(ra, dec, roll, time, mode, **kwargs):
        if mode == ACSMode.SCIENCE:
            assert kwargs["target"] is not target
            checks.append(time)
            if time > sim.ustart + 900:
                return ("tracker", "science")
        return None

    monkeypatch.setattr(sim, "_attitude_constraint_name_for_attitude", constraint)
    admitted = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    assert admitted is not None and admitted.end <= sim.ustart + 900
    # Four rolls, at most one linear scan each, regardless of shortened visits.
    assert len(checks) < 4 * 900 / sim.acs.momentum_budget.step


def test_idle_escape_tries_destination_solar_roll_first(scenario, monkeypatch):
    sim, _ = scenario
    monkeypatch.setattr(
        sim.acs, "_idle_safe_attitude_candidates", lambda time: [(180.0, 10.0)]
    )
    optimum = Mock(return_value=123.0)
    monkeypatch.setattr(
        "conops.simulation.momentum_planning.optimum_body_roll", optimum
    )
    planner = MomentumObservationPlanner(sim)
    recovery = Mock(return_value=object())
    monkeypatch.setattr(planner, "recovery", recovery)
    planner._idle_escape(planner.executor.state, planner.budget.current)
    assert recovery.call_args.args[2] == (180.0, 10.0, 123.0)
    assert optimum.call_args.args[:3] == (180.0, 10.0, sim.ustart)
    assert optimum.call_args.kwargs["drive_state"] is sim.acs.solar_array_drive_state


def test_pass_conflict_allows_safe_idle_wait_with_torque_enabled(scenario, monkeypatch):
    sim, _ = scenario
    ephem = DeterministicEphemeris(
        SCENARIO_BEGIN, SCENARIO_BEGIN + timedelta(hours=1), 2
    )
    sim.ephem = sim.acs.ephem = sim.config.constraint.ephem = ephem
    sim.config.spacecraft_bus.attitude_control.stored_momentum.gravity_gradient_enabled = True
    executor = sim.acs._executor
    executor.momentum = MomentumBudget(sim.config, ephem, executor.state)
    cutoff = sim.ustart + 3000
    monkeypatch.setattr(
        sim,
        "_attitude_constraint_name_for_attitude",
        lambda ra, dec, roll, time, mode, **kwargs: (
            ("keepout", "hardware") if time >= cutoff else None
        ),
    )
    monkeypatch.setattr(
        sim._idle_safety, "first_violation", lambda attitude, start: cutoff
    )
    monkeypatch.setattr(
        sim.acs, "_idle_safe_attitude_candidates", lambda time: [(180, 0)]
    )
    monkeypatch.setattr(sim.acs, "_idle_safe_roll_candidates", lambda roll: [0])
    pending_pass = Mock()
    pending_pass.tracking_profile_slew_deadlines.return_value = [
        (None, sim.ustart + 30)
    ]
    monkeypatch.setattr(sim.acs.passrequests, "next_pass", lambda time: pending_pass)
    assert MomentumObservationPlanner(sim).idle_recovery(sim.ustart) is None
    assert not sim.acs.fault_management.events


def test_failed_momentum_initialization_returns_diagnostic_failure(scenario):
    sim, _ = scenario
    sim.ppt = None
    sim.acs._executor = None
    sim.config.spacecraft_bus.attitude_control.stored_momentum.initial_momentum_body_n_m_s = (
        100,
        0,
        0,
    )
    assert sim.calc() is False
    assert any(
        event.name == "attitude_execution"
        and "Actuator momentum exceeds" in event.cause
        for event in sim.fault_management.events
    )


def test_slow_slew_limits_are_cached_and_follow_config_mutation(scenario):
    sim, _ = scenario
    slew = Slew(config=sim.config, rate_fraction=0.5)
    first = slew.motion_limits
    assert slew.motion_limits is first
    config = sim.config.spacecraft_bus.attitude_control
    config.max_slew_rate = 1
    config.max_slew_rate_body = (1, 2, 3)
    config.slew_acceleration_body = (4, 5, 6)
    config.settle_time = 17
    second = slew.motion_limits
    assert second is not first and slew.motion_limits is second
    assert second.max_slew_rate == 0.5
    assert second.max_slew_rate_body == (0.5, 1, 1.5)
    assert second.slew_acceleration_body == (4, 5, 6)
    assert second.settle_time == 17
    slew.rate_fraction = 0.25
    assert slew.motion_limits.max_slew_rate == 0.25
    slew.rate_fraction = 1
    assert slew.motion_limits is config


def test_braking_deferral_readmits_and_updates_executed_command(scenario):
    from conops.common import ACSCommandType
    from conops.simulation.acs_command import ACSCommand
    from conops.simulation.attitude import AttitudeTrajectory

    sim, target = scenario
    admitted = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    assert admitted is not None
    executor = sim.acs._executor
    limits = sim.config.spacecraft_bus.attitude_control.model_copy(
        update={"max_slew_rate": 0.5}
    )
    intervening = AttitudeTrajectory.turn_from_state(executor.state, (20, 0, 0), limits)
    executor.install(intervening)
    command = ACSCommand(
        command_type=ACSCommandType.SLEW_TO_TARGET,
        execution_time=sim.ustart + 5,
        slew=admitted.slew,
    )
    sim.acs.enqueue_command(command)
    sim.acs._process_commands(sim.ustart + 5)
    assert command.execution_time > sim.ustart + 5
    assert not sim.acs.executed_commands
    sim.acs._process_commands(sim.ustart + 60)
    assert sim.acs.executed_commands[-1] is command
    assert command.slew is sim.acs.current_slew
    assert command.slew is not admitted.slew
    assert command.slew.slewstart == sim.ustart + 60
    assert command.slew.startra > 0
    assert target.begin == command.slew.slewstart
    assert sim._momentum_recovery.slewstart == target.end


def test_delayed_target_that_misses_deadline_is_cancelled(scenario):
    sim, target = scenario
    target.deadline = sim.ustart + 200
    admitted = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    assert admitted is not None
    target.end = admitted.end
    sim._apply_slew_metadata(target, admitted.slew)
    sim._track_ppt_in_timeline()
    assert len(sim.plan) == 1
    assert sim.acs._start_slew(admitted.slew, sim.ustart + 60) is None
    assert sim.ppt is None
    assert not sim.plan.entries
    assert sim.acs.current_slew is None
    assert sim._momentum_recovery is None
    assert not sim.acs.fault_management.events


def test_runtime_scoped_checks_keep_the_admitted_target(scenario, monkeypatch):
    sim, target = scenario
    admitted = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    sim.acs._start_slew(admitted.slew, sim.ustart)
    monkeypatch.setattr(type(target), "uses_mounted_attitude", lambda self: True)
    checked = Mock(return_value=["late telescope keepout"])
    monkeypatch.setattr(type(target), "attitude_constraint_names", checked)
    sim.acs.pointing(sim.ustart + 240)
    assert checked.call_count == 1
    assert checked.call_args.args[-1] == ACSMode.SCIENCE
    assert sim.acs.last_slew.at is target


def test_interrupted_saa_wait_has_empty_collection_window(scenario):
    sim, target = scenario
    sim.config.payload.observation_timing.setup_seconds = 60
    sim.acs.saa = Mock(insaa=lambda time: time < sim.ustart + 480)
    admitted = MomentumObservationPlanner(sim).observation(target, sim.ustart)
    target.end = sim.ustart + 300
    sim._apply_slew_metadata(target, admitted.slew)
    assert target.collection_begin <= target.end
    assert target.collection_end == target.collection_begin


def test_execution_momentum_failure_ends_run_with_retained_diagnostics(
    scenario, monkeypatch
):
    sim, _ = scenario
    sim.ppt = None
    record = sim._record_pointing_data

    def tighten_capacity(*args):
        record(*args)
        sim.acs.momentum_budget.capacity = 0.01

    monkeypatch.setattr(sim, "_record_pointing_data", tighten_capacity)
    assert sim.calc() is False
    assert len(sim.telemetry.housekeeping) == 1
    assert sim.acs._executor.state.utime == sim.ustart
    assert any(
        event.name == "attitude_execution" for event in sim.fault_management.events
    )


def test_plan_replay_reports_invalid_initial_momentum(scenario):
    from conops import DITL
    from conops.targets import Plan

    sim, target = scenario
    sim.config.spacecraft_bus.attitude_control.stored_momentum.initial_momentum_body_n_m_s = (
        100,
        0,
        0,
    )
    target.begin, target.end = sim.ustart, sim.ustart + 600
    plan = Plan(entries=[target])
    replay = DITL(config=sim.config, ephem=sim.ephem, plan=plan)
    replay.acs.ra, replay.acs.dec, replay.acs.roll = 0, 0, 0
    assert replay.calc() is False
    assert any(
        event.name == "attitude_execution" for event in replay.fault_management.events
    )


def test_expired_science_deadline_rejects_before_scanning_a_negative_hold(
    scenario, monkeypatch
):
    sim, target = scenario
    monkeypatch.setattr(
        sim, "_next_science_deadline", lambda *args, **kwargs: (sim.ustart - 1, "pass")
    )
    assert MomentumObservationPlanner(sim).observation(target, sim.ustart) is None


def test_run_with_intervening_motion_keeps_plan_and_execution_consistent(
    scenario, monkeypatch
):
    sim, target = scenario
    sim.ppt = None
    target.exptime = 300
    visible = sim.ustart + 120
    monkeypatch.setattr(type(target), "next_vis", lambda self, now: max(now, visible))
    intervening = Slew(
        config=sim.config, endra=180, rate_fraction=0.5, obstype=ObsType.IDLE
    )
    pointing = sim.acs.pointing
    started = False

    def with_intervening_guidance(time):
        nonlocal started
        if time == sim.ustart + 60 and not started:
            started = True
            sim.acs._start_slew(intervening, time)
        return pointing(time)

    monkeypatch.setattr(sim.acs, "pointing", with_intervening_guidance)
    assert sim.calc()
    entries = [entry for entry in sim.plan if entry.obsid == target.obsid]
    assert len(entries) == 1
    assert entries[0].begin == sim.ustart + 180
    assert entries[0].exposure == 300
    assert not sim.validate_plan_matches_execution()
