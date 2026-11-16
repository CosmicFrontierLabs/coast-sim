"""Emergency recharge preserves truthful scheduled-hold execution records."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pytest

from conops import QueueDITL
from conops.common import ACSMode, ObsType
from conops.config import (
    ActivityAttitude,
    Battery,
    Instrument,
    PowerDraw,
    SolarPanel,
    SolarPanelSet,
)
from conops.config.instrument import InstrumentMode
from conops.targets import Plan, ScheduledActivity

from .planning_scenario import BEGIN, HOUR, MIN, T0, make_config


def configuration(hours=9):
    config = make_config(hours)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.constraint.eclipse_intervals = [(T0, T0 + 6 * HOUR)]
    config.battery = Battery(
        watthour=12000, max_depth_of_discharge=0.4, recharge_threshold=0.95
    )
    config.spacecraft_bus.power_draw = PowerDraw(
        nominal_power=200, eclipse_power=600, power_mode={int(ACSMode.CHARGING): 100}
    )
    config.payload.instruments = [
        Instrument(
            name="camera",
            modes=[
                InstrumentMode(name="IDLE", power_w=10),
                InstrumentMode(name="SCIENCE", power_w=100),
                InstrumentMode(name="standby", power_w=10),
                InstrumentMode(name="calibration", power_w=400),
                InstrumentMode(name="CHARGING", power_w=10),
            ],
        )
    ]
    config.solar_panel = SolarPanelSet(
        panels=[SolarPanel(normal=(1, 0, 0), max_power=8000)], conversion_efficiency=1
    )
    return config


def run_case(
    *,
    critical=False,
    kind="idle",
    explicit=False,
    hours=9,
    critical_offset=20 * MIN,
    critical_margin_wh=100,
    calibration_offset=0,
):
    config = configuration(hours)
    blocks = [
        ScheduledActivity(
            name="Eclipse standby",
            begin=T0,
            duration_seconds=6 * HOUR,
            fixed=True,
            instrument_modes={"camera": "standby"},
        ),
        ScheduledActivity(
            name="Calibration",
            begin=T0 + 6 * HOUR + calibration_offset,
            duration_seconds=HOUR,
            fixed=True,
            instrument_modes={"camera": "calibration"},
            data_gb=0.25,
            safe_hold=kind == "safe",
            desaturation=kind == "dump",
            attitude=ActivityAttitude(kind="fixed", ra_deg=105, dec_deg=10, roll_deg=0)
            if explicit
            else ActivityAttitude(),
        ),
    ]

    class PerturbedQueue(QueueDITL):
        def _allocate_if_due(self, utime):
            super()._allocate_if_due(utime)
            if critical and utime == T0 + 6 * HOUR + critical_offset:
                self.battery.charge_level = (
                    self.battery.minimum_charge_level * self.battery.watthour
                    + critical_margin_wh
                )

    run = PerturbedQueue(
        config=config,
        begin=BEGIN,
        end=BEGIN + timedelta(hours=hours),
        activities=blocks,
    )
    pose = (105, 10, 0)
    run.acs.ra, run.acs.dec, run.acs.roll = pose
    run.acs._hold_idle_attitude(*pose, T0)
    run._add_activity_data = Mock(wraps=run._add_activity_data)
    run._unload_stored_momentum = Mock(wraps=run._unload_stored_momentum)
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
    return run


@pytest.mark.parametrize("kind", ["idle", "safe"])
def test_six_hour_eclipse_block_finishes_before_emergency_recharge(kind):
    run = run_case(kind=kind)
    [entry] = [e for e in run.plan if e.name == "Calibration"]
    assert entry.activity_hold_begin == T0 + 6 * HOUR
    assert float(entry.end) == T0 + 7 * HOUR
    assert entry.activity_interruption is None
    assert not entry.activity_resumed
    charges = [e for e in run.plan if e.obstype == ObsType.CHARGE]
    assert charges and min(float(e.begin) for e in charges) >= float(entry.end)
    assert any("charging waits for Calibration" in e.description for e in run.log)
    expected = ACSMode.SAFE if kind == "safe" else ACSMode.IDLE
    samples = [
        h
        for h in run.telemetry.housekeeping
        if entry.activity_hold_begin <= h.timestamp.timestamp() < float(entry.end)
    ]
    assert len(samples) == 60
    assert all(int(h.acs_mode) == int(expected) for h in samples)
    assert all(h.instrument_modes == {"camera": "calibration"} for h in samples)
    assert min(run.batterylevel) >= run.battery.minimum_charge_level


@pytest.mark.parametrize("kind", ["idle", "safe", "dump"])
@pytest.mark.parametrize("explicit", [False, True])
def test_critical_deficit_interrupts_and_retains_unexecuted_work(
    kind, explicit, tmp_path
):
    run = run_case(critical=True, kind=kind, explicit=explicit)
    entries = [e for e in run.plan if e.name == "Calibration"]
    assert len(entries) == 2
    first, resumed = entries
    interruption = first.activity_interruption
    assert interruption is not None
    assert interruption.time == T0 + 6 * HOUR + 20 * MIN
    assert not first.activity_finishes
    assert resumed.activity_resumed and resumed.activity_finishes
    assert first.activity_partial and resumed.activity_partial
    expected_remaining = HOUR if kind == "dump" else 40 * MIN
    assert interruption.remaining_seconds == expected_remaining
    assert interruption.disposition == ("restart" if kind == "dump" else "resume")
    assert float(resumed.end) - resumed.activity_hold_begin == expected_remaining
    assert first.fixed_data_gb == 0.25 and resumed.fixed_data_gb == 0
    assert [
        c.args[0] for c in run._add_activity_data.call_args_list if c.args[0] > 0
    ] == [0.25]
    assert run._paused_activity is None
    assert min(run.batterylevel) >= run.battery.minimum_charge_level
    assert run._unload_stored_momentum.call_count == (1 if kind == "dump" else 0)
    charges = [e for e in run.plan if e.obstype == ObsType.CHARGE]
    assert charges and any(
        float(first.end) <= float(c.begin) < float(c.end) <= float(resumed.begin)
        for c in charges
    )
    restored = Plan.load(run.plan.save(tmp_path / "execution.json"))
    restored_parts = [e for e in restored if e.name == "Calibration"]
    assert restored_parts[0].activity_interruption == interruption
    assert restored_parts[1].activity_resumed


@pytest.mark.parametrize("existing_charge", [False, True])
def test_a_due_block_waits_for_critical_recharge_without_zero_length_execution(
    existing_charge,
):
    run = run_case(
        critical=True,
        critical_offset=0,
        calibration_offset=10 * MIN if existing_charge else 0,
    )
    [entry] = [e for e in run.plan if e.name == "Calibration"]
    assert float(entry.end) - entry.activity_hold_begin == HOUR
    assert entry.activity_interruption is None and not entry.activity_resumed
    assert any(
        "Calibration deferred until emergency charging ends" in e.description
        for e in run.log
    )
    charges = [e for e in run.plan if e.obstype == ObsType.CHARGE]
    assert charges and all(float(c.end) <= float(entry.begin) for c in charges)
    assert [
        c.args[0] for c in run._add_activity_data.call_args_list if c.args[0] > 0
    ] == [0.25]
    assert all(float(e.end) > float(e.begin) for e in run.plan)


def test_unfinished_recharge_retains_the_remaining_work_at_the_run_boundary():
    run = run_case(critical=True, hours=6.5, critical_margin_wh=20)
    [entry] = [e for e in run.plan if e.name == "Calibration"]
    assert not entry.activity_finishes
    assert entry.activity_interruption.remaining_seconds == 40 * MIN
    assert run._paused_activity.remaining_seconds == 40 * MIN
    assert all(float(e.end) <= T0 + 6.5 * HOUR for e in run.plan)


def test_interrupted_execution_plan_replays_with_actual_modes_and_data(tmp_path):
    from conops import DITL

    run = run_case(critical=True, explicit=True)
    plan = Plan.load(run.plan.save(tmp_path / "execution.json"))
    replay = DITL(
        config=run.config, plan=plan, begin=BEGIN, end=BEGIN + timedelta(hours=9)
    )
    replay.acs.ra, replay.acs.dec, replay.acs.roll = (105, 10, 0)
    replay.acs._hold_idle_attitude(105, 10, 0, T0)
    assert replay.calc()
    assert replay.validate_plan_matches_execution() == []


def test_finishing_forecast_prices_named_load_and_the_charge_ingress(monkeypatch):
    from conops.schedulers import SchedulingContext
    from conops.simulation.activity_charging import can_finish_before_recharge

    config = configuration()
    evaluate = SolarPanelSet.evaluate_executed_attitude

    def shadowed_during_slew(self, *args, **kwargs):
        illumination, generation, state = evaluate(self, *args, **kwargs)
        return (
            (illumination, 0.0, state)
            if kwargs.get("acs_mode") == ACSMode.SLEWING
            else (illumination, generation, state)
        )

    monkeypatch.setattr(
        SolarPanelSet, "evaluate_executed_attitude", shadowed_during_slew
    )
    context = SchedulingContext(config, BEGIN, BEGIN + timedelta(hours=9))
    state = config.solar_panel.initial_drive_state()
    floor = config.battery.minimum_charge_level * config.battery.watthour
    kwargs = dict(
        now=T0 + 6 * HOUR,
        hold_begin=T0 + 6 * HOUR,
        hold_end=T0 + 6 * HOUR + 20 * MIN,
        attitude=(105, 10, 0),
        ingress=None,
        mode=ACSMode.IDLE,
        drive_state=state,
    )
    # 200 Wh covers the hold, but five Wh more cannot also cover the turn.
    assert not can_finish_before_recharge(
        config,
        context,
        instrument_modes={"camera": "calibration"},
        charge_wh=floor + 205,
        **kwargs,
    )
    assert can_finish_before_recharge(
        config,
        context,
        instrument_modes={"camera": "calibration"},
        charge_wh=floor + 250,
        **kwargs,
    )
    assert not can_finish_before_recharge(
        config,
        context,
        instrument_modes={"camera": "calibration"},
        charge_wh=floor + 90,
        **kwargs,
    )
    assert can_finish_before_recharge(
        config,
        context,
        instrument_modes={"camera": "standby"},
        charge_wh=floor + 90,
        **kwargs,
    )
    assert state == config.solar_panel.initial_drive_state()


def test_finish_forecast_advances_finite_drives_without_changing_live_state():
    from conops.config import SingleAxisSolarArrayDrive, SolarArrayDriveControl
    from conops.schedulers import SchedulingContext
    from conops.simulation.activity_charging import can_finish_before_recharge

    config = configuration()
    context = SchedulingContext(config, BEGIN, BEGIN + timedelta(hours=9))
    floor = config.battery.minimum_charge_level * config.battery.watthour
    kwargs = dict(
        now=T0 + 6 * HOUR,
        hold_begin=T0 + 6 * HOUR,
        hold_end=T0 + 6 * HOUR + 20 * MIN,
        attitude=(105, 10, 0),
        ingress=None,
        mode=ACSMode.IDLE,
        instrument_modes={"camera": "calibration"},
        charge_wh=floor + 50,
    )
    assert not can_finish_before_recharge(
        config, context, drive_state=config.solar_panel.initial_drive_state(), **kwargs
    )
    panel = config.solar_panel.panels[0]
    panel.single_axis_drive = SingleAxisSolarArrayDrive(
        rotation_axis=(0, 0, 1),
        min_angle_deg=-180,
        max_angle_deg=180,
        max_rate_deg_per_s=0.5,
    )
    panel.drive_control = SolarArrayDriveControl(
        sun_tracking_modes=[ACSMode.IDLE, ACSMode.CHARGING]
    )
    state = config.solar_panel.initial_drive_state()
    assert can_finish_before_recharge(config, context, drive_state=state, **kwargs)
    assert state == config.solar_panel.initial_drive_state()


def test_a_deferred_block_waits_for_handoff_without_losing_its_override():
    original = QueueDITL._activity_waits_for
    waited = False

    def wait_once(run, activity, utime):
        nonlocal waited
        if (
            activity.name == "Calibration"
            and activity.begin > T0 + 6 * HOUR
            and run.charging_ppt is None
            and not run.battery.battery_alert
            and not waited
        ):
            waited = True
            return "the handoff under way"
        return original(run, activity, utime)

    with patch.object(QueueDITL, "_activity_waits_for", wait_once):
        run = run_case(critical=True, critical_offset=0)
    assert waited
    [entry] = [e for e in run.plan if e.name == "Calibration"]
    assert entry.activity_finishes and not entry.activity_partial
    assert float(entry.end) - float(entry.begin) == HOUR
    assert float(entry.end) - entry.activity_hold_begin == HOUR
    assert not run._charge_deferred_activities
    assert not run._activities_left
    assert [
        c.args[0] for c in run._add_activity_data.call_args_list if c.args[0] > 0
    ] == [0.25]
