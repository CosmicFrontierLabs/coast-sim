"""Charges must improve the complete battery forecast, including both turns."""

from datetime import timedelta

import pytest

from conops.common import ACSMode, ObsType
from conops.common.vector import angular_separation
from conops.config import Battery, Constraint, PowerDraw, SolarPanel, SolarPanelSet
from conops.schedulers import PriorityScheduler
from conops.schedulers.priority_scheduler import StartState

from .planning_scenario import BEGIN, HOUR, MIN, T0, make_config, make_target
from .planning_scenario import planner as _planner


def builder(monkeypatch, *, capacity=1000, deficit=0, slew_watts=180):
    config = make_config(4, open_sky=True)
    config.battery = Battery(
        watthour=capacity, recharge_threshold=0.7, max_depth_of_discharge=0.3
    )
    config.spacecraft_bus.power_draw = PowerDraw(
        nominal_power=90,
        power_mode={int(ACSMode.CHARGING): 80, int(ACSMode.SLEWING): slew_watts},
    )
    config.solar_panel = SolarPanelSet(
        panels=[SolarPanel(normal=(1, 0, 0), max_power=600)], conversion_efficiency=1
    )
    ephem = config.constraint.ephem
    sun = (float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0]))
    # A controlled sunlit span lets us distinguish generation at the charge
    # attitude from generation while left at the dark departure attitude.
    monkeypatch.setattr(Constraint, "in_eclipse", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        SolarPanelSet,
        "power",
        lambda self, t, ra, dec, ephem, **kw: (
            600 if angular_separation(ra, dec, *sun) < 30 else 0
        ),
    )
    scheduler = PriorityScheduler(
        config,
        [],
        BEGIN,
        BEGIN + timedelta(hours=4),
        include_passes=False,
        start_deficit_wh=deficit,
        start_state=StartState(time=T0, attitude=(105, 10, 0)),
    )
    scheduler._reserve_fixed()
    return scheduler


def test_necessary_charge_is_shortened_without_assuming_energy_above_full(monkeypatch):
    scheduler = builder(monkeypatch, deficit=100)
    before = scheduler._all_charge_timeline([])
    planned = scheduler._charged_timeline([])
    assert planned
    assert sum(b.end - b.ready for b in planned) < sum(b.end - b.ready for b in before)
    assert scheduler._within_level(scheduler._timeline_energy(planned))
    assert all(b.end - b.ready >= 5 * MIN for b in planned)


@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
def test_real_orbit_planners_skip_charging_when_the_full_forecast_fits(kind):
    config = make_config(4)
    # No panels: even complete idle and observation draws fit comfortably.
    # Set panels so the former opportunistic policy can place a charge too.
    config.battery = Battery(watthour=100000)
    config.solar_panel = SolarPanelSet(
        panels=[SolarPanel(normal=(1, 0, 0), max_power=600)]
    )
    target = make_target(config, 100, 105, 10, minutes=20)
    scheduler, options = _planner(kind)
    planner = scheduler(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(hours=4),
        include_passes=False,
        **options,
    )
    plan = planner.schedule()
    assert any(e.obstype == ObsType.AT for e in plan)
    assert not any(e.obstype == ObsType.CHARGE for e in plan)
    assert planner._within_level(planner._timeline_energy(planner.timeline))


@pytest.mark.parametrize(
    "capacity,deficit,slew_watts,charges",
    [
        (12000, 0, 180, False),  # full battery and the horizon fits
        (1000, 0, 180, True),  # full battery but a later deficit
        (1000, 0, 50000, False),  # the turns cost more than the charge gains
        (1000, 250, 180, False),  # passive generation at the charge pose recovers
    ],
)
def test_a_charge_is_planned_only_when_it_improves_the_forecast(
    monkeypatch, capacity, deficit, slew_watts, charges
):
    scheduler = builder(
        monkeypatch, capacity=capacity, deficit=deficit, slew_watts=slew_watts
    )
    if deficit:
        scheduler._origin.attitude_out = scheduler._battery.charging_attitude(T0)
    candidates = scheduler._all_charge_timeline([])
    planned = scheduler._charged_timeline([])
    assert any(b.is_charge for b in planned) == charges
    if slew_watts > 180:
        assert candidates
        assert scheduler._charge_excess(candidates) > scheduler._charge_excess([])
    else:
        assert scheduler._within_level(scheduler._timeline_energy([])) != charges
        assert scheduler._within_level(scheduler._timeline_energy(planned))


@pytest.mark.parametrize("reserved", ["visit", "charge"])
def test_charge_planning_does_not_rewrite_a_reserved_block(monkeypatch, reserved):
    from conops.schedulers.priority_scheduler import _Block
    from conops.targets import PlanEntry

    if reserved == "charge":
        scheduler = builder(monkeypatch, capacity=12000)
        [block] = scheduler._all_charge_timeline([])
    else:
        scheduler = builder(monkeypatch)
        pose = (105.0, 10.0, 0.0)
        start = T0 + 2 * HOUR
        slew = scheduler.ctx.slew(pose, pose, start, obsid=100)
        step = PlanEntry(
            config=scheduler.config,
            obsid=100,
            obstype=ObsType.AT,
            ra=pose[0],
            dec=pose[1],
            roll=pose[2],
            begin=start,
            end=start + 20 * MIN,
            collection_begin=start,
            collection_end=start + 20 * MIN,
        )
        block = _Block(
            entry=step,
            steps=[step],
            step_slews=[slew],
            slew=slew,
            attitude_in=pose,
            attitude_out=pose,
            ready=start,
            end=float(step.end),
        )
        assert scheduler._all_charge_timeline([block])  # speculative charges tried
    before, step_slews = block.entry.model_dump_json(), list(block.step_slews)
    planned = scheduler._charged_timeline([block])
    assert block.entry.model_dump_json() == before
    assert block.step_slews == step_slews
    if reserved == "charge":  # kept although not needed
        assert [b.entry.obsid for b in planned if b.is_charge] == [block.entry.obsid]


@pytest.mark.parametrize("desaturation", [False, True], ids=["idle", "desat"])
def test_charge_trials_reconnect_an_activity_in_its_hold_mode(
    monkeypatch,
    desaturation,
):
    from conops.targets import ScheduledActivity

    configured = builder(monkeypatch)
    planner = PriorityScheduler(
        configured.config,
        [],
        BEGIN,
        BEGIN + timedelta(hours=4),
        include_passes=False,
        start_deficit_wh=100,
        start_state=StartState(time=T0, attitude=(105, 10, 0)),
        activities=[
            ScheduledActivity(
                name="Calibration",
                begin=T0 + 2 * HOUR,
                duration_seconds=30 * MIN,
                desaturation=desaturation,
            )
        ],
    )
    planner._reserve_fixed()
    activity = next(b for b in planner.timeline if b.is_activity)
    before = activity.entry.model_dump_json()
    candidates = planner._all_charge_timeline(planner.timeline)
    index = next(
        i
        for i, block in enumerate(candidates[:-1])
        if block.is_charge and candidates[i + 1].is_activity
    )
    hold_modes = []
    original_hold_check = planner.ctx.first_hold_violation

    def hold_check(attitude, begin, end, mode, *args, **kwargs):
        hold_modes.append(mode)
        return original_hold_check(attitude, begin, end, mode, *args, **kwargs)

    monkeypatch.setattr(planner.ctx, "first_hold_violation", hold_check)
    # The later charge starts as the activity ends, leaving no time to turn
    # from a changed hold attitude. Removing the ingress charge must refuse
    # that complete chain; the isolated activity can safely inherit the origin.
    assert planner._replace_charge(candidates, index, None) is None
    trial = planner._replace_charge(candidates[: index + 2], index, None)
    assert trial is not None
    assert hold_modes[-1] == (ACSMode.DESAT if desaturation else ACSMode.IDLE)
    reconnected = next(b for b in trial if b.is_activity)
    assert reconnected.ready == T0 + 2 * HOUR
    assert reconnected.end - reconnected.ready == 30 * MIN
    assert activity.entry.model_dump_json() == before
    planned = planner._charged_timeline(planner.timeline)
    assert planner._within_level(planner._timeline_energy(planned))
    assert activity.entry.model_dump_json() == before
