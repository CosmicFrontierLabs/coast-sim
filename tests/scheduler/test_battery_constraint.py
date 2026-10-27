"""Plans keep the battery above its planning level, charging when they must.

The ledger's segment arithmetic is tested in test_consumable.py; here it is
the battery's mapping onto it, with additions ordered in time, and the
schedulers' use of it.
"""

from datetime import datetime, timedelta

import pytest

from conops.common import ObsType
from conops.common.enums import ACSMode
from conops.config import (
    Battery,
    MissionConfig,
    PowerDraw,
    SingleAxisSolarArrayDrive,
    SolarArrayDriveControl,
    SolarArrayDriveState,
    SolarPanel,
    SolarPanelSet,
)
from conops.ditl import DITL, RollingHorizonDITL
from conops.ditl.plan_validator import entry_obstype
from conops.schedulers import (
    BatteryLedger,
    LocalSearchScheduler,
    PriorityScheduler,
    SchedulingContext,
)
from conops.schedulers.battery import MIN_CHARGE_SECONDS, Addition
from conops.targets import PlanEntry, Pointing

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target
from .test_priority_scheduler import _science

HOURS = 14
STEP = 60
WATTHOUR = 1000.0
LEVEL = 0.7
"""The default planning level is the depth-of-discharge floor."""
LIMIT_WH = (1.0 - LEVEL) * WATTHOUR
SIDE = (0.0, 1.0, 0.0)
"""A side-mounted panel: rolled towards the Sun, it generates while observing."""
BORESIGHT = (1.0, 0.0, 0.0)
"""A panel on the boresight: it generates only pointing at the Sun."""


def _battery_config(
    normal: tuple[float, float, float] = BORESIGHT,
    *,
    level: float | None | str = "default",
    panels: bool = True,
    stations: bool = False,
    watthour: float = WATTHOUR,
) -> MissionConfig:
    """The scenario with a 1 kWh battery, one 600 W panel and modal draws:
    200 W observing, 230 W slewing, 140 W idle and 130 W charging. The
    battery alert is set at the planning level, so an alert in a run means
    the plan let the battery below it."""
    config = _config(HOURS, stations=stations)
    config.solar_panel = SolarPanelSet(
        panels=[SolarPanel(name="panel", normal=normal, max_power=600.0)]
        if panels
        else []
    )
    config.battery = Battery(
        watthour=watthour,
        max_depth_of_discharge=1.0 - LEVEL,
        recharge_threshold=LEVEL,
    )
    if level != "default":
        config.battery.planning_level = level  # type: ignore[assignment]
    config.spacecraft_bus.power_draw = PowerDraw(
        nominal_power=100.0, power_mode={0: 150.0, 1: 180.0, 4: 80.0, 6: 90.0}
    )
    return config


def _targets(config: MissionConfig, minutes: float = 200) -> list[Pointing]:
    return [
        _target(config, 100 + i, ra, dec, merit=90 - i, minutes=minutes, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH)
    ]


def _ledger(config: MissionConfig, start_deficit: float = 0.0) -> BatteryLedger:
    return BatteryLedger(
        config, T0, T0 + HOURS * HOUR, start_deficit_wh=start_deficit, step_size=STEP
    )


def _empty(ledger: BatteryLedger) -> tuple[tuple[Addition, ...], ...]:
    return tuple(() for _ in range(ledger.segments))


def _with_hold(
    ledger: BatteryLedger,
    planned: tuple[tuple[Addition, ...], ...],
    begin: float,
    end: float,
) -> tuple[tuple[Addition, ...], ...]:
    """``planned`` with the spacecraft held at each sunlit run's charging
    attitude through ``[begin, end)``, as a plan with charges would have it."""
    for stretch in ledger.stretches:
        if stretch.attitude is None:
            continue
        hold = ledger.hold_addition(stretch.attitude, ACSMode.IDLE, begin, end)
        if hold is not None:
            return ledger.add(planned, ledger.segment(begin), hold)
    return planned


def _science_entry(
    config: MissionConfig, begin: float, minutes: float, ra: float = 105.0
) -> PlanEntry:
    """A snapshot as the scheduler would place it: at the power-optimal roll."""
    end = begin + minutes * MIN
    target = Pointing(config=config, ra=ra, dec=10.0, obsid=1)
    ctx = SchedulingContext(
        config, BEGIN, BEGIN + timedelta(hours=HOURS), step_size=STEP
    )
    roll = ctx.instrument_roll(target, begin)
    return PlanEntry(
        config=config,
        ra=ra,
        dec=10.0,
        roll=roll,
        begin=begin,
        slewtime=120,
        end=end,
        obsid=1,
        obstype=ObsType.AT,
        collection_begin=begin + 150,
        collection_end=end - 30,
    )


def _execute(config: MissionConfig, plan) -> DITL:
    ditl = DITL(
        config=config, plan=plan, begin=BEGIN, end=BEGIN + timedelta(hours=HOURS)
    )
    ditl.step_size = STEP
    ditl.calc()
    return ditl


def _charges(plan) -> list[PlanEntry]:
    return [e for e in plan if entry_obstype(e) == ObsType.CHARGE]


def _observations(plan) -> list[PlanEntry]:
    return [e for e in plan if entry_obstype(e) == ObsType.AT]


def _timing(plan) -> list[tuple[int, float, float]]:
    return [(int(e.obsid), float(e.begin), float(e.end)) for e in plan]


class TestPlanningLevel:
    def test_defaults_to_the_depth_of_discharge_floor(self) -> None:
        battery = Battery(watthour=500.0, max_depth_of_discharge=0.4)
        assert battery.planning_level == pytest.approx(0.6)
        assert battery.planning_deficit_wh == pytest.approx(200.0)
        loaded = Battery.model_validate_json(battery.model_dump_json())
        assert loaded.planning_level == pytest.approx(0.6)

    def test_can_be_set_or_disabled(self) -> None:
        assert Battery(watthour=500.0, planning_level=0.9).planning_deficit_wh == (
            pytest.approx(50.0)
        )
        assert Battery(planning_level=None).planning_deficit_wh is None

    def test_disabled_survives_a_save_and_load(self, tmp_path) -> None:
        """None is a setting, not an absence: a save that leaves out None
        fields keeps it, so a load does not restore the derived default."""
        config = _battery_config(level=None)
        path = str(tmp_path / "config.yaml")
        config.to_yaml_file(path)
        assert MissionConfig.from_yaml_file(path).battery.planning_level is None
        with open(path) as f:
            assert "planning_level: null" in f.read()

    def test_recharge_level_defaults_to_full_and_is_above_the_level(self) -> None:
        assert Battery().planning_recharge_level == 1.0
        assert Battery(watthour=500.0).planning_recharge_deficit_wh == 0.0
        with pytest.raises(ValueError, match="planning_recharge_level"):
            Battery(planning_level=0.8, planning_recharge_level=0.7)


class TestLedger:
    def test_segments_start_at_each_sunrise(self) -> None:
        ledger = _ledger(_battery_config())
        runs = ledger.sunlit_runs(T0, T0 + HOURS * HOUR)
        assert len(runs) > 5
        assert ledger.bounds[1:-1] == [b for b, _ in runs]
        assert ledger.limit == pytest.approx(LIMIT_WH)

    def test_idle_time_with_no_attitude_only_draws(self) -> None:
        """A plan read on its own: idle time draws the idle power and the
        deficit only rises. The schedulers give idle time the attitude held."""
        ledger = _ledger(_battery_config())
        levels = ledger.levels(_empty(ledger))
        assert levels == sorted(levels)
        hour = ledger.levels(_empty(ledger))[-1] / (HOURS)
        assert hour == pytest.approx(140.0, rel=0.05)

    def test_a_hold_generates_at_its_attitude(self) -> None:
        """Held at the charging attitude, a sunlit run puts energy back; in
        eclipse the same hold draws the idle power."""
        ledger = _ledger(_battery_config())
        stretch = ledger.stretches[1]
        assert stretch.attitude is not None
        sunlit = ledger.hold_addition(
            stretch.attitude, ACSMode.IDLE, stretch.begin, stretch.end
        )
        eclipse = ledger.hold_addition(
            stretch.attitude, ACSMode.IDLE, stretch.end, ledger.stretches[2].begin
        )
        assert sunlit is not None and eclipse is not None
        assert sunlit.amount < -100.0
        assert all(delta < 0 for delta in sunlit.deltas)
        assert eclipse.amount == pytest.approx(
            140.0 * (ledger.stretches[2].begin - stretch.end) / 3600.0
        )

    def test_without_panels_the_sun_is_idle_time_too(self) -> None:
        ledger = _ledger(_battery_config(panels=False))
        assert all(stretch.attitude is None for stretch in ledger.stretches)
        assert ledger.charges_in(T0, T0 + HOURS * HOUR) == []

    def test_a_science_addition_is_the_simulations_arithmetic(self) -> None:
        """Slewing power over the slew, then science power less what the
        panel generates at the attitude, step by step."""
        config = _battery_config(SIDE)
        ledger = _ledger(config)
        begin = ledger.stretches[1].begin
        entry = _science_entry(config, begin, 20)
        addition = ledger.addition(entry)
        assert addition is not None
        assert len(addition.deltas) == 20
        assert addition.amount == pytest.approx(sum(addition.deltas))
        # Slewing at 230 W for two minutes, no generation counted.
        assert addition.deltas[0] == pytest.approx(230.0 / 60.0)
        # Observing at 200 W with a sun-rolled side panel: net generation.
        assert all(delta < 0 for delta in addition.deltas[2:])

    def test_a_contacts_ingress_slew_draws_pass_power(self) -> None:
        """ACS reports the slew into a contact as PASS, and the braking after
        it as SLEWING; the ledger counts each as ACS runs it."""
        config = _battery_config(panels=False)
        config.spacecraft_bus.power_draw.power_mode[int(ACSMode.PASS)] = 400.0
        ledger = _ledger(config)
        begin = ledger.stretches[1].begin
        end = begin + 10 * MIN
        entry = PlanEntry(
            config=config,
            ra=0.0,
            dec=0.0,
            roll=0.0,
            begin=begin,
            slewtime=120,
            end=end,
            obsid=50000,
            obstype=ObsType.GSP,
        )
        addition = ledger.addition(
            entry,
            rest=end + 2 * MIN,
            path=lambda t: (0.0, 0.0, 0.0),
            path_start=(0.0, 0.0, 0.0),
        )
        assert addition is not None
        eclipse = ledger.in_eclipse(begin)
        pass_watts = ledger._watts(ACSMode.PASS, eclipse)
        slew_watts = ledger._watts(ACSMode.SLEWING, eclipse)
        assert pass_watts != slew_watts
        assert len(addition.deltas) == 12
        assert addition.deltas[0] == pytest.approx(pass_watts / 60.0)
        assert addition.deltas[5] == pytest.approx(pass_watts / 60.0)
        assert addition.deltas[-1] == pytest.approx(slew_watts / 60.0)

    def test_charging_is_not_counted_above_the_recharge_level(self) -> None:
        """A plan never assumes the battery above ``planning_recharge_level``:
        held at the charging attitude, the deficit comes down to that floor
        and no further, and a deficit already below it is left where it is."""
        config = _battery_config(BORESIGHT)
        config.battery.planning_recharge_level = 0.8
        floor = 0.2 * WATTHOUR
        full = _ledger(_battery_config(BORESIGHT))
        stretch_end = full.stretches[1].end
        assert (
            full.level_at(_with_hold(full, _empty(full), T0, stretch_end), stretch_end)
            == 0.0
        )
        drained = _ledger(config, start_deficit=floor + 100.0)
        planned = _with_hold(drained, _empty(drained), T0, stretch_end)
        assert drained.level_at(planned, stretch_end) == pytest.approx(floor)
        charged = _ledger(config, start_deficit=50.0)
        planned = _with_hold(charged, _empty(charged), T0, stretch_end)
        at_sunrise = charged.level_at(planned, charged.stretches[1].begin)
        assert 50.0 < at_sunrise < floor, "the eclipse before drew"
        assert charged.level_at(planned, stretch_end) == pytest.approx(at_sunrise)

    def test_a_surplus_before_a_draw_is_lost_as_the_battery_is_full(self) -> None:
        """A block that generates in sunlight and then draws in eclipse ends
        with the eclipse's draw on the deficit, not the net of the two."""
        config = _battery_config(SIDE)
        ledger = _ledger(config)
        stretch = ledger.stretches[1]
        # Starts 10 minutes before the eclipse, runs 30 minutes into it.
        entry = _science_entry(config, stretch.end - 10 * MIN, 40)
        addition = ledger.addition(entry)
        assert addition is not None
        sunlit = [d for d in addition.deltas[2:] if d < 0]
        assert len(sunlit) >= 5, "the sunlit minutes generate"
        # Held at the charging attitude until then, the battery is full.
        planned = _with_hold(ledger, _empty(ledger), T0, addition.begin)
        level_before = ledger.level_at(planned, addition.begin)
        assert level_before < 1.0
        planned = ledger.add(planned, ledger.segment(addition.begin), addition)
        level_after = ledger.level_at(planned, addition.end)
        # The walk ends above the net sum: the surplus beyond what the deficit
        # could absorb went nowhere.
        assert level_after - level_before > addition.amount + 10.0

    def test_fits_refuses_what_takes_the_battery_below_the_level(self) -> None:
        config = _battery_config(BORESIGHT)
        ledger = _ledger(config)
        begin = ledger.stretches[1].begin
        short = ledger.addition(_science_entry(config, begin, 20))
        long = ledger.addition(_science_entry(config, begin, 120))
        assert short is not None and long is not None
        segment = ledger.segment(begin)
        # Held at the charging attitude before and after, as a plan would.
        before = _with_hold(ledger, _empty(ledger), T0, begin)
        end = T0 + HOURS * HOUR
        assert ledger.fits(_with_hold(ledger, before, short.end, end), segment, short)
        assert not ledger.fits(_with_hold(ledger, before, long.end, end), segment, long)

    def test_time_to_fit_waits_for_charging_in_the_gap(self) -> None:
        """A snapshot refused at the start of a sunlit run fits after enough
        of the run has charged the battery."""
        config = _battery_config(BORESIGHT)
        ledger = _ledger(config, start_deficit=LIMIT_WH - 20.0)
        # The horizon starts in sunlight, so the first run starts at once.
        begin = ledger.stretches[0].begin
        assert begin == T0
        addition = ledger.addition(_science_entry(config, begin, 20))
        assert addition is not None
        excess = LIMIT_WH - 20.0 + addition.amount - LIMIT_WH
        assert excess > 0
        later = ledger.time_to_fit(
            _empty(ledger), addition, ledger.stretches[0].end, excess
        )
        assert later is not None
        assert later > begin + MIN_CHARGE_SECONDS

    def test_no_limit_fits_anything(self) -> None:
        config = _battery_config(BORESIGHT, level=None)
        ledger = _ledger(config)
        assert not ledger.active
        addition = ledger.addition(_science_entry(config, T0, 600))
        assert addition is not None
        assert ledger.fits(_empty(ledger), 0, addition)


class TestPriorityScheduler:
    def test_a_side_panel_observes_all_day_without_charging(self) -> None:
        """Rolled towards the Sun, the panel generates more than observing
        draws, so the plan fills the day and has no idle time to charge in."""
        config = _battery_config(SIDE, stations=True)
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        plan = scheduler.schedule()
        assert not _charges(plan)
        assert len(_science(plan)) > 30
        ditl = _execute(config, plan)
        assert min(ditl.batterylevel) >= LEVEL
        assert ditl.validate_plan_matches_execution() == []

    def test_a_boresight_panel_plans_charges_and_stays_above_the_level(self) -> None:
        """Observing draws without generating, so the plan charges in its idle
        sunlit time and refuses the science the battery cannot carry."""
        config = _battery_config(BORESIGHT, stations=True)
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        plan = scheduler.schedule()
        charges = _charges(plan)
        assert charges
        assert all(float(c.end) - float(c.begin) >= MIN_CHARGE_SECONDS for c in charges)
        assert _science(plan)
        assert any("Not enough charge" in e.description for e in scheduler.log.events)
        ditl = _execute(config, plan)
        assert min(ditl.batterylevel) >= LEVEL - 1e-6
        assert not any(ditl.batteryalert)
        assert ditl.validate_plan_matches_execution() == []
        # Charges execute in charging mode at their attitude.
        modes = [ACSMode(int(m)) for m in ditl.mode]
        assert ACSMode.CHARGING in modes

    def test_a_contact_that_draws_more_than_a_slew_executes_above_the_level(
        self,
    ) -> None:
        """The slews into contacts run in PASS mode: with a contact draw well
        above the slewing draw, the plan still executes above the level."""
        config = _battery_config(BORESIGHT, stations=True)
        config.spacecraft_bus.power_draw.power_mode[int(ACSMode.PASS)] = 450.0
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        plan = scheduler.schedule()
        assert _charges(plan) and _science(plan)
        ditl = _execute(config, plan)
        assert min(ditl.batterylevel) >= LEVEL - 1e-6
        assert ditl.validate_plan_matches_execution() == []

    def test_without_the_limit_the_battery_goes_below_the_level(self) -> None:
        config = _battery_config(BORESIGHT, level=None)
        plan = PriorityScheduler(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        ).schedule()
        assert not _charges(plan)
        ditl = _execute(config, plan)
        assert min(ditl.batterylevel) < LEVEL

    def test_without_panels_nothing_is_charged_and_the_plan_is_as_before(self) -> None:
        """A battery the plan cannot run down, and nothing to charge it with."""
        config = _battery_config(panels=False, level=None, watthour=100_000.0)
        before = PriorityScheduler(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        ).schedule()
        config = _battery_config(panels=False, watthour=100_000.0)
        plan = PriorityScheduler(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        ).schedule()
        assert not _charges(plan)
        assert _timing(plan) == _timing(before)

    def test_a_low_battery_at_the_start_is_counted(self) -> None:
        config = _battery_config(BORESIGHT)
        targets = _targets(config)
        full = PriorityScheduler(
            config, targets, BEGIN, BEGIN + timedelta(hours=HOURS)
        ).schedule()
        low = PriorityScheduler(
            config,
            targets,
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            start_deficit_wh=LIMIT_WH - 60.0,
        ).schedule()
        assert float(_observations(low)[0].begin) > float(_observations(full)[0].begin)

    def test_a_snapshot_refused_at_the_start_of_a_run_is_placed_later_in_it(
        self,
    ) -> None:
        """Starting low, the first snapshot waits for the sunlit idle time
        before it to charge, rather than for the next gap."""
        config = _battery_config(BORESIGHT)
        scheduler = PriorityScheduler(
            config,
            _targets(config)[:1],
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            start_deficit_wh=LIMIT_WH - 80.0,
        )
        plan = scheduler.schedule()
        first = float(_observations(plan)[0].begin)
        assert T0 + MIN_CHARGE_SECONDS < first < T0 + 3 * HOUR
        assert _charges(plan)
        assert float(_charges(plan)[0].end) <= first


class TestLocalSearch:
    def test_the_search_keeps_every_plan_within_the_level(self) -> None:
        config = _battery_config(BORESIGHT, stations=True)
        scheduler = LocalSearchScheduler(
            config,
            _targets(config),
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            time_limit=600,
            max_iterations=40,
            seed=1,
        )
        plan = scheduler.schedule()
        assert scheduler.score >= scheduler.initial_score
        assert _charges(plan)
        ditl = _execute(config, plan)
        assert min(ditl.batterylevel) >= LEVEL - 1e-6
        assert ditl.validate_plan_matches_execution() == []


class TestRollingHorizon:
    def test_each_replan_starts_from_the_battery_it_will_have(self) -> None:
        config = _battery_config(BORESIGHT, stations=True)
        ditl = RollingHorizonDITL(
            config,
            _targets(config),
            begin=BEGIN,
            end=BEGIN + timedelta(hours=HOURS),
            horizon=timedelta(hours=6),
            replan_interval=timedelta(hours=3),
            commit_lead_time=timedelta(minutes=10),
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert len(ditl.replans) >= 4
        assert min(ditl.batterylevel) >= LEVEL - 1e-6
        assert _charges(ditl.plan)
        assert ditl.validate_plan_matches_execution() == []


def _driven_config(rate: float = 1.0) -> MissionConfig:
    """The side-panel scenario with contacts and the panel on a single-axis
    drive about the boresight that tracks the Sun during contacts and charges
    and holds its angle in every other mode: a contact leaves the panel
    turned, and the science after it generates at that angle, not the initial
    one. A charge tracks too: a contact can leave the panel turned onto the
    boresight, which the Sun constraint keeps from the Sun, and a charge
    holding that angle barely charges. The drive turns at ``rate`` degrees
    per second."""
    config = _battery_config(SIDE, stations=True)
    config.solar_panel = SolarPanelSet(
        panels=[
            SolarPanel(
                name="panel",
                normal=SIDE,
                max_power=600.0,
                single_axis_drive=SingleAxisSolarArrayDrive(
                    rotation_axis=(0.0, 0.0, 1.0),
                    min_angle_deg=-90.0,
                    max_angle_deg=90.0,
                    max_rate_deg_per_s=rate,
                    initial_angle_deg=0.0,
                ),
                drive_control=SolarArrayDriveControl(
                    sun_tracking_modes=[ACSMode.PASS, ACSMode.CHARGING]
                ),
            )
        ]
    )
    return config


class TestSolarArrayDrive:
    def test_a_hold_is_counted_at_the_angles_it_is_reached_with(self) -> None:
        """Idle time holds the drive, so a hold reached at the starting angle
        is its own deltas; reached with the drive turned, it generates
        differently and leaves the drive where it found it."""
        ledger = _ledger(_driven_config())
        assert ledger.driven
        stretch = ledger.stretches[1]
        assert stretch.attitude is not None
        hold = ledger.hold_addition(
            stretch.attitude, ACSMode.IDLE, stretch.begin, stretch.end
        )
        assert hold is not None
        deltas, left = ledger.driven_deltas(hold, ledger.drive_start)
        assert deltas == pytest.approx(hold.deltas) and left == ledger.drive_start
        turned, left = ledger.driven_deltas(hold, (60.0,))
        assert left == (60.0,)
        assert sum(turned) != pytest.approx(sum(deltas))

    def test_the_plan_executes_as_the_ledger_predicts(self) -> None:
        """Contacts turn the drive towards the Sun and science holds it there,
        so what the science after a contact generates depends on the contact;
        the walk carries the drive along, and the executed battery's lowest
        point is the ledger's."""
        config = _driven_config()
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        plan = scheduler.schedule()
        assert _science(plan)
        predicted = min(scheduler._battery.charge_levels(scheduler._energy))
        ditl = _execute(config, plan)
        assert ditl.acs.solar_array_drive_state.angles_deg[0] != 0.0
        assert min(ditl.batterylevel) == pytest.approx(predicted, abs=0.01)
        assert min(ditl.batterylevel) >= LEVEL - 1e-6
        assert ditl.validate_plan_matches_execution() == []
        assert _charges(plan)

    def test_a_replan_forecasts_committed_contacts_along_their_profiles(
        self,
    ) -> None:
        """A replan starts from where the committed activities leave the
        drive. A committed contact turns it along its tracking profile, so
        the forecast must fly that profile rather than the entry's fixed
        attitude, or it hands the next plan the wrong angle and the plan
        overestimates its generation."""
        config = _driven_config()
        ditl = RollingHorizonDITL(
            config,
            _targets(config),
            begin=BEGIN,
            end=BEGIN + timedelta(hours=HOURS),
            horizon=timedelta(hours=6),
            replan_interval=timedelta(hours=2),
            commit_lead_time=timedelta(minutes=30),
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert len(ditl.replans) > 3
        assert not ditl.acs.in_safe_mode
        assert min(ditl.batterylevel) >= LEVEL - 1e-6
        assert ditl.validate_plan_matches_execution() == []

    def test_a_replan_forecasts_the_slew_into_a_committed_contact(self) -> None:
        """ACS flies the slew into a contact in PASS mode, in which the drive
        tracks the Sun, so the drive turns during the slew as well as during
        the contact. A drive slow enough that the contact does not bring it
        to the Sun anyway shows whether the forecast flies the slew: without
        it, the forecast handed the next plan an angle 12 degrees off."""
        config = _driven_config(rate=0.2)
        forecasts: list[tuple[float, float]] = []

        class Recording(PriorityScheduler):
            def __init__(self, *args: object, **kwargs: object) -> None:
                state = kwargs["start_drive_state"]
                assert isinstance(state, SolarArrayDriveState)
                begin = args[2]
                assert isinstance(begin, datetime)
                forecasts.append((begin.timestamp(), float(state.angles_deg[0])))
                super().__init__(*args, **kwargs)  # type: ignore[arg-type]

        ditl = RollingHorizonDITL(
            config,
            _targets(config),
            begin=BEGIN,
            end=BEGIN + timedelta(hours=HOURS),
            horizon=timedelta(hours=6),
            replan_interval=timedelta(hours=2),
            commit_lead_time=timedelta(minutes=30),
            scheduler=Recording,
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        executed = {
            hk.timestamp.timestamp(): hk.solar_array_drive_angles_deg
            for hk in ditl.telemetry.housekeeping
        }
        turned = 0
        for start, forecast in forecasts[1:]:
            angles = executed.get(start)
            assert angles is not None, "a plan starts on a simulation step"
            assert forecast == pytest.approx(angles[0], abs=0.5)
            turned += angles[0] != 0.0
        assert turned, "no committed contact turned the drive before a replan"

    def test_a_replan_forecasts_the_idle_time_before_it_takes_over(self) -> None:
        """Nothing committed still leaves idle time between a replan and
        the plan it builds taking over, and ACS flies that too: the drive
        tracks the Sun through it when its control lists IDLE, and the
        panels generate as it turns. A forecast that counts idle time as
        the idle draw alone, with the drive frozen, hands the next plan a
        battery 20 Wh low and a drive 36 degrees off."""
        config = _driven_config(rate=0.2)
        config.battery = Battery(
            watthour=500.0, max_depth_of_discharge=1.0 - LEVEL, recharge_threshold=LEVEL
        )
        config.solar_panel.panels[0].drive_control = SolarArrayDriveControl(
            sun_tracking_modes=[ACSMode.PASS, ACSMode.IDLE]
        )
        forecasts: list[tuple[float, float, float]] = []

        class Recording(PriorityScheduler):
            def __init__(self, *args: object, **kwargs: object) -> None:
                state = kwargs["start_drive_state"]
                assert isinstance(state, SolarArrayDriveState)
                deficit = kwargs["start_deficit_wh"]
                assert isinstance(deficit, float)
                begin = args[2]
                assert isinstance(begin, datetime)
                forecasts.append(
                    (begin.timestamp(), deficit, float(state.angles_deg[0]))
                )
                super().__init__(*args, **kwargs)  # type: ignore[arg-type]

        ditl = RollingHorizonDITL(
            config,
            [],
            begin=BEGIN + timedelta(hours=12),
            end=BEGIN + timedelta(hours=13),
            horizon=timedelta(minutes=20),
            replan_interval=timedelta(minutes=10),
            commit_lead_time=timedelta(minutes=5),
            include_passes=False,
            scheduler=Recording,
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        # Housekeeping at a step reports the state after it, so the state a
        # plan takes over is the sample one step before it starts.
        executed = {
            hk.timestamp.timestamp() + STEP: (
                hk.battery_level,
                hk.solar_array_drive_angles_deg,
            )
            for hk in ditl.telemetry.housekeeping
        }
        turned = 0
        for start, deficit, forecast in forecasts[1:]:
            sample = executed.get(start)
            assert sample is not None, "a plan starts on a simulation step"
            level, angles = sample
            assert level is not None and angles is not None
            assert deficit == pytest.approx((1.0 - level) * 500.0, abs=0.01)
            assert forecast == pytest.approx(angles[0], abs=0.5)
            turned += angles[0] != 0.0
        assert turned, "idle time never turned the drive before a replan"


class TestValidator:
    def test_a_charge_executed_elsewhere_is_a_mismatch(self) -> None:
        config = _battery_config(BORESIGHT)
        plan = PriorityScheduler(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        ).schedule()
        charge = _charges(plan)[0]
        ditl = _execute(config, plan)
        assert ditl.validate_plan_matches_execution() == []
        charge.ra = (float(charge.ra) + 40.0) % 360.0
        mismatches = ditl.validate_plan_matches_execution()
        assert mismatches
        assert all("charge" in m.message for m in mismatches)

    def test_idle_in_sunlight_during_a_charge_is_a_mismatch(self) -> None:
        """IDLE is accepted over a charge only in eclipse, where ACS reports no
        charging; the sunlit samples must be in CHARGING mode."""
        config = _battery_config(BORESIGHT)
        plan = PriorityScheduler(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        ).schedule()
        ditl = _execute(config, plan)
        assert ditl.validate_plan_matches_execution() == []
        charging = [
            i
            for i, m in enumerate(ditl.mode)
            if ACSMode(int(m)) is ACSMode.CHARGING
            and not config.constraint.in_eclipse(ra=0.0, dec=0.0, time=ditl.utime[i])
        ]
        assert charging
        for i in charging:
            ditl.mode[i] = int(ACSMode.IDLE)
            ditl.obsid[i] = 0
        mismatches = ditl.validate_plan_matches_execution()
        assert mismatches
        assert all("charge" in m.message for m in mismatches)
