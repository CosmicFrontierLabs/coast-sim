"""Scheduled ground-station passes during science, with consistent recorder
forecasts, allocation budgets and dispatch."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from conops import DITL, QueueDITL, create_ditl
from conops.common import AntennaType, ObsType, whole_steps
from conops.common.enums import ACSMode
from conops.config import (
    AllocationSettings,
    AntennaPointing,
    BandCapability,
    CommunicationsSystem,
    HousekeepingData,
    MissionConfig,
    ObservationTiming,
    OnboardRecorder,
    SchedulerConfig,
    SchedulerMode,
    ScienceData,
)
from conops.config.observation_categories import (
    ObservationCategories,
    ObservationCategory,
)
from conops.ditl import RollingHorizonDITL
from conops.schedulers import (
    CpSatScheduler,
    LocalSearchScheduler,
    LongRangeAllocator,
    PriorityScheduler,
)
from conops.schedulers.priority_scheduler import StartState
from conops.schedulers.recorder import RecorderLedger
from conops.simulation import ContactSpan, PassTimes, StationCoverage
from conops.targets import Plan, Pointing

from .planning_scenario import BEGIN, HOUR, MIN, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target
from .test_priority_scheduler import _execute, _science

HOURS = 14
STEP = 60
CAPACITY = 20.0
LIMIT = 0.9 * CAPACITY
HIGH_RATE = 0.01
"""Gbps: a 20-minute snapshot is 12 Gb, over half the limit."""
LOW_RATE = 0.001
RATE_MBPS = 10.0
"""The default stations' S-band rate, which the spacecraft matches."""


def _comms(antenna_type: AntennaType, **antenna: float) -> CommunicationsSystem:
    return CommunicationsSystem(
        name="S-band",
        band_capabilities=[BandCapability(band="S", downlink_rate_mbps=RATE_MBPS)],
        antenna_pointing=AntennaPointing(antenna_type=antenna_type, **antenna),
    )


def _recorder_config(
    antenna_type: AntennaType | None = AntennaType.OMNI,
    hours: int = HOURS,
    **antenna: float,
) -> MissionConfig:
    """The scenario with stations, a small recorder, some housekeeping and
    the given antenna; a body-fixed antenna without ``antenna_type``."""
    config = _config(hours, stations=True)
    config.recorder = OnboardRecorder(capacity_gb=CAPACITY)
    config.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0.0001)
    if antenna_type is not None:
        config.spacecraft_bus.communications = _comms(antenna_type, **antenna)
    return config


def _data_target(
    config: MissionConfig, obsid: int, ra: float, dec: float, rate: float, **kw: float
) -> Pointing:
    target = _target(config, obsid, ra, dec, **kw)
    target.science_data = {"Default Instrument": ScienceData(rate_gbps=rate)}
    return target


def _targets(config: MissionConfig) -> list[Pointing]:
    return [
        _data_target(
            config, 1, 105.0, 10.0, HIGH_RATE, merit=90, minutes=200, snapshot=20
        ),
        _data_target(
            config, 2, 110.0, 10.0, LOW_RATE, merit=50, minutes=200, snapshot=20
        ),
        _data_target(
            config, 3, 115.0, 5.0, LOW_RATE, merit=60, minutes=100, snapshot=20
        ),
    ]


def _planned_downlink(coverage: StationCoverage, begin: float, end: float) -> float:
    return sum(
        span.rate_mbps * (span.end - span.begin) / 1000.0
        for span in coverage.spans(begin, end)
    )


@pytest.fixture(scope="module")
def omni_run() -> QueueDITL:
    """One executed scenario for the read-only science, plot and stats checks."""
    config = _recorder_config()
    ditl = create_ditl(config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS))
    assert isinstance(ditl, QueueDITL)
    ditl.step_size = STEP
    assert ditl.calc()
    return ditl


@pytest.fixture(scope="module")
def contact_window(omni_run: QueueDITL) -> tuple[datetime, datetime]:
    """A short execution interval containing a booked, usable pass."""
    assert omni_run.coverage is not None
    first = omni_run.coverage.scheduled_passes[0]
    return (
        datetime.fromtimestamp(max(T0, first.begin - 10 * MIN), timezone.utc),
        datetime.fromtimestamp(first.end + 10 * MIN, timezone.utc),
    )


class TestExecution:
    @pytest.mark.parametrize("antenna_type", [AntennaType.OMNI, AntennaType.GIMBALED])
    def test_future_days_are_booked_before_execution(
        self, antenna_type, monkeypatch
    ) -> None:
        config = _recorder_config(antenna_type, hours=48, gimbal_range_deg=180.0)
        for station in config.ground_stations.stations:
            station.schedule_probability = 0.5
        config.ground_stations.get("SGS").schedule_probability = 1.0
        # The ephemeris books both days before execution; only a short run is
        # needed to verify that execution consumes those existing bookings.
        ditl = create_ditl(config, _targets(config), BEGIN, BEGIN + timedelta(hours=2))
        assert isinstance(ditl.passes, PassTimes)
        assert ditl.coverage is not None
        assert ditl.passes is ditl.coverage.pass_times
        booked = [(p.begin, p.end, p.station) for p in ditl.passes.passes]
        assert any(begin >= T0 + 24 * HOUR for begin, _, _ in booked)
        assert all(not p.requires_tracking for p in ditl.passes.passes)

        def unexpected_prediction(*args, **kwargs):
            raise AssertionError("Passes must be booked before execution")

        monkeypatch.setattr(PassTimes, "get", unexpected_prediction)
        monkeypatch.setattr(PassTimes, "predict", unexpected_prediction)
        assert ditl.calc()
        assert [(p.begin, p.end, p.station) for p in ditl.passes.passes] == booked
        assert ditl.plan.ground_contacts == booked
        assert ACSMode.PASS not in set(ditl.mode)

    @pytest.mark.parametrize("antenna_type", [AntennaType.OMNI, AntennaType.GIMBALED])
    def test_no_station_bookings_means_no_forecast_or_executed_downlink(
        self, antenna_type
    ) -> None:
        config = _recorder_config(antenna_type, gimbal_range_deg=180.0)
        for station in config.ground_stations.stations:
            station.schedule_probability = 0.0
        ditl = create_ditl(config, _targets(config), BEGIN, BEGIN + timedelta(hours=2))
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.coverage is not None
        assert ditl.coverage.visible.any()
        assert ditl.coverage.contacts == []
        assert ditl.data_downlinked_gb[-1] == 0.0
        assert ACSMode.PASS not in set(ditl.mode)
        ledger = RecorderLedger(
            config,
            T0,
            T0 + HOURS * HOUR,
            [],
            start_volume_gb=0.0,
            coverage=ditl.coverage,
        )
        assert sum(ledger.downlink_gb) == 0.0
        allocator = LongRangeAllocator(
            config,
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            coverage=ditl.coverage,
            solver="greedy",
        )
        assert sum(allocator.allocate(_targets(config), T0).downlink) == 0.0

    def test_an_omni_antenna_downlinks_without_a_pass(
        self, omni_run: QueueDITL
    ) -> None:
        """Dispatch with an omni antenna never enters PASS mode, yet downlinks
        in every span of coverage, at the station's rate, and keeps the
        recorder within its planning limit."""
        ditl = omni_run

        assert ditl.coverage is not None
        assert ACSMode.PASS not in set(ditl.mode)
        assert ditl.executed_passes.passes == []
        end = T0 + HOURS * HOUR
        downlinked = ditl.data_downlinked_gb[-1]
        assert downlinked > 0.0
        # No more than the coverage allows, and most of it: the recorder was
        # never empty for long during a contact.
        assert downlinked <= _planned_downlink(ditl.coverage, T0, end) + 1e-9
        assert downlinked > 0.5 * _planned_downlink(ditl.coverage, T0, end)
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6
        assert max(ditl.recorder_alert) < 2
        observed = {hk.obsid for hk in ditl.telemetry.housekeeping if hk.obsid}
        assert {1, 2, 3} <= observed

    def test_downlink_happens_in_science_mode(self, omni_run: QueueDITL) -> None:
        """The spacecraft stays on its target through a contact."""
        ditl = omni_run
        assert ditl.coverage is not None
        first = ditl.coverage.spans(T0, T0 + HOURS * HOUR)[0]
        k = ditl.utime.index(first.begin)
        steps_in_contact = range(k, ditl.utime.index(first.end))
        assert any(ditl.mode[i] == ACSMode.SCIENCE for i in steps_in_contact)
        assert ditl.data_downlinked_gb[k] > ditl.data_downlinked_gb[k - 1]

    def test_a_gimballed_antenna_downlinks_only_within_its_range(
        self, contact_window: tuple[datetime, datetime]
    ) -> None:
        """A gimbal that reaches everywhere downlinks as an omni antenna does;
        one that cannot move downlinks only when the attitude flown happens
        to put its boresight on the station."""
        config = _recorder_config(AntennaType.GIMBALED, gimbal_range_deg=180.0)
        begin, end = contact_window
        free = create_ditl(config, _targets(config), begin, end)
        free.step_size = STEP
        assert free.calc()
        config = _recorder_config(AntennaType.GIMBALED, gimbal_range_deg=1.0)
        stuck = create_ditl(config, _targets(config), begin, end)
        stuck.step_size = STEP
        assert stuck.calc()

        config = _recorder_config()
        reference = create_ditl(config, _targets(config), begin, end)
        reference.step_size = STEP
        assert reference.calc()
        assert free.data_downlinked_gb[-1] > 0.0
        assert free.data_downlinked_gb[-1] == pytest.approx(
            reference.data_downlinked_gb[-1]
        )
        assert stuck.data_downlinked_gb[-1] < 0.5 * free.data_downlinked_gb[-1]

    def test_a_body_fixed_antenna_keeps_its_passes(self) -> None:
        config = _recorder_config(AntennaType.FIXED)
        first = PassTimes(config).predict()[0]
        # Fly the ingress and the pass itself without simulating the many
        # hours before the first usable tracking opportunity.
        begin = datetime.fromtimestamp(max(T0, first.begin - 10 * MIN), timezone.utc)
        end = datetime.fromtimestamp(first.end + 10 * MIN, timezone.utc)
        ditl = create_ditl(config, _targets(config), begin, end)
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.coverage is None
        assert ACSMode.PASS in set(ditl.mode)
        assert ditl.data_downlinked_gb[-1] > 0.0

    def test_a_station_not_scheduled_for_tracking_downlinks(
        self, contact_window: tuple[datetime, datetime]
    ) -> None:
        config = _recorder_config()
        for station in config.ground_stations.stations:
            station.schedule_for_tracking = False
        begin, end = contact_window
        ditl = create_ditl(config, _targets(config), begin, end)
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.data_downlinked_gb[-1] > 0.0

    def test_the_timeline_plot_shows_the_coverage_as_ground_contact(
        self, omni_run: QueueDITL
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from conops.visualization.mpl.ditl_timeline import (
            _extract_contacts,
            _extract_ground_passes,
            plot_ditl_timeline,
        )
        from conops.visualization.plotly.ditl_timeline import (
            _extract_contacts as _extract_contacts_plotly,
        )
        from conops.visualization.plotly.ditl_timeline import (
            plot_ditl_timeline_plotly,
        )

        ditl = omni_run
        assert ditl.coverage is not None
        t_start = float(ditl.plan[0].begin)
        spans = ditl.coverage.spans(T0, T0 + HOURS * HOUR)
        assert len(spans) > 1
        # No tracking activities; one bar per usable link span, by station.
        assert _extract_ground_passes(ditl, t_start, 0.0) == []
        bars = _extract_contacts(ditl, t_start, 0.0)
        assert _extract_contacts_plotly(ditl, t_start, 0.0) == bars
        assert [(b.start, b.duration, b.station, b.fraction) for b in bars] == [
            pytest.approx(
                ((s.begin - t_start) / 3600, (s.end - s.begin) / 3600, s.station, 1.0)
            )
            for s in spans
        ]
        fig, ax = plot_ditl_timeline(ditl)
        assert "Ground Contact" in [t.get_text() for t in ax.get_yticklabels()]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert {f"Contact: {s.station}" for s in spans} <= set(labels)
        plt.close(fig)
        figure = plot_ditl_timeline_plotly(ditl)
        names = {trace.name for trace in figure.data}
        assert {f"Contact: {s.station}" for s in spans} <= names

    def test_statistics_report_the_coverage(
        self, capsys: pytest.CaptureFixture[str], omni_run: QueueDITL
    ) -> None:
        omni_run.print_statistics()
        out = capsys.readouterr().out
        assert "GROUND STATION PASS STATISTICS" in out
        assert "% of the run" in out


class TestDispatchRoom:
    def test_the_room_runs_to_the_next_span_of_coverage(self) -> None:
        """The contacts the room is carried through are the spans of
        coverage: before the first, the housekeeping until it begins has to
        fit; in contact, the room is what is free now."""
        config = _recorder_config()
        ditl = create_ditl(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        assert ditl.coverage is not None
        # The run's end, which calc() would set.
        ditl.uend = T0 + HOURS * HOUR
        first = ditl.coverage.spans(T0, T0 + HOURS * HOUR)[0]
        free = LIMIT - config.recorder.current_volume_gb
        before = ditl._recorder_room_gb(T0, 0.0, 0.0, 0.0)
        assert before is not None
        assert before == pytest.approx(
            free
            - config.housekeeping_while_waiting(
                whole_steps(first.begin - T0, ditl.step_size)
            ),
            rel=1e-6,
        )
        # In contact, nothing accrues before the next contact: it is now.
        assert ditl._recorder_room_gb(
            first.begin + MIN, 0.0, 0.0, 0.0
        ) == pytest.approx(free, rel=1e-6)


class TestPlanning:
    def test_weeklong_omni_contact_does_not_refit_at_every_step(self, monkeypatch):
        hours = 7 * 24
        config = _recorder_config(hours=hours)
        config.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0.0)
        config.constraint.sun_constraint = None
        config.constraint.earth_constraint = None
        config.constraint.invalidate_combined_constraint_cache()
        config.recorder.current_volume_gb = 17.0
        # This link cannot clear enough room for a snapshot over the whole
        # week. Retrying at 10,080 individual credits is both futile and costly.
        coverage = Mock(spec=StationCoverage)
        coverage.step_size = STEP
        coverage.repoints = []
        coverage.scheduled_passes = [ContactSpan(T0, T0 + hours * HOUR, "MAL", 1e-5)]
        coverage.rate_spans.return_value = coverage.scheduled_passes
        target = _data_target(config, 1, 105.0, 10.0, 0.08, minutes=1, snapshot=1)
        scheduler = PriorityScheduler(
            config,
            [target],
            BEGIN,
            BEGIN + timedelta(hours=hours),
            include_passes=False,
            coverage=coverage,
            start_state=StartState(time=T0, attitude=(0.0, 0.0, 0.0)),
        )
        fit = scheduler._fit_in_gap
        calls = 0

        def bounded_fit(*args, **kwargs):
            nonlocal calls
            calls += 1
            assert calls < 40, "Recorder retries must not walk telemetry steps"
            return fit(*args, **kwargs)

        monkeypatch.setattr(scheduler, "_fit_in_gap", bounded_fit)
        plan = scheduler.schedule()
        assert not _science(plan)
        assert scheduler._ledger.segments <= 3

    @pytest.mark.parametrize(
        "scheduler_class", [PriorityScheduler, LocalSearchScheduler, CpSatScheduler]
    )
    def test_science_during_a_contact_cannot_spend_future_downlink(
        self, scheduler_class
    ):
        options = {}
        if scheduler_class is CpSatScheduler:
            pytest.importorskip("ortools")
            options = {
                "solver_time_limit": 0.5,
                "workers": 1,
                "seed": 1,
                "max_iterations": 0,
            }
        elif scheduler_class is LocalSearchScheduler:
            options = {"max_iterations": 3, "seed": 1}
        config = _recorder_config()
        config.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0.0)
        config.recorder.current_volume_gb = 17.0
        coverage = StationCoverage.for_config(config)
        assert coverage is not None
        first = coverage.spans(T0, T0 + HOURS * HOUR)[0]
        begin = datetime.fromtimestamp(first.begin, timezone.utc)
        end = datetime.fromtimestamp(first.end + 5 * MIN, timezone.utc)
        target = _data_target(config, 1, 105.0, 10.0, 0.08, minutes=1, snapshot=1)
        scheduler = scheduler_class(
            config,
            [target],
            begin,
            end,
            coverage=coverage,
            include_passes=False,
            **options,
        )
        plan = scheduler.schedule()
        assert _science(plan)
        ditl = DITL(config, plan=plan, begin=begin, end=end)
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        assert max(ditl.recorder_volume_gb) < LIMIT
        assert (
            max(ditl.recorder_volume_gb)
            <= max(scheduler._ledger.peaks(scheduler._volumes)) + 1e-6
        )

    @pytest.mark.parametrize("antenna_type", [AntennaType.OMNI, AntennaType.GIMBALED])
    @pytest.mark.parametrize("probability", [0.0, 0.5])
    def test_bookings_survive_saved_plan_replay_without_a_seed(
        self, antenna_type, probability, tmp_path
    ) -> None:
        config = _recorder_config(antenna_type, gimbal_range_deg=180.0)
        config.random_seed = None
        for station in config.ground_stations.stations:
            station.schedule_probability = probability
        config.scheduler = SchedulerConfig(mode=SchedulerMode.PLANNED)
        ditl = create_ditl(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        assert ditl.coverage is not None
        booked = ditl.plan.ground_contacts
        assert booked is not None
        assert booked == [
            (p.begin, p.end, p.station) for p in ditl.coverage.scheduled_passes
        ]
        loaded = Plan.load(ditl.plan.save(tmp_path / "plan.json"))
        assert loaded.ground_contacts == booked

        # Even if a later probability draw would reject every pass, replay
        # must use the bookings saved with the plan. Assign after construction
        # as well, which exercises the common Plan.load workflow.
        for station in config.ground_stations.stations:
            station.schedule_probability = 0.0
        replay = DITL(config=config, begin=BEGIN, end=BEGIN + timedelta(hours=HOURS))
        replay.plan = loaded
        assert replay.calc()
        assert replay.coverage is not None
        assert replay.coverage.contacts == ditl.coverage.contacts
        assert replay.validate_plan_matches_execution() == []
        assert ACSMode.PASS not in set(replay.mode)

    def _plan(self, antenna_type: AntennaType = AntennaType.OMNI):
        config = _recorder_config(antenna_type)
        hungry = _data_target(
            config, 1, 105.0, 10.0, HIGH_RATE, merit=90, minutes=40, snapshot=20
        )
        frugal = _data_target(
            config, 2, 110.0, 10.0, LOW_RATE, merit=50, minutes=200, snapshot=20
        )
        scheduler = PriorityScheduler(
            config,
            [hungry, frugal],
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            step_size=STEP,
        )
        return config, scheduler, scheduler.schedule()

    def test_the_ledger_takes_its_contacts_from_the_coverage(self) -> None:
        config = _recorder_config()
        coverage = StationCoverage.for_config(config)
        assert coverage is not None
        end = T0 + HOURS * HOUR
        ledger = RecorderLedger(
            config, T0, end, [], start_volume_gb=0.0, coverage=coverage
        )
        spans = coverage.spans(T0, end)
        assert ledger.segments == len(spans) + 1
        assert sum(ledger.downlink_gb) == pytest.approx(
            _planned_downlink(coverage, T0, end)
        )
        # Housekeeping at the nominal rate throughout: in contact as well.
        assert sum(ledger.housekeeping_gb) + sum(
            ledger.contact_housekeeping_gb
        ) == pytest.approx(config.housekeeping_generated(end - T0))
        assert RecorderLedger(config, T0, end, [], start_volume_gb=0.0).segments == 1

    def test_the_plan_stays_within_the_limit_and_executes(self) -> None:
        config, scheduler, plan = self._plan()

        assert scheduler.ctx.coverage is not None
        assert not any(entry.obstype == ObsType.GSP for entry in plan)
        assert (
            scheduler._ledger.segments
            == len(scheduler.ctx.coverage.spans(T0, T0 + HOURS * HOUR)) + 1
        )
        assert all(
            peak <= LIMIT + 1e-6 for peak in scheduler._ledger.peaks(scheduler._volumes)
        )
        assert len(_science(plan)) >= 5
        # Both high-rate snapshots exceed the recorder's limit together, so
        # a contact must make room for the second within the science gap.
        # Check that behavior without assuming particular orbital pass times.
        hungry = sorted(
            (e for e in _science(plan) if int(e.obsid) == 1),
            key=lambda e: float(e.begin),
        )
        assert len(hungry) == 2
        assert (
            sum(
                scheduler._ledger.science_volume(
                    e, e.collection_seconds_between(e.begin, e.end)
                )
                for e in hungry
            )
            > LIMIT
        )
        _, ledger, _ = scheduler._ledger.members[0]
        assert ledger.segment(float(hungry[0].collection_begin)) < ledger.segment(
            float(hungry[1].collection_begin)
        )
        ditl = _execute(config, plan, HOURS, STEP)
        assert ditl.validate_plan_matches_execution() == []
        assert ACSMode.PASS not in set(ditl.mode)
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6
        assert max(ditl.recorder_volume_gb) > 0.5 * LIMIT
        assert ditl.data_downlinked_gb[-1] > 0.0

    def test_the_scheduler_takes_the_coverage_it_is_given(self) -> None:
        config = _recorder_config()
        coverage = StationCoverage.for_config(config)
        scheduler = PriorityScheduler(
            config,
            _targets(config),
            BEGIN,
            BEGIN + timedelta(hours=2),
            coverage=coverage,
        )
        scheduler.schedule()
        assert scheduler.ctx.coverage is coverage

    def test_a_body_fixed_antenna_plans_its_passes(self) -> None:
        config, scheduler, plan = self._plan(AntennaType.FIXED)
        assert scheduler.ctx.coverage is None
        assert any(entry.obstype == ObsType.GSP for entry in plan)


class TestRollingHorizon:
    def test_first_replan_reuses_unseeded_bookings(self, monkeypatch):
        config = _recorder_config()
        config.random_seed = None
        for station in config.ground_stations.stations:
            station.schedule_probability = 0.5
        ditl = RollingHorizonDITL(
            config,
            [],
            begin=BEGIN,
            end=BEGIN + timedelta(minutes=5),
            horizon=timedelta(minutes=3),
            replan_interval=timedelta(minutes=2),
        )
        booked = list(ditl.coverage.pass_times.passes)

        def no_prediction(*args, **kwargs):
            raise AssertionError("Replanning must reuse the run's bookings")

        monkeypatch.setattr(PassTimes, "predict", no_prediction)
        assert ditl.calc()
        assert ditl._run_passes == booked

    def test_the_run_shares_its_coverage_with_each_plan(self) -> None:
        config = _recorder_config()
        seen: list[object] = []

        class Recording(PriorityScheduler):
            def __init__(self, *args: object, **kwargs: object) -> None:
                seen.append(kwargs["coverage"])
                super().__init__(*args, **kwargs)  # type: ignore[arg-type]

        ditl = RollingHorizonDITL(
            config,
            [_data_target(config, 1, 105.0, 10.0, LOW_RATE, minutes=60)],
            begin=BEGIN,
            end=BEGIN + timedelta(hours=10),
            horizon=timedelta(hours=4),
            replan_interval=timedelta(hours=3),
            scheduler=Recording,
        )
        ditl.step_size = STEP
        assert ditl.calc()

        assert ditl.coverage is not None
        assert len(seen) >= 3
        assert all(coverage is ditl.coverage for coverage in seen)
        assert ACSMode.PASS not in set(ditl.mode)
        assert ditl.data_downlinked_gb[-1] > 0.0
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6


class TestAllocator:
    @pytest.mark.parametrize("antenna_type", [AntennaType.OMNI, AntennaType.GIMBALED])
    def test_supplied_bookings_govern_the_budget(self, antenna_type):
        config = _recorder_config(antenna_type, gimbal_range_deg=180.0)
        booked = PassTimes(config).predict()[:1]
        assert booked
        allocator = LongRangeAllocator(
            config,
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            passes=[],
            solver="greedy",
        )
        targets = _targets(config)
        assert sum(allocator.allocate(targets, T0).downlink) == 0.0
        # Set bookings after coverage has already been constructed: callers
        # such as rolling replanning supply their run's bookings this way.
        allocator.passes = booked
        allocation = allocator.allocate(targets, T0)
        coverage = allocator.ctx.coverage
        assert coverage is not None
        assert [(p.begin, p.end, p.station) for p in coverage.pass_times.passes] == [
            (p.begin, p.end, p.station) for p in booked
        ]
        assert sum(allocation.downlink) == pytest.approx(
            coverage.downlink_gb(T0, T0 + HOURS * HOUR)
        )
        assert sum(allocation.downlink) > 0.0
        allocator.passes = []
        assert sum(allocator.allocate(targets, T0).downlink) == 0.0

    def test_dispatch_shares_its_bookings_with_the_allocator(self) -> None:
        config = _recorder_config()
        config.random_seed = None
        for station in config.ground_stations.stations:
            station.schedule_probability = 0.5
        config.scheduler.allocation = AllocationSettings(solver="greedy")
        ditl = create_ditl(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        assert isinstance(ditl, QueueDITL)
        assert ditl.calc()
        assert ditl.allocator is not None
        assert ditl.allocator.ctx.coverage is ditl.coverage
        assert ditl.coverage is not None
        assert ditl.plan.ground_contacts == [
            (p.begin, p.end, p.station) for p in ditl.coverage.scheduled_passes
        ]

    def test_the_data_budget_downlinks_through_the_coverage(self) -> None:
        config = _recorder_config(hours=48)
        allocator = LongRangeAllocator(
            config,
            BEGIN,
            BEGIN + timedelta(hours=48),
            bin_length=timedelta(hours=24),
            solver="greedy",
        )
        allocation = allocator.allocate(_targets(config), T0)

        assert allocator.ctx.coverage is not None
        assert allocation.data_limit == pytest.approx(LIMIT)
        coverage = allocator.ctx.coverage
        assert allocation.downlink == pytest.approx(
            [
                _planned_downlink(coverage, T0, T0 + 24 * HOUR),
                _planned_downlink(coverage, T0 + 24 * HOUR, T0 + 48 * HOUR),
            ]
        )
        assert min(allocation.downlink) > 0.0
        # Housekeeping at the nominal rate for the whole bin.
        assert allocation.housekeeping == pytest.approx(
            [config.housekeeping_generated(24 * HOUR)] * 2
        )

    def test_the_rolling_run_gives_the_allocator_its_coverage(self) -> None:
        config = _recorder_config()
        allocator = LongRangeAllocator(
            config,
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            bin_length=timedelta(hours=7),
            solver="greedy",
        )
        ditl = RollingHorizonDITL(
            config,
            _targets(config),
            begin=BEGIN,
            end=BEGIN + timedelta(hours=HOURS),
            horizon=timedelta(hours=7),
            replan_interval=timedelta(hours=7),
            allocator=allocator,
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert allocator.ctx.coverage is ditl.coverage
        assert ditl.allocation is not None
        assert ditl.allocation.downlink[-1] > 0.0


class TestRangeDependentRates:
    """A rate that varies with range is applied per step in the simulation,
    per span in the ledger and per bin in the budget."""

    TABLE = [(500.0, 20.0), (1500.0, 10.0), (3000.0, 2.0)]

    def _ranged_config(self, hours: int = HOURS) -> MissionConfig:
        config = _recorder_config(hours=hours)
        for station in config.ground_stations.stations:
            station.bands = [
                BandCapability(
                    band="S",
                    downlink_rate_mbps=20.0,
                    downlink_rate_vs_range_km=self.TABLE,
                )
            ]
        config.spacecraft_bus.communications = CommunicationsSystem(
            name="S-band",
            band_capabilities=[BandCapability(band="S", downlink_rate_mbps=50.0)],
            antenna_pointing=AntennaPointing(antenna_type=AntennaType.OMNI),
        )
        return config

    def test_the_run_downlinks_at_the_rate_at_each_range(self) -> None:
        config = self._ranged_config()
        ditl = create_ditl(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.coverage is not None
        end = T0 + HOURS * HOUR
        planned = ditl.coverage.downlink_gb(T0, end)
        assert 0.5 * planned < ditl.data_downlinked_gb[-1] <= planned + 1e-9
        # The first contact's steps downlink different amounts.
        first = ditl.coverage.spans(T0, end)[0]
        k = ditl.utime.index(first.begin)
        steps = [
            ditl.data_downlinked_gb[i + 1] - ditl.data_downlinked_gb[i]
            for i in range(k, ditl.utime.index(first.end) - 1)
        ]
        assert len({round(s, 6) for s in steps}) > 1
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6

    def test_the_ledger_and_the_budget_take_the_same_downlink(self) -> None:
        config = self._ranged_config(hours=48)
        coverage = StationCoverage.for_config(config)
        assert coverage is not None
        end = T0 + 48 * HOUR
        ledger = RecorderLedger(
            config, T0, end, [], start_volume_gb=0.0, coverage=coverage
        )
        assert sum(ledger.downlink_gb) == pytest.approx(coverage.downlink_gb(T0, end))
        allocator = LongRangeAllocator(
            config,
            BEGIN,
            BEGIN + timedelta(hours=48),
            bin_length=timedelta(hours=24),
            solver="greedy",
            coverage=coverage,
        )
        allocation = allocator.allocate(_targets(config), T0)
        assert allocation.downlink == pytest.approx(
            [
                coverage.downlink_gb(T0, T0 + 24 * HOUR),
                coverage.downlink_gb(T0 + 24 * HOUR, end),
            ]
        )


class TestAntennaRepoints:
    """Repoints are logged, counted in sensitive observations, and kept out
    of observations whose category forbids them."""

    @staticmethod
    def _config(hours: int = HOURS) -> MissionConfig:
        config = _recorder_config(hours=hours)
        config.spacecraft_bus.communications = CommunicationsSystem(
            name="S-band",
            band_capabilities=[BandCapability(band="S", downlink_rate_mbps=RATE_MBPS)],
            antenna_pointing=AntennaPointing(
                antenna_type=AntennaType.GIMBALED,
                gimbal_range_deg=180.0,
                # Every three minutes of a contact, a minute long.
                repoint_interval_vs_range_km=[(0.0, 180.0)],
                repoint_seconds=60.0,
            ),
            acquisition_seconds=60.0,
        )
        config.observation_categories = ObservationCategories(
            categories=[
                ObservationCategory(
                    name="Sensitive",
                    obsid_min=1,
                    obsid_max=3,
                    antenna_repoints="forbidden",
                ),
                ObservationCategory(
                    name="Watched",
                    obsid_min=3,
                    obsid_max=4,
                    antenna_repoints="reported",
                ),
            ]
        )
        return config

    @staticmethod
    def _overlaps(cuts: list[tuple[float, float]], begin: float, end: float) -> bool:
        return any(c0 < end and c1 > begin or begin <= c0 < end for c0, c1 in cuts)

    @pytest.mark.parametrize("path", ["queue", "dispatch"])
    @pytest.mark.parametrize("cut_begin", [-30, 30])
    def test_repoint_finished_during_setup_does_not_reject_collection(
        self, monkeypatch, path, cut_begin
    ):
        config = self._config(hours=1)
        config.payload.observation_timing = ObservationTiming(setup_seconds=300)
        ditl = QueueDITL(config, begin=BEGIN, end=BEGIN + timedelta(hours=1))
        ditl.ustart, ditl.uend = T0, T0 + HOUR
        target = _data_target(config, 1, 105.0, 10.0, LOW_RATE, minutes=1, snapshot=1)
        target.visibility()
        ditl.queue.targets = [target]
        calls = []

        def cut_at(time):
            calls.append(time)
            return (T0 + cut_begin, T0 + 120) if time < T0 + 120 else None

        monkeypatch.setattr(ditl.coverage, "next_repoint_cut", cut_at)
        # The fast candidate check is enabled only when there are repoints.
        from conops.simulation.coverage import Repoint

        ditl.coverage.repoints = [Repoint(T0 + cut_begin, T0 + 120, "MAL", "tracking")]
        if path == "queue":
            assert ditl.queue.get(0.0, 0.0, T0) is target
        else:
            assert (
                ditl._fetch_new_ppt_inner(T0, 0.0, 0.0, enqueue_slew=False) is not None
            )
        assert calls and all(time >= T0 + 300 for time in calls)

    def test_a_plan_keeps_forbidden_observations_between_repoints(self) -> None:
        config = self._config()
        targets = _targets(config)
        scheduler = PriorityScheduler(
            config, targets, BEGIN, BEGIN + timedelta(hours=HOURS), step_size=STEP
        )
        plan = scheduler.schedule()
        coverage = scheduler.ctx.coverage
        assert coverage is not None
        cuts = coverage.repoint_cuts(T0, T0 + HOURS * HOUR)
        assert len(cuts) > 10
        science = _science(plan)
        forbidden = [e for e in science if int(e.obsid) in (1, 2)]
        watched = [e for e in science if int(e.obsid) == 3]
        assert forbidden and watched
        for entry in forbidden:
            assert not self._overlaps(
                cuts, float(entry.collection_begin), float(entry.collection_end)
            )
        # The watched category is placed without regard to repoints: the
        # same targets without the rule take the same room and some
        # observation of the run does contain a repoint.
        free = self._config()
        free.observation_categories = ObservationCategories(categories=[])
        free_plan = PriorityScheduler(
            free, _targets(free), BEGIN, BEGIN + timedelta(hours=HOURS), step_size=STEP
        ).schedule()
        assert any(
            self._overlaps(cuts, float(e.collection_begin), float(e.collection_end))
            for e in _science(free_plan)
        )
        ditl = _execute(config, plan, HOURS, STEP)
        assert ditl.validate_plan_matches_execution() == []
        assert all(
            setting != "forbidden" for _, _, setting in ditl.repoints_in_observations
        )
        assert any(
            e.event_type == "ANTENNA" and "tracking" in e.description
            for e in ditl.log.events
        )

    @pytest.fixture(scope="class")
    @classmethod
    def repoint_dispatch(cls) -> QueueDITL:
        config = cls._config()
        ditl = create_ditl(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        return ditl

    def test_dispatch_keeps_forbidden_observations_between_repoints(
        self, repoint_dispatch: QueueDITL
    ) -> None:
        ditl = repoint_dispatch
        assert ditl.coverage is not None
        assert ditl.queue.coverage is ditl.coverage
        observed = {hk.obsid for hk in ditl.telemetry.housekeeping if hk.obsid}
        assert {1, 2, 3} <= observed
        settings = [setting for _, _, setting in ditl.repoints_in_observations]
        assert "forbidden" not in settings
        assert "reported" in settings
        events = [e for e in ditl.log.events if e.event_type == "ANTENNA"]
        assert any("during observation 3 (reported)" in e.description for e in events)
        assert len(events) == len(ditl.coverage.repoints_in(T0, T0 + HOURS * HOUR))

    def test_statistics_report_the_repoints(
        self, capsys: pytest.CaptureFixture[str], repoint_dispatch: QueueDITL
    ) -> None:
        repoint_dispatch.print_statistics()
        out = capsys.readouterr().out
        assert "Antenna Repoints:" in out
        assert "Repoints During Sensitive Observations:" in out
        assert "(0 in observations that forbid them)" in out


def test_a_ditl_without_coverage_has_the_legacy_downlink() -> None:
    config = _recorder_config(None)
    ditl = DITL(
        config=config,
        ephem=config.constraint.ephem,
        begin=BEGIN,
        end=BEGIN + timedelta(hours=1),
    )
    assert ditl.coverage is None
