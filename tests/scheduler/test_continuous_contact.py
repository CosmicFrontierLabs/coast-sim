"""Contacts without a pass: an omni or gimballed antenna downlinks whenever a
station is in view, and the plans, the data budget and dispatch count on it."""

from datetime import timedelta

import pytest

from conops import DITL, create_ditl
from conops.common import AntennaType, ObsType
from conops.common.enums import ACSMode
from conops.config import (
    AntennaPointing,
    BandCapability,
    CommunicationsSystem,
    HousekeepingData,
    MissionConfig,
    OnboardRecorder,
    ScienceData,
)
from conops.ditl import RollingHorizonDITL
from conops.schedulers import LongRangeAllocator, PriorityScheduler
from conops.schedulers.recorder import RecorderLedger
from conops.simulation import StationCoverage
from conops.targets import Pointing

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


class TestExecution:
    def test_an_omni_antenna_downlinks_without_a_pass(self) -> None:
        """Dispatch with an omni antenna never enters PASS mode, yet downlinks
        in every span of coverage, at the station's rate, and keeps the
        recorder within its planning limit."""
        config = _recorder_config()
        ditl = create_ditl(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        ditl.step_size = STEP
        assert ditl.calc()

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

    def test_downlink_happens_in_science_mode(self) -> None:
        """The spacecraft stays on its target through a contact."""
        config = _recorder_config()
        ditl = create_ditl(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.coverage is not None
        first = ditl.coverage.spans(T0, T0 + HOURS * HOUR)[0]
        k = ditl.utime.index(first.begin)
        steps_in_contact = range(k, ditl.utime.index(first.end))
        assert any(ditl.mode[i] == ACSMode.SCIENCE for i in steps_in_contact)
        assert ditl.data_downlinked_gb[k] > ditl.data_downlinked_gb[k - 1]

    def test_a_gimballed_antenna_downlinks_only_within_its_range(self) -> None:
        """A gimbal that reaches everywhere downlinks as an omni antenna does;
        one that cannot move downlinks only when the attitude flown happens
        to put its boresight on the station."""
        config = _recorder_config(AntennaType.GIMBALED, gimbal_range_deg=180.0)
        free = create_ditl(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        free.step_size = STEP
        assert free.calc()
        config = _recorder_config(AntennaType.GIMBALED, gimbal_range_deg=1.0)
        stuck = create_ditl(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        stuck.step_size = STEP
        assert stuck.calc()

        omni = _recorder_config()
        reference = create_ditl(
            omni, _targets(omni), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        reference.step_size = STEP
        assert reference.calc()
        assert free.data_downlinked_gb[-1] == pytest.approx(
            reference.data_downlinked_gb[-1]
        )
        assert stuck.data_downlinked_gb[-1] < 0.5 * free.data_downlinked_gb[-1]

    def test_a_body_fixed_antenna_keeps_its_passes(self) -> None:
        config = _recorder_config(AntennaType.FIXED)
        ditl = create_ditl(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.coverage is None
        assert ACSMode.PASS in set(ditl.mode)
        assert ditl.data_downlinked_gb[-1] > 0.0

    def test_a_station_not_scheduled_for_tracking_downlinks(self) -> None:
        config = _recorder_config()
        for station in config.ground_stations.stations:
            station.schedule_for_tracking = False
        ditl = create_ditl(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.data_downlinked_gb[-1] > 0.0

    def test_statistics_report_the_coverage(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = _recorder_config()
        ditl = create_ditl(config, _targets(config), BEGIN, BEGIN + timedelta(hours=2))
        ditl.step_size = STEP
        assert ditl.calc()
        ditl.print_statistics()
        out = capsys.readouterr().out
        assert "GROUND STATION COVERAGE" in out
        assert "% of the run" in out
        assert "GROUND STATION PASS STATISTICS" not in out


class TestDispatchRoom:
    def test_the_room_runs_to_the_next_span_of_coverage(self) -> None:
        config = _recorder_config()
        ditl = create_ditl(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        assert ditl.coverage is not None
        # The run's end, which calc() would set.
        ditl.uend = T0 + HOURS * HOUR
        first = ditl.coverage.spans(T0, T0 + HOURS * HOUR)[0]
        free = LIMIT - config.recorder.current_volume_gb
        before = ditl._recorder_room_gb(T0)
        assert before is not None
        assert before == pytest.approx(
            free - config.housekeeping_generated(first.begin - T0)
        )
        # In contact, nothing accrues before the next contact: it is now.
        assert ditl._recorder_room_gb(first.begin + MIN) == pytest.approx(free)


class TestPlanning:
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
        assert sum(ledger.contact_housekeeping_gb) == pytest.approx(
            config.housekeeping_generated(sum(s.end - s.begin for s in spans))
        )
        assert RecorderLedger(config, T0, end, [], start_volume_gb=0.0).segments == 1

    def test_the_plan_stays_within_the_limit_and_executes(self) -> None:
        config, scheduler, plan = self._plan()

        assert scheduler.ctx.coverage is not None
        assert not any(entry.obstype == ObsType.GSP for entry in plan)
        assert scheduler._ledger.segments > 5
        assert all(
            peak <= LIMIT + 1e-6 for peak in scheduler._ledger.peaks(scheduler._volumes)
        )
        assert len(_science(plan)) >= 5
        # The recorder has room for three snapshots before the first contact;
        # the next is placed after a contact, which is no block on the timeline.
        coverage = scheduler.ctx.coverage
        first = coverage.spans(T0, T0 + HOURS * HOUR)[0]
        science = sorted(_science(plan), key=lambda e: float(e.begin))
        assert float(science[2].end) < first.begin
        assert float(science[3].begin) >= first.begin
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


def test_a_ditl_without_coverage_has_the_legacy_downlink() -> None:
    config = _recorder_config(None)
    ditl = DITL(
        config=config,
        ephem=config.constraint.ephem,
        begin=BEGIN,
        end=BEGIN + timedelta(hours=1),
    )
    assert ditl.coverage is None
