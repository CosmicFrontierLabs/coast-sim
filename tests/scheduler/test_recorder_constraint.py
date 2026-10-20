"""Plans keep the onboard recorder within its planning limit.

The ledger's arithmetic is tested in test_consumable.py; here it is the
recorder's mapping onto it, and the schedulers' use of it.
"""

from datetime import timedelta

import pytest

from conops.common import ObsType
from conops.common.enums import ACSMode
from conops.config import (
    BandCapability,
    CommunicationsSystem,
    DataGeneration,
    GroundStation,
    GroundStationRegistry,
    HousekeepingData,
    Instrument,
    MissionConfig,
    OnboardRecorder,
    Payload,
    ScienceData,
    SpacecraftBus,
)
from conops.ditl import RollingHorizonDITL
from conops.schedulers import LocalSearchScheduler, PriorityScheduler
from conops.schedulers.recorder import RecorderLedger, contact_downlink_gb
from conops.targets import PlanEntry, Pointing

from .planning_scenario import BEGIN, HOUR, MIN, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target
from .test_priority_scheduler import _execute, _science

HOURS = 14
STEP = 60
CAPACITY = 20.0
LIMIT = 0.9 * CAPACITY
"""The default planning limit is the red threshold."""
HIGH_RATE = 0.01
"""Gbps: a 20-minute snapshot is 12 Gb, over half the limit."""
LOW_RATE = 0.001
"""Gbps: a 20-minute snapshot is 1.2 Gb."""


def _recorder_config(limit: float | None | str = "default") -> MissionConfig:
    """The scenario with contacts, a small recorder and some housekeeping."""
    config = _config(HOURS, stations=True)
    config.recorder = OnboardRecorder(capacity_gb=CAPACITY)
    if limit != "default":
        config.recorder.planning_limit = limit  # type: ignore[assignment]
    config.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0.0001)
    return config


def _data_target(
    config: MissionConfig, obsid: int, ra: float, dec: float, rate: float, **kw: float
) -> Pointing:
    target = _target(config, obsid, ra, dec, **kw)
    target.science_data = {"Default Instrument": ScienceData(rate_gbps=rate)}
    return target


def _peaks_of(scheduler: PriorityScheduler) -> list[float]:
    return scheduler._ledger.peaks(scheduler._volumes)


def _station(code: str, rate_mbps: float) -> GroundStation:
    return GroundStation(
        code=code,
        name=code,
        latitude_deg=0.0,
        longitude_deg=0.0,
        bands=[BandCapability(band="S", downlink_rate_mbps=rate_mbps)],
    )


def _contact(begin: float, end: float, station: str) -> PlanEntry:
    """A plan's ground-contact entry, linked for its whole length."""
    return PlanEntry(
        obstype=ObsType.GSP,
        station=station,
        begin=begin,
        end=end,
        contact_begin=begin,
        contact_end=end,
    )


def _first_contact(plan) -> float:
    return min(float(e.contact_begin) for e in plan if e.contact_begin is not None)


class TestPlanningLimit:
    def test_defaults_to_the_red_threshold(self) -> None:
        recorder = OnboardRecorder(capacity_gb=50.0, red_threshold=0.8)
        assert recorder.planning_limit == 0.8
        assert recorder.planning_limit_gb == pytest.approx(40.0)
        loaded = OnboardRecorder.model_validate_json(recorder.model_dump_json())
        assert loaded.planning_limit == 0.8

    def test_can_be_set_or_disabled(self) -> None:
        assert (
            OnboardRecorder(capacity_gb=50.0, planning_limit=0.5).planning_limit_gb
            == 25.0
        )
        assert OnboardRecorder(planning_limit=None).planning_limit_gb is None


class TestLedger:
    @pytest.fixture
    def config(self) -> MissionConfig:
        return MissionConfig(
            spacecraft_bus=SpacecraftBus(
                data_generation=HousekeepingData(
                    rate_gbps=0.001, rate_mode={"PASS": 0.003}
                )
            ),
            payload=Payload(
                instruments=[
                    Instrument(
                        name="camera",
                        data_generation=DataGeneration(
                            science=ScienceData(rate_gbps=0.1, per_observation_gb=1.0)
                        ),
                    )
                ]
            ),
            recorder=OnboardRecorder(capacity_gb=100.0, planning_limit=0.8),
            ground_stations=GroundStationRegistry(
                stations=[
                    _station("TWENTY", 200.0),
                    _station("THIRTY", 300.0),
                    _station("FIFTY", 500.0),
                    _station("ONE", 10.0),
                ]
            ),
        )

    def test_contacts_cut_the_horizon_into_segments(
        self, config: MissionConfig
    ) -> None:
        ledger = RecorderLedger(
            config,
            0.0,
            1000.0,
            [_contact(600.0, 700.0, "THIRTY"), _contact(200.0, 300.0, "TWENTY")],
            start_volume_gb=5.0,
        )
        assert ledger.segments == 3
        assert ledger.bounds == [0.0, 200.0, 600.0, 1000.0]
        assert ledger.downlink_gb == [0.0, 20.0, 30.0]
        times = (0, 199, 200, 599, 600, 1000)
        assert [ledger.segment(t) for t in times] == [0, 0, 1, 1, 2, 2]
        assert ledger.limit_gb == pytest.approx(80.0)

    def test_housekeeping_at_the_pass_rate_during_contacts(
        self, config: MissionConfig
    ) -> None:
        ledger = RecorderLedger(
            config, 0.0, 1000.0, [_contact(200.0, 300.0, "TWENTY")], start_volume_gb=0.0
        )
        assert ledger.housekeeping_gb == pytest.approx([0.2, 0.7])
        assert ledger.contact_housekeeping_gb == pytest.approx([0.0, 0.3])

    def test_contacts_outside_the_horizon_are_ignored(
        self, config: MissionConfig
    ) -> None:
        ledger = RecorderLedger(
            config, 100.0, 900.0, [_contact(950.0, 1000.0, "ONE")], start_volume_gb=0.0
        )
        assert ledger.segments == 1

    def test_a_contact_straddling_the_horizon_counts_for_the_part_inside(
        self, config: MissionConfig
    ) -> None:
        """As when a replan starts partway through a committed contact."""
        ledger = RecorderLedger(
            config,
            250.0,
            900.0,
            [_contact(200.0, 300.0, "TWENTY")],
            start_volume_gb=0.0,
        )
        assert ledger.bounds == [250.0, 250.0, 900.0]
        assert ledger.downlink_gb == pytest.approx([0.0, 10.0])
        assert ledger.contact_housekeeping_gb == pytest.approx([0.0, 0.15])

    def test_entries_without_a_link_are_not_contacts(
        self, config: MissionConfig
    ) -> None:
        science = PlanEntry(begin=200.0, end=300.0)
        ledger = RecorderLedger(config, 0.0, 1000.0, [science], start_volume_gb=0.0)
        assert ledger.segments == 1

    def test_no_limit_fits_anything(self, config: MissionConfig) -> None:
        config.recorder.planning_limit = None
        ledger = RecorderLedger(config, 0.0, 1000.0, [], start_volume_gb=0.0)
        assert not ledger.active
        assert ledger.fits((0.0,), 0, 1e9)

    def test_science_volume_is_the_targets(self, config: MissionConfig) -> None:
        ledger = RecorderLedger(config, 0.0, 1000.0, [], start_volume_gb=0.0)
        own = PlanEntry()
        bright = PlanEntry(science_data={"camera": ScienceData(rate_gbps=0.5)})
        assert ledger.science_volume(own, 60.0) == pytest.approx(7.0)
        assert ledger.science_volume(bright, 60.0) == pytest.approx(30.0)

    def test_initial_volumes_charge_entries_to_their_segments(
        self, config: MissionConfig
    ) -> None:
        ledger = RecorderLedger(
            config, 0.0, 1000.0, [_contact(500.0, 600.0, "ONE")], start_volume_gb=0.0
        )
        entries = [
            PlanEntry(begin=0, end=100, collection_begin=10, collection_end=70),
            PlanEntry(begin=700, end=800, collection_begin=710, collection_end=720),
            PlanEntry(begin=500, end=600),  # a contact: no collection
        ]
        assert ledger.initial_volumes(entries) == pytest.approx((7.0, 2.0))


class TestDownlink:
    def test_a_contact_downlinks_its_rate_for_its_length(self) -> None:
        config = _config(1, stations=True)
        assert contact_downlink_gb(config, "MAL", 480.0) == pytest.approx(4.8)
        assert contact_downlink_gb(config, "NOWHERE", 480.0) == 0.0

    def test_the_spacecraft_limits_the_rate(self) -> None:
        station = GroundStation(
            code="X",
            name="X",
            latitude_deg=0.0,
            longitude_deg=0.0,
            bands=[
                BandCapability(band="S", downlink_rate_mbps=10.0),
                BandCapability(band="X", downlink_rate_mbps=100.0),
            ],
        )
        comms = CommunicationsSystem(
            band_capabilities=[
                BandCapability(band="S", downlink_rate_mbps=4.0),
                BandCapability(band="X", downlink_rate_mbps=50.0),
            ]
        )
        assert station.effective_downlink_rate_mbps(None) == 100.0
        assert station.effective_downlink_rate_mbps(comms) == 50.0
        only_ka = CommunicationsSystem(
            band_capabilities=[BandCapability(band="Ka", downlink_rate_mbps=500.0)]
        )
        assert station.effective_downlink_rate_mbps(only_ka) is None


class TestPriorityScheduler:
    def _plan(self, limit: float | None | str = "default"):
        config = _recorder_config(limit)
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

    def test_the_recorder_stays_within_its_limit(self) -> None:
        config, scheduler, plan = self._plan()

        assert config.recorder.planning_limit_gb == pytest.approx(LIMIT)
        assert all(peak <= LIMIT + 1e-6 for peak in _peaks_of(scheduler))
        ditl = _execute(config, plan, HOURS, STEP)
        assert ditl.validate_plan_matches_execution() == []
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6
        assert max(ditl.recorder_volume_gb) > 0.5 * LIMIT

    def test_without_the_limit_the_recorder_goes_into_the_red(self) -> None:
        config, _, plan = self._plan(None)

        ditl = _execute(config, plan, HOURS, STEP)
        assert max(ditl.recorder_volume_gb) > LIMIT

    def test_high_merit_high_data_science_is_placed_first(self) -> None:
        """The data-hungry target is not kept out by its volume: it outranks
        the frugal one, so it takes what room there is before the first
        contact, and the frugal one fills the rest."""
        _, _, plan = self._plan()
        first = _first_contact(plan)
        before = [e for e in _science(plan) if float(e.begin) < first]

        assert sum(e.obsid == 1 for e in before) == 1
        # 18 Gb, less 12 for the hungry snapshot and 2.9 of housekeeping over
        # the 8 hours to the contact, takes two 1.2 Gb frugal snapshots.
        assert sum(e.obsid == 2 for e in before) == 2
        assert any(e.obsid == 1 and float(e.begin) > first for e in _science(plan))

    def test_snapshots_are_not_shortened_to_fit(self) -> None:
        _, _, plan = self._plan()
        hungry = [e for e in _science(plan) if e.obsid == 1]
        assert len(hungry) == 2
        assert all(e.exposure == pytest.approx(20 * MIN) for e in hungry)

    def test_a_full_recorder_at_the_start_is_counted(self) -> None:
        config = _recorder_config()
        hungry = _data_target(config, 1, 105.0, 10.0, HIGH_RATE, minutes=20)
        scheduler = PriorityScheduler(
            config,
            [hungry],
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            step_size=STEP,
            start_volume_gb=10.0,
        )
        plan = scheduler.schedule()
        placed = _science(plan)
        assert len(placed) == 1
        assert float(placed[0].begin) > _first_contact(plan)

    def test_locked_entries_count_towards_the_recorder(self) -> None:
        config = _recorder_config()
        hungry = _data_target(config, 1, 105.0, 10.0, HIGH_RATE, minutes=20)
        locked = PlanEntry(
            config=config,
            obsid=7,
            ra=105.0,
            dec=10.0,
            roll=0.0,
            begin=T0 + 1 * HOUR,
            slewtime=180,
            end=T0 + 1 * HOUR + 180 + 20 * MIN,
            collection_begin=T0 + 1 * HOUR + 180,
            collection_end=T0 + 1 * HOUR + 180 + 20 * MIN,
            science_data={"Default Instrument": ScienceData(rate_gbps=HIGH_RATE)},
        )
        scheduler = PriorityScheduler(
            config,
            [hungry],
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            step_size=STEP,
            locked=[locked],
        )
        plan = scheduler.schedule()
        assert scheduler._volumes[0] == pytest.approx(12.0)
        placed = [e for e in _science(plan) if e.obsid == 1]
        assert len(placed) == 1
        assert float(placed[0].begin) > _first_contact(plan)


class TestLocalSearch:
    def test_the_search_keeps_the_recorder_within_its_limit(self) -> None:
        config = _recorder_config()
        targets = [
            _data_target(config, 1, 105.0, 10.0, HIGH_RATE, merit=90, minutes=200),
            _data_target(config, 2, 110.0, 10.0, LOW_RATE, merit=50, minutes=200),
            _data_target(config, 3, 115.0, 5.0, LOW_RATE, merit=60, minutes=100),
        ]
        scheduler = LocalSearchScheduler(
            config,
            targets,
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            step_size=STEP,
            max_iterations=400,
            time_limit=60.0,
        )
        plan = scheduler.schedule()

        assert scheduler.score >= scheduler.initial_score
        best = scheduler._best.final
        peaks = scheduler._ledger.peaks(best.volumes)
        assert all(peak <= LIMIT + 1e-6 for peak in peaks)
        # The spliced volumes are what decoding from the start gives.
        fresh = scheduler._decode(scheduler._best.sequence).final
        assert fresh.volumes == pytest.approx(best.volumes)
        ditl = _execute(config, plan, HOURS, STEP)
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6


class TestRollingHorizon:
    def test_each_replan_starts_from_the_recorders_volume(self) -> None:
        config = _recorder_config()
        config.recorder.current_volume_gb = 7.5
        starts: list[float] = []

        class Recording(PriorityScheduler):
            def __init__(self, *args: object, **kwargs: object) -> None:
                starts.append(float(kwargs["start_volume_gb"]))  # type: ignore[arg-type]
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

        assert starts[0] == 7.5
        assert len(starts) >= 3
        assert starts[1] != starts[0]
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6
        assert ACSMode.PASS in {hk.acs_mode for hk in ditl.telemetry.housekeeping}

    def test_a_replan_counts_what_committed_activities_still_generate(self) -> None:
        """A new plan starts after a commit lead time; the recorder volume it
        starts from includes what the committed activities generate until then."""
        config = _recorder_config()
        starts: list[float] = []
        volumes: list[float] = []

        class Recording(PriorityScheduler):
            def __init__(self, *args: object, **kwargs: object) -> None:
                starts.append(float(kwargs["start_volume_gb"]))  # type: ignore[arg-type]
                volumes.append(float(config.recorder.current_volume_gb))
                super().__init__(*args, **kwargs)  # type: ignore[arg-type]

        targets = [
            _data_target(config, 1, 105.0, 10.0, HIGH_RATE, merit=90, minutes=200),
            _data_target(config, 2, 110.0, 10.0, LOW_RATE, merit=50, minutes=200),
        ]
        ditl = RollingHorizonDITL(
            config,
            targets,
            begin=BEGIN,
            end=BEGIN + timedelta(hours=HOURS),
            horizon=timedelta(hours=6),
            replan_interval=timedelta(hours=2),
            commit_lead_time=timedelta(minutes=30),
            scheduler=Recording,
        )
        ditl.step_size = STEP
        assert ditl.calc()

        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6
        # At least one replan started from more than was on the recorder when
        # it was built: what the committed snapshot went on to generate.
        assert any(s > v + 1.0 for s, v in zip(starts, volumes, strict=True))
