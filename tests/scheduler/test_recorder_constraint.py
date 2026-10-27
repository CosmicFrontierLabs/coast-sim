"""Plans keep the onboard recorder within its planning limit.

The ledger's arithmetic is tested in test_consumable.py; here it is the
recorder's mapping onto it, and the schedulers' use of it.
"""

from datetime import timedelta
from pathlib import Path

import pytest

from conops import create_ditl
from conops.common import ObsType
from conops.common.enums import ACSMode
from conops.config import (
    BandCapability,
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
from conops.ditl import QueueDITL, RollingHorizonDITL
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


def _recorder_config(
    limit: float | None | str = "default", *, stations: bool = True
) -> MissionConfig:
    """The scenario with contacts, a small recorder and some housekeeping."""
    config = _config(HOURS, stations=stations)
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

    def test_a_disabled_limit_survives_a_save_and_load(self, tmp_path: Path) -> None:
        """A YAML save leaves out None fields, but a disabled limit is a
        setting: left out, it would load as the red threshold."""
        config = MissionConfig(recorder=OnboardRecorder(planning_limit=None))
        dumped = config.model_dump(mode="json", exclude_none=True)
        assert "planning_limit" in dumped["recorder"]
        assert dumped["recorder"]["planning_limit"] is None
        path = tmp_path / "config.yaml"
        config.to_yaml_file(str(path))
        loaded = MissionConfig.from_yaml_file(str(path))
        assert loaded.recorder.planning_limit is None
        assert loaded.recorder.planning_limit_gb is None
        kept = MissionConfig(recorder=OnboardRecorder(planning_limit=0.5))
        kept.to_yaml_file(str(path))
        assert MissionConfig.from_yaml_file(str(path)).recorder.planning_limit == 0.5


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

    def test_the_slew_into_a_contact_adds_the_pass_rates_excess(
        self, config: MissionConfig
    ) -> None:
        """ACS runs the slew into a contact in PASS mode: the 60 s slew adds
        the difference between the rates, with the contact's other entries,
        in the segment it starts in; the ledger's own accrual is nominal."""
        contact = PlanEntry(
            obstype=ObsType.GSP,
            station="TWENTY",
            begin=140.0,
            slewtime=60,
            end=300.0,
            contact_begin=200.0,
            contact_end=300.0,
        )
        ledger = RecorderLedger(config, 0.0, 1000.0, [contact], start_volume_gb=0.0)
        assert ledger.bounds == [0.0, 200.0, 1000.0]
        assert ledger.housekeeping_gb == pytest.approx([0.2, 0.7])
        assert ledger.contact_housekeeping_gb == pytest.approx([0.0, 0.3])
        assert ledger.ingress_volume(contact) == pytest.approx(0.12)
        assert ledger.ingress_segment(contact) == 0
        assert ledger.initial_volumes([contact]) == pytest.approx((0.12, 0.0))
        # As it would be flown from elsewhere, longer.
        assert ledger.ingress_volume(contact, begin=100.0, slewtime=100.0) == (
            pytest.approx(0.2)
        )
        # The wait on the track before the contact is not in PASS mode.
        assert ledger.ingress_volume(contact, begin=100.0, slewtime=30.0) == (
            pytest.approx(0.06)
        )
        # Only the part inside the horizon.
        later = RecorderLedger(config, 170.0, 1000.0, [contact], start_volume_gb=0.0)
        assert later.ingress_volume(contact) == pytest.approx(0.06)

    def test_on_the_step_grid_each_step_counts_whole_in_its_mode(
        self, config: MissionConfig
    ) -> None:
        """With 60 s steps from 0: the slew from 120 to 200 occupies the
        steps at 120 and 180, so 120 s at the PASS rate; the link from 200
        to 300 occupies the step at 240, so 60 s at the PASS rate, and
        downlinks for its own 100 s; the horizon's steps are the 17 up to 960."""
        contact = PlanEntry(
            obstype=ObsType.GSP,
            station="TWENTY",
            begin=120.0,
            slewtime=80,
            end=300.0,
            contact_begin=200.0,
            contact_end=300.0,
        )
        ledger = RecorderLedger(
            config, 0.0, 1000.0, [contact], start_volume_gb=0.0, step_size=60
        )
        assert ledger.bounds == [0.0, 240.0, 1020.0]
        assert ledger.downlink_gb == pytest.approx([0.0, 20.0])
        assert ledger.housekeeping_gb == pytest.approx([0.24, 0.72])
        assert ledger.contact_housekeeping_gb == pytest.approx([0.0, 0.18])
        assert ledger.ingress_volume(contact) == pytest.approx(0.24)
        assert [ledger.segment(t) for t in (0, 239, 240, 1000)] == [0, 0, 1, 1]
        # The grid passes through the origin, not the horizon's start.
        shifted = RecorderLedger(
            config, 30.0, 1000.0, [contact], start_volume_gb=0.0, step_size=60
        )
        assert shifted.bounds == [30.0, 210.0, 1050.0]
        from_origin = RecorderLedger(
            config,
            30.0,
            1000.0,
            [contact],
            start_volume_gb=0.0,
            step_size=60,
            origin=0.0,
        )
        assert from_origin.bounds == [60.0, 240.0, 1020.0]

    def test_a_replan_during_a_contact_downlinks_only_what_is_left(
        self, config: MissionConfig
    ) -> None:
        """A 90 s link at 200 Mbps is 18 Gb; a horizon starting 60 s in
        gets the 30 s left, 6 Gb, though on the step grid the link occupies
        the two steps from 0 to 120."""
        ledger = RecorderLedger(
            config,
            60.0,
            1000.0,
            [_contact(0.0, 90.0, "TWENTY")],
            start_volume_gb=0.0,
            step_size=60,
        )
        assert ledger.bounds == [60.0, 60.0, 1020.0]
        assert ledger.downlink_gb == pytest.approx([0.0, 6.0])
        assert ledger.contact_housekeeping_gb == pytest.approx([0.0, 0.09])

    def test_what_the_last_contact_step_generates_after_the_link_stays_on(
        self, config: MissionConfig
    ) -> None:
        """The link from 200 to 290 downlinks the step at 240's housekeeping
        only up to 290: the 10 s after it, still at the PASS rate, go on the
        recorder after the link, so a contact that empties the recorder
        leaves them."""
        ledger = RecorderLedger(
            config,
            0.0,
            1000.0,
            [_contact(200.0, 290.0, "TWENTY")],
            start_volume_gb=1.0,
            step_size=60,
        )
        assert ledger.bounds == [0.0, 240.0, 1020.0]
        assert ledger.downlink_gb == pytest.approx([0.0, 18.0])
        assert ledger.contact_housekeeping_gb == pytest.approx([0.0, 0.15])
        assert ledger.housekeeping_gb == pytest.approx([0.24, 0.73 + 0.02])
        assert ledger.levels((0.0, 0.0)) == pytest.approx([1.24, 0.75])

    def test_a_pass_rate_below_nominal_makes_a_negative_ingress(
        self, config: MissionConfig
    ) -> None:
        config.spacecraft_bus.data_generation = HousekeepingData(
            rate_gbps=0.001, rate_mode={"PASS": 0.0005}
        )
        contact = PlanEntry(
            obstype=ObsType.GSP,
            station="TWENTY",
            begin=140.0,
            slewtime=60,
            end=300.0,
            contact_begin=200.0,
            contact_end=300.0,
        )
        ledger = RecorderLedger(config, 0.0, 1000.0, [contact], start_volume_gb=0.0)
        assert ledger.ingress_volume(contact) == pytest.approx(-0.03)
        assert ledger.initial_volumes([contact]) == pytest.approx((-0.03, 0.0))

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

    def test_a_snapshot_that_reaches_the_limit_exactly_is_refused(self) -> None:
        """The red alert fires at the limit, so a plan stops short of it:
        with no housekeeping, a 20-minute snapshot of exactly 18 Gb on an
        empty recorder fits nowhere."""
        config = _recorder_config()
        config.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0.0)
        exact = _data_target(config, 1, 105.0, 10.0, LIMIT / (20 * MIN), minutes=20)
        scheduler = PriorityScheduler(
            config, [exact], BEGIN, BEGIN + timedelta(hours=HOURS), step_size=STEP
        )
        plan = scheduler.schedule()
        assert not _science(plan)
        assert all(peak < LIMIT for peak in _peaks_of(scheduler))

    def test_the_slew_into_a_contact_counts_at_the_pass_rate(self) -> None:
        """Housekeeping only in PASS mode, at 1 Mbps, and a snapshot just
        under the limit: counted at the nominal rate, the slew into the first
        contact would let the recorder reach the red before acquisition."""
        config = _recorder_config()
        config.spacecraft_bus.data_generation = HousekeepingData(
            rate_gbps=0.0, rate_mode={"PASS": 0.001}
        )
        nearly = _data_target(
            config, 1, 105.0, 10.0, (LIMIT - 0.05) / (20 * MIN), minutes=20
        )
        scheduler = PriorityScheduler(
            config, [nearly], BEGIN, BEGIN + timedelta(hours=HOURS), step_size=STEP
        )
        plan = scheduler.schedule()
        placed = _science(plan)
        assert len(placed) == 1
        assert float(placed[0].begin) > _first_contact(plan)
        ditl = _execute(config, plan, HOURS, STEP)
        assert ditl.validate_plan_matches_execution() == []
        assert max(ditl.recorder_volume_gb) < LIMIT
        assert max(ditl.recorder_alert) < 2

    def _near_limit(self, rate: float):
        """A snapshot of 20 minutes at ``rate`` with housekeeping at
        0.1 Mbps, and 1 Mbps in PASS mode, on a recorder planned to 15 Gb."""
        config = _recorder_config()
        config.recorder.capacity_gb = 50.0 / 3.0
        config.spacecraft_bus.data_generation = HousekeepingData(
            rate_gbps=0.0001, rate_mode={"PASS": 0.001}
        )
        target = _data_target(config, 1, 105.0, 10.0, rate, minutes=20)
        scheduler = PriorityScheduler(
            config, [target], BEGIN, BEGIN + timedelta(hours=HOURS), step_size=STEP
        )
        return config, scheduler, scheduler.schedule()

    def test_the_slew_into_a_contact_is_counted_as_executed(self) -> None:
        """Counted for its own length at the PASS rate, the slew into the
        first contact let a snapshot be placed that put the recorder over
        the limit when executed: the simulation generates housekeeping for
        whole steps in the mode each starts in, so the 151 s slew counts for
        three 60 s steps. Counted that way, the plan executes as predicted."""
        config, scheduler, plan = self._near_limit(0.009996)
        limit = config.recorder.planning_limit_gb
        assert limit == pytest.approx(15.0)
        assert all(peak < limit for peak in _peaks_of(scheduler))
        ditl = _execute(config, plan, HOURS, STEP)
        assert ditl.validate_plan_matches_execution() == []
        assert max(ditl.recorder_volume_gb) < limit
        assert max(ditl.recorder_alert) < 2

    def test_the_predicted_peak_is_the_executed_one(self) -> None:
        config, scheduler, plan = self._near_limit(0.0099)
        placed = _science(plan)
        first = _first_contact(plan)
        assert len(placed) == 1 and float(placed[0].begin) < first
        ditl = _execute(config, plan, HOURS, STEP)
        assert ditl.validate_plan_matches_execution() == []
        executed = max(
            v
            for v, t in zip(ditl.recorder_volume_gb, ditl.utime, strict=True)
            if t < first
        )
        assert executed == pytest.approx(_peaks_of(scheduler)[0], abs=0.002)

    def test_the_volumes_kept_while_planning_are_those_of_the_plan(self) -> None:
        """Snapshots placed before a contact change the slew into it; what
        the ledger holds for it must be what the final timeline's is."""
        _, scheduler, _ = self._plan()
        entries = [block.entry for block in scheduler.timeline]
        assert scheduler._volumes == pytest.approx(
            scheduler._ledger.initial_volumes(entries)
        )
        assert any(
            block.is_pass and block.entry.slewtime for block in scheduler.timeline
        )

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

    def test_a_recorder_already_over_the_limit_is_reported(self) -> None:
        """Nothing placed can bring the recorder down; the plan says so and
        places science only after a contact has."""
        config = _recorder_config()
        hungry = _data_target(config, 1, 105.0, 10.0, LOW_RATE, minutes=20)
        scheduler = PriorityScheduler(
            config,
            [hungry],
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            step_size=STEP,
            start_volume_gb=LIMIT + 1.0,
        )
        plan = scheduler.schedule()
        assert any(
            "its planning limit being" in e.description for e in scheduler.log.events
        )
        first = _first_contact(plan)
        assert all(float(e.begin) > first for e in _science(plan))

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
        # The spliced volumes are what decoding from the start gives, and
        # what the decoded timeline, with its slews into contacts, comes to.
        fresh = scheduler._decode(scheduler._best.sequence).final
        assert fresh.volumes == pytest.approx(best.volumes)
        entries = [block.entry for block in best.timeline]
        assert best.volumes == pytest.approx(scheduler._ledger.initial_volumes(entries))
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


class TestDispatch:
    def _targets(self, config: MissionConfig) -> list[Pointing]:
        """Twenty-minute snapshots: 12 Gb each for the hungry target, 1.2 Gb
        for the frugal ones, on an 18 Gb limit."""
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

    def test_dispatch_selects_only_what_the_recorder_can_hold(self) -> None:
        """Queue dispatch passes over a snapshot the recorder cannot hold from
        its fill now, so the executed recorder stays below the planning limit
        and never goes into the red, and the hungry target still gets its
        snapshots once the contacts have made room."""
        config = _recorder_config()
        ditl = create_ditl(
            config, self._targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        ditl.step_size = STEP
        assert ditl.calc()

        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6
        assert max(ditl.recorder_alert) < 2
        observed = {hk.obsid for hk in ditl.telemetry.housekeeping if hk.obsid}
        assert {1, 2, 3} <= observed
        assert any(
            "passed over" in e.description
            for e in ditl.log.events
            if e.event_type == "QUEUE"
        )

    def test_without_the_limit_dispatch_fills_the_recorder(self) -> None:
        config = _recorder_config(None)
        ditl = create_ditl(
            config, self._targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert max(ditl.recorder_volume_gb) > LIMIT

    @staticmethod
    def _collected(ditl: QueueDITL, obsid: int) -> float:
        """Seconds of science the run collected on ``obsid``."""
        return sum(
            hk.collection_seconds or 0.0
            for hk in ditl.telemetry.housekeeping
            if hk.obsid == obsid
        )

    def test_a_shortened_snapshot_is_executed_as_shortened(self) -> None:
        """A snapshot the recorder holds part of, more than its minimum, is
        shortened to what it holds, and the dispatcher executes it shortened
        rather than whole: a 40-minute snapshot at 0.01 Gb/s, 24 Gb, on the
        18 Gb limit less two hours of housekeeping with no contact in them,
        17.28 Gb, collects 1728 s. The rest waits for a contact that never
        comes, and the recorder stays under the limit."""
        config = _recorder_config()
        target = _data_target(
            config, 1, 105.0, 10.0, HIGH_RATE, merit=90, minutes=40, ss_min=1
        )
        ditl = create_ditl(config, [target], BEGIN, BEGIN + timedelta(hours=2))
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        assert not any(float(p.begin) < ditl.uend for p in ditl.acs.passrequests.passes)

        collected = self._collected(ditl, 1)
        assert 60.0 <= collected < 2400.0
        assert collected == pytest.approx(1728.0, abs=STEP)
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6
        assert max(ditl.recorder_alert) < 2

    def test_housekeeping_is_reserved_at_the_executed_mode_rates(self) -> None:
        """The slew into a contact, which ACS flies in PASS mode, generates at
        the PASS rate, and that counts against the room: a snapshot the limit
        holds with housekeeping at the nominal rate, but not with the ingress
        at ten times it, is shortened, so the recorder stays under the limit
        and the contact is kept rather than lost to safe mode."""
        config = _recorder_config(0.75)
        config.spacecraft_bus.data_generation = HousekeepingData(
            rate_gbps=0.0001, rate_mode={ACSMode.PASS: 0.001}
        )
        target = _data_target(
            config, 1, 105.0, 10.0, 0.009996, merit=90, minutes=20, ss_min=1
        )
        ditl = create_ditl(config, [target], BEGIN, BEGIN + timedelta(hours=HOURS))
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()

        assert max(ditl.recorder_volume_gb) <= 0.75 * CAPACITY + 1e-6
        assert max(ditl.recorder_alert) < 2
        assert ACSMode.SAFE not in {hk.acs_mode for hk in ditl.telemetry.housekeeping}
        assert ACSMode.PASS in {hk.acs_mode for hk in ditl.telemetry.housekeeping}

    def test_science_mode_housekeeping_counts_against_the_room(self) -> None:
        """What the visit generates at the SCIENCE rate beyond the nominal
        one counts too: without contacts, a snapshot whose science alone
        just fits the room is shortened by what its housekeeping adds."""
        config = _recorder_config(stations=False)
        config.spacecraft_bus.data_generation = HousekeepingData(
            rate_gbps=0.0001, rate_mode={ACSMode.SCIENCE: 0.002}
        )
        # 17.28 Gb in 20 minutes: the room, at the nominal rate, for two hours.
        target = _data_target(
            config, 1, 105.0, 10.0, 0.0144, merit=90, minutes=20, ss_min=1
        )
        ditl = create_ditl(config, [target], BEGIN, BEGIN + timedelta(hours=2))
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()

        collected = self._collected(ditl, 1)
        assert 60.0 <= collected < 1200.0
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6
        assert max(ditl.recorder_alert) < 2

    def test_the_reserve_ends_with_the_run(self) -> None:
        """A contact after the run does not extend the reserve: with the
        14-hour ephemeris and a two-hour run, the first contact is after
        the run, so the room at the start is the limit less two hours of
        housekeeping, not eight, and a 27-minute snapshot at 0.01 Gb/s,
        16.2 Gb, is executed whole."""
        config = _recorder_config()
        target = _data_target(config, 1, 105.0, 10.0, HIGH_RATE, merit=90, minutes=27)
        ditl = create_ditl(config, [target], BEGIN, BEGIN + timedelta(hours=2))
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()

        passes = ditl.acs.passrequests.passes
        assert any(float(p.begin) > ditl.uend for p in passes)
        assert not any(float(p.begin) < ditl.uend for p in passes)
        room = ditl._recorder_room_gb(T0, 0.0, 0.0, 0.0)
        assert room == pytest.approx(
            LIMIT
            - ditl.recorder.current_volume_gb
            - config.housekeeping_generated(2 * HOUR)
        )
        assert self._collected(ditl, 1) == pytest.approx(27 * MIN, abs=STEP)
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6

    def test_the_slew_charged_is_the_one_flown(self) -> None:
        """Selection charges the candidate's slew at the SLEWING rate for the
        slew it will fly, not a geometric estimate that can be a step
        shorter, and admission checks the limit again with the slew it
        flies: with SLEWING housekeeping fifty times the nominal rate, a
        target waiting until just before a contact is shortened to what the
        recorder holds and the run stays under the limit."""
        config = _recorder_config()
        config.spacecraft_bus.data_generation = HousekeepingData(
            rate_gbps=0.0001, rate_mode={ACSMode.SLEWING: 0.005}
        )
        target = _data_target(
            config, 1, 105.0, 10.0, HIGH_RATE, merit=90, minutes=40, ss_min=1
        )
        target.earliest_start = T0 + 29100.0
        ditl = create_ditl(config, [target], BEGIN, BEGIN + timedelta(hours=HOURS))
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()

        assert self._collected(ditl, 1) >= 60.0
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6
        assert ACSMode.SAFE not in {hk.acs_mode for hk in ditl.telemetry.housekeeping}

    @pytest.mark.parametrize(
        ("mode", "nominal", "hours", "stations"),
        [(ACSMode.PASS, 0.0001, HOURS, True), (ACSMode.SCIENCE, 0.001, 2, False)],
    )
    def test_a_mode_rate_below_nominal_credits_no_saving(
        self, mode: ACSMode, nominal: float, hours: int, stations: bool
    ) -> None:
        """A mode that generates less than the nominal rate saves data only
        over the seconds it is certain to run, and the reserve counts spans
        by their upper bounds, so no saving is credited: a snapshot sized to
        the room is not over-admitted on the strength of a saving that
        execution need not deliver."""
        config = _recorder_config(stations=stations)
        config.spacecraft_bus.data_generation = HousekeepingData(
            rate_gbps=nominal, rate_mode={mode: 0.0}
        )
        target = _data_target(
            config, 1, 105.0, 10.0, HIGH_RATE, merit=90, minutes=40, ss_min=1
        )
        ditl = create_ditl(config, [target], BEGIN, BEGIN + timedelta(hours=hours))
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()

        assert self._collected(ditl, 1) >= 60.0
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6
        assert ACSMode.SAFE not in {hk.acs_mode for hk in ditl.telemetry.housekeeping}

    def test_the_time_waiting_is_reserved_at_the_idle_rate(self) -> None:
        """Outside the candidate's slew and science the dispatcher idles,
        and the simulation generates housekeeping at the IDLE rate then:
        with the IDLE rate ten times the nominal one and no contact to
        empty the recorder, the room counts the wait at the IDLE rate and
        the snapshot is shortened so the run stays under the limit."""
        config = _recorder_config(stations=False)
        config.spacecraft_bus.data_generation = HousekeepingData(
            rate_gbps=0.0001, rate_mode={ACSMode.IDLE: 0.001}
        )
        target = _data_target(
            config, 1, 105.0, 10.0, HIGH_RATE, merit=90, minutes=40, ss_min=1
        )
        ditl = create_ditl(config, [target], BEGIN, BEGIN + timedelta(hours=2))
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()

        assert 60.0 <= self._collected(ditl, 1) < 2400.0
        assert max(ditl.recorder_volume_gb) <= LIMIT + 1e-6
        assert ACSMode.SAFE not in {hk.acs_mode for hk in ditl.telemetry.housekeeping}
