"""create_ditl builds the simulation a configuration's scheduler section selects."""

import importlib.util
from datetime import timedelta

import pytest
from rust_ephem import TLEEphemeris

from conops import DITL, QueueDITL, RollingHorizonDITL, create_ditl
from conops.common import ObsType
from conops.config import (
    MissionConfig,
    Payload,
    ReplanSettings,
    SchedulerConfig,
    SchedulerKind,
    SchedulerMode,
    TargetConfig,
    Telescope,
)
from conops.config.observation_categories import ObservationCategory
from conops.schedulers import SCHEDULERS, LocalSearchScheduler, PriorityScheduler
from conops.targets import Pointing
from conops.targets.merit import MeritModel

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 2
END = BEGIN + timedelta(hours=HOURS)


def _setup(scheduler: SchedulerConfig) -> tuple[MissionConfig, list[Pointing]]:
    config = _config(HOURS)
    config.scheduler = scheduler
    targets = [
        _target(config, 100 + i, ra, dec, merit=90 - 10 * i, minutes=40)
        for i, (ra, dec) in enumerate(PATCH)
    ]
    return config, targets


def _run(ditl: QueueDITL | DITL | RollingHorizonDITL) -> None:
    assert ditl.calc()
    assert ditl.validate_plan_matches_execution() == []


class TestDispatch:
    def test_a_target_can_name_a_plain_instrument(self) -> None:
        """A plain instrument has no mounting, so the queued target keeps its
        name and its science coordinates are the body attitude."""
        from conops.config import Instrument

        config, targets = _setup(SchedulerConfig())
        config.payload.instruments = [
            Instrument(name="Camera"),
            Telescope(name="Scope", boresight=(0.0, 0.0, 1.0)),
        ]
        targets[0].instrument_name = "Camera"

        ditl = create_ditl(config, targets[:1], BEGIN, END)

        assert isinstance(ditl, QueueDITL)
        queued = ditl.queue.targets[0]
        assert queued.instrument_name == "Camera"
        assert queued.science_telescope() is None
        assert not queued.uses_mounted_attitude()
        assert queued.target_body_attitude(0.0) == (
            targets[0].ra,
            targets[0].dec,
            0.0,
        )

    def test_queues_every_target_with_its_settings(self) -> None:
        config, targets = _setup(SchedulerConfig())
        targets[0].deadline = T0 + 30 * MIN
        targets[1].earliest_start = T0 + 20 * MIN
        # Merit is what counts, not the legacy fom, which is not kept in step.
        targets[2].merit = 33.0

        ditl = create_ditl(config, targets, BEGIN, END)

        assert isinstance(ditl, QueueDITL)
        queued = {int(t.obsid): t for t in ditl.queue.targets}
        assert set(queued) == {int(t.obsid) for t in targets}
        assert [queued[int(t.obsid)].merit for t in targets] == [
            t.merit for t in targets
        ]
        assert queued[102].merit == 33.0
        assert queued[100].deadline == T0 + 30 * MIN
        assert queued[101].earliest_start == T0 + 20 * MIN
        assert queued[100].exptime == targets[0].exptime
        _run(ditl)

    def test_queued_targets_keep_their_observation_history(self) -> None:
        monitor_category = ObservationCategory(
            name="Monitor", obsid_min=1, obsid_max=2, cadence_seconds=HOUR
        )
        config = _config(
            HOURS,
            weights=TargetConfig(cadence_weight=50),
            categories=[monitor_category],
        )
        monitor = _target(config, 1, *PATCH[0], merit=60, minutes=40)
        # Observed for 5 minutes, ending 10 minutes before the run starts.
        monitor.record_collection(T0 - 10 * MIN, 5 * MIN)

        ditl = create_ditl(config, [monitor], BEGIN, END)

        assert isinstance(ditl, QueueDITL)
        (queued,) = ditl.queue.targets
        assert queued is not monitor
        assert queued.collected_seconds == 5 * MIN
        assert queued.last_collection_time == T0 - 10 * MIN
        assert queued.exptime == 35 * MIN
        assert queued.exporig == 40 * MIN
        # Ten minutes into its hourly cadence, not due as if never observed.
        cadence = MeritModel(config).value_terms(queued, T0).cadence
        assert cadence == pytest.approx(50 * (10 * MIN) / HOUR)

    def test_a_target_from_an_earlier_run_is_not_left_invisible(self) -> None:
        """A target not visible during one run is ranked by its merit in the
        next, not by the penalty the queue gave it while hidden."""
        config = _config(HOURS)
        target = _target(config, 1, 30, 40, merit=60, minutes=40, snapshot=5, ss_min=1)
        split = BEGIN + timedelta(minutes=10)
        first = create_ditl(config, [target], BEGIN, split)
        _run(first)
        assert isinstance(first, QueueDITL)
        (previous,) = first.queue.targets
        assert previous.collected_seconds == 0  # Not visible in the first run.
        assert previous.merit == 60

        second = create_ditl(config, first.queue.targets, split, END)
        _run(second)

        assert isinstance(second, QueueDITL)
        (queued,) = second.queue.targets
        assert queued.collected_seconds == 40 * MIN

    def test_a_target_from_an_earlier_run_does_not_keep_its_bonus(self) -> None:
        """A target selected with a cadence bonus is ranked by its merit in the
        next run, not by the value it was selected with."""
        monitor_category = ObservationCategory(
            name="Monitor", obsid_min=1, obsid_max=2, cadence_seconds=HOUR
        )
        config = _config(
            HOURS,
            weights=TargetConfig(cadence_weight=1),
            categories=[monitor_category],
        )
        monitor = _target(config, 1, *PATCH[0], merit=60, minutes=60, snapshot=5)
        monitor.record_collection(T0 - HOUR, 5 * MIN)
        split = BEGIN + timedelta(minutes=8)
        first = create_ditl(config, [monitor], BEGIN, split)
        _run(first)
        assert isinstance(first, QueueDITL)
        (previous,) = first.queue.targets
        assert previous.merit_breakdown is not None
        assert previous.merit_breakdown.value == pytest.approx(61)
        assert previous.merit == 60
        fresh = _target(config, 3, *PATCH[1], merit=60.5, minutes=20, snapshot=5)

        second = create_ditl(config, [previous, fresh], split, END)
        _run(second)

        assert isinstance(second, QueueDITL)
        science = [e for e in second.plan if e.obstype != ObsType.GSP]
        assert int(science[0].obsid) == 3

    def test_a_target_made_with_only_a_merit_is_ranked_by_it(self) -> None:
        """Not by the default of fom, the legacy name of merit."""
        config, _ = _setup(SchedulerConfig())
        lower = Pointing(ra=PATCH[0][0], dec=PATCH[0][1], obsid=100, merit=50)
        higher = Pointing(ra=PATCH[1][0], dec=PATCH[1][1], obsid=101, merit=60)
        for target in (lower, higher):
            target.exptime = 20 * MIN

        ditl = create_ditl(config, [lower, higher], BEGIN, END)
        _run(ditl)

        science = [e for e in ditl.plan if e.obstype != ObsType.GSP]
        assert [int(e.obsid) for e in science[:2]] == [101, 100]

    def test_a_target_made_without_a_config_is_queued(self) -> None:
        config, _ = _setup(SchedulerConfig())
        bare = Pointing(ra=PATCH[0][0], dec=PATCH[0][1], obsid=100, merit=50)
        bare.exptime = 20 * MIN

        ditl = create_ditl(config, [bare], BEGIN, END)

        assert isinstance(ditl, QueueDITL)
        (queued,) = ditl.queue.targets
        assert queued.windows
        _run(ditl)

    def test_queued_targets_use_the_simulation_ephemeris(self) -> None:
        """Targets made with the configuration's ephemeris, run with a new one
        for the next day, are visible during the run, not the day before."""
        config, targets = _setup(SchedulerConfig())
        begin = BEGIN + timedelta(days=1)
        end = begin + timedelta(hours=HOURS)
        ephem = TLEEphemeris(
            tle="examples/example.tle", begin=begin, end=end, step_size=60
        )

        ditl = create_ditl(config, targets, begin, end, ephem=ephem)

        assert isinstance(ditl, QueueDITL)
        for queued in ditl.queue.targets:
            assert queued.ephem is ephem
            assert queued.windows
            assert all(
                begin.timestamp() <= start and stop <= end.timestamp()
                for start, stop in queued.windows
            )
        _run(ditl)

    def test_the_run_leaves_the_targets_passed_in_unchanged(self) -> None:
        config, targets = _setup(SchedulerConfig())

        ditl = create_ditl(config, targets, BEGIN, END)
        _run(ditl)

        assert isinstance(ditl, QueueDITL)
        assert sum(t.collected_seconds for t in ditl.queue.targets) > 0
        assert all(t.collected_seconds == 0 for t in targets)
        assert all(t.exptime == 40 * MIN for t in targets)


class TestPlanned:
    @pytest.mark.parametrize(
        ("scheduler", "name"),
        [
            (SchedulerConfig(mode=SchedulerMode.PLANNED), "priority"),
            (
                SchedulerConfig(
                    mode=SchedulerMode.PLANNED,
                    kind=SchedulerKind.LOCAL_SEARCH,
                    time_limit=600,
                    max_iterations=50,
                ),
                "local_search",
            ),
        ],
    )
    def test_builds_and_executes_the_configured_plan(
        self, scheduler: SchedulerConfig, name: str
    ) -> None:
        config, targets = _setup(scheduler)

        ditl = create_ditl(config, targets, BEGIN, END)

        assert isinstance(ditl, DITL)
        assert ditl.plan.metadata is not None
        assert ditl.plan.metadata["scheduler"] == name
        assert len(ditl.plan) > 0
        _run(ditl)

    @pytest.mark.skipif(
        importlib.util.find_spec("ortools") is None, reason="needs OR-Tools"
    )
    def test_cp_sat_scheduler(self) -> None:
        config, targets = _setup(
            SchedulerConfig(
                mode=SchedulerMode.PLANNED,
                kind=SchedulerKind.CP_SAT,
                solver_time_limit=3,
                workers=1,
                seed=1,
            )
        )

        ditl = create_ditl(config, targets, BEGIN, END)

        assert ditl.plan.metadata is not None
        assert ditl.plan.metadata["scheduler"] == "cp_sat"
        _run(ditl)

    def test_without_passes_the_plan_has_none(self) -> None:
        config, targets = _setup(
            SchedulerConfig(mode=SchedulerMode.PLANNED, include_passes=False)
        )

        ditl = create_ditl(config, targets, BEGIN, END)

        assert all(e.obstype.name != "GSP" for e in ditl.plan)
        _run(ditl)


class TestRolling:
    def test_passes_the_replanning_settings_through(self) -> None:
        config, targets = _setup(
            SchedulerConfig(
                mode=SchedulerMode.ROLLING,
                kind=SchedulerKind.LOCAL_SEARCH,
                include_passes=False,
                time_limit=600,
                max_iterations=20,
                replanning=ReplanSettings(
                    horizon_seconds=5400,
                    replan_interval_seconds=1800,
                    commit_lead_time_seconds=600,
                    rapid_replans=False,
                    allow_interrupts=False,
                ),
            )
        )

        ditl = create_ditl(config, targets, BEGIN, END)

        assert isinstance(ditl, RollingHorizonDITL)
        assert ditl.horizon == timedelta(minutes=90)
        assert ditl.replan_interval == timedelta(minutes=30)
        assert ditl.commit_lead_time == timedelta(minutes=10)
        assert not ditl.rapid_replans and not ditl.allow_interrupts
        assert not ditl.include_passes
        assert ditl.scheduler is LocalSearchScheduler
        assert ditl.scheduler_options == {
            "successor_retries": 3,
            "time_limit": 600,
            "max_iterations": 20,
        }
        _run(ditl)
        assert len(ditl.replans) == 4


class TestRuntimeBinding:
    """Targets loaded from JSON have no configuration attached; every mode
    gives them the simulation's, so they keep the telescope they ask for."""

    def _loaded_targets(
        self, scheduler: SchedulerConfig
    ) -> tuple[MissionConfig, list[Pointing]]:
        config, targets = _setup(scheduler)
        config.payload = Payload(
            instruments=[
                Telescope(name="Front", boresight=(1.0, 0.0, 0.0)),
                Telescope(name="Rear", boresight=(-1.0, 0.0, 0.0)),
            ]
        )
        for target in targets:
            target.instrument_name = "Rear"
        loaded = [Pointing.model_validate_json(t.model_dump_json()) for t in targets]
        assert all(target.config is None for target in loaded)
        return config, loaded

    @staticmethod
    def _instruments(ditl: DITL | RollingHorizonDITL) -> set[str | None]:
        return {e.instrument_name for e in ditl.plan if e.obstype != ObsType.GSP}

    def test_planned_targets_keep_their_telescope(self) -> None:
        config, targets = self._loaded_targets(
            SchedulerConfig(mode=SchedulerMode.PLANNED)
        )

        ditl = create_ditl(config, targets, BEGIN, END)

        assert isinstance(ditl, DITL)
        assert len(ditl.plan) > 0
        assert self._instruments(ditl) == {"Rear"}
        # The targets passed in are left as they were.
        assert all(target.config is None for target in targets)
        _run(ditl)

    def test_rolling_targets_keep_their_telescope(self) -> None:
        config, targets = self._loaded_targets(
            SchedulerConfig(mode=SchedulerMode.ROLLING, include_passes=False)
        )

        ditl = create_ditl(config, targets, BEGIN, END)
        _run(ditl)

        assert isinstance(ditl, RollingHorizonDITL)
        assert self._instruments(ditl) == {"Rear"}
        # Rolling replanning updates the targets passed in.
        assert sum(target.collected_seconds for target in targets) > 0


class TestDefaults:
    def test_horizon_defaults_to_the_ephemeris(self) -> None:
        config, targets = _setup(SchedulerConfig(mode=SchedulerMode.PLANNED))

        ditl = create_ditl(config, targets)

        assert ditl.begin == BEGIN
        assert ditl.end == END

    def test_requires_an_ephemeris(self) -> None:
        config, targets = _setup(SchedulerConfig())
        config.constraint.ephem = None

        with pytest.raises(ValueError, match="ephemeris"):
            create_ditl(config, targets, BEGIN, END)


def test_registry_covers_every_scheduler_kind() -> None:
    assert set(SCHEDULERS) == set(SchedulerKind)
    assert SCHEDULERS[SchedulerKind.PRIORITY] is PriorityScheduler
