"""PriorityPlanner follows the dynamic merit terms: cadence and completion deficit."""

from collections.abc import Mapping, Sequence
from datetime import timedelta

import pytest

from conops import DITL
from conops.common import ObsType
from conops.config import MissionConfig, TargetConfig
from conops.config.observation_categories import ObservationCategory
from conops.schedulers import PriorityPlanner
from conops.targets import Plan, PlanEntry, Pointing

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 4
MONITOR = ObservationCategory(
    name="Monitor", obsid_min=1, obsid_max=2, cadence_seconds=HOUR
)
PROGRAMS = [
    ObservationCategory(name="A", obsid_min=100, obsid_max=200, time_share=0.5),
    ObservationCategory(name="B", obsid_min=200, obsid_max=300, time_share=0.5),
]


def _monitor_config(cadence_weight: float) -> MissionConfig:
    return _config(
        HOURS,
        weights=TargetConfig(cadence_weight=cadence_weight),
        categories=[MONITOR],
    )


def _monitored(config: MissionConfig) -> list[Pointing]:
    """A cadence target wanting three ten-minute visits, and filler."""
    monitor = _target(config, 1, *PATCH[0], merit=60, minutes=30, snapshot=10)
    filler = [
        _target(config, 10 + i, ra, dec, merit=40, minutes=60, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH[1:])
    ]
    return [monitor, *filler]


def _programs_config(deficit_weight: float) -> MissionConfig:
    return _config(
        HOURS,
        weights=TargetConfig(completion_deficit_weight=deficit_weight),
        categories=PROGRAMS,
    )


def _programs(config: MissionConfig) -> list[Pointing]:
    """Program A outranks B, and either could fill the horizon alone."""
    a = [
        _target(config, 100 + i, ra, dec, merit=100, minutes=120, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH[:3])
    ]
    b = [
        _target(config, 200 + i, ra, dec, merit=60, minutes=120, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH[2:])
    ]
    return [*a, *b]


def _plan(
    config: MissionConfig,
    targets: Sequence[Pointing],
    **options: Mapping[int, float] | Sequence[PlanEntry],
) -> Plan:
    planner = PriorityPlanner(
        config,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=HOURS),
        **options,  # type: ignore[arg-type]
    )
    return planner.schedule()


def _visits(plan: Plan, obsid: int) -> list[tuple[float, float]]:
    return sorted(
        (float(e.collection_begin), float(e.collection_end))
        for e in plan
        if e.obsid == obsid
        and e.collection_begin is not None
        and e.collection_end is not None
    )


def _share(plan: Plan, low: int, high: int) -> float:
    seconds = {True: 0.0, False: 0.0}
    for e in plan:
        if e.obstype == ObsType.GSP or e.collection_begin is None:
            continue
        assert e.collection_end is not None
        span = float(e.collection_end) - float(e.collection_begin)
        seconds[low <= e.obsid < high] += span
    total = seconds[True] + seconds[False]
    return seconds[True] / total if total else 0.0


def _executes_exactly(config: MissionConfig, plan: Plan, hours: int = HOURS) -> bool:
    ditl = DITL(
        config=config,
        ephem=config.constraint.ephem,
        plan=Plan.model_validate_json(plan.model_dump_json()),
        begin=BEGIN,
        end=BEGIN + timedelta(hours=hours),
    )
    ditl.step_size = 60
    assert ditl.calc()
    return ditl.validate_plan_matches_execution() == []


class TestCadence:
    def test_visits_wait_for_the_cadence(self) -> None:
        config = _monitor_config(cadence_weight=50)

        plan = _plan(config, _monitored(config))

        visits = _visits(plan, 1)
        assert len(visits) == 3
        for (_, end), (begin, _) in zip(visits, visits[1:]):
            assert begin >= end + HOUR
        assert _executes_exactly(config, plan)

    def test_without_a_cadence_weight_visits_are_not_spaced(self) -> None:
        config = _monitor_config(cadence_weight=0)

        plan = _plan(config, _monitored(config))

        visits = _visits(plan, 1)
        assert any(
            begin < end + HOUR for (_, end), (begin, _) in zip(visits, visits[1:])
        )

    def test_the_last_collected_visit_counts(self) -> None:
        config = _monitor_config(cadence_weight=50)
        targets = _monitored(config)
        targets[0].record_collection(T0 - 30 * MIN, 60.0)

        plan = _plan(config, targets)

        first, _ = _visits(plan, 1)[0]
        assert first >= T0 + 30 * MIN


class TestCadenceOrdering:
    def test_a_recently_observed_target_outranks_filler_worth_less_than_its_visits(
        self,
    ) -> None:
        """Filler is worth more than the target's cadence value at the start of
        the horizon, but less than each of its visits, which wait until due."""
        hours = 3
        config = _config(
            hours, weights=TargetConfig(cadence_weight=50), categories=[MONITOR]
        )
        monitor = _target(config, 1, *PATCH[0], merit=60, minutes=30, snapshot=10)
        monitor.record_collection(T0 - 30 * MIN, 60.0)
        filler = [
            _target(config, 10 + i, ra, dec, merit=100, minutes=60, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH[1:])
        ]

        plan = PriorityPlanner(
            config, [monitor, *filler], BEGIN, BEGIN + timedelta(hours=hours)
        ).schedule()

        # The third visit is not due until too late to fit in the horizon.
        visits = _visits(plan, 1)
        assert len(visits) == 2
        assert visits[0][0] >= T0 + 30 * MIN
        for (_, end), (begin, _) in zip(visits, visits[1:]):
            assert begin >= end + HOUR
        assert all(e.merit == pytest.approx(110.0) for e in plan if int(e.obsid) == 1)
        assert _executes_exactly(config, plan, hours)


class TestCadenceWithLockedVisits:
    """Planned visits keep a cadence from locked visits of the same target."""

    @staticmethod
    def _locked(config: MissionConfig, start: float) -> PlanEntry:
        return PlanEntry(
            config=config,
            name="locked",
            ra=PATCH[0][0],
            dec=PATCH[0][1],
            roll=0.0,
            obsid=1,
            obstype=ObsType.AT,
            begin=start,
            end=start + 10 * MIN,
            collection_begin=start,
            collection_end=start + 10 * MIN,
            ss_min=60,
        )

    def test_the_next_visit_waits_a_cadence_after_a_locked_one(self) -> None:
        config = _monitor_config(cadence_weight=50)
        locked = self._locked(config, T0 + 10 * MIN)

        plan = _plan(config, _monitored(config), locked=[locked])

        visits = _visits(plan, 1)
        planned = [v for v in visits if v != (T0 + 10 * MIN, T0 + 20 * MIN)]
        assert planned
        assert all(begin >= T0 + 20 * MIN + HOUR for begin, _ in planned)
        assert _executes_exactly(config, plan)

    def test_visits_before_a_later_locked_one_end_a_cadence_before_it(
        self,
    ) -> None:
        config = _monitor_config(cadence_weight=50)
        lock_start = T0 + 90 * MIN
        locked = self._locked(config, lock_start)

        plan = _plan(config, _monitored(config), locked=[locked])

        planned = [
            v for v in _visits(plan, 1) if v != (lock_start, lock_start + 10 * MIN)
        ]
        before = [v for v in planned if v[0] < lock_start]
        assert before
        assert all(end + HOUR <= lock_start for _, end in before)
        assert all(
            begin >= lock_start + 10 * MIN + HOUR
            for begin, _ in planned
            if begin > lock_start
        )

    def test_a_shorter_visit_fits_before_a_locked_one(self) -> None:
        """A full snapshot would end too close to the locked visit; a shorter
        one ends a cadence before it."""
        hours = 2
        config = _config(
            hours,
            weights=TargetConfig(cadence_weight=50),
            categories=[MONITOR],
        )
        lock_start = T0 + 90 * MIN
        locked = self._locked(config, lock_start)
        monitor = _target(
            config, 1, *PATCH[0], merit=60, minutes=80, snapshot=40, ss_min=5
        )
        plan = PriorityPlanner(
            config,
            [monitor],
            BEGIN,
            BEGIN + timedelta(hours=hours),
            locked=[locked],
        ).schedule()

        before = [v for v in _visits(plan, 1) if v[0] < lock_start]
        assert before
        assert all(end + HOUR <= lock_start for _, end in before)
        assert _executes_exactly(config, plan, hours)


class TestCompletionDeficit:
    def test_shares_balance_programs(self) -> None:
        plain_config = _programs_config(deficit_weight=0)
        balanced_config = _programs_config(deficit_weight=100)

        plain = _plan(plain_config, _programs(plain_config))
        balanced = _plan(balanced_config, _programs(balanced_config))

        assert _share(balanced, 200, 300) > _share(plain, 200, 300) + 0.2
        assert _executes_exactly(balanced_config, balanced)

    def test_collected_science_counts_towards_shares(self) -> None:
        config = _programs_config(deficit_weight=100)
        targets = _programs(config)
        targets[0].record_collection(T0 - HOUR, 10 * HOUR)

        plan = _plan(config, targets)

        assert _share(plan, 200, 300) > 0.8


class TestCommittedObservations:
    """Observations committed outside the plan count as visits and shares."""

    def test_a_committed_visit_delays_the_next(self) -> None:
        config = _monitor_config(cadence_weight=50)

        plan = _plan(
            config,
            _monitored(config),
            reserved_visits={1: T0 + 20 * MIN},
        )

        first, _ = _visits(plan, 1)[0]
        assert first >= T0 + 80 * MIN

    def test_committed_seconds_count_towards_shares(self) -> None:
        plain_config = _programs_config(deficit_weight=100)
        reserved_config = _programs_config(deficit_weight=100)

        plain = _plan(plain_config, _programs(plain_config))
        reserved = _plan(
            reserved_config,
            _programs(reserved_config),
            reserved_seconds={100: HOUR},
        )

        # An hour of A already committed: the plan gives B more.
        assert _share(reserved, 200, 300) > _share(plain, 200, 300) + 0.05


def _rolling_cadence(monkeypatch: pytest.MonkeyPatch | None) -> list[float]:
    """Run a monitored target under rolling replanning; return its visit gaps."""
    from conops.ditl import RollingHorizonDITL

    if monkeypatch is not None:
        monkeypatch.setattr(
            RollingHorizonDITL, "_reserved_visits", staticmethod(lambda *_: {})
        )
    config = _monitor_config(cadence_weight=50)
    ditl = RollingHorizonDITL(
        config,
        _monitored(config),
        begin=BEGIN,
        end=BEGIN + timedelta(hours=HOURS),
        horizon=timedelta(hours=2),
        replan_interval=timedelta(minutes=30),
        # Long enough that a replan finds a visit committed but not collected.
        commit_lead_time=timedelta(minutes=30),
    )
    ditl.step_size = 60
    assert ditl.calc()
    # A visit is a run of consecutive steps collecting for the target.
    starts: list[float] = []
    last: float | None = None
    for record in ditl.telemetry.housekeeping:
        if record.obsid != 1 or not record.collection_seconds:
            continue
        utime = record.timestamp.timestamp()
        if last is None or utime - last > 90.0:
            starts.append(utime)
        last = utime
    return [b - a for a, b in zip(starts, starts[1:])]


class TestRollingCadence:
    def test_replans_keep_visits_apart(self) -> None:
        gaps = _rolling_cadence(None)

        assert gaps
        assert min(gaps) >= HOUR

    def test_without_committed_visits_a_replan_revisits_early(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guards the test above: it must fail if committed visits are ignored."""
        assert min(_rolling_cadence(monkeypatch)) < HOUR
