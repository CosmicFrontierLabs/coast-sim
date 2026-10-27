"""Time budgets on a category or an observation group: a cap over the run on
the science their targets collect, honoured by dispatch, the planners and the
allocator, and reported in the summaries. Groups also bound their members to
a window and prefer to keep them together."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from conops import QueueDITL, create_ditl
from conops.common import ObsType
from conops.config import ObservationGroup, ObservationGroups, TargetConfig
from conops.config.observation_categories import ObservationCategory
from conops.schedulers import LongRangeAllocator, PriorityScheduler
from conops.targets import ObservationSequence, Pointing, SequenceStep

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0, planner, science_entries
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 6
END = BEGIN + timedelta(hours=HOURS)
BUDGET = 50 * MIN
MEMBERS = [1, 2, 3]
CAPPED = ObservationCategory(
    name="Capped", obsid_min=1, obsid_max=10, time_budget_seconds=BUDGET
)


def _budgeted(source: str, hours: int = HOURS, **group):  # type: ignore[no-untyped-def]
    """A config whose targets 1-3 share a budget through ``source``."""
    if source == "category":
        return _config(hours, categories=[CAPPED])
    config = _config(hours)
    config.observation_groups = ObservationGroups(
        groups=[
            ObservationGroup(
                name="Capped", members=MEMBERS, time_budget_seconds=BUDGET, **group
            )
        ]
    )
    return config


def _scenario(source: str, *, collected_before: float = 0.0):  # type: ignore[no-untyped-def]
    """Three budgeted targets of an hour each that outrank a filler."""
    config = _budgeted(source)
    capped = [
        _target(config, k, *PATCH[k], merit=100 - k, minutes=60, snapshot=20, ss_min=5)
        for k in MEMBERS
    ]
    if collected_before:
        capped[0].record_collection(T0 - HOUR, collected_before)
    filler = _target(config, 100, *PATCH[4], merit=10, minutes=300, snapshot=20)
    return config, [*capped, filler]


def _capped_seconds(plan) -> float:  # type: ignore[no-untyped-def]
    return sum(
        e.collection_seconds_between(float(e.begin), float(e.end))
        for e in science_entries(plan)
        if 1 <= int(e.obsid) < 10
    )


def _dispatch(config, targets):  # type: ignore[no-untyped-def]
    ditl = create_ditl(config, targets, BEGIN, END)
    assert isinstance(ditl, QueueDITL)
    assert ditl.calc()
    return ditl


@pytest.mark.parametrize("source", ["category", "group"])
class TestBudgets:
    def test_dispatch_stops_at_the_budget(self, source: str) -> None:
        ditl = _dispatch(*_scenario(source))
        taken = _capped_seconds(ditl.plan)
        assert BUDGET - 60 <= taken <= BUDGET + 1e-6
        # The filler takes over; the last capped observation is sized to the budget left.
        assert any(int(e.obsid) == 100 for e in ditl.plan)
        assert any(
            e.exposure < 20 * MIN
            for e in ditl.plan
            if e.obstype == ObsType.AT and 1 <= int(e.obsid) < 10
        )
        if source == "category":
            row = next(r for r in ditl.summary_by_category() if r.category == "Capped")
        else:
            row = next(r for r in ditl.summary_by_group() if r.group == "Capped")
            assert row.members == 3
        assert row.budget_hours == pytest.approx(BUDGET / 3600)
        assert row.budget_used == pytest.approx(taken / BUDGET)

    @pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
    def test_the_plan_caps_the_budget(self, source: str, kind: str) -> None:
        config, targets = _scenario(source)
        scheduler, options = planner(kind)
        built = scheduler(config, targets, BEGIN, END, **options)
        plan = built.schedule()
        assert _capped_seconds(plan) <= BUDGET + 1e-6
        assert any(int(e.obsid) == 100 for e in plan)
        if kind != "cp_sat":
            assert BUDGET - 60 <= _capped_seconds(plan)
        if kind == "priority" and source == "category":
            assert any("time budget leaves" in e.description for e in built.log.events)

    @pytest.mark.parametrize("solver", ["greedy", "milp"])
    def test_the_allocation_caps_the_budget(self, source: str, solver: str) -> None:
        if solver == "milp":
            pytest.importorskip("ortools")
        window = {"window_start": T0 + 24 * HOUR} if source == "group" else {}
        config = _budgeted(source, 48, **window)
        targets = [
            _target(config, k, *PATCH[k], merit=100 - k, minutes=120, snapshot=20)
            for k in MEMBERS
        ]
        targets.append(
            _target(config, 100, *PATCH[4], merit=10, minutes=600, snapshot=20)
        )
        allocator = LongRangeAllocator(
            config, BEGIN, BEGIN + timedelta(days=2), solver=solver
        )
        allocation = allocator.allocate(targets, T0)
        grouped = {k: allocation.seconds.get(k, {}) for k in MEMBERS}
        assert 0 < sum(sum(b.values()) for b in grouped.values()) <= BUDGET + 1e-6
        assert sum(allocation.seconds.get(100, {}).values()) > 0
        if window:  # nothing allocated before the group's window
            assert all(
                allocation.bins[k][1] > T0 + 24 * HOUR
                for per_bin in grouped.values()
                for k in per_bin
            )


class TestCategoryBudget:
    def test_science_collected_before_the_run_counts(self) -> None:
        ditl = _dispatch(*_scenario("category", collected_before=30 * MIN))
        taken = _capped_seconds(ditl.plan)
        assert BUDGET - 30 * MIN - 60 <= taken <= BUDGET - 30 * MIN + 1e-6

    def test_a_visit_is_taken_only_whole(self) -> None:
        config = _config(HOURS, categories=[CAPPED])
        visit = Pointing(
            config=config,
            obsid=1,
            name="visit",
            ra=PATCH[0][0],
            dec=PATCH[0][1],
            merit=100.0,
            sequence=ObservationSequence(
                steps=[SequenceStep(name="s", duration_seconds=30 * MIN)]
            ),
            visits=3,
        )
        filler = _target(config, 100, *PATCH[4], merit=10, minutes=300, snapshot=20)
        ditl = _dispatch(config, [visit, filler])
        visits = [e for e in ditl.plan if int(e.obsid) == 1 and e.step is not None]
        assert len(visits) == 1, "a second visit would not fit the budget whole"


def _group_targets(config, merits=(60, 50, 50), filler=55):  # type: ignore[no-untyped-def]
    """Three group members and a filler that outranks all but the first."""
    members = [
        _target(config, k, *PATCH[k], merit=m, minutes=20, snapshot=20, ss_min=5)
        for k, m in zip(MEMBERS, merits, strict=True)
    ]
    other = _target(config, 100, *PATCH[4], merit=filler, minutes=300, snapshot=20)
    return [*members, other]


def _grouped(config, group: ObservationGroup, **targets: float):  # type: ignore[no-untyped-def]
    config.observation_groups = ObservationGroups(groups=[group])
    if targets:
        config.targets = TargetConfig(**targets)  # type: ignore[arg-type]
    return config


def _at(plan):  # type: ignore[no-untyped-def]
    return [e for e in plan if e.obstype.name == "AT"]


class TestGroups:
    def test_groups_are_validated(self) -> None:
        group = ObservationGroup(
            name="bin", members=[1, 2], window_start=T0, window_deadline=T0 + HOUR
        )
        groups = ObservationGroups(groups=[group])
        assert groups.group_of(2) is group and groups.group_of(3) is None
        assert group.bounds == (T0, T0 + HOUR)
        with pytest.raises(ValidationError, match="twice"):
            ObservationGroup(name="bin", members=[1, 1])
        with pytest.raises(ValidationError, match="end after"):
            ObservationGroup(
                name="bin", members=[1], window_start=T0, window_deadline=T0
            )
        with pytest.raises(ValidationError, match="one group"):
            ObservationGroups(
                groups=[
                    ObservationGroup(name="a", members=[1]),
                    ObservationGroup(name="b", members=[1, 2]),
                ]
            )
        dated = ObservationGroup(name="bin", members=[1], window_start=BEGIN)
        assert dated.window_start == T0

    def test_dispatch_keeps_the_members_in_the_window(self) -> None:
        config = _grouped(
            _config(HOURS),
            ObservationGroup(
                name="bin",
                members=MEMBERS,
                window_start=T0 + 2 * HOUR,
                window_deadline=T0 + 3 * HOUR,
            ),
        )
        science = _at(_dispatch(config, _group_targets(config)).plan)
        assert int(science[0].obsid) == 100, "the filler goes first: the window is shut"
        for entry in science:
            if int(entry.obsid) in MEMBERS:
                assert entry.collection_begin is not None
                assert T0 + 2 * HOUR <= float(entry.collection_begin) <= T0 + 3 * HOUR
        assert any(int(e.obsid) in MEMBERS for e in science)

    def test_contiguity_keeps_the_group_together_in_dispatch(self) -> None:
        """The filler outranks the second and third members, so without the
        bonus it follows the first member; with it the group is observed
        one target after another."""
        config = _grouped(
            _config(HOURS),
            ObservationGroup(name="bin", members=MEMBERS),
            group_contiguity_weight=10.0,
        )
        order = [
            int(e.obsid) for e in _at(_dispatch(config, _group_targets(config)).plan)
        ]
        assert order[0] == 1 and set(order[:3]) == set(MEMBERS) and order[3] == 100
        plain = _grouped(_config(HOURS), ObservationGroup(name="bin", members=MEMBERS))
        order = [
            int(e.obsid) for e in _at(_dispatch(plain, _group_targets(plain)).plan)
        ]
        assert order[:2] == [1, 100]

    def test_the_priority_plan_honours_window_budget_and_contiguity(self) -> None:
        config = _grouped(
            _config(HOURS),
            ObservationGroup(
                name="bin",
                members=MEMBERS,
                window_start=T0 + 2 * HOUR,
                time_budget_seconds=30 * MIN,
            ),
        )
        plan = PriorityScheduler(
            config, _group_targets(config), BEGIN, END, step_size=60
        ).schedule()
        members = [e for e in _at(plan) if int(e.obsid) in MEMBERS]
        assert members
        assert all(float(e.collection_begin) >= T0 + 2 * HOUR for e in members)
        taken = sum(
            e.collection_seconds_between(float(e.begin), float(e.end)) for e in members
        )
        assert 30 * MIN - 60 <= taken <= 30 * MIN + 1e-6
        # Once a member is placed the others follow, so their entries are consecutive.
        science = sorted(_at(plan), key=lambda e: float(e.begin))
        positions = [k for k, e in enumerate(science) if int(e.obsid) in MEMBERS]
        assert positions == list(range(positions[0], positions[0] + len(positions)))
