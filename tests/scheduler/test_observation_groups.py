"""Observation groups: a window, a budget and a preference for contiguity
over a set of targets, honoured by dispatch, the planners and the allocator,
and reported by group."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from conops import QueueDITL, create_ditl
from conops.config import ObservationGroup, ObservationGroups, TargetConfig
from conops.schedulers import (
    LocalSearchScheduler,
    LongRangeAllocator,
    PriorityScheduler,
)

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 6
END = BEGIN + timedelta(hours=HOURS)
MEMBERS = [1, 2, 3]


def _targets(config, merits=(60, 50, 50), filler=55):  # type: ignore[no-untyped-def]
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


def _science(plan):  # type: ignore[no-untyped-def]
    return [e for e in plan if e.obstype.name == "AT"]


def _dispatch(config, targets):  # type: ignore[no-untyped-def]
    ditl = create_ditl(config, targets, BEGIN, END)
    assert isinstance(ditl, QueueDITL)
    ditl.step_size = 60
    assert ditl.calc()
    return ditl


class TestConfiguration:
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


class TestDispatch:
    def test_the_window_bounds_the_members(self) -> None:
        config = _grouped(
            _config(HOURS),
            ObservationGroup(
                name="bin",
                members=MEMBERS,
                window_start=T0 + 2 * HOUR,
                window_deadline=T0 + 3 * HOUR,
            ),
        )
        ditl = _dispatch(config, _targets(config))
        science = _science(ditl.plan)
        assert int(science[0].obsid) == 100, "the filler goes first: the window is shut"
        for entry in science:
            if int(entry.obsid) in MEMBERS:
                assert entry.collection_begin is not None
                assert T0 + 2 * HOUR <= float(entry.collection_begin) <= T0 + 3 * HOUR
        assert any(int(e.obsid) in MEMBERS for e in science)

    def test_the_budget_caps_the_group(self) -> None:
        config = _grouped(
            _config(HOURS),
            ObservationGroup(name="bin", members=MEMBERS, time_budget_seconds=30 * MIN),
        )
        ditl = _dispatch(config, _targets(config))
        taken = sum(
            e.collection_seconds_between(float(e.begin), float(e.end))
            for e in _science(ditl.plan)
            if int(e.obsid) in MEMBERS
        )
        assert 30 * MIN - 60 <= taken <= 30 * MIN + 1e-6
        row = next(r for r in ditl.summary_by_group() if r.group == "bin")
        assert row.members == 3 and row.budget_hours == pytest.approx(0.5)
        assert row.budget_used == pytest.approx(taken / (30 * MIN))

    def test_contiguity_keeps_the_group_together(self) -> None:
        """The filler outranks the second and third members, so without the
        bonus it follows the first member; with it the group is observed
        one target after another."""
        config = _grouped(
            _config(HOURS),
            ObservationGroup(name="bin", members=MEMBERS),
            group_contiguity_weight=10.0,
        )
        ditl = _dispatch(config, _targets(config))
        order = [int(e.obsid) for e in _science(ditl.plan)][:4]
        assert order[0] == 1 and set(order[:3]) == set(MEMBERS) and order[3] == 100
        plain = _grouped(_config(HOURS), ObservationGroup(name="bin", members=MEMBERS))
        order = [
            int(e.obsid) for e in _science(_dispatch(plain, _targets(plain)).plan)
        ][:4]
        assert order[:2] == [1, 100]


class TestPlanners:
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
            config, _targets(config), BEGIN, END, step_size=60
        ).schedule()
        members = [e for e in _science(plan) if int(e.obsid) in MEMBERS]
        assert members
        assert all(float(e.collection_begin) >= T0 + 2 * HOUR for e in members)
        taken = sum(
            e.collection_seconds_between(float(e.begin), float(e.end)) for e in members
        )
        assert 30 * MIN - 60 <= taken <= 30 * MIN + 1e-6
        # Contiguity: once a member is placed the others follow in the
        # placement order, so the members' entries are consecutive.
        science = sorted(_science(plan), key=lambda e: float(e.begin))
        positions = [k for k, e in enumerate(science) if int(e.obsid) in MEMBERS]
        assert positions == list(range(positions[0], positions[0] + len(positions)))

    @pytest.mark.parametrize("scheduler", ["local_search", "cp_sat"])
    def test_the_search_planners_cap_the_group(self, scheduler: str) -> None:
        config = _grouped(
            _config(HOURS),
            ObservationGroup(name="bin", members=MEMBERS, time_budget_seconds=30 * MIN),
        )
        if scheduler == "cp_sat":
            pytest.importorskip("ortools")
            from conops.schedulers import CpSatScheduler

            built = CpSatScheduler(
                config,
                _targets(config),
                BEGIN,
                END,
                step_size=60,
                solver_time_limit=2.0,
            )
        else:
            built = LocalSearchScheduler(
                config, _targets(config), BEGIN, END, step_size=60, time_limit=1.0
            )
        plan = built.schedule()
        taken = sum(
            e.collection_seconds_between(float(e.begin), float(e.end))
            for e in _science(plan)
            if int(e.obsid) in MEMBERS
        )
        assert taken <= 30 * MIN + 1e-6


class TestAllocator:
    def test_the_allocation_keeps_the_window_and_the_budget(self) -> None:
        config = _grouped(
            _config(48),
            ObservationGroup(
                name="bin",
                members=MEMBERS,
                window_start=T0 + 24 * HOUR,
                time_budget_seconds=40 * MIN,
            ),
        )
        targets = [
            _target(config, k, *PATCH[k], merit=90, minutes=120, snapshot=20)
            for k in MEMBERS
        ]
        targets.append(
            _target(config, 100, *PATCH[4], merit=10, minutes=600, snapshot=20)
        )
        allocator = LongRangeAllocator(
            config, BEGIN, BEGIN + timedelta(days=2), solver="greedy"
        )
        allocation = allocator.allocate(targets, T0)
        grouped = {k: allocation.seconds.get(k, {}) for k in MEMBERS}
        assert sum(sum(b.values()) for b in grouped.values()) <= 40 * MIN + 1e-6
        for per_bin in grouped.values():
            for k in per_bin:
                assert allocation.bins[k][1] > T0 + 24 * HOUR, (
                    "allocated before the window"
                )
