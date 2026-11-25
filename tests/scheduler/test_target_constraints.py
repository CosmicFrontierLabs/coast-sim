"""Constraints set on one target: a science constraint that replaces the
mission's, an extra constraint applied as well, and a stray-light limit."""

from datetime import timedelta

import pytest
from pydantic import ValidationError
from rust_ephem import EarthLimbConstraint, MoonConstraint
from rust_ephem.constraints import PointSourceTransmittance, StrayLightConstraint

from conops import create_ditl
from conops.common.enums import ACSMode
from conops.config.constraint import AttitudeConstraintScope
from conops.targets import Plan

from .planning_scenario import (
    BEGIN,
    CONSTRAINED,
    make_config,
    make_target,
    planner,
    replay,
)

WIDE = EarthLimbConstraint(min_angle=45)
"""Much wider than the scenario's 10 degree Earth limb limit."""
END = BEGIN + timedelta(hours=6)


def stray_light(max_ratio: float = 3.0) -> StrayLightConstraint:
    table = PointSourceTransmittance.from_pst(
        off_axis_deg=[1.0, 30.0, 90.0, 180.0],
        azimuth_deg=[0.0],
        pst=[[1e-3], [1e-6], [1e-9], [1e-10]],
    )
    return StrayLightConstraint(pst=table, conversion_factor=3e-9, max_ratio=max_ratio)


def visible_seconds(target) -> float:
    target.visibility()
    return sum(end - begin for begin, end in target.windows)


def inside(windows, begin: float, end: float) -> bool:
    return any(a <= begin and end <= b for a, b in windows)


def science_spans(entries, obsid: int) -> list[tuple[float, float]]:
    return [
        (e.collection_begin, e.collection_end)
        for e in entries
        if e.obsid == obsid and e.collection_begin is not None
    ]


# ── The derived constraint ───────────────────────────────────────────────


def test_a_target_without_constraints_of_its_own_uses_the_missions():
    config = make_config(6)
    target = make_target(config, 1, *CONSTRAINED[0])
    assert not target.has_own_constraints()
    assert target.observing_constraint() is config.constraint
    assert target.own_constraints_key() is None


def test_an_extra_constraint_is_applied_with_the_missions_science_constraint():
    config = make_config(6)
    config.constraint.science_constraint = MoonConstraint(min_angle=5)
    target = make_target(config, 1, *CONSTRAINED[0])
    target.extra_constraint = WIDE
    own = target.observing_constraint()
    assert own is not config.constraint
    assert own.science_constraint == MoonConstraint(min_angle=5) | WIDE
    # Everything else is the mission's.
    assert own.sun_constraint is config.constraint.sun_constraint
    assert own.ephem is config.constraint.ephem
    assert config.constraint.science_constraint == MoonConstraint(min_angle=5)


def test_a_science_constraint_replaces_the_missions():
    config = make_config(6)
    config.constraint.science_constraint = WIDE
    target = make_target(config, 1, *CONSTRAINED[0])
    target.science_constraint = MoonConstraint(min_angle=5)
    assert target.observing_constraint().science_constraint == MoonConstraint(
        min_angle=5
    )


def test_targets_with_equal_constraints_share_one_derived_constraint():
    config = make_config(6)
    a = make_target(config, 1, *CONSTRAINED[0])
    b = make_target(config, 2, *CONSTRAINED[1])
    c = make_target(config, 3, *CONSTRAINED[2])
    a.extra_constraint = EarthLimbConstraint(min_angle=45)
    b.extra_constraint = EarthLimbConstraint(min_angle=45)
    c.extra_constraint = EarthLimbConstraint(min_angle=30)
    assert a.observing_constraint() is b.observing_constraint()
    assert a.observing_constraint() is not c.observing_constraint()
    assert a.own_constraints_key() == b.own_constraints_key()


def test_the_derived_constraint_follows_a_change_to_the_missions():
    config = make_config(6)
    target = make_target(config, 1, *CONSTRAINED[0])
    target.extra_constraint = WIDE
    before = target.observing_constraint()
    assert target.observing_constraint() is before
    config.constraint.earth_constraint = EarthLimbConstraint(min_angle=20)
    after = target.observing_constraint()
    assert after is not before
    assert after.earth_constraint == EarthLimbConstraint(min_angle=20)
    target.extra_constraint = EarthLimbConstraint(min_angle=30)
    assert target.observing_constraint().science_constraint == EarthLimbConstraint(
        min_angle=30
    )


# ── The stray-light limit ────────────────────────────────────────────────


def test_a_targets_stray_light_ratio_replaces_the_limit_wherever_it_appears():
    config = make_config(6)
    config.constraint.science_constraint = MoonConstraint(min_angle=5) | stray_light(3)
    target = make_target(config, 1, *CONSTRAINED[0])
    target.stray_light_max_ratio = 0.5
    science = target.observing_constraint().science_constraint
    moon, scattered = science.constraints
    assert moon == MoonConstraint(min_angle=5)
    assert scattered == stray_light(0.5)
    # The mission's own limit is untouched.
    assert config.constraint.science_constraint.constraints[1].max_ratio == 3


def test_a_stray_light_ratio_decides_whether_an_attitude_is_allowed():
    config = make_config(6)
    config.constraint.science_constraint = stray_light(3)
    utime = (BEGIN + timedelta(hours=1)).timestamp()
    scopes = [AttitudeConstraintScope.IMAGING_QUALITY]

    def names(max_ratio: float) -> list[str]:
        target = make_target(config, 1, 110.0, 10.0)
        target.stray_light_max_ratio = max_ratio
        return target.attitude_constraint_names(
            scopes, (110.0, 10.0, 0.0), utime, ACSMode.SCIENCE
        )

    assert names(1e12) == []
    assert "Science" in names(1e-12)


def test_a_stray_light_ratio_with_no_stray_light_constraint_is_refused():
    config = make_config(6)
    with pytest.raises(ValidationError, match="stray_light_max_ratio"):
        type(make_target(config, 1, 110.0, 10.0))(
            config=config, obsid=1, ra=110.0, dec=10.0, stray_light_max_ratio=2.0
        )
    with pytest.raises(ValidationError):
        type(make_target(config, 1, 110.0, 10.0))(
            config=config, obsid=1, ra=110.0, dec=10.0, stray_light_max_ratio=0.0
        )


# ── Visibility ───────────────────────────────────────────────────────────


def test_an_extra_constraint_narrows_only_its_own_targets_windows():
    config = make_config(6)
    plain = make_target(config, 1, *CONSTRAINED[0])
    narrowed = make_target(config, 2, *CONSTRAINED[0])
    narrowed.extra_constraint = WIDE
    assert 0 < visible_seconds(narrowed) < visible_seconds(plain)


def test_a_looser_science_constraint_widens_only_its_own_targets_windows():
    config = make_config(6)
    config.constraint.science_constraint = WIDE
    held = make_target(config, 1, *CONSTRAINED[0])
    loosened = make_target(config, 2, *CONSTRAINED[0])
    loosened.science_constraint = EarthLimbConstraint(min_angle=11)
    assert visible_seconds(loosened) > visible_seconds(held) > 0


# ── Runs ─────────────────────────────────────────────────────────────────


def targets_for(config, constrained: bool):
    """Two requests for the same position; the second takes the wide limit."""
    a = make_target(config, 1, *CONSTRAINED[0], minutes=300, snapshot=20)
    b = make_target(config, 2, *CONSTRAINED[0], minutes=300, snapshot=20)
    if constrained:
        b.extra_constraint = WIDE
    return [a, b]


def narrow_windows(config):
    probe = make_target(config, 99, *CONSTRAINED[0])
    probe.extra_constraint = WIDE
    probe.visibility()
    return [(a, b) for a, b in probe.windows]


def test_dispatch_observes_a_target_only_where_its_own_constraint_allows():
    config = make_config(6)
    run = create_ditl(config, targets_for(config, True), BEGIN, END)
    run.step_size = 60
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
    windows = narrow_windows(config)
    own, other = science_spans(run.plan, 2), science_spans(run.plan, 1)
    assert own and all(inside(windows, *span) for span in own)
    # The same position without the constraint is observed outside them.
    assert any(not inside(windows, *span) for span in other)
    assert all(e.extra_constraint == WIDE for e in run.plan if e.obsid == 2)


@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
def test_a_planner_places_a_target_only_where_its_own_constraint_allows(kind):
    config = make_config(6)
    scheduler, options = planner(kind)
    plan = scheduler(
        config, targets_for(config, True), BEGIN, END, step_size=60, **options
    ).schedule()
    windows = narrow_windows(config)
    own, other = science_spans(plan, 2), science_spans(plan, 1)
    assert own and all(inside(windows, *span) for span in own)
    assert any(not inside(windows, *span) for span in other)
    replay(config, plan, end=END)


def test_a_looser_science_constraint_lets_dispatch_observe_where_the_missions_forbids():
    config = make_config(6)
    config.constraint.science_constraint = WIDE
    held = make_target(config, 1, *CONSTRAINED[0], minutes=300, snapshot=20)
    loosened = make_target(config, 2, *CONSTRAINED[0], minutes=300, snapshot=20)
    loosened.science_constraint = EarthLimbConstraint(min_angle=11)
    held.visibility()
    windows = [(a, b) for a, b in held.windows]
    run = create_ditl(config, [held, loosened], BEGIN, END)
    run.step_size = 60
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
    assert all(inside(windows, *span) for span in science_spans(run.plan, 1))
    assert any(not inside(windows, *span) for span in science_spans(run.plan, 2))


@pytest.mark.parametrize("kind", ["priority", "local_search"])
def test_a_looser_science_constraint_lets_a_planner_place_where_the_missions_forbids(
    kind,
):
    config = make_config(6)
    config.constraint.science_constraint = WIDE
    held = make_target(config, 1, *CONSTRAINED[0], minutes=300, snapshot=20)
    loosened = make_target(config, 2, *CONSTRAINED[0], minutes=300, snapshot=20)
    loosened.science_constraint = EarthLimbConstraint(min_angle=11)
    held.visibility()
    windows = [(a, b) for a, b in held.windows]
    scheduler, options = planner(kind)
    plan = scheduler(
        config, [held, loosened], BEGIN, END, step_size=60, **options
    ).schedule()
    assert all(inside(windows, *span) for span in science_spans(plan, 1))
    assert any(not inside(windows, *span) for span in science_spans(plan, 2))
    replay(config, plan, end=END)


# ── Plan files ───────────────────────────────────────────────────────────


def test_a_saved_plan_keeps_each_entrys_constraints_and_replays(tmp_path):
    config = make_config(6)
    scheduler, options = planner("priority")
    plan = scheduler(
        config, targets_for(config, True), BEGIN, END, step_size=60, **options
    ).schedule()
    loaded = Plan.load(plan.save(tmp_path / "plan.json"))
    assert [e.extra_constraint for e in loaded] == [e.extra_constraint for e in plan]
    assert any(e.extra_constraint == WIDE for e in loaded)
    loaded.bind_runtime(config)
    replay(config, loaded, end=END)


def test_a_target_without_constraints_writes_none_of_the_fields():
    config = make_config(6)
    dumped = make_target(config, 1, *CONSTRAINED[0]).model_dump(mode="json")
    assert not {
        "science_constraint",
        "extra_constraint",
        "stray_light_max_ratio",
    } & set(dumped)
