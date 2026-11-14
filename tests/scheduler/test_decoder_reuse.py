"""Reusing a suffix must preserve exact scores and resource histories."""

from datetime import timedelta

from conops.config import TargetConfig
from conops.schedulers import LocalSearchScheduler

from .planning_scenario import BEGIN, PATCH, make_config, make_target


def scheduler(*, battery=True):
    config = make_config(
        2, weights=TargetConfig(urgency_weight=0, completion_deficit_weight=0)
    )
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    if not battery:
        config.battery.planning_level = None
    targets = [
        make_target(config, n, *PATCH[0], merit=merit, minutes=5)
        for n, merit in enumerate(
            [32.3900381556679, 15.093408900710946, 65.09693795925497], 1
        )
    ]
    return LocalSearchScheduler(
        config,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=2),
        time_limit=0,
        max_iterations=0,
        include_passes=False,
    )


def test_unchanged_sequence_retains_its_exact_score():
    planner = scheduler(battery=False)
    start = planner._start_from_priority_plan()
    repeated = planner._redecode(start, list(start.sequence), 0, 0, 0)
    fresh = planner._decode(start.sequence)
    assert repeated.final.score == fresh.final.score == start.final.score


def test_small_recorder_difference_requires_fresh_decoding():
    planner = scheduler(battery=False)
    initial = planner._start_from_priority_plan().states[0]
    assert planner._ledger.active
    left = initial.model_copy(update={"volumes": (0.3,)})
    right = initial.model_copy(update={"volumes": (0.3 + 5e-10,)})
    assert not planner._same_state(left, right)


def test_different_prefix_score_requires_fresh_decoding():
    planner = scheduler(battery=False)
    initial = planner._start_from_priority_plan().states[0]
    assert not planner._same_state(initial, initial.model_copy(update={"score": 1.0}))


def test_static_panel_history_does_not_block_identical_suffix_reuse():
    planner = scheduler()
    start = planner._start_from_priority_plan()
    repeated = planner._step(start.states[0], start.sequence[0])
    assert not planner._battery.driven
    assert planner._same_state(repeated, start.states[1])
