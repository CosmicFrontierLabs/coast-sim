"""A future committed visit must coexist with earlier collected history."""

from datetime import timedelta

import pytest

from conops.ditl import RollingHorizonDITL
from conops.targets import (
    ObservationSequence,
    PlanEntry,
    Pointing,
    SequenceStep,
    VisitWindow,
)

from .planning_scenario import (
    BEGIN,
    HOUR,
    MIN,
    T0,
    collected_seconds,
    make_config,
    planner,
)


def _planner(kind, config, target, **options):
    scheduler, defaults = planner(kind)
    begin, end = BEGIN + timedelta(minutes=80), BEGIN + timedelta(hours=4)
    return scheduler(
        config, [target], begin, end, include_passes=False, **options, **defaults
    )


def _case():
    config = make_config(4, open_sky=True)
    target = Pointing(
        config=config,
        obsid=700,
        ra=105,
        dec=10,
        ss_min=10 * MIN,
        ss_max=10 * MIN,
        visit_windows=[
            VisitWindow(start=T0 + HOUR, deadline=T0 + 70 * MIN),
            VisitWindow(start=T0 + 90 * MIN, deadline=T0 + 150 * MIN),
            VisitWindow(start=T0 + 3 * HOUR, deadline=T0 + 210 * MIN),
        ],
    )
    target.record_collection(T0 + 70 * MIN, 10 * MIN, visit=0)
    return config, target


@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
@pytest.mark.parametrize(
    "setup,reserved,expected",
    [
        # Window 1 committed beyond the horizon: only window 2 remains.
        (None, (10 * MIN, 100 * MIN, {700: 1}), [2]),
        # Sequence overhead: the recorded index, not the end time, names window 1.
        ("sequence", (10 * MIN, 190 * MIN, {700: 1}), [2, 2, 2]),
        # An older reservation does not erase newer completed history.
        ("history", (None, 70 * MIN, {700: 0}), [2]),
        # A pending cut attempt reserves no whole window, so a whole retry is allowed.
        (None, (5 * MIN, 105 * MIN, {}), [1, 2]),
    ],
)
def test_reserved_visits_with_completed_history_place_the_remaining_windows(
    kind, setup, reserved, expected
):
    config, target = _case()
    if setup == "sequence":
        target.sequence = ObservationSequence(
            steps=[
                SequenceStep(name="science A", duration_seconds=5 * MIN),
                SequenceStep(
                    name="calibration", duration_seconds=50 * MIN, science=False
                ),
                SequenceStep(name="science B", duration_seconds=5 * MIN),
            ]
        )
    elif setup == "history":
        target.record_collection(T0 + 100 * MIN, 10 * MIN, visit=1)
    seconds, visit_end, windows = reserved
    options = {"reserved_seconds": {700: seconds}} if seconds else {}
    scheduler = _planner(
        kind,
        config,
        target,
        reserved_visits={700: T0 + visit_end},
        reserved_visit_windows=windows,
        **options,
    )
    assert [e.visit for e in scheduler.schedule() if e.obsid == 700] == expected
    assert target.visits_done == ({0, 1} if setup == "history" else {0})


@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
def test_explicit_no_completed_windows_overrides_elapsed_time_inference(kind):
    config = make_config(4, open_sky=True)
    target = Pointing(
        config=config,
        obsid=700,
        ra=105,
        dec=10,
        ss_min=600,
        ss_max=600,
        visit_windows=[VisitWindow(start=T0 + 30 * MIN, deadline=T0 + 180 * MIN)],
    )
    scheduler = _planner(
        kind,
        config,
        target,
        reserved_seconds={700: 300},
        reserved_visits={700: T0 + 85 * MIN},
        reserved_visit_windows={},
    )
    assert [e.visit for e in scheduler.schedule() if e.obsid == 700] == [0]


@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
def test_rolling_replan_keeps_one_whole_visit_per_window(kind):
    config, target = _case()
    target.reset()
    scheduler, options = planner(kind)
    run = RollingHorizonDITL(
        config,
        [target],
        begin=BEGIN,
        end=BEGIN + timedelta(hours=4),
        horizon=timedelta(hours=3),
        replan_interval=timedelta(minutes=80),
        commit_lead_time=timedelta(minutes=30),
        scheduler=scheduler,
        scheduler_options=options,
        include_passes=False,
    )
    run.step_size = 60
    assert run.calc()
    assert not run.validate_plan_matches_execution()
    entries = [e for e in run.plan if e.obsid == 700 and e.exposure > 0]
    assert [e.visit for e in entries] == [0, 1, 2]
    assert collected_seconds(run) == 30 * MIN


def test_rolling_reservation_uses_only_future_science_and_recorded_indices():
    config, target = _case()
    run = RollingHorizonDITL(
        config, [target], begin=BEGIN, end=BEGIN + timedelta(hours=4)
    )

    def entry(visit, stop, *, calibration=False):
        return PlanEntry(
            obsid=700,
            visit=visit,
            begin=T0,
            end=stop,
            collection_begin=T0,
            collection_end=stop,
            calibration=calibration,
        )

    now = T0 + 80 * MIN
    entries = [
        entry(0, now),
        entry(1, now + MIN),
        entry(2, now + 2 * MIN, calibration=True),
        entry(None, now + 3 * MIN),
    ]
    assert run._reserved_visit_windows(entries, now) == {700: 1}
    assert run._reserved_visit_windows(entries, now + MIN) == {}
    # A future cut attempt collecting half a visit does not reserve its window.
    cut = PlanEntry(
        obsid=700,
        visit=1,
        begin=now + 19 * MIN,
        end=now + 26 * MIN,
        collection_begin=now + 20 * MIN,
        collection_end=now + 25 * MIN,
    )
    assert run._reserved_visit_windows([cut], now) == {}


@pytest.mark.parametrize("last_seconds", [0, 150, 300])
def test_sequence_reservation_counts_its_science_steps_without_calibration(
    last_seconds,
):
    config, target = _case()
    target.sequence = ObservationSequence(
        steps=[
            SequenceStep(name="science A", duration_seconds=300),
            SequenceStep(name="calibration", duration_seconds=50 * MIN, science=False),
            SequenceStep(name="science B", duration_seconds=300),
        ]
    )
    run = RollingHorizonDITL(
        config, [target], begin=BEGIN, end=BEGIN + timedelta(hours=4)
    )

    def entry(name, start, duration, *, calibration=False):
        return PlanEntry(
            obsid=700,
            visit=1,
            step=name,
            begin=start,
            end=start + duration,
            collection_begin=start,
            collection_end=start + duration,
            calibration=calibration,
        )

    entries = [
        entry("science A", T0 + 90 * MIN, 300),
        entry("calibration", T0 + 95 * MIN, 50 * MIN, calibration=True),
        entry("science B", T0 + 145 * MIN, last_seconds),
    ]
    expected = {700: 1} if last_seconds == 300 else {}
    assert run._reserved_visit_windows(entries, T0 + 120 * MIN) == expected
