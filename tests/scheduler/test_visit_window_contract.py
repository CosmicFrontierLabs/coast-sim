"""Window-specific requests retain budgets and require a whole visit attempt."""

import pytest

from conops.config import ObservationTiming
from conops.targets import Pointing, TargetQueue, VisitWindow

from .planning_scenario import HOUR, MIN, PATCH, T0, make_config
from .test_visit_windows import HOURS, STARTS, VISIT, _timed_target


@pytest.mark.parametrize(
    "own_request", [{"roll_offset_deg": 0}, {"ra": 105, "dec": 10}]
)
def test_window_pointing_keeps_custom_timing(own_request):
    config = make_config(3)
    timing = ObservationTiming(setup_seconds=600, cleanup_seconds=90)
    target = Pointing(
        config=config,
        own_timing=timing,
        ss_min=600,
        ss_max=600,
        visit_windows=[
            VisitWindow(start=T0 + HOUR, deadline=T0 + 2 * HOUR, **own_request)
        ],
    )
    visit = target.visit_pointing(0)
    assert visit is not target
    assert visit.own_timing == timing
    assert visit.observation_timing.setup_seconds == 600
    assert visit.observation_timing.cleanup_seconds == 90


@pytest.mark.parametrize(
    "own_request", [{"roll_offset_deg": 0}, {"ra": 105, "dec": 10}]
)
def test_window_pointing_keeps_opening_protection(own_request):
    config = make_config(3)
    target = Pointing(
        config=config,
        protect_opening=True,
        ss_min=600,
        ss_max=600,
        visit_windows=[
            VisitWindow(start=T0 + HOUR, deadline=T0 + 2 * HOUR, **own_request)
        ],
    )
    assert target.visit_pointing(0).protect_opening is True


def test_protected_opening_uses_the_window_pointing_visibility():
    config = make_config(3)
    target = Pointing(
        config=config,
        obsid=900,
        ra=210,
        dec=-15,
        protect_opening=True,
        ss_min=600,
        ss_max=600,
        visit_windows=[
            VisitWindow(
                start=T0 + HOUR, deadline=T0 + 2 * HOUR, ra=PATCH[0][0], dec=PATCH[0][1]
            )
        ],
    )
    queue = TargetQueue(config=config, ephem=config.constraint.ephem)
    queue.append(target)
    assert target.visible(T0 + HOUR, T0 + HOUR) is False
    assert target.visit_pointing(0).visible(T0 + HOUR, T0 + HOUR + 10 * MIN)
    openings = queue.request_openings(T0)
    assert len(openings) == 1
    assert openings[0][0] == T0 + HOUR
    assert openings[0][2] == target.obsid


def test_cut_attempts_cannot_add_up_to_one_whole_visit():
    target = _timed_target(make_config(HOURS))
    begin = STARTS[0] + 60
    target.record_collection(begin + 354, 354)
    target.record_interruption(begin + 354)
    target.record_collection(begin + 900, 506)
    target.record_interruption(begin + 900)
    assert target.visits_done == set()
    assert target.next_visit(begin + 900) == 0
    assert target.collected_seconds == 860
    target.record_collection(begin + 1600, VISIT * MIN)
    target.record_interruption(begin + 1600)
    assert target.visits_done == {0}
    assert target.collected_seconds == 860 + VISIT * MIN


def test_reset_clears_the_previous_visit_attempt():
    config = make_config(HOURS)
    target = _timed_target(config)
    queue = TargetQueue(config=config, ephem=config.constraint.ephem)
    queue.append(target)
    begin = STARTS[0] + 60
    target.record_collection(begin + VISIT * MIN, VISIT * MIN)
    assert target.visits_done == {0}
    queue.reset()
    target.record_collection(begin + 360, 360)
    target.record_interruption(begin + 360)
    assert target.visits_done == set()
    assert target.next_visit(begin + 360) == 0
    assert target.collected_seconds == 360


@pytest.mark.parametrize("windows", [1, 2])
def test_querying_done_cannot_prevent_an_interrupted_visit_from_reopening(windows):
    target = _timed_target(make_config(HOURS))
    target.visit_windows = target.visit_windows[:windows]
    begin = STARTS[0] + 60
    target.record_collection(begin + 120, 120)
    # Public status queries and serialization must not make a provisional
    # final visit permanently complete before a later interruption.
    target.model_dump()
    target.record_interruption(begin + 120)
    assert target.visits_done == set()
    assert target.done is False
    assert target.next_visit(begin + 120) == 0
    assert target.collected_seconds == 120
