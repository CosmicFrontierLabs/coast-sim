"""Interruptions keep the visit credited by collection across later openings."""

import pytest

from conops import QueueDITL, create_ditl
from conops.targets import Pointing, VisitWindow

from .planning_scenario import PATCH, T0, make_config


@pytest.mark.parametrize("own_pointing", [False, True])
def test_interruption_after_next_window_opens_reopens_the_collected_visit(own_pointing):
    target = Pointing(
        ss_min=600,
        ss_max=600,
        visit_windows=[
            VisitWindow(
                start=100,
                deadline=200,
                **({"ra": 105, "dec": 10} if own_pointing else {}),
            ),
            VisitWindow(start=300, deadline=400),
        ],
    )
    target.visit = 0
    selected = target.visit_pointing(0)
    # A start window limits admission, not the end of collection. This first
    # visit is still under way when the second window opens.
    selected.record_collection(350, 200, visit=0)
    selected.record_interruption(350)
    assert target.visits_done == set()
    assert target.collected_seconds == 200
    assert target.exptime == 1200


def test_whole_visit_stays_complete_when_it_ends_after_the_next_window_opens():
    target = Pointing(
        ss_min=600,
        ss_max=600,
        visit_windows=[
            VisitWindow(start=100, deadline=200),
            VisitWindow(start=300, deadline=400),
        ],
    )
    target.visit = 0
    target.record_collection(750, 600, visit=0)
    target.record_interruption(750)
    assert target.visits_done == {0}
    assert target.collected_seconds == 600
    target.record_interruption(751)
    assert target.visits_done == {0}


def test_cut_visit_log_names_the_flown_window_after_the_next_one_opens():
    config = make_config(3)
    target = Pointing(
        config=config,
        ra=PATCH[0][0],
        dec=PATCH[0][1],
        obsid=900,
        name="Windowed target",
        ss_min=600,
        ss_max=600,
        visit_windows=[
            VisitWindow(start=T0 + 100, deadline=T0 + 200),
            VisitWindow(start=T0 + 300, deadline=T0 + 400),
        ],
    )
    run = create_ditl(config, [target])
    assert isinstance(run, QueueDITL)
    selected = run.queue.targets[0]
    selected.visit = 0
    selected.record_collection(T0 + 350, 200, visit=0)
    run.ppt = selected
    run._terminate_ppt(T0 + 350, "Interrupted", interrupted=True)
    events = [e.description for e in run.log.events if "cut short:" in e.description]
    assert events == [
        "Visit 1 of Windowed target cut short: its window has closed, so it is missed"
    ]
