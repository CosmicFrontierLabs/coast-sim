"""A sequence's exact ingress must still meet its parent's start window."""

from datetime import timedelta

import pytest

from conops import QueueDITL, create_ditl
from conops.ditl.queue_ditl import _ObservationRequest
from conops.targets import (
    ObservationSequence,
    Pointing,
    SequenceStep,
    VisitWindow,
)

from .planning_scenario import BEGIN, T0, make_config


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("first_science", [False, True])
def test_first_sequence_step_keeps_the_parent_admission_deadline(
    windowed: bool, first_science: bool
) -> None:
    config = make_config(2)
    parent = Pointing(
        config=config,
        obsid=700,
        ra=105,
        dec=10,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(name="first", duration_seconds=60, science=first_science),
                SequenceStep(name="second", duration_seconds=60),
            ]
        ),
        **(
            {"visit_windows": [VisitWindow(start=T0, deadline=T0 + 30)]}
            if windowed
            else {"earliest_start": T0, "deadline": T0 + 30}
        ),
    )
    if windowed:
        parent.visit = 0
    queue = create_ditl(config, [parent], BEGIN, BEGIN + timedelta(hours=2))
    assert isinstance(queue, QueueDITL)
    step = queue._begin_sequence(parent, T0)
    queue.ppt = step
    assert queue._ppt_deadline(step) == T0 + 30
    slew = queue._new_ppt_slew(step, T0)
    slew.slewstart, slew.slewend, slew.slewtime = T0, T0 + 60, 60
    assert (
        queue._validate_ppt_slew(
            step,
            slew,
            T0,
            _ObservationRequest(start=T0, requested=60, recorder_room=None),
        )
        is None
    )
    assert any("after its deadline" in e.description for e in queue.log.events)
    # Once admitted, later steps may begin after the parent's latest start.
    assert queue._sequence is not None
    queue._sequence.index = 1
    queue.ppt = parent.step_pointing(1)
    assert queue._ppt_deadline(queue.ppt) is None
