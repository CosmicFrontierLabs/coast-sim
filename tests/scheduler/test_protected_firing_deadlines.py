"""Only unconstrained firings may defer around an indivisible visit."""

from datetime import timedelta

import pytest

from conops import QueueDITL
from conops.config import ActivityAttitude, ObservationTiming
from conops.targets import (
    ObservationSequence,
    Pointing,
    ScheduledActivity,
    SequenceStep,
)

from .planning_scenario import BEGIN, T0, make_config


@pytest.mark.parametrize(
    "fixed,slide_seconds,fits", [(False, 0, True), (True, 0, False), (False, 60, False)]
)
def test_firing_start_windows_remain_deadlines_for_protected_visits(
    fixed, slide_seconds, fits
):
    config = make_config(2)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.payload.observation_timing = ObservationTiming()
    config.spacecraft_bus.attitude_control.settle_time = 0
    parent = Pointing(
        config=config,
        obsid=700,
        ra=105,
        dec=10,
        roll_deg=0,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(name="first", duration_seconds=60),
                SequenceStep(name="second", duration_seconds=60),
            ]
        ),
    )
    activity = ScheduledActivity(
        name="Pointed firing",
        obsid=998001,
        begin=T0 + 30,
        duration_seconds=60,
        slide_seconds=slide_seconds,
        fixed=fixed,
        desaturation=True,
        attitude=ActivityAttitude(kind="fixed", ra_deg=105, dec_deg=10, roll_deg=0),
    )
    queue = QueueDITL(
        config=config,
        begin=BEGIN,
        end=BEGIN + timedelta(minutes=6),
        activities=[activity],
    )
    queue.step_size = 60
    queue.ustart, queue.uend = T0, T0 + 360
    step = queue._begin_sequence(parent, T0)
    step.begin, step.end = T0, T0 + 60
    step.collection_begin, step.collection_end = T0, T0 + 60
    queue.ppt = step
    slew = queue._new_ppt_slew(step, T0)
    slew.slewstart = slew.slewend = T0
    slew.endra, slew.enddec, slew.endroll = 105, 10, 0
    assert queue._sequence_fits_window(slew) is fits
