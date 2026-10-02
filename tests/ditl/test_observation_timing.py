from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from conops.common import ACSMode
from conops.config import ObservationTiming
from conops.targets import PlanEntry


@pytest.mark.parametrize(
    "budgets, expected",
    [
        ((43, 2, 10), [0, 27, 60, 48]),
        ((0, 0, 0), [10, 60, 60, 60]),
    ],
)
@pytest.mark.parametrize("saved_window", [False, True])
def test_replay_collection_and_data_exclude_setup_and_teardown(
    ditl, budgets, expected, saved_window
):
    start = float(ditl.ephem.utime[0])
    ditl.begin = datetime.fromtimestamp(start, tz=timezone.utc)
    ditl.end = datetime.fromtimestamp(start + 240, tz=timezone.utc)
    ditl.step_size = 60
    ditl.config.payload.observation_timing = ObservationTiming(
        setup_seconds=budgets[0], cleanup_seconds=budgets[1], handoff_seconds=budgets[2]
    )
    entry = PlanEntry(begin=start, end=start + 240, slewtime=50, roll=0, obsid=7)
    if saved_window:
        entry.set_collection_window(ditl.config.payload.observation_timing)
        entry = PlanEntry.model_validate_json(entry.model_dump_json())
        ditl.config.payload.observation_timing = ObservationTiming()
    ditl.plan.which_ppt.return_value = entry
    ditl.acs.pointing.return_value = (0, 0, 0, 7)
    ditl.payload.data_generated = Mock(side_effect=lambda seconds: seconds * 0.01)

    assert ditl.calc()
    assert [hk.collection_seconds for hk in ditl.telemetry.housekeeping] == expected
    assert ditl.data_generated_gb[-1] == pytest.approx(sum(expected) * 0.01)
    assert entry.exposure == sum(expected)


@pytest.mark.parametrize(
    "actual_slew, setup, expected",
    [
        (65, 0, [0, 55, 60, 48]),
        (90, 43, [0, 0, 47, 48]),
        (20, 43, [0, 27, 60, 48]),
        (200, 43, [0, 0, 0, 0]),
    ],
)
def test_replay_waits_for_executed_slew_and_saved_setup(
    ditl, actual_slew, setup, expected
):
    start = float(ditl.ephem.utime[0])
    ditl.begin = datetime.fromtimestamp(start, tz=timezone.utc)
    ditl.end = datetime.fromtimestamp(start + 240, tz=timezone.utc)
    entry = PlanEntry(begin=start, end=start + 240, slewtime=50, roll=0, obsid=7)
    entry.set_collection_window(
        ObservationTiming(setup_seconds=setup, cleanup_seconds=2, handoff_seconds=10)
    )
    saved = (entry.collection_begin, entry.collection_end)
    ditl.plan.which_ppt.return_value = entry
    ditl.acs.pointing.return_value = (0, 0, 0, 7)
    ditl.acs.current_slew = ditl.acs.last_slew = Mock(
        obsid=7, slewend=start + actual_slew
    )
    ditl.acs.get_mode.side_effect = lambda t: (
        ACSMode.SLEWING if t < start + actual_slew else ACSMode.SCIENCE
    )
    ditl.payload.data_generated.side_effect = lambda seconds: seconds * 0.01

    assert ditl.calc()
    assert [hk.collection_seconds for hk in ditl.telemetry.housekeeping] == expected
    assert ditl.data_generated_gb[-1] == pytest.approx(sum(expected) * 0.01)
    assert (entry.collection_begin, entry.collection_end) == saved


@pytest.mark.parametrize("slew", [None, Mock(obsid=8, slewend=0)])
def test_slewing_without_matching_execution_does_not_collect(ditl, slew):
    ditl.ppt = PlanEntry(begin=100, end=400, obsid=7)
    ditl.ppt.set_collection_window(ObservationTiming())
    ditl.acs.current_slew = slew
    ditl.uend = 400
    assert ditl._collection_seconds_for_step(200, ACSMode.SLEWING, 7) == 0
