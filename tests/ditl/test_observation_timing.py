from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from conops.config import ObservationTiming
from conops.targets import PlanEntry


@pytest.mark.parametrize(
    "budgets, expected",
    [
        ((43, 2, 10), [0, 27, 60, 48]),
        ((0, 0, 0), [10, 60, 60, 60]),
    ],
)
def test_replay_collection_and_data_exclude_setup_and_teardown(ditl, budgets, expected):
    start = float(ditl.ephem.utime[0])
    ditl.begin = datetime.fromtimestamp(start, tz=timezone.utc)
    ditl.end = datetime.fromtimestamp(start + 240, tz=timezone.utc)
    ditl.step_size = 60
    ditl.config.payload.observation_timing = ObservationTiming(
        setup_seconds=budgets[0], cleanup_seconds=budgets[1], handoff_seconds=budgets[2]
    )
    entry = PlanEntry(begin=start, end=start + 240, slewtime=50, roll=0, obsid=7)
    ditl.plan.which_ppt.return_value = entry
    ditl.acs.pointing.return_value = (0, 0, 0, 7)
    ditl.payload.data_generated = Mock(side_effect=lambda seconds: seconds * 0.01)

    assert ditl.calc()
    assert [hk.collection_seconds for hk in ditl.telemetry.housekeeping] == expected
    assert ditl.data_generated_gb[-1] == pytest.approx(sum(expected) * 0.01)
    assert entry.exposure == sum(expected)
