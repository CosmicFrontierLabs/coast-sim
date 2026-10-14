"""Interrupted visits must not retire unfinished campaign integration."""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from conops import MissionConfig
from conops.targets import Pointing, Queue


@pytest.mark.parametrize("interruption", ["charging", "pass"])
@pytest.mark.parametrize(
    "remaining, cancelled", [(7015, False), (75, False), (0, False), (7015, True)]
)
def test_interruption_preserves_target_budget_and_eligibility(
    queue_ditl, interruption, remaining, cancelled
):
    config = MissionConfig()
    config.constraint.ephem = queue_ditl.ephem
    target = Pointing(
        config=config,
        ra=10.0,
        dec=20.0,
        obsid=123,
        begin=1000.0,
        end=2000.0,
        slewtime=100.0,
        ss_min=60.0,
    )
    target.exptime = remaining
    target.done = cancelled
    target.windows = [[1000.0, 10000.0]]
    entry = target.model_copy()
    queue_ditl.ppt = target
    queue_ditl.plan = [entry]
    charging = Pointing(ra=100.0, dec=50.0, obsid=999001)
    queue_ditl.emergency_charging.create_charging_pointing = Mock(return_value=charging)

    if interruption == "charging":
        queue_ditl._initiate_charging(1600.0, target.ra, target.dec)
        assert queue_ditl.ppt is charging
    else:
        queue_ditl._terminate_science_ppt_for_pass(1600.0)
        assert queue_ditl.ppt is None

    assert entry.end == 1600.0
    assert target.exptime == remaining
    assert target.done == (cancelled or remaining == 0)
    queue = Queue(config=config, ephem=queue_ditl.ephem)
    queue.targets = [target]
    selected = queue.get(target.ra, target.dec, 1700.0)
    assert selected is (None if target.done else target)


@pytest.mark.parametrize(
    "end_offset, last_checked",
    [(59.5, 60), (60, 60), (60.1, 120), (127, 180), (180, 180)],
)
def test_locked_roll_check_brackets_fractional_end(
    queue_ditl, end_offset, last_checked
):
    start = 1000.0
    ephem = queue_ditl.acs.ephem
    ephem.step_size = 60
    ephem.timestamp = [
        datetime.fromtimestamp(start + offset, timezone.utc)
        for offset in (0, 60, 120, 180)
    ]
    ephem.index = lambda time: int((time.timestamp() - start) // 60)
    queue_ditl.ppt = Pointing()
    check = Mock(return_value=None)
    queue_ditl._constraint_name_for_science_attitude = check

    assert (
        queue_ditl._check_locked_roll_window(
            start + 7, start + end_offset, (10, 20, 30)
        )
        is None
    )
    checked = [call.args[-1] for call in check.call_args_list]
    assert checked == [start + offset for offset in range(0, last_checked + 1, 60)]


def test_locked_roll_detects_first_bad_sample_after_fractional_end(queue_ditl):
    """A boundary crossed after the last in-window sample must not be missed."""
    start = 1000.0
    ephem = queue_ditl.acs.ephem
    ephem.step_size = 60
    ephem.timestamp = [
        datetime.fromtimestamp(start + offset, timezone.utc) for offset in (0, 60, 120)
    ]
    ephem.index = lambda time: int((time.timestamp() - start) // 60)
    queue_ditl.ppt = Pointing()
    queue_ditl._constraint_name_for_science_attitude = lambda ra, dec, roll, t: (
        ("ST Soft", "imaging_quality") if t > start + 60 else None
    )

    assert queue_ditl._check_locked_roll_window(start, start + 67, (10, 20, 30)) == (
        start + 120,
        "ST Soft",
        "imaging_quality",
    )
