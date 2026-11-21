"""An actual slew must not enlarge the science window admitted for a visit."""

import pytest

from conops import ACSMode, QueueDITL, Slew
from conops.common.enums import ObsType
from conops.config import ObservationTiming
from conops.targets import Pointing


def admitted_visit(queue_ditl: QueueDITL, actual_slew: int):
    begin = 1000.0
    queue_ditl.ustart, queue_ditl.uend = begin, begin + 4000
    timing = ObservationTiming(setup_seconds=60, cleanup_seconds=30, handoff_seconds=20)
    target = Pointing(config=queue_ditl.config, obsid=1001, exptime=600, ss_max=600)
    target.own_timing = timing
    target.begin = begin
    target.slewtime = 1200
    target.end = begin + 1200 + timing.total_seconds + 600
    target.set_collection_window(timing)
    admitted_begin, admitted_end = target.collection_begin, target.collection_end
    assert admitted_begin is not None and admitted_end is not None
    queue_ditl.ppt = target
    entry = target.model_copy(deep=True)
    queue_ditl.plan.append(entry)

    slew = Slew(config=queue_ditl.config)
    slew.obstype = ObsType.PPT
    slew.obsid = target.obsid
    slew.slewstart = begin
    slew.slewtime = actual_slew
    slew.slewend = begin + actual_slew
    queue_ditl.acs.last_slew = slew
    queue_ditl._sync_slew_metadata(slew)
    queue_ditl.step_size = 10
    return target, entry, admitted_begin, admitted_end


@pytest.mark.parametrize(
    "actual_slew, expected_seconds", [(300, 600), (1400, 400), (2000, 0)]
)
def test_executed_slew_preserves_admitted_science_bounds(
    queue_ditl: QueueDITL, actual_slew: int, expected_seconds: int
) -> None:
    target, entry, admitted_begin, admitted_end = admitted_visit(
        queue_ditl, actual_slew
    )
    begin = float(target.begin)

    # Replay collection readiness through both an early and a late actual
    # arrival. Waiting/late setup can reduce the admitted science, never add it.
    collected = sum(
        queue_ditl._collection_seconds_for_step(t, ACSMode.SCIENCE)
        for t in range(int(begin), int(target.end), 10)
    )
    assert collected == expected_seconds
    assert target.collection_begin >= admitted_begin
    assert (
        target.collection_end - target.collection_begin <= admitted_end - admitted_begin
    )
    assert (entry.collection_begin, entry.collection_end) == (
        target.collection_begin,
        target.collection_end,
    )
    assert target.end == entry.end == begin + 1910


@pytest.mark.parametrize("charge_deadline, expected_seconds", [(2600, 290), (2100, 0)])
def test_recharge_cut_does_not_reopen_collection_before_its_admitted_start(
    queue_ditl: QueueDITL,
    monkeypatch: pytest.MonkeyPatch,
    charge_deadline: int,
    expected_seconds: int,
) -> None:
    target, entry, _, _ = admitted_visit(queue_ditl, 300)
    monkeypatch.setattr(
        queue_ditl, "_next_charge_science_deadline", lambda _time: charge_deadline
    )
    queue_ditl._reserve_pending_charge_cleanup(1900)
    collected = sum(
        queue_ditl._collection_seconds_for_step(t, ACSMode.SCIENCE)
        for t in range(1000, charge_deadline, 10)
    )
    assert collected == expected_seconds
    assert target.end == entry.end == charge_deadline
    assert target.collection_end == charge_deadline - 50
    assert (entry.collection_begin, entry.collection_end) == (
        target.collection_begin,
        target.collection_end,
    )
