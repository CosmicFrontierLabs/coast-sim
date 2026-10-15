"""Validate collection windows even when no telemetry timestamp falls inside."""

from datetime import datetime, timezone

import pytest

from conops.common import ACSMode, ObsType
from conops.ditl.queue_ditl import PlanExecutionMismatchError
from conops.ditl.telemetry import Housekeeping
from conops.targets import PlanEntry


@pytest.mark.parametrize("obstype", [ObsType.AT, ObsType.TOO])
@pytest.mark.parametrize(
    "collection,recorded_obsid,expected_valid",
    [(0, 42, False), (39, 42, True), (39, 99, False), (None, 42, False)],
)
def test_collection_entirely_between_ticks(
    queue_ditl, obstype, collection, recorded_obsid, expected_valid
):
    # The +0 tick can credit [20,59) after slew completion, but its SLEWING
    # mode alone cannot prove that the collection happened.
    entry = PlanEntry(
        obsid=42,
        ra=0,
        dec=0,
        roll=0,
        begin=1000,
        end=1059,
        slewtime=20,
        obstype=obstype,
    )
    queue_ditl.plan.append(entry)
    queue_ditl.step_size = 60
    queue_ditl.utime = [1000, 1060]
    queue_ditl.ra = queue_ditl.dec = queue_ditl.roll = [0, 0]
    queue_ditl.obsid = [recorded_obsid, 0]
    queue_ditl.mode = [ACSMode.SLEWING, ACSMode.IDLE]
    queue_ditl.telemetry.housekeeping = [
        Housekeeping(
            ra=0,
            dec=0,
            roll=0,
            timestamp=datetime.fromtimestamp(1000, tz=timezone.utc),
            acs_mode=ACSMode.SLEWING,
            obsid=recorded_obsid,
            collection_seconds=collection,
        ),
        Housekeeping(
            ra=0,
            dec=0,
            roll=0,
            timestamp=datetime.fromtimestamp(1060, tz=timezone.utc),
            acs_mode=ACSMode.IDLE,
            obsid=0,
            collection_seconds=0,
        ),
    ]

    mismatches = queue_ditl.validate_plan_matches_execution()
    if expected_valid:
        assert mismatches == []
    else:
        assert any("no_science_execution" in str(m) for m in mismatches)
        with pytest.raises(PlanExecutionMismatchError, match="no_science_execution"):
            queue_ditl._assert_plan_matches_execution()


@pytest.mark.parametrize("offset", [-60, 60])
def test_collection_outside_window_cannot_validate_short_observation(
    queue_ditl, offset
):
    entry = PlanEntry(
        obsid=42,
        ra=0,
        dec=0,
        roll=0,
        begin=1000,
        end=1059,
        slewtime=20,
        obstype=ObsType.AT,
    )
    queue_ditl.plan.append(entry)
    queue_ditl.step_size = 60
    queue_ditl.utime = [1000 + offset]
    queue_ditl.ra = queue_ditl.dec = queue_ditl.roll = [0]
    queue_ditl.obsid = [42]
    queue_ditl.mode = [ACSMode.SLEWING]
    queue_ditl.telemetry.housekeeping = [
        Housekeeping(
            ra=0,
            dec=0,
            roll=0,
            timestamp=datetime.fromtimestamp(1000 + offset, tz=timezone.utc),
            obsid=42,
            acs_mode=ACSMode.SLEWING,
            collection_seconds=39,
        )
    ]
    assert any(
        "no_science_execution" in str(m)
        for m in queue_ditl.validate_plan_matches_execution()
    )


def test_science_mode_without_collection_does_not_validate(queue_ditl):
    entry = PlanEntry(
        obsid=42,
        ra=0,
        dec=0,
        roll=0,
        begin=1000,
        end=1060,
        slewtime=0,
        obstype=ObsType.AT,
    )
    queue_ditl.plan.append(entry)
    queue_ditl.step_size = 60
    queue_ditl.utime = [1000]
    queue_ditl.ra = queue_ditl.dec = queue_ditl.roll = [0]
    queue_ditl.obsid = [42]
    queue_ditl.mode = [ACSMode.SCIENCE]
    queue_ditl.telemetry.housekeeping = [
        Housekeeping(
            ra=0,
            dec=0,
            roll=0,
            timestamp=datetime.fromtimestamp(1000, tz=timezone.utc),
            obsid=42,
            acs_mode=ACSMode.SCIENCE,
            collection_seconds=0,
        )
    ]
    assert any(
        "no_science_execution" in str(m)
        for m in queue_ditl.validate_plan_matches_execution()
    )


def test_preceding_task_with_same_obsid_is_not_collection_evidence(queue_ditl):
    queue_ditl.plan.append(
        PlanEntry(
            obsid=42,
            ra=0,
            dec=0,
            roll=0,
            begin=1020,
            end=1059,
            slewtime=0,
            obstype=ObsType.AT,
        )
    )
    queue_ditl.step_size = 60
    queue_ditl.utime = [1000]
    queue_ditl.ra = queue_ditl.dec = queue_ditl.roll = [0]
    queue_ditl.obsid = [42]
    queue_ditl.mode = [ACSMode.SLEWING]
    queue_ditl.telemetry.housekeeping = [
        Housekeeping(
            timestamp=datetime.fromtimestamp(1000, tz=timezone.utc),
            ra=0,
            dec=0,
            roll=0,
            obsid=42,
            acs_mode=ACSMode.SLEWING,
            collection_seconds=5,
        )
    ]
    assert any(
        "no_science_execution" in str(m)
        for m in queue_ditl.validate_plan_matches_execution()
    )
