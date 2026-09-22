from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from conops.config import ObservationTiming
from conops.ditl.telemetry import Housekeeping
from conops.targets import PlanEntry


def test_replay_collection_and_data_exclude_setup_and_teardown(ditl):
    start = float(ditl.ephem.utime[0])
    ditl.begin = datetime.fromtimestamp(start, tz=timezone.utc)
    ditl.end = datetime.fromtimestamp(start + 240, tz=timezone.utc)
    ditl.step_size = 60
    ditl.config.payload.observation_timing = ObservationTiming(
        setup_seconds=43, cleanup_seconds=2, handoff_seconds=10
    )
    entry = PlanEntry(begin=start, end=start + 240, slewtime=50, roll=0, obsid=7)
    ditl.plan.which_ppt.return_value = entry
    ditl.acs.pointing.return_value = (0, 0, 0, 7)
    ditl.payload.data_generated = Mock(side_effect=lambda seconds: seconds * 0.01)

    assert ditl.calc()
    assert [hk.collection_seconds for hk in ditl.telemetry.housekeeping] == [
        0,
        27,
        60,
        48,
    ]
    assert ditl.data_generated_gb[-1] == pytest.approx(1.35)
    assert entry.exposure == 135
    assert ditl.observation_time_totals() == {
        "observation_slew_seconds": 50,
        "setup_seconds": 43,
        "collection_seconds": 135,
        "cleanup_seconds": 2,
        "handoff_seconds": 10,
    }
    assert [hk.setup_seconds for hk in ditl.telemetry.housekeeping] == [10, 33, 0, 0]
    assert [hk.cleanup_seconds for hk in ditl.telemetry.housekeeping] == [0, 0, 0, 2]
    assert [hk.handoff_seconds for hk in ditl.telemetry.housekeeping] == [0, 0, 0, 10]
    # Missing phase telemetry cannot silently appear as zero overhead.
    ditl.telemetry.housekeeping[0].setup_seconds = None
    assert ditl.observation_time_totals() is None


def test_statistics_report_generic_overheads_separately(ditl_basic, capsys):
    for time in ditl_basic.utime:
        ditl_basic.telemetry.housekeeping.append(
            Housekeeping(
                timestamp=datetime.fromtimestamp(time, tz=timezone.utc),
                observation_slew_seconds=10,
                setup_seconds=5,
                collection_seconds=40,
                cleanup_seconds=2,
                handoff_seconds=3,
            )
        )
    totals = ditl_basic.observation_time_totals()
    assert totals["setup_seconds"] == 300
    assert totals["cleanup_seconds"] == 120
    assert totals["handoff_seconds"] == 180
    ditl_basic.print_statistics()
    output = capsys.readouterr().out
    assert "OBSERVATION TIME ACCOUNTING" in output
    assert "Setup" in output and "Cleanup" in output and "Handoff margin" in output

    ditl_basic.telemetry.housekeeping[0].setup_seconds = 65
    assert ditl_basic.observation_time_totals() is None


def test_statistics_do_not_invent_overheads_from_legacy_modes(ditl_basic):
    for time in ditl_basic.utime:
        ditl_basic.telemetry.housekeeping.append(
            Housekeeping(
                timestamp=datetime.fromtimestamp(time, tz=timezone.utc),
            )
        )
    assert ditl_basic.observation_time_totals() is None
