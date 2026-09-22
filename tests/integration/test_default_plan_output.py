from unittest.mock import patch

from conops.simulation.acs import ACS
from scripts.check_default_plan_output import (
    DEFAULT_BASELINE,
    DEFAULT_NUMERIC_ABS_TOL,
    build_default_plan_payload,
    compare_to_baseline,
)


def test_default_plan_output_matches_baseline() -> None:
    actual = build_default_plan_payload()
    diffs = compare_to_baseline(
        actual, DEFAULT_BASELINE, abs_tol=DEFAULT_NUMERIC_ABS_TOL
    )
    assert not diffs, "\n".join(diffs[:25])


def test_recording_motion_does_not_change_schedule_or_samples() -> None:
    recorded = build_default_plan_payload()
    with patch.object(ACS, "_record_attitude_interval", lambda self, utime: None):
        unrecorded = build_default_plan_payload()
    assert recorded["attitude_timeseries"].pop("resolved_intervals")
    assert unrecorded["attitude_timeseries"].pop("resolved_intervals") == []
    assert recorded == unrecorded
