"""Collection validation distinguishes legacy data from broken modern telemetry."""

from datetime import datetime, timezone

import numpy as np
import pytest

from conops.common import ACSMode, ObsType
from conops.config import MissionConfig
from conops.ditl.plan_validator import PlanExecutionValidator
from conops.ditl.telemetry import Housekeeping
from conops.targets import Plan, PlanEntry


def _housekeeping(utime, collection=0):
    return Housekeeping(
        timestamp=datetime.fromtimestamp(utime, tz=timezone.utc),
        ra=0,
        dec=0,
        roll=0,
        obsid=42,
        acs_mode=ACSMode.SCIENCE,
        collection_seconds=collection,
    )


@pytest.fixture
def make_validator():
    def make(
        *,
        utime=(1000, 1060),
        mode=None,
        begin=None,
        end=None,
        slewtime=0,
        collection_window=(None, None),
        **kwargs,
    ):
        return PlanExecutionValidator(
            config=MissionConfig(),
            plan=Plan(
                entries=[
                    PlanEntry(
                        obsid=42,
                        ra=0,
                        dec=0,
                        roll=0,
                        begin=utime[0] if begin is None else begin,
                        end=utime[-1] + 60 if end is None else end,
                        slewtime=slewtime,
                        collection_begin=collection_window[0],
                        collection_end=collection_window[1],
                        obstype=ObsType.AT,
                    )
                ]
            ),
            utime=utime,
            ra=[0] * len(utime),
            dec=[0] * len(utime),
            roll=[0] * len(utime),
            obsid=[42] * len(utime),
            mode=[ACSMode.SCIENCE] * len(utime) if mode is None else mode,
            end_time=utime[-1] + 60,
            passes=(),
            attitude_violation_at=lambda _index, _mode: None,
            **kwargs,
        )

    return make


@pytest.mark.parametrize("records", ["omitted", "empty", "legacy"])
@pytest.mark.parametrize("begin,end", [(1020, 1059), (1120, 1159)])
def test_legacy_empty_science_window_keeps_mode_only_behavior(
    make_validator, records, begin, end
):
    kwargs = {}
    if records == "empty":
        kwargs["housekeeping"] = ()
    elif records == "legacy":
        kwargs.update(
            housekeeping=[_housekeeping(t, None) for t in (1000, 1060)], step_size=60
        )
    validator = make_validator(
        begin=begin, end=end, mode=[ACSMode.SLEWING] * 2, **kwargs
    )
    assert validator.validate() == []


@pytest.mark.parametrize("legacy_records", [False, True])
@pytest.mark.parametrize("mode", [ACSMode.SCIENCE, ACSMode.SLEWING])
def test_legacy_sampled_window_still_checks_mode(make_validator, legacy_records, mode):
    validator = make_validator(
        mode=[mode] * 2,
        housekeeping=[_housekeeping(t, None) for t in (1000, 1060)]
        if legacy_records
        else (),
        step_size=60,
    )
    mismatches = validator.validate()
    if mode == ACSMode.SCIENCE:
        assert mismatches == []
    else:
        assert any("no_science_execution" in str(m) for m in mismatches)


def test_missing_window_records_in_modern_telemetry_cannot_use_legacy_fallback(
    make_validator,
):
    validator = make_validator(
        end=1059,
        mode=[ACSMode.SCIENCE, ACSMode.IDLE],
        housekeeping=[_housekeeping(1000, None), _housekeeping(1060, 0)],
        step_size=60,
    )
    assert any("no_science_execution" in str(m) for m in validator.validate())


@pytest.mark.parametrize("times", [(1000,), (1000, 1060, 1120)])
def test_housekeeping_length_mismatch_is_reported(make_validator, times):
    validator = make_validator(
        housekeeping=[_housekeeping(t) for t in times], step_size=60
    )
    mismatches = validator.validate()
    assert len(mismatches) == 1
    assert "telemetry_length_mismatch" in str(mismatches[0])
    assert "housekeeping" in str(mismatches[0])


@pytest.mark.parametrize("offset", [0.001, 60])
def test_shifted_housekeeping_is_reported_instead_of_ignored(make_validator, offset):
    validator = make_validator(
        housekeeping=[_housekeeping(t + offset) for t in (1000, 1060)], step_size=60
    )
    mismatches = validator.validate()
    assert len(mismatches) == 1
    assert "telemetry_timestamp_mismatch" in str(mismatches[0])


@pytest.mark.parametrize("collection", [0, 0.1])
def test_submicrosecond_timestamp_preserves_recorded_zero(make_validator, collection):
    utime = 1.7e9 + 0.0000002
    housekeeping = [_housekeeping(utime, collection)]
    assert housekeeping[0].timestamp.timestamp() != utime
    validator = make_validator(utime=[utime], housekeeping=housekeeping, step_size=0.1)
    mismatches = validator.validate()
    if collection > 0:
        assert mismatches == []
    else:
        assert len(mismatches) == 1
        assert "no_science_execution" in str(mismatches[0])


@pytest.mark.parametrize("collection", [0, 0.07])
def test_fractional_steps_preserve_substep_collection(make_validator, collection):
    times = np.arange(1.7e9 + 0.3, 1.7e9 + 1.3, 0.1).tolist()
    housekeeping = [_housekeeping(t, collection) for t in times]
    begin = next(
        t for hk, t in zip(housekeeping, times) if hk.timestamp.timestamp() != t
    )
    validator = make_validator(
        utime=times,
        mode=[ACSMode.SLEWING] * len(times),
        begin=begin,
        end=begin + 0.09,
        collection_window=(begin + 0.02, begin + 0.09),
        housekeeping=housekeeping,
        step_size=0.1,
    )
    mismatches = validator.validate()
    if collection > 0:
        assert mismatches == []
    else:
        assert len(mismatches) == 1
        assert "no_science_execution" in str(mismatches[0])


@pytest.mark.parametrize("step_size", [None, 0, -60, float("nan"), float("inf")])
def test_housekeeping_requires_positive_finite_step_size(make_validator, step_size):
    kwargs = {} if step_size is None else {"step_size": step_size}
    with pytest.raises(ValueError, match="step_size must be finite and positive"):
        make_validator(housekeeping=[_housekeeping(t) for t in (1000, 1060)], **kwargs)


def test_positive_step_size_includes_collection_before_first_science_tick(
    make_validator,
):
    validator = make_validator(
        end=1059,
        slewtime=20,
        mode=[ACSMode.SLEWING, ACSMode.IDLE],
        housekeeping=[_housekeeping(1000, 39), _housekeeping(1060, 0)],
        step_size=60,
    )
    assert validator.validate() == []
