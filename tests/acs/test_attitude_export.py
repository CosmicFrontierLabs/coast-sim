from unittest.mock import Mock

import pytest

from conops.common.enums import ACSCommandType, ObsType
from conops.common.vector import attitude_to_quat
from conops.config import AttitudeControlSystem
from conops.simulation.acs_command import ACSCommand
from conops.simulation.attitude_profile import same_rotation
from conops.simulation.slew import Slew


@pytest.fixture
def executing_acs(acs, monkeypatch):
    monkeypatch.setattr(acs, "_check_constraints", lambda t: None)
    monkeypatch.setattr(acs, "_enforce_idle_constraint_safe_attitude", lambda t: None)
    slew = Slew(
        acs_config=AttitudeControlSystem(
            max_slew_rate=2, slew_acceleration=1, settle_time=0
        ),
        endra=270,
        enddec=0,
        endroll=0,
    )
    acs.enqueue_command(
        ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET, execution_time=990, slew=slew
        )
    )
    acs.pointing(1000)
    return acs


def test_actual_slew_and_post_slew_hold_are_recorded(executing_acs):
    acs = executing_acs
    acs.pointing(1060)
    assert len(acs.executed_attitude_intervals) == 1
    interval = acs.executed_attitude_intervals[0]
    assert (interval.start_utime, interval.end_utime, interval.motion.start_utime) == (
        1000,
        1060,
        1000,
    )
    for time in (1000, 1001, 1020, 1047, 1060):
        assert same_rotation(
            interval.motion.quaternion_at(time),
            attitude_to_quat(*acs.last_slew.attitude(time)),
        )


def test_preemption_does_not_claim_unexecuted_nominal_motion(executing_acs):
    acs = executing_acs
    acs.pointing(1002)
    interrupted = acs.last_slew
    replacement = Slew(acs_config=interrupted.acs_config, endra=190)
    acs.enqueue_command(
        ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1010,
            slew=replacement,
        )
    )
    acs.pointing(1010)
    acs.pointing(1015)
    assert [(i.start_utime, i.end_utime) for i in acs.executed_attitude_intervals] == [
        (1000, 1002),
        (1010, 1015),
    ]


def test_repeated_tick_discontinuity_removes_invalid_interval(executing_acs):
    acs = executing_acs
    acs.pointing(1060)
    assert acs.executed_attitude_intervals
    acs._hold_idle_attitude(0, 0, 0, 1060)
    acs.pointing(1060)
    assert not acs.executed_attitude_intervals


@pytest.mark.parametrize("override", ["safe", "charge", "pass"])
def test_tracking_not_exported_as_known_motion(executing_acs, override):
    acs = executing_acs
    if override == "safe":
        acs.in_safe_mode = True
    elif override == "charge":
        acs.last_slew.obstype = ObsType.CHARGE
    else:
        acs.current_pass = Mock()
    acs._record_attitude_interval(1000)
    assert acs._pending_attitude_interval is None


def test_export_history_is_not_reused_when_time_rewinds(executing_acs):
    acs = executing_acs
    acs.pointing(1060)
    acs.pointing(1000)
    assert not acs.executed_attitude_intervals
