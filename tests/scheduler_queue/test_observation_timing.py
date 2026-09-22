import math
from unittest.mock import Mock

import pytest

from conops.common import ACSMode, ObsType
from conops.config import AttitudeControlSystem, ObservationTiming
from conops.simulation.passes import Pass
from conops.targets import PlanEntry


@pytest.fixture
def timed_ditl(queue_ditl):
    timing = ObservationTiming(setup_seconds=43, cleanup_seconds=2, handoff_seconds=10)
    queue_ditl.config.payload.observation_timing = timing
    entry = PlanEntry(begin=100, end=400, slewtime=50, obstype=ObsType.AT)
    entry.set_collection_window(timing)
    entry.exptime = 195
    queue_ditl.ppt = entry
    queue_ditl.uend = 1000
    queue_ditl.step_size = 60
    return queue_ditl


@pytest.mark.parametrize(
    ("time", "expected"), [(100, 0), (160, 27), (220, 60), (340, 48), (388, 0)]
)
def test_partial_step_science_accounting(timed_ditl, time, expected):
    assert timed_ditl._collection_seconds_for_step(time, ACSMode.SCIENCE) == expected


@pytest.mark.parametrize(
    "mode", [ACSMode.PASS, ACSMode.SAA, ACSMode.SAFE, ACSMode.CHARGING]
)
def test_no_collection_in_other_operational_modes(timed_ditl, mode):
    assert timed_ditl._collection_seconds_for_step(220, mode) == 0


def test_data_generation_and_remaining_exposure_use_collection_seconds(timed_ditl):
    timed_ditl.payload.data_generated = Mock(side_effect=lambda seconds: seconds * 0.1)
    timed_ditl._handle_data_management(160, ACSMode.SCIENCE)
    timed_ditl.payload.data_generated.assert_called_once_with(27)
    assert timed_ditl.ppt.exptime == 168
    assert timed_ditl.data_generated_gb[-1] == pytest.approx(2.7)
    timed_ditl.payload.data_generated.reset_mock()
    timed_ditl._handle_data_management(388, ACSMode.SCIENCE)
    timed_ditl.payload.data_generated.assert_not_called()
    assert timed_ditl.ppt.exptime == 168


def test_observation_holds_attitude_until_cleanup_and_handoff_finish(timed_ditl):
    timed_ditl.ppt.exptime = 0
    timed_ditl._attitude_constraint_name_for_attitude = Mock(return_value=None)
    timed_ditl._terminate_ppt = Mock()
    timed_ditl._check_ppt_termination(388)
    timed_ditl._terminate_ppt.assert_not_called()
    timed_ditl._check_ppt_termination(400)
    timed_ditl._terminate_ppt.assert_called_once()


def test_charge_forecast_reserves_cleanup_before_data_is_counted(timed_ditl):
    timed_ditl.plan.append(timed_ditl.ppt.model_copy())
    timed_ditl._next_charge_science_deadline = Mock(return_value=280)
    timed_ditl._reserve_pending_charge_cleanup(220)
    assert timed_ditl.ppt.end == 280
    assert timed_ditl.ppt.collection_end == 268
    assert timed_ditl.plan[-1].collection_end == 268
    assert timed_ditl._collection_seconds_for_step(220, ACSMode.SCIENCE) == 48


def test_charge_cannot_retroactively_erase_collection(timed_ditl):
    timed_ditl._next_charge_science_deadline = Mock(return_value=225)
    with pytest.raises(ValueError, match="Recharge interrupts observation"):
        timed_ditl._reserve_pending_charge_cleanup(220)


def test_unanticipated_interrupt_fails_closed(timed_ditl):
    timed_ditl.ppt.ss_min = 100
    timed_ditl.plan.append(timed_ditl.ppt.model_copy())
    with pytest.raises(ValueError, match="interrupted before reserved cleanup"):
        timed_ditl._close_last_plan_entry(280)


def test_pass_alternate_profile_reserves_cleanup_before_ingress(timed_ditl):
    timed_ditl.uend = 2600
    timed_ditl.ephem.step_size = 60
    timed_ditl.config.spacecraft_bus.attitude_control = AttitudeControlSystem(
        max_slew_rate=2.0, slew_acceleration=0.125, settle_time=36.0
    )
    profiles = [[(10.0, 20.0, 30.0)], [(10.0, 20.0, 170.0)]]
    upcoming = Pass.model_construct(
        config=timed_ditl.config,
        ephem=timed_ditl.ephem,
        station="GS",
        begin=2000.0,
        length=600.0,
        gsstartra=10.0,
        gsstartdec=20.0,
        gsstartroll=30.0,
        tracking_attitude_profiles=profiles,
    )
    timed_ditl.acs.passrequests.next_pass = Mock(return_value=upcoming)
    entry = PlanEntry(
        begin=1000,
        end=2000,
        slewtime=50,
        ra=10,
        dec=20,
        roll=30,
        obstype=ObsType.AT,
        ss_min=100,
    )
    deadline = timed_ditl._next_pass_science_deadline(
        1050, target=entry, target_roll=entry.roll
    )
    assert deadline is not None
    entry.end = deadline
    entry.set_collection_window(timed_ditl.config.payload.observation_timing)
    timed_ditl.ppt = entry
    timed_ditl.plan.append(entry.model_copy())

    ingress = math.ceil(deadline / timed_ditl.step_size) * timed_ditl.step_size
    assert upcoming.tracking_profiles_due_for_slew(ingress, 10, 20, 30) == [profiles[1]]
    default_deadline = next(upcoming.tracking_profile_slew_deadlines(1050, 10, 20, 30))[
        1
    ]
    assert default_deadline - 12 > ingress  # Old admission still claimed collection.
    timed_ditl._close_last_plan_entry(ingress)
    assert entry.collection_end + 12 <= ingress
    assert timed_ditl.plan[-1].end == deadline


def test_interrupt_with_zero_teardown_only_cancels_future_collection(timed_ditl):
    timing = ObservationTiming()
    timed_ditl.config.payload.observation_timing = timing
    timed_ditl.ppt.ss_min = 100
    timed_ditl.ppt.set_collection_window(timing)
    timed_ditl.plan.append(timed_ditl.ppt.model_copy())
    timed_ditl._close_last_plan_entry(280)
    assert timed_ditl.plan[-1].collection_end == 280
    assert timed_ditl.plan[-1].exposure == 130


def test_aborted_slew_does_not_export_zero_collection_observation(timed_ditl):
    timed_ditl.ppt.end = 150
    timed_ditl.ppt.set_collection_window(timed_ditl.config.payload.observation_timing)
    timed_ditl.plan.append(timed_ditl.ppt.model_copy())
    timed_ditl._close_last_plan_entry(150)
    assert not timed_ditl.plan
