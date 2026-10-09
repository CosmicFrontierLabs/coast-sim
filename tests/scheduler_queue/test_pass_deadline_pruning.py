"""Only skip ingress geometry when a conservative bound proves it cannot win."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from conops.common.enums import SlewAlgorithm
from conops.config import AttitudeControlSystem
from conops.ditl.queue_ditl import _ScienceDeadlineInputs
from conops.simulation.passes import Pass, pass_slew_trigger_buffer
from conops.simulation.slew import Slew
from conops.targets import PlanEntry


@pytest.fixture
def deadline_ditl(queue_ditl):
    queue_ditl.config.spacecraft_bus.attitude_control = AttitudeControlSystem(
        max_slew_rate=2.0, slew_acceleration=0.125, settle_time=36.0
    )
    queue_ditl.ephem.step_size = 60.0
    upcoming = Pass.model_construct(
        config=queue_ditl.config,
        ephem=queue_ditl.ephem,
        station="GS",
        begin=2000.0,
        length=600.0,
        utime=[2000.0, 2060.0],
        tracking_attitude_profiles=[
            [(10.0, 20.0, 30.0), (20.0, 30.0, 40.0)],
            [(10.0, 20.0, 170.0), (20.0, 30.0, 180.0)],
        ],
    )
    queue_ditl.acs.passrequests.next_pass = Mock(return_value=upcoming)
    queue_ditl.ppt = PlanEntry(ra=10.0, dec=20.0, roll=30.0)
    return queue_ditl, upcoming


@pytest.mark.parametrize(
    "winner", ["simulation end", "visibility window", "charge opportunity"]
)
def test_earlier_deadline_skips_all_profile_slews(deadline_ditl, winner):
    ditl, _ = deadline_ditl
    inputs = _ScienceDeadlineInputs(
        simulation_end=1700 if winner == "simulation end" else 3000,
        charge_deadline=1700 if winner == "charge opportunity" else None,
    )
    with (
        patch.object(
            ditl,
            "_current_ppt_visibility_deadline",
            return_value=1700 if winner == "visibility window" else None,
        ),
        patch.object(
            Pass,
            "_slew_time_to_target",
            side_effect=AssertionError("unneeded ingress geometry"),
        ),
    ):
        assert ditl._next_science_deadline(
            1000, 1000, target_roll=30, deadline_inputs=inputs
        ) == (1700, winner)


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_bound_boundary_keeps_equality_on_full_path(deadline_ditl, offset):
    ditl, upcoming = deadline_ditl
    earliest = (
        upcoming.begin
        - Slew.duration_upper_bound(upcoming.config.spacecraft_bus.attitude_control)
        - pass_slew_trigger_buffer(upcoming.ephem.step_size)
    )
    exact = ditl._next_pass_science_deadline(1000, target_roll=30)
    with patch.object(
        Pass,
        "_slew_time_to_target",
        autospec=True,
        side_effect=Pass._slew_time_to_target,
    ) as slew:
        result = ditl._next_pass_science_deadline(
            1000, target_roll=30, earlier_deadline=earliest + offset
        )
    assert (slew.call_count == 0) == (offset < 0)
    assert result == (None if offset < 0 else exact)
    assert exact >= earliest


@pytest.mark.parametrize("directional", [False, True])
@pytest.mark.parametrize("epoch", [1000, 2000, 2050])
def test_matches_exhaustive_deadlines_and_reasons(deadline_ditl, directional, epoch):
    ditl, upcoming = deadline_ditl
    acs = ditl.config.spacecraft_bus.attitude_control
    if directional:
        acs.max_slew_rate_body = (0.5, 2.0, 3.0)
        acs.slew_acceleration_body = (0.2, 0.03, 0.1)
    rng = np.random.default_rng(283)
    for _ in range(40):
        # Mounted instruments use the resolved body attitude, not RA/Dec/roll.
        ditl.ppt.spacecraft_attitude = tuple(rng.uniform([0, -90, 0], [360, 90, 360]))
        exact = min(
            deadline
            for _, deadline in upcoming.tracking_profile_slew_deadlines(
                epoch, *ditl.ppt.spacecraft_attitude, for_admission=True
            )
        )
        simulation_end, visibility_end, charge_deadline = rng.uniform(800, 3000, 3)
        # Include exact pass/charge ties to check the reason priority as well.
        if rng.random() < 0.5:
            charge_deadline = exact
        expected = min(
            [
                (simulation_end, "simulation end"),
                (visibility_end, "visibility window"),
                (exact, "pass"),
                (charge_deadline, "charge opportunity"),
            ],
            key=lambda item: item[0],
        )
        with patch.object(
            ditl, "_current_ppt_visibility_deadline", return_value=visibility_end
        ):
            assert (
                ditl._next_science_deadline(
                    epoch,
                    epoch,
                    target_roll=30,
                    deadline_inputs=_ScienceDeadlineInputs(
                        simulation_end=simulation_end, charge_deadline=charge_deadline
                    ),
                )
                == expected
            )


def test_constraint_avoiding_admission_retains_existing_bound(deadline_ditl):
    ditl, _ = deadline_ditl
    ditl.config.spacecraft_bus.attitude_control.slew_algorithm = (
        SlewAlgorithm.CONSTRAINT_AVOIDING
    )
    with patch.object(
        Pass,
        "_slew_time_to_target",
        side_effect=AssertionError("time-dependent paths must be bounded"),
    ):
        assert (
            ditl._next_pass_science_deadline(
                1000, target_roll=30, earlier_deadline=1000
            )
            == 1631
        )


def test_unknown_algorithm_does_not_use_quaternion_shortcut(deadline_ditl):
    ditl, _ = deadline_ditl
    ditl.config.spacecraft_bus.attitude_control = AttitudeControlSystem.model_construct(
        slew_algorithm="future_algorithm"
    )
    with pytest.raises(ValueError, match="No slew duration bound"):
        ditl._next_pass_science_deadline(1000, target_roll=30, earlier_deadline=1000)


@pytest.mark.parametrize("changed", ["rate", "step", "begin"])
def test_bound_uses_current_pass_and_kinematics(deadline_ditl, changed):
    ditl, upcoming = deadline_ditl
    assert (
        ditl._next_pass_science_deadline(1000, target_roll=30, earlier_deadline=1700)
        is None
    )
    if changed == "rate":
        ditl.config.spacecraft_bus.attitude_control.max_slew_rate = 0.1
    elif changed == "step":
        upcoming.ephem.step_size = 300
    else:
        upcoming.begin = 1700
    exact = ditl._next_pass_science_deadline(1000, target_roll=30)
    assert exact <= 1700
    assert (
        ditl._next_pass_science_deadline(1000, target_roll=30, earlier_deadline=1700)
        == exact
    )


def test_empty_profiles_still_have_no_deadline(deadline_ditl):
    ditl, upcoming = deadline_ditl
    upcoming.tracking_attitude_profiles = []
    assert (
        ditl._next_pass_science_deadline(1000, target_roll=30, earlier_deadline=3000)
        is None
    )


@pytest.mark.parametrize("field", ["max_slew_rate", "slew_acceleration"])
def test_unavailable_bound_falls_back_to_original_calculation(deadline_ditl, field):
    ditl, _ = deadline_ditl
    # The scalar motion model supports a best-effort fallback for zero limits.
    setattr(ditl.config.spacecraft_bus.attitude_control, field, 0.0)
    exact = ditl._next_pass_science_deadline(1000, target_roll=30)
    with patch.object(
        Pass,
        "_slew_time_to_target",
        autospec=True,
        side_effect=Pass._slew_time_to_target,
    ) as slew:
        assert (
            ditl._next_pass_science_deadline(
                1000, target_roll=30, earlier_deadline=1000
            )
            == exact
        )
    assert slew.call_count == 2
