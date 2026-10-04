"""Tests for DITL executing every entry of a supplied plan."""

from unittest.mock import Mock

import pytest

from conops import DITL, ACSMode, Plan
from conops.common import ObsType
from conops.common.enums import ACSCommandType
from conops.simulation.acs_command import ACSCommand
from conops.simulation.passes import Pass
from conops.targets.plan_entry import PlanEntry


def _science(begin: float, end: float, obsid: int, **kwargs: object) -> PlanEntry:
    return PlanEntry(
        ra=10.0 * obsid,
        dec=5.0,
        roll=30.0,
        obsid=obsid,
        begin=begin,
        end=end,
        obstype=kwargs.pop("obstype", ObsType.AT),
        **kwargs,
    )


def _commands(ditl: DITL, command_type: ACSCommandType) -> list[ACSCommand]:
    return [
        call.args[0]
        for call in ditl.acs.enqueue_command.call_args_list
        if call.args[0].command_type == command_type
    ]


@pytest.fixture
def t(ditl: DITL) -> list[float]:
    """The four simulated step times of the fixture DITL."""
    return [float(u) for u in ditl.ephem.utime[:4]]


class TestScienceEntries:
    def test_every_science_entry_is_slewed_to_at_its_begin(
        self, ditl: DITL, t: list[float]
    ) -> None:
        ditl.plan = Plan(entries=[_science(t[0], t[2], 1), _science(t[2], t[3], 2)])

        ditl.calc()

        calls = ditl.acs._enqueue_slew.call_args_list
        assert [(c.args[2], c.args[3]) for c in calls] == [(1, t[0]), (2, t[2])]

    def test_queue_science_types_slew_as_science_pointings(
        self, ditl: DITL, t: list[float]
    ) -> None:
        ditl.plan = Plan(
            entries=[
                _science(t[0], t[1], 1, obstype=ObsType.AT),
                _science(t[1], t[2], 2, obstype=ObsType.TOO),
                _science(t[2], t[3], 3, obstype=ObsType.PPT),
            ]
        )

        ditl.calc()

        obstypes = [c.kwargs["obstype"] for c in ditl.acs._enqueue_slew.call_args_list]
        assert obstypes == [ObsType.PPT] * 3

    def test_science_ends_when_the_entry_ends(self, ditl: DITL, t: list[float]) -> None:
        ditl.plan = Plan(entries=[_science(t[0], t[2], 1)])

        ditl.calc()

        assert ditl.acs.end_science_observation.call_count == 1

    def test_entries_are_commanded_in_begin_order(
        self, ditl: DITL, t: list[float]
    ) -> None:
        ditl.plan = Plan(entries=[_science(t[2], t[3], 2), _science(t[0], t[2], 1)])

        ditl.calc()

        obsids = [c.args[2] for c in ditl.acs._enqueue_slew.call_args_list]
        assert obsids == [1, 2]

    def test_entry_finished_before_the_simulation_is_not_commanded(
        self, ditl: DITL, t: list[float]
    ) -> None:
        ditl.plan = Plan(entries=[_science(t[0] - 600, t[0], 1)])

        ditl.calc()

        ditl.acs._enqueue_slew.assert_not_called()

    def test_entry_already_running_at_start_is_commanded_at_the_first_step(
        self, ditl: DITL, t: list[float]
    ) -> None:
        ditl.plan = Plan(entries=[_science(t[0] - 600, t[2], 1)])

        ditl.calc()

        assert ditl.acs._enqueue_slew.call_args.args[3] == t[0]

    def test_pointing_is_reevaluated_after_commanding(
        self, ditl: DITL, t: list[float]
    ) -> None:
        """A slew due at a step must take effect in that step's telemetry."""
        ditl.plan = Plan(entries=[_science(t[1], t[3] + 60, 1)])

        ditl.calc()

        assert ditl.acs.pointing.call_count == len(ditl.utime) + 1

    def test_nothing_is_commanded_in_safe_mode(
        self, ditl: DITL, t: list[float]
    ) -> None:
        ditl.acs.in_safe_mode = True
        ditl.plan = Plan(entries=[_science(t[0], t[3], 1)])

        ditl.calc()

        ditl.acs._enqueue_slew.assert_not_called()


class TestChargeEntries:
    def test_charge_entry_requests_and_ends_charging(
        self, ditl: DITL, t: list[float]
    ) -> None:
        entry = PlanEntry(
            ra=45.0,
            dec=23.5,
            roll=12.0,
            obsid=999001,
            begin=t[1],
            end=t[3],
            obstype=ObsType.CHARGE,
        )
        ditl.plan = Plan(entries=[entry])

        ditl.calc()

        ditl.acs.request_battery_charge.assert_called_once_with(
            t[1], 45.0, 23.5, 12.0, 999001
        )
        ditl.acs.request_end_battery_charge.assert_called_once_with(t[3])
        ditl.acs._enqueue_slew.assert_not_called()


class TestUnexecutableEntries:
    def test_unsupported_obstype_is_logged_and_skipped(
        self, ditl: DITL, t: list[float]
    ) -> None:
        ditl.plan = Plan(
            entries=[_science(t[0], t[3], 1, obstype=ObsType.IDLE)],
        )

        ditl.calc()

        ditl.acs._enqueue_slew.assert_not_called()
        assert any(
            "cannot execute" in event.description and event.event_type == "ERROR"
            for event in ditl.log.events
        )


class TestPassEntries:
    @pytest.fixture
    def gspass(self, t: list[float]) -> Pass:
        gspass = Pass(station="SGS", begin=t[2], length=60.0, obsid=0xFFFF)
        gspass.utime = [t[2], t[3]]
        gspass.ra = [100.0, 101.0]
        gspass.dec = [-20.0, -21.0]
        gspass.roll = [5.0, 6.0]
        gspass.gsstartra, gspass.gsstartdec, gspass.gsstartroll = 100.0, -20.0, 5.0
        return gspass

    @pytest.fixture
    def unplanned(self, t: list[float]) -> Pass:
        return Pass(station="KSG", begin=t[0] + 3600, length=600.0)

    @pytest.fixture
    def contact(self, gspass: Pass, t: list[float]) -> PlanEntry:
        return PlanEntry(
            ra=100.0,
            dec=-20.0,
            roll=5.0,
            obsid=gspass.obsid,
            begin=t[1],
            end=t[3] + 60,
            obstype=ObsType.GSP,
            station="SGS",
            contact_begin=gspass.begin,
            contact_end=gspass.end,
            track_start_ra=100.0,
            track_start_dec=-20.0,
            track_start_roll=5.0,
        )

    @pytest.fixture
    def passing_acs(self, ditl: DITL, gspass: Pass) -> Mock:
        """Mock ACS that holds the planned tracking attitude and runs START_PASS."""
        acs = ditl.acs
        acs.ra, acs.dec, acs.roll = 100.0, -20.0, 5.0

        def enqueue(command: ACSCommand) -> None:
            if command.command_type == ACSCommandType.START_PASS:
                acs.current_pass = gspass

        acs.enqueue_command.side_effect = enqueue
        return acs

    def test_only_planned_passes_are_kept(
        self,
        ditl: DITL,
        gspass: Pass,
        unplanned: Pass,
        contact: PlanEntry,
        passing_acs: Mock,
    ) -> None:
        ditl.acs.passrequests.passes = [gspass, unplanned]
        ditl.plan = Plan(entries=[contact])

        ditl.calc()

        assert ditl.acs.passrequests.passes == [gspass]

    def test_passes_are_predicted_when_none_exist(
        self, ditl: DITL, gspass: Pass, contact: PlanEntry, passing_acs: Mock
    ) -> None:
        def predict(year: int, day: int, length: int) -> None:
            ditl.acs.passrequests.passes = [gspass]

        ditl.acs.passrequests.passes = []
        ditl.acs.passrequests.get.side_effect = predict
        ditl.plan = Plan(entries=[contact])

        ditl.calc()

        ditl.acs.passrequests.get.assert_called_once_with(2018, 331, 1)
        assert ditl.acs.passrequests.passes == [gspass]

    def test_contact_slews_starts_and_ends_on_plan(
        self,
        ditl: DITL,
        gspass: Pass,
        contact: PlanEntry,
        passing_acs: Mock,
        t: list[float],
    ) -> None:
        ditl.acs.passrequests.passes = [gspass]
        ditl.plan = Plan(entries=[contact])

        ditl.calc()

        (slew_command,) = _commands(ditl, ACSCommandType.SLEW_TO_TARGET)
        assert slew_command.execution_time == t[1]
        assert slew_command.slew is not None
        assert slew_command.slew.obstype == ObsType.GSP
        assert (
            slew_command.slew.endra,
            slew_command.slew.enddec,
            slew_command.slew.endroll,
        ) == (100.0, -20.0, 5.0)
        assert [
            c.execution_time for c in _commands(ditl, ACSCommandType.START_PASS)
        ] == [t[2]]
        # The contact runs until the end of the simulation, so it is not ended.
        assert _commands(ditl, ACSCommandType.END_PASS) == []

    def test_contact_ends_at_entry_end(
        self,
        ditl: DITL,
        gspass: Pass,
        contact: PlanEntry,
        passing_acs: Mock,
        t: list[float],
    ) -> None:
        contact.end = t[3]
        ditl.acs.passrequests.passes = [gspass]
        ditl.plan = Plan(entries=[contact])

        ditl.calc()

        ends = _commands(ditl, ACSCommandType.END_PASS)
        assert [c.execution_time for c in ends] == [t[3]]

    def test_contact_waits_for_the_tracking_attitude(
        self,
        ditl: DITL,
        gspass: Pass,
        contact: PlanEntry,
        passing_acs: Mock,
    ) -> None:
        passing_acs.ra = 0.0  # still slewing toward the track
        ditl.acs.passrequests.passes = [gspass]
        ditl.plan = Plan(entries=[contact])

        ditl.calc()

        assert _commands(ditl, ACSCommandType.START_PASS) == []

    def test_planned_tracking_profile_is_selected(
        self,
        ditl: DITL,
        gspass: Pass,
        contact: PlanEntry,
        passing_acs: Mock,
    ) -> None:
        other = [(200.0, 10.0, 90.0), (201.0, 11.0, 91.0)]
        planned = list(zip(gspass.ra, gspass.dec, gspass.roll, strict=True))
        gspass.tracking_attitude_profiles = [other, planned]
        ditl.acs.passrequests.passes = [gspass]
        ditl.plan = Plan(entries=[contact])

        ditl.calc()

        assert (gspass.gsstartra, gspass.gsstartdec, gspass.gsstartroll) == planned[0]

    def test_unmatched_contact_is_logged_and_not_commanded(
        self,
        ditl: DITL,
        unplanned: Pass,
        contact: PlanEntry,
        passing_acs: Mock,
    ) -> None:
        ditl.acs.passrequests.passes = [unplanned]
        ditl.plan = Plan(entries=[contact])

        ditl.calc()

        assert ditl.acs.passrequests.passes == []
        assert _commands(ditl, ACSCommandType.SLEW_TO_TARGET) == []
        assert any(
            "No predicted pass matches" in event.description
            for event in ditl.log.events
        )


class TestValidation:
    @pytest.fixture(autouse=True)
    def _tolerance(self, ditl: DITL) -> None:
        ditl.config.spacecraft_bus.attitude_control.slew_accuracy = 0.01

    def test_validate_before_calc_reports_nothing(self, ditl: DITL) -> None:
        ditl.plan = Plan()
        ditl._attitude_rate_violations = Mock(return_value=[])  # type: ignore[method-assign]

        assert ditl.validate_plan_matches_execution() == []

    def test_ppt_entries_are_validated_as_science(
        self, ditl: DITL, t: list[float]
    ) -> None:
        """A PPT entry that never reached SCIENCE must be reported."""
        ditl.acs.get_mode.return_value = ACSMode.IDLE
        ditl.plan = Plan(entries=[_science(t[0], t[3], 1, obstype=ObsType.PPT)])

        ditl.calc()
        mismatches = ditl.validate_plan_matches_execution()

        assert any("no_science_execution" in str(m) for m in mismatches)
