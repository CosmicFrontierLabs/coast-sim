"""A spacecraft parked where the PASS scopes are violated still reaches its contacts."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from conops import DITL, ACSMode, Pass, QueueDITL
from conops.common import ACSCommandType, ObsType

UTIME = 1000.0
PARKED = (10.0, 20.0, 42.0)
PROFILE = [(100.0, 50.0, 37.0)]


@pytest.fixture
def parked(queue_ditl: QueueDITL) -> QueueDITL:
    """Idle, with a contact at 1200 s and the PASS scopes violated where it is parked."""
    ditl = queue_ditl
    ditl.ephem.step_size = 60
    ditl.ephem.timestamp = [
        datetime.fromtimestamp(UTIME + 60.0 * k, timezone.utc) for k in range(12)
    ]
    ditl.config.spacecraft_bus.attitude_control.slew_time = Mock(return_value=120.0)
    gspass = Pass(
        config=ditl.config,
        ephem=ditl.ephem,
        station="GS_TEST",
        begin=1200.0,
        length=600.0,
        utime=[1200.0],
        ra=[PROFILE[0][0]],
        dec=[PROFILE[0][1]],
        roll=[PROFILE[0][2]],
        gsstartra=PROFILE[0][0],
        gsstartdec=PROFILE[0][1],
        gsstartroll=PROFILE[0][2],
        obsid=4242,
        tracking_attitude_profiles=[PROFILE],
    )
    ditl.acs.passrequests.current_pass = Mock(return_value=None)
    ditl.acs.passrequests.next_pass = Mock(return_value=gspass)
    ditl.acs.acsmode = ACSMode.IDLE
    ditl.ppt = None

    def violated(
        ra: float, dec: float, roll: float, utime: float, mode: ACSMode, **_: object
    ) -> tuple[str, str] | None:
        near = abs(ra - PARKED[0]) < 5.0
        return ("Panel", "power_generation") if mode == ACSMode.PASS and near else None

    ditl._attitude_constraint_name_for_attitude = Mock(side_effect=violated)
    return ditl


def _manage(ditl: QueueDITL, utime: float = UTIME) -> bool:
    with (
        patch.object(Pass, "time_to_slew", return_value=True),
        patch.object(Pass, "tracking_profiles_due_for_slew", return_value=[PROFILE]),
    ):
        return ditl._check_and_manage_passes(utime, *PARKED)


def test_flown_as_a_reposition(parked: QueueDITL) -> None:
    assert _manage(parked)
    command = parked.acs.enqueue_command.call_args[0][0]
    assert command.command_type == ACSCommandType.SLEW_TO_TARGET
    slew = command.slew
    assert slew.obstype == ObsType.IDLE
    assert (slew.endra, slew.enddec, slew.endroll) == PROFILE[0]
    assert slew.slewend <= 1200.0
    (entry,) = list(parked.plan)
    assert entry.obstype == ObsType.GSP
    assert entry.begin == UTIME
    assert entry.slewtime == round(slew.slewtime)
    assert parked._gsp_activity_in_progress(1500.0)


def _violating(ditl: QueueDITL, mode: ACSMode) -> None:
    original = ditl._attitude_constraint_name_for_attitude.side_effect

    def violated(
        ra: float, dec: float, roll: float, utime: float, at: ACSMode, **kw: object
    ) -> tuple[str, str] | None:
        if at == mode:
            return "Sun", "hardware_safety"
        return original(ra, dec, roll, utime, at, **kw)

    ditl._attitude_constraint_name_for_attitude.side_effect = violated


@pytest.mark.parametrize("mode", [ACSMode.SLEWING, ACSMode.IDLE])
def test_not_on_an_unsafe_path_or_hold(parked: QueueDITL, mode: ACSMode) -> None:
    _violating(parked, mode)
    assert not _manage(parked)
    parked.acs.enqueue_command.assert_not_called()
    assert len(parked.plan) == 0


def test_not_when_it_would_arrive_late(parked: QueueDITL) -> None:
    # The slew from here at 1000 s takes 120 s: from 1120 s it ends too late.
    assert not _manage(parked, utime=1120.0)
    parked.acs.enqueue_command.assert_not_called()


def test_not_once_the_contact_has_begun(parked: QueueDITL) -> None:
    assert not _manage(parked, utime=1240.0)
    parked.acs.enqueue_command.assert_not_called()


@pytest.mark.parametrize("mode", [ACSMode.SCIENCE, ACSMode.CHARGING, ACSMode.SAA])
def test_also_from_an_attitude_held_for_something_else(
    parked: QueueDITL, mode: ACSMode
) -> None:
    parked.acs.acsmode = mode
    assert _manage(parked)
    assert parked.acs.enqueue_command.call_args[0][0].slew.obstype == ObsType.IDLE


def test_charging_gives_way_to_the_contact(parked: QueueDITL) -> None:
    parked.acs.acsmode = ACSMode.CHARGING
    charging = Mock(obsid=999000)
    parked.ppt = parked.charging_ppt = charging
    assert _manage(parked)
    assert parked.charging_ppt is None
    assert parked.ppt is None
    assert [entry.obstype for entry in parked.plan] == [ObsType.GSP]


def test_not_while_dumping_momentum(parked: QueueDITL) -> None:
    parked.acs.acsmode = ACSMode.DESAT
    assert not _manage(parked)
    parked.acs.enqueue_command.assert_not_called()


def test_the_plan_entry_follows_the_slew_flown(parked: QueueDITL) -> None:
    assert _manage(parked)
    slew = parked.acs.enqueue_command.call_args[0][0].slew
    (entry,) = list(parked.plan)
    assert entry.contact_ingress == "reposition"
    slew.slewstart, slew.slewtime, slew.slewdist = UTIME + 5, 77.0, 12.5
    parked._sync_slew_metadata(slew)
    assert (entry.begin, entry.slewtime, entry.slewdist) == (UTIME + 5, 77, 12.5)


def test_a_path_that_turns_unsafe_later_is_still_rejected(parked: QueueDITL) -> None:
    """From a compliant parked attitude the PASS scopes hold along the whole path."""

    def violated(
        ra: float, dec: float, roll: float, utime: float, mode: ACSMode, **_: object
    ) -> tuple[str, str] | None:
        return (
            ("Panel", "power_generation")
            if mode == ACSMode.PASS and utime > UTIME
            else None
        )

    parked._attitude_constraint_name_for_attitude.side_effect = violated
    assert not _manage(parked)
    parked.acs.enqueue_command.assert_not_called()


def test_a_compliant_ingress_is_unchanged(parked: QueueDITL) -> None:
    parked._attitude_constraint_name_for_attitude.side_effect = None
    parked._attitude_constraint_name_for_attitude.return_value = None
    assert _manage(parked)
    command = parked.acs.enqueue_command.call_args[0][0]
    assert command.slew.obstype == ObsType.GSP


@pytest.mark.parametrize(
    ("ingress", "obstype"), [("reposition", ObsType.IDLE), (None, ObsType.GSP)]
)
def test_plan_execution_flies_the_ingress_the_plan_records(
    parked: QueueDITL, ingress: str | None, obstype: ObsType
) -> None:
    gspass = parked.acs.passrequests.next_pass(UTIME)
    entry = Mock(
        contact_begin=1200.0,
        contact_end=1800.0,
        begin=UTIME,
        end=1800.0,
        contact_ingress=ingress,
    )
    parked.acs.ra, parked.acs.dec, parked.acs.roll = PARKED
    ditl = DITL.__new__(DITL)
    ditl.config, ditl.acs = parked.config, parked.acs
    ditl._entry_passes = {id(entry): gspass}
    assert ditl._command_pass_entry(entry, UTIME)
    slew = parked.acs.enqueue_command.call_args[0][0].slew
    assert slew.obstype == obstype
    assert (slew.endra, slew.enddec, slew.endroll) == PROFILE[0]
