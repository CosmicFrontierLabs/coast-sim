"""Exercise charging/contact handoffs through the real scheduler and ACS."""

from datetime import timedelta

import pytest
import rust_ephem

from conops import QueueDITL
from conops.common import ACSMode, ObsType
from conops.common.enums import ACSCommandType
from conops.config import (
    AttitudeControlSystem,
    Battery,
    GroundStation,
    GroundStationRegistry,
    MissionConfig,
    RadiatorConfiguration,
    SolarPanel,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
)
from conops.simulation.passes import Pass
from conops.targets import Pointing
from scripts.check_default_plan_output import (
    SCENARIO_BEGIN,
    DeterministicConstraint,
    DeterministicEphemeris,
)


def test_charging_preserves_pass_ingress_and_resumes_after_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    begin = SCENARIO_BEGIN
    end = begin + timedelta(seconds=600)
    start = begin.timestamp()
    ephem = DeterministicEphemeris(begin, end, step_size_seconds=60)
    config = MissionConfig(
        constraint=DeterministicConstraint(),
        ground_stations=GroundStationRegistry(
            stations=[
                GroundStation(code="TEST", name="Test", latitude_deg=0, longitude_deg=0)
            ]
        ),
        solar_panel=SolarPanelSet(
            panels=[SolarPanel(normal=(1, 0, 0), max_power=1000)]
        ),
        battery=Battery(watthour=100_000),
        spacecraft_bus=SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                max_slew_rate=1, slew_acceleration=0.5, settle_time=10
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
    )
    config.constraint.ephem = ephem
    ditl = QueueDITL(config=config, ephem=ephem, begin=begin, end=end)
    ditl.battery.charge_level = 0.94 * ditl.battery.watthour

    # Reproduce a registered charge request that never became the active PPT.
    # Pass ingress must take ownership without losing its physical slew state.
    pending_charge = Pointing(
        config=config,
        begin=start - 60,
        end=end.timestamp(),
        obsid=999000,
        ra=0,
        dec=0,
        roll=0,
        obstype=ObsType.CHARGE,
    )
    ditl.charging_ppt = pending_charge
    ditl.emergency_charging.current_charging_ppt = pending_charge
    contact = Pass(
        config=config,
        ephem=ephem,
        station="TEST",
        begin=start + 300,
        length=180,
        gsstartra=180,
        gsstartdec=0,
        gsstartroll=120,
        gsendra=180,
        gsenddec=0,
        gsendroll=120,
        utime=[start + offset for offset in (300, 360, 420, 480)],
        ra=[180] * 4,
        dec=[0] * 4,
        roll=[120] * 4,
    )
    ditl.acs.passrequests.passes = [contact]
    monkeypatch.setattr(
        rust_ephem,
        "get_eop_provenance",
        lambda: {"ut1": {"available": False}, "polar_motion": {"available": False}},
    )

    assert ditl.calc()
    assert ditl.validate_plan_matches_execution() == []
    assert ditl._attitude_rate_violations() == []
    assert pending_charge.done
    assert pending_charge.end == start

    commands = ditl.acs.executed_commands
    ingress_slews = [
        command.slew
        for command in commands
        if command.command_type == ACSCommandType.SLEW_TO_TARGET
        and command.slew is not None
        and command.slew.obstype == ObsType.GSP
    ]
    assert len(ingress_slews) == 1
    ingress = ingress_slews[0]
    assert ingress.slewstart == start
    assert ingress.slewend < contact.begin

    gap_indices = [
        i
        for i, time in enumerate(ditl.utime)
        if ingress.slewend <= time < contact.begin
    ]
    assert gap_indices
    for i in gap_indices:
        assert ditl.mode[i] == ACSMode.IDLE
        assert ditl.batterylevel[i] < ditl.battery.recharge_threshold
        assert (ditl.ra[i], ditl.dec[i], ditl.roll[i]) == pytest.approx((180, 0, 120))

    contact_indices = [
        i for i, time in enumerate(ditl.utime) if contact.begin <= time <= contact.end
    ]
    assert len(contact_indices) == 4
    for i in contact_indices:
        assert ditl.mode[i] == ACSMode.PASS
        assert ditl.obsid[i] == contact.obsid
        assert (ditl.ra[i], ditl.dec[i], ditl.roll[i]) == pytest.approx((180, 0, 120))

    # Pass windows include their final sample; release occurs on the next tick.
    release_time = contact.end + ephem.step_size
    pass_commands = [
        (command.command_type, command.execution_time)
        for command in commands
        if command.command_type in (ACSCommandType.START_PASS, ACSCommandType.END_PASS)
    ]
    assert pass_commands == [
        (ACSCommandType.START_PASS, contact.begin),
        (ACSCommandType.END_PASS, release_time),
    ]
    assert [
        command.execution_time
        for command in commands
        if command.command_type == ACSCommandType.START_BATTERY_CHARGE
    ] == [release_time]
    assert ditl.acs.last_slew is not None
    assert ditl.acs.last_slew.obstype == ObsType.CHARGE
