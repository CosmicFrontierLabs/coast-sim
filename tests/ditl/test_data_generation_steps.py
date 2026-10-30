"""Data the DITL generates each step from the configuration and the target."""

from unittest.mock import Mock

import pytest

from conops.common.enums import ACSMode
from conops.config import (
    AntennaPointing,
    BandCapability,
    CommunicationsSystem,
    DataGeneration,
    GroundStation,
    GroundStationRegistry,
    HousekeepingData,
    Instrument,
    OnboardRecorder,
    Payload,
    ScienceData,
    SpacecraftBus,
)
from conops.config.config import MissionConfig
from conops.targets import PlanEntry

STEP = 60


@pytest.fixture
def ditl(ditl_instance):
    """A DITL with a real configuration: housekeeping, and two instruments."""
    ditl, _, _ = ditl_instance
    ditl.config = MissionConfig(
        spacecraft_bus=SpacecraftBus(
            data_generation=HousekeepingData(rate_gbps=0.001, rate_mode={"SAFE": 0.002})
        ),
        payload=Payload(
            instruments=[
                Instrument(
                    name="camera",
                    data_generation=DataGeneration(
                        science=ScienceData(rate_gbps=0.1, per_observation_gb=2.0)
                    ),
                ),
                Instrument(
                    name="spectrometer",
                    data_generation=DataGeneration(science=ScienceData(rate_gbps=0.01)),
                ),
            ]
        ),
    )
    ditl.recorder = OnboardRecorder(capacity_gb=1e6)
    ditl.uend = 1e9
    ditl.ppt = None
    return ditl


def _step(ditl, utime: float, collection_seconds: float, mode=ACSMode.SCIENCE):
    generated, _ = ditl._process_data_management(
        utime, mode, STEP, collection_seconds=collection_seconds
    )
    return generated


def test_housekeeping_without_collection(ditl) -> None:
    assert _step(ditl, 0, 0, ACSMode.SLEWING) == pytest.approx(0.001 * STEP)
    assert _step(ditl, STEP, 0, ACSMode.SAFE) == pytest.approx(0.002 * STEP)
    assert ditl.recorder.current_volume_gb == pytest.approx(0.003 * STEP)


def test_per_observation_data_comes_once_per_observation(ditl) -> None:
    first = PlanEntry(begin=0, end=600, obsid=1)
    second = PlanEntry(begin=600, end=1200, obsid=2)
    housekeeping = 0.001 * STEP
    collected = 0.11 * STEP

    ditl.ppt = first
    assert _step(ditl, 0, STEP) == pytest.approx(housekeeping + collected + 2.0)
    assert _step(ditl, STEP, STEP) == pytest.approx(housekeeping + collected)
    ditl.ppt = second
    assert _step(ditl, 600, 0) == pytest.approx(housekeeping)
    assert _step(ditl, 660, STEP) == pytest.approx(housekeeping + collected + 2.0)


def test_a_new_visit_to_the_same_target_is_a_new_observation(ditl) -> None:
    target = PlanEntry(begin=0, end=600, obsid=1)
    ditl.ppt = target
    _step(ditl, 0, STEP)
    target.begin, target.end = 3000, 3600
    assert _step(ditl, 3000, STEP) == pytest.approx(0.001 * STEP + 0.11 * STEP + 2.0)


def test_the_target_sets_its_instruments_science_data(ditl) -> None:
    ditl.ppt = PlanEntry(
        begin=0,
        end=600,
        obsid=1,
        science_data={"camera": ScienceData(rate_gbps=0.5)},
    )
    # The camera's own per-observation data is replaced too; the spectrometer's
    # rate is unchanged.
    assert _step(ditl, 0, 30) == pytest.approx(0.001 * STEP + 0.51 * 30)


def test_housekeeping_stops_at_the_end_of_the_simulation(ditl) -> None:
    ditl.uend = 20
    assert _step(ditl, 0, 0, ACSMode.SLEWING) == pytest.approx(0.001 * 20)


def _contact(ditl, begin: float, end: float, *, rate_mbps: float, hk_gbps: float):
    """Put the DITL in a contact from ``begin`` to ``end`` at ``rate_mbps``,
    with housekeeping generated at ``hk_gbps`` in every mode."""
    ditl.config = MissionConfig(
        spacecraft_bus=SpacecraftBus(
            data_generation=HousekeepingData(rate_gbps=hk_gbps)
        ),
        ground_stations=GroundStationRegistry(
            stations=[
                GroundStation(
                    code="GS",
                    name="GS",
                    latitude_deg=0.0,
                    longitude_deg=0.0,
                    bands=[BandCapability(band="X", downlink_rate_mbps=rate_mbps)],
                )
            ]
        ),
    )
    contact = Mock()
    contact.station = "GS"
    contact.begin = begin
    contact.end = end
    ditl.acs.current_pass = contact


def test_a_full_contact_near_capacity_keeps_the_recorder_level(ditl) -> None:
    """31 of 32 Gb, housekeeping at 0.1 Gbps and a 100 Mbps link for the whole
    60 s step: the link carries what is generated and the volume stays at 31 Gb.
    Adding the step's data before removing the downlink lost 5 Gb to the cap."""
    _contact(ditl, 0, 3600, rate_mbps=100.0, hk_gbps=0.1)
    ditl.recorder = OnboardRecorder(capacity_gb=32.0, current_volume_gb=31.0)

    generated, downlinked = ditl._process_data_management(
        0, ACSMode.PASS, STEP, collection_seconds=0
    )

    assert generated == pytest.approx(6.0)
    assert downlinked == pytest.approx(6.0)
    assert ditl.recorder.current_volume_gb == pytest.approx(31.0)


def test_a_contact_ending_mid_step_sends_only_what_it_saw(ditl) -> None:
    """An empty recorder, housekeeping at 1 Mbps and a 100 Mbps contact over the
    first 30 s of a 60 s step: 0.03 Gb goes down the link and the 0.03 Gb
    generated after the contact stays on the recorder."""
    _contact(ditl, -600, 30, rate_mbps=100.0, hk_gbps=0.001)
    ditl.recorder = OnboardRecorder(capacity_gb=32.0)

    generated, downlinked = ditl._process_data_management(
        0, ACSMode.PASS, STEP, collection_seconds=0
    )

    assert generated == pytest.approx(0.06)
    assert downlinked == pytest.approx(0.03)
    assert ditl.recorder.current_volume_gb == pytest.approx(0.03)


def test_a_contact_starting_mid_step_stores_what_came_before_it(ditl) -> None:
    """A 100 Mbps contact over the last 30 s of the step empties the 1 Gb on
    the recorder and the 0.03 Gb generated before it, as well as the 0.03 Gb
    generated during it: the step ends with the recorder empty."""
    _contact(ditl, 30, 3600, rate_mbps=100.0, hk_gbps=0.001)
    ditl.recorder = OnboardRecorder(capacity_gb=32.0, current_volume_gb=1.0)

    generated, downlinked = ditl._process_data_management(
        0, ACSMode.PASS, STEP, collection_seconds=0
    )

    assert generated == pytest.approx(0.06)
    assert downlinked == pytest.approx(1.06)
    assert ditl.recorder.current_volume_gb == pytest.approx(0.0)


def test_the_same_contact_in_one_step_or_many_ends_the_same(ditl) -> None:
    """A 60 s step and sixty 1 s steps over the same contact leave the same
    volume and downlink the same amount."""
    results = []
    for step in (60, 1):
        _contact(ditl, 15, 45, rate_mbps=100.0, hk_gbps=0.05)
        ditl.recorder = OnboardRecorder(capacity_gb=32.0, current_volume_gb=31.0)
        downlinked = sum(
            ditl._process_data_management(
                utime, ACSMode.PASS, step, collection_seconds=0
            )[1]
            for utime in range(0, 60, step)
        )
        results.append((downlinked, ditl.recorder.current_volume_gb))
    assert results[0] == pytest.approx(results[1])


@pytest.mark.parametrize("step", [60, 1])
def test_repoint_gaps_can_overflow_between_contacts(ditl, step) -> None:
    from conops.common import AntennaType
    from tests.simulation.test_coverage import _walk

    _contact(ditl, 0, 60, rate_mbps=100.0, hk_gbps=0.01)
    ditl.recorder = OnboardRecorder(capacity_gb=0.05)
    coverage = _walk([[1]], [100.0], table=[(7000.0, 20.0)], repoint_seconds=10.0)
    ditl.config.spacecraft_bus.communications = CommunicationsSystem(
        antenna_pointing=AntennaPointing(
            antenna_type=AntennaType.GIMBALED, gimbal_range_deg=180.0
        )
    )
    ditl.acs.ra = ditl.acs.dec = ditl.acs.roll = 0.0
    coverage.config = ditl.config
    coverage.range_km = coverage.earth_range_km.reshape(1, -1)
    coverage._to_station[:] = [1.0, 0.0, 0.0]
    assert [(p.begin, p.end) for p in coverage.contacts] == [
        (0.0, 20.0),
        (30.0, 40.0),
        (50.0, 60.0),
    ]
    ditl.coverage = coverage
    ditl.log = Mock()
    generated = downlinked = 0.0
    for time in range(0, 60, step):
        incoming, outgoing = ditl._process_data_management(
            time, ACSMode.SAFE, step, collection_seconds=0
        )
        generated += incoming
        downlinked += outgoing
    assert generated == pytest.approx(0.6)
    assert downlinked == pytest.approx(0.5)
    assert ditl.recorder.current_volume_gb == pytest.approx(0.0)
    assert generated - downlinked == pytest.approx(0.1)
