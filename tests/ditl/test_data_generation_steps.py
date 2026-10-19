"""Data the DITL generates each step from the configuration and the target."""

import pytest

from conops.common.enums import ACSMode
from conops.config import (
    DataGeneration,
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
