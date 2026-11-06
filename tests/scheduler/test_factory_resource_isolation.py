"""Fresh factory simulations must start with the configured resource state."""

from datetime import timedelta

import pytest

from conops import create_ditl
from conops.config import DataStorage, DataStore, HousekeepingData, OnboardRecorder
from conops.config.scheduler import SchedulerKind, SchedulerMode

from .planning_scenario import BEGIN, PATCH, make_config, make_target


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("mode", list(SchedulerMode))
def test_factory_owns_resource_state_and_preserves_configured_initial_values(
    mode: SchedulerMode, named: bool
) -> None:
    config = make_config(1)
    config.scheduler.mode = mode
    config.scheduler.kind = SchedulerKind.PRIORITY
    config.scheduler.include_passes = False
    config.battery.charge_level = 0.8 * config.battery.watthour
    config.recorder.current_volume_gb = 12.0
    if named:
        names = [instrument.name for instrument in config.payload.instruments]
        config.data_storage = DataStorage(
            stores={
                "science": DataStore(
                    owner="payload",
                    recorder=OnboardRecorder(capacity_gb=1000, current_volume_gb=12),
                ),
                "bus": DataStore(
                    owner="spacecraft_bus",
                    recorder=OnboardRecorder(capacity_gb=100, current_volume_gb=3),
                ),
            },
            bus_housekeeping="bus",
            instrument_science=dict.fromkeys(names, "science"),
            instrument_housekeeping=dict.fromkeys(names, "bus"),
            downlinks=[],
        )
    # Exercise the view cache too: copied views must refer to their own stores.
    initial_volumes = {
        name: view.recorder.current_volume_gb
        for name, view in config.recorder_views().items()
    }
    first = create_ditl(config, begin=BEGIN, end=BEGIN + timedelta(hours=1))
    first.battery.charge_level = 0.4 * first.battery.watthour
    for view in first.config.recorder_views().values():
        view.recorder.add_data(10.0)
    second = create_ditl(config, begin=BEGIN, end=BEGIN + timedelta(hours=1))
    assert second.battery.battery_level == pytest.approx(0.8)
    assert config.battery.battery_level == pytest.approx(0.8)
    for name, view in second.config.recorder_views().items():
        assert view.recorder.current_volume_gb == initial_volumes[name]
        assert (
            config.recorder_views()[name].recorder.current_volume_gb
            == initial_volumes[name]
        )
        assert (
            first.config.recorder_views()[name].recorder.current_volume_gb
            == initial_volumes[name] + 10
        )


def test_reusing_configuration_repeats_the_actual_dispatch_execution() -> None:
    config = make_config(1)
    config.scheduler.mode = SchedulerMode.DISPATCH
    config.scheduler.include_passes = False
    config.battery.charge_level = 0.8 * config.battery.watthour
    config.recorder.current_volume_gb = 12.0
    config.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0.001)
    target = make_target(config, 701, *PATCH[0], minutes=10)
    records = []
    for _ in range(2):
        run = create_ditl(
            config, targets=[target], begin=BEGIN, end=BEGIN + timedelta(hours=1)
        )
        assert run.battery.battery_level == pytest.approx(0.8)
        assert run.recorder.current_volume_gb == 12.0
        assert run.calc()
        assert run.recorder.current_volume_gb > 12.0
        assert run.battery.battery_level < 0.8
        records.append(
            {
                "entries": [e.model_dump(mode="json") for e in run.plan],
                "housekeeping": [
                    h.model_dump(mode="json") for h in run.telemetry.housekeeping
                ],
            }
        )
    assert records[0] == records[1]
    assert config.battery.battery_level == pytest.approx(0.8)
    assert config.recorder.current_volume_gb == 12.0
