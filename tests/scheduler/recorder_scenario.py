"""Shared split-recorder storage for the recorder planning tests."""

from conops.config import DataStorage, DataStore, OnboardRecorder


def storage(instrument="camera"):
    return DataStorage(
        stores={
            "payload": DataStore(
                owner="payload",
                installed_capacity_gb=64000,
                recorder=OnboardRecorder(
                    current_volume_gb=0, capacity_gb=64000, planning_limit=0.8
                ),
            ),
            "bus": DataStore(
                owner="spacecraft_bus",
                recorder=OnboardRecorder(
                    current_volume_gb=0, capacity_gb=10, planning_limit=0.8
                ),
            ),
        },
        bus_housekeeping="bus",
        instrument_science={instrument: "payload"},
        instrument_housekeeping={instrument: "bus"},
        downlinks=[],
    )
