"""Independent computer storage must not borrow another store's room or link."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from conops.common.enums import ACSMode
from conops.config import (
    BandCapability,
    CommunicationsSystem,
    DataGeneration,
    DataStorage,
    DataStore,
    DownlinkRoute,
    GroundStation,
    GroundStationRegistry,
    HousekeepingData,
    Instrument,
    MissionConfig,
    OnboardRecorder,
    Payload,
    ScienceData,
    SpacecraftBus,
)
from conops.ditl import QueueDITL, RollingHorizonDITL
from conops.ditl.ditl_mixin import DITLMixin
from conops.schedulers import LocalSearchScheduler, PriorityScheduler
from conops.schedulers.recorder import RecorderLedgers, contact_downlink_gb
from conops.targets import PlanEntry, Pointing, TargetQueue

from .planning_scenario import BEGIN, make_config
from .test_recorder_constraint import (
    HOURS,
    STEP,
    _data_target,
)


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


def config():
    return MissionConfig(
        data_storage=storage(),
        spacecraft_bus=SpacecraftBus(
            data_generation=HousekeepingData(rate_gbps=0.01),
            communications=CommunicationsSystem(
                band_capabilities=[BandCapability(band="Ka", downlink_rate_mbps=100)]
            ),
        ),
        payload=Payload(
            instruments=[
                Instrument(
                    name="camera",
                    data_generation=DataGeneration(
                        science=ScienceData(rate_gbps=0.1, per_observation_gb=2),
                        housekeeping=HousekeepingData(rate_gbps=0.001),
                    ),
                )
            ]
        ),
        ground_stations=GroundStationRegistry(
            stations=[
                GroundStation(
                    code="GS",
                    name="GS",
                    latitude_deg=0,
                    longitude_deg=0,
                    bands=[BandCapability(band="Ka", downlink_rate_mbps=200)],
                )
            ]
        ),
    )


def test_roundtrip_and_legacy_assignment(tmp_path):
    c = config()
    c.name = "updated"
    dumped = c.model_dump(mode="json", exclude_none=True)
    assert "recorder" not in dumped
    loaded = MissionConfig.model_validate(dumped)
    assert loaded.recorder_views()["payload"].recorder.capacity_gb == 64000
    c.to_yaml_file(str(tmp_path / "config.yaml"))
    assert (
        MissionConfig.from_yaml_file(str(tmp_path / "config.yaml")).data_storage
        == c.data_storage
    )
    with pytest.raises(ValidationError, match="either recorder or data_storage"):
        MissionConfig(data_storage=storage(), recorder=OnboardRecorder())
    legacy = MissionConfig(
        recorder=OnboardRecorder(
            current_volume_gb=0, capacity_gb=20, planning_limit=None
        )
    )
    assert (
        MissionConfig.model_validate(legacy.model_dump()).recorder.planning_limit
        is None
    )


def test_inventory_is_not_an_unlimited_recorder():
    c = MissionConfig(
        data_storage=DataStorage(
            stores={
                "payload": DataStore(owner="payload", installed_capacity_gb=64000),
                "bus": DataStore(owner="spacecraft_bus"),
            }
        )
    )
    with pytest.raises(
        ValueError,
        match="Unresolved data storage configuration.*stores.payload.recorder.*downlinks",
    ):
        c.recorder_views()
    for constructor in (
        lambda: PriorityScheduler(c, [], BEGIN, BEGIN + timedelta(hours=1)).schedule(),
        lambda: DITLMixin._init_subsystems(SimpleNamespace(config=c)),
    ):
        with pytest.raises(ValueError, match="Unresolved data storage"):
            constructor()


def test_source_routing_honors_modes_and_science_overrides():
    c = config()
    views = c.recorder_views()
    assert views["bus"].housekeeping_generated(10) == pytest.approx(0.11)
    assert views["payload"].housekeeping_generated(10) == 0
    assert views["payload"].science_generated(10) == 3
    assert views["bus"].science_generated(10) == 0
    assert (
        views["payload"].science_generated(10, {"camera": ScienceData(rate_gbps=1)})
        == 10
    )
    with pytest.raises(ValueError):
        views["bus"].science_generated(10, {"typo": ScienceData(rate_gbps=1)})


@pytest.mark.parametrize(
    "shares", [{"payload": 1, "bus": 1}, {"payload": float("nan")}, {"payload": -0.1}]
)
def test_cannot_allocate_a_link_twice(shares):
    with pytest.raises(ValueError):
        DownlinkRoute(station="GS", band="Ka", shares=shares)


def test_separate_capacity_and_shared_physical_link():
    c = config()
    c.data_storage.downlinks = [
        DownlinkRoute(station="GS", band="Ka", shares={"payload": 0.75, "bus": 0.25})
    ]
    views = c.recorder_views()
    assert [contact_downlink_gb(views[n], "GS", 100) for n in ("payload", "bus")] == [
        7.5,
        2.5,
    ]
    ledger = RecorderLedgers(c, 0, 100, [])
    initial = ledger.initial_volumes([])
    assert ledger.fits_all(initial, [(0, 1000)])
    assert not ledger.fits_all(initial, [(1, 7)])  # 1.1 Gb housekeeping + 7 > 8
    with pytest.raises(ValueError, match="by store"):
        RecorderLedgers(c, 0, 100, [], start_volume_gb=5)


def runtime(c):
    d = object.__new__(QueueDITL)
    d.config = c
    d.uend = 1000
    d.ustart = 0
    d.step_size = 10
    d._data_observation = None
    d._data_collected = 0
    d.ppt = PlanEntry(begin=0, end=100, obsid=1)
    d.acs = SimpleNamespace(current_pass=None, passrequests=SimpleNamespace(passes=[]))
    d._init_subsystems()
    return d


def test_execution_fixed_science_once_and_shared_link_conservation():
    c = config()
    d = runtime(c)
    for t in [0, 10]:
        d._process_data_management(t, ACSMode.SCIENCE, 10, collection_seconds=10)
    views = c.recorder_views()
    assert views["payload"].recorder.current_volume_gb == 4
    assert views["bus"].recorder.current_volume_gb == pytest.approx(0.22)
    c.data_storage.downlinks = [
        DownlinkRoute(station="GS", band="Ka", shares={"payload": 0.75, "bus": 0.25})
    ]
    views["bus"].recorder.current_volume_gb = 5
    d.acs.current_pass = SimpleNamespace(station="GS", begin=20, end=30)
    generated, downlinked = d._process_data_management(
        20, ACSMode.PASS, 10, collection_seconds=0
    )
    assert generated == pytest.approx(0.11)
    assert downlinked == pytest.approx(1.0)
    assert views["payload"].recorder.current_volume_gb == pytest.approx(3.25)
    assert views["bus"].recorder.current_volume_gb == pytest.approx(4.86)
    # An empty store's unused reservation never becomes another full link.
    views["payload"].recorder.current_volume_gb = 0
    d.acs.current_pass.end = 40
    _, downlinked = d._process_data_management(
        30, ACSMode.PASS, 10, collection_seconds=0
    )
    assert downlinked == pytest.approx(0.25)


def test_dispatch_room_and_rolling_state_remain_separate():
    c = config()
    d = runtime(c)
    d.uend = 100
    c.recorder_views()["bus"].recorder.current_volume_gb = 7
    rooms = d._recorder_room_gb(0, 0, 0, 0)
    assert rooms["bus"] < 0 and rooms["payload"] > 50000
    queue = TargetQueue(config=c)
    assert queue.collection_within(Pointing(), rooms) < 0
    before = {n: v.recorder.current_volume_gb for n, v in c.recorder_views().items()}
    projected = d._recorder_volume_at(0, 10)
    assert projected["bus"] == pytest.approx(7.11)
    rolling = object.__new__(RollingHorizonDITL)
    rolling.config = c
    rolling.step_size = 10
    rolling.ustart = 0
    projected = rolling._recorder_volume_at(0, 10, [])
    assert projected == pytest.approx({"payload": 0, "bus": 7.11})
    assert before == {
        n: v.recorder.current_volume_gb for n, v in c.recorder_views().items()
    }


@pytest.mark.parametrize("scheduler_class", [PriorityScheduler, LocalSearchScheduler])
def test_planners_cannot_borrow_payload_space_for_housekeeping(scheduler_class):
    c = make_config(HOURS, stations=False)
    c.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0.0001)
    c.data_storage = storage("Default Instrument")
    c.data_storage.stores["bus"].recorder = OnboardRecorder(
        current_volume_gb=0, capacity_gb=0.1, planning_limit=0.8
    )
    target = _data_target(c, 1, 105, 10, 0.001, minutes=20)
    scheduler = scheduler_class(
        c,
        [target],
        BEGIN,
        BEGIN + timedelta(hours=HOURS),
        step_size=STEP,
        include_passes=False,
    )
    plan = scheduler.schedule()
    assert not any(e.collection_begin is not None for e in plan)
    assert any(
        "bus" in e.description and "planning limit" in e.description
        for e in scheduler.log.events
    )


def test_chunk_solver_enforces_each_store():
    pytest.importorskip("ortools")
    import numpy as np
    from ortools.sat.python import cp_model

    from conops.schedulers.cpsat_scheduler import _ChunkModel

    from .test_cpsat_recorder import ORIGIN, _recorder, _snapshot, _Task

    tasks = [
        _Task(None, ORIGIN, (0, 0)),
        _snapshot(1, (0, 3000), 0),
        _snapshot(2, (0, 3000), 0),
    ]
    payload = _recorder([], [], limit=100)._replace(
        data=[(0, 0), (0, 20_000_000), (0, 20_000_000)]
    )
    bus = _recorder([], [], limit=1)._replace(data=[(0, 0), (0, 900_000), (0, 900_000)])
    chunk = _ChunkModel(
        tasks, np.zeros((3, 3)), 60, 20000, {1: 1e9, 2: 1e9}, [payload, bus]
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    assert solver.solve(chunk.model) == cp_model.OPTIMAL
    assert sum(bool(solver.value(v)) for v in chunk.present[1:]) == 1


def test_faults_and_overflow_are_per_store():
    from unittest.mock import Mock

    from conops.ditl.telemetry import Housekeeping

    c = config()
    d = runtime(c)
    bus = c.recorder_views()["bus"].recorder
    bus.current_volume_gb = bus.capacity_gb
    generated, downlinked = d._process_data_management(
        0, ACSMode.IDLE, 10, collection_seconds=0
    )
    snapshot = d._recorder_telemetry()
    assert snapshot["bus"]["lost_gb"] == pytest.approx(generated)
    assert snapshot["bus"]["generated_gb"] == pytest.approx(0.11)
    assert snapshot["payload"]["lost_gb"] == 0
    assert downlinked == 0
    assert d.recorder.get_alert_level() == 2
    run = c.fault_management.new_run()
    acs = Mock(in_safe_mode=False, acsmode=ACSMode.IDLE)
    acs.ephem = Mock(step_size=10)
    classified = run.check(Housekeeping(timestamp=BEGIN, recorders=snapshot), acs)
    assert classified["recorders.bus.fill_fraction"] == "red"
    assert classified["recorders.payload.fill_fraction"] == "nominal"
    assert run.safe_mode_requested


def test_bad_routes_and_capacity_fail_explicitly():
    with pytest.raises(ValueError, match="exceeds installed"):
        DataStore(
            owner="payload",
            installed_capacity_gb=10,
            recorder=OnboardRecorder(current_volume_gb=0, capacity_gb=20),
        )
    with pytest.raises(ValueError, match="unknown store"):
        DataStorage(
            stores={"payload": DataStore(owner="payload")}, bus_housekeeping="typo"
        )
    c = config()
    c.data_storage.downlinks = [
        DownlinkRoute(station="GS", band="S", shares={"bus": 1})
    ]
    with pytest.raises(ValueError, match="no compatible S downlink"):
        c.recorder_views()


def test_named_recorder_plots_use_each_capacity():
    import matplotlib.pyplot as plt

    from conops.ditl.telemetry import Housekeeping, Telemetry
    from conops.visualization import (
        plot_data_management_telemetry,
        plot_data_management_telemetry_plotly,
    )

    d = runtime(config())
    d.telemetry = Telemetry()
    d.telemetry.housekeeping.append(
        Housekeeping(
            timestamp=BEGIN,
            recorders=d._recorder_telemetry(),
            recorder_volume_gb=0,
            recorder_fill_fraction=0,
            recorder_alert=0,
        )
    )
    d.data_generated_gb = [0]
    d.data_downlinked_gb = [0]
    fig, axes = plot_data_management_telemetry(d, show_summary=False)
    assert {line.get_label() for line in axes[0].lines} == {
        "payload",
        "bus",
        "payload capacity",
        "bus capacity",
    }
    plt.close(fig)
    interactive = plot_data_management_telemetry_plotly(d)
    assert {trace.name for trace in interactive.data} >= {
        "payload volume_gb",
        "bus volume_gb",
    }


def test_named_recorders_do_not_default_hardware_inputs():
    with pytest.raises(ValueError, match="explicit capacity_gb, current_volume_gb"):
        DataStore(owner="payload", recorder=OnboardRecorder())
    with pytest.raises(ValueError, match="explicit current_volume_gb"):
        DataStore(owner="payload", recorder=OnboardRecorder(capacity_gb=64000))


def test_mutated_routes_are_revalidated_before_use():
    c = config()
    c.data_storage.instrument_science["camera"] = "typo"
    with pytest.raises(ValueError, match="unknown store"):
        c.recorder_views()
    c.data_storage.instrument_science["camera"] = "payload"
    c.data_storage.downlinks = [
        DownlinkRoute(station="GS", band="Ka", shares={"payload": 0.75, "bus": 0.25})
    ]
    c.data_storage.downlinks[0].shares["bus"] = 1.0
    with pytest.raises(ValueError, match="more than one physical downlink"):
        c.recorder_views()


def test_nested_config_also_rejects_ambiguous_storage():
    from pydantic import BaseModel

    class Wrapped(BaseModel):
        config: MissionConfig

    with pytest.raises(ValueError, match="either recorder or data_storage"):
        Wrapped.model_validate({"config": {"recorder": {}, "data_storage": storage()}})
