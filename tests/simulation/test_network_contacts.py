"""Overlapping station opportunities must not create artificial outages."""

import numpy as np
import pytest

from conops.common import AntennaType
from conops.config import (
    AntennaPointing,
    BandCapability,
    CommunicationsSystem,
    GroundStation,
    GroundStationRegistry,
)
from conops.simulation import ContactSpan, PassTimes, StationCoverage
from conops.simulation.passes import StationGeometry
from conops.targets import Plan
from tests.scheduler.planning_scenario import T0, make_config


def network(
    monkeypatch, antenna_type=AntennaType.OMNI, *, windows=None, rates=None, hours=2
):
    config = make_config(hours)
    config.random_seed = 2029
    config.spacecraft_bus.communications = CommunicationsSystem(
        band_capabilities=[BandCapability(band="S", downlink_rate_mbps=1000)],
        antenna_pointing=AntennaPointing(
            antenna_type=antenna_type, gimbal_range_deg=180
        ),
    )
    steps = len(config.constraint.ephem.timestamp)
    windows = windows or [(0, 35), (25, 70), (60, 110), (100, steps)]
    rates = rates or [5, 10, 20, 100]
    config.ground_stations = GroundStationRegistry(
        stations=[
            GroundStation(
                code=f"S{k}",
                name=f"Station {k}",
                latitude_deg=0,
                longitude_deg=0,
                min_elevation_deg=10,
                bands=[BandCapability(band="S", downlink_rate_mbps=rate)],
            )
            for k, rate in enumerate(rates)
        ]
    )

    def geometry(self, station, startindex, endindex, *, out=None):
        k = int(station.code[1:])
        lo, hi = windows[k]
        elevation = np.full(steps, -90.0)
        elevation[lo:hi] = 30.0
        vectors = np.zeros((steps, 3))
        vectors[:, 0] = 1.0
        result = StationGeometry(
            elevation[startindex:endindex],
            np.full(endindex - startindex, 100000.0),
            vectors[startindex:endindex],
        )
        if out is not None:
            for target, source in zip(out, result, strict=True):
                target[:] = source
            return out
        return result

    monkeypatch.setattr(PassTimes, "station_geometry", geometry)
    return config, steps


@pytest.mark.parametrize("antenna_type", [AntennaType.OMNI, AntennaType.GIMBALED])
def test_continuous_chain_keeps_every_visible_step(monkeypatch, antenna_type):
    config, steps = network(monkeypatch, antenna_type)
    coverage = StationCoverage(config)
    assert coverage.visible.any(axis=0).all()
    assert coverage.contacts == [
        ContactSpan(T0, T0 + 35 * 60, "S0", 5),
        ContactSpan(T0 + 35 * 60, T0 + 70 * 60, "S1", 10),
        ContactSpan(T0 + 70 * 60, T0 + 110 * 60, "S2", 20),
        ContactSpan(T0 + 110 * 60, T0 + steps * 60, "S3", 100),
    ]
    assert coverage.fraction(T0, T0 + steps * 60) == 1
    assert coverage.dropped_overlapping_passes == []
    # A faster station rising does not cause a premature handover.
    assert coverage.contact_at(T0 + 30 * 60).station == "S0"


@pytest.mark.parametrize("entrypoint", ["predict", "get"])
@pytest.mark.parametrize("antenna_type", [AntennaType.OMNI, AntennaType.GIMBALED])
def test_pass_scheduler_books_continuous_handovers(
    monkeypatch, entrypoint, antenna_type
):
    config, steps = network(monkeypatch, antenna_type)
    scheduler = PassTimes(config)
    if entrypoint == "predict":
        scheduler.predict()
    else:
        scheduler.get(2025, 305)
    assert [(p.begin, p.end, p.station) for p in scheduler.passes] == [
        (T0, T0 + 35 * 60, "S0"),
        (T0 + 35 * 60, T0 + 70 * 60, "S1"),
        (T0 + 70 * 60, T0 + 110 * 60, "S2"),
        (T0 + 110 * 60, T0 + steps * 60, "S3"),
    ]
    assert all(not p.requires_tracking for p in scheduler.passes)
    assert scheduler.current_pass(T0 + 34 * 60).station == "S0"
    assert scheduler.current_pass(T0 + 35 * 60).station == "S1"
    assert scheduler.next_pass(T0).station == "S1"
    assert not scheduler.dropped_overlapping_passes
    # Coverage consumes the scheduler's bookings; it makes no selection of its own.
    coverage = StationCoverage(config)
    assert [(s.begin, s.end, s.station) for s in coverage.scheduled_passes] == [
        (p.begin, p.end, p.station) for p in scheduler.passes
    ]


@pytest.mark.parametrize("entrypoint", ["predict", "get"])
def test_pass_scheduler_keeps_short_network_windows(monkeypatch, entrypoint):
    config, steps = network(monkeypatch, windows=[(0, 2), (2, 121)], rates=[10, 10])
    scheduler = PassTimes(config)
    if entrypoint == "predict":
        scheduler.predict()
    else:
        scheduler.get(2025, 305)
    assert [(p.begin, p.end, p.station) for p in scheduler.passes] == [
        (T0, T0 + 120, "S0"),
        (T0 + 120, T0 + steps * 60, "S1"),
    ]


def test_pass_scheduler_can_return_unscheduled_physical_windows(monkeypatch):
    config, _ = network(monkeypatch)
    for station in config.ground_stations.stations:
        station.schedule_probability = 0
    scheduler = PassTimes(config)
    scheduler.get(2025, 305, apply_scheduling=False)
    assert [p.station for p in scheduler.passes] == ["S0", "S1", "S2", "S3"]
    assert scheduler.passes[0].end > scheduler.passes[1].begin


def test_day_by_day_scheduling_keeps_the_link_across_midnight(monkeypatch):
    config, steps = network(
        monkeypatch, windows=[(0, 1500), (1400, 2881)], rates=[10, 20], hours=48
    )
    scheduler = PassTimes(config)
    scheduler.get(2025, 305)
    scheduler.get(2025, 306)
    whole = PassTimes(config)
    whole.predict()
    # Two requested days exclude the ephemeris's final endpoint sample.
    expected = [(T0, T0 + 1500 * 60, "S0"), (T0 + 1500 * 60, T0 + 2880 * 60, "S1")]
    assert [(p.begin, p.end, p.station) for p in scheduler.passes] == expected
    assert [(p.begin, p.end, p.station) for p in whole.passes] == [
        expected[0],
        (expected[1][0], T0 + steps * 60, "S1"),
    ]


def test_handover_selects_best_current_rate_and_ties_are_stable(monkeypatch):
    config, steps = network(
        monkeypatch, windows=[(0, 20), (10, 121), (10, 121)], rates=[5, 10, 20]
    )
    coverage = StationCoverage(config)
    assert [s.station for s in coverage.contacts] == ["S0", "S2"]
    config.ground_stations.stations[1].bands[0].downlink_rate_mbps = 20
    config.ground_stations.stations.reverse()
    tied = StationCoverage(config)
    assert [s.station for s in tied.contacts] == ["S0", "S1"]
    assert tied.contacts[-1].end == T0 + steps * 60


def test_short_visibility_is_not_discarded_as_an_eight_minute_pass(monkeypatch):
    config, steps = network(monkeypatch, windows=[(0, 2), (2, 121)], rates=[10, 10])
    coverage = StationCoverage(config)
    assert coverage.fraction(T0, T0 + steps * 60) == 1
    assert coverage.contacts[0] == ContactSpan(T0, T0 + 120, "S0", 10)


@pytest.mark.parametrize("antenna_type", [AntennaType.OMNI, AntennaType.GIMBALED])
def test_handover_costs_and_saved_contact_replay(monkeypatch, antenna_type, tmp_path):
    config, steps = network(monkeypatch, antenna_type)
    comms = config.spacecraft_bus.communications
    comms.acquisition_seconds = 25
    comms.handover_seconds = 15
    coverage = StationCoverage(config)
    expected = [
        ContactSpan(T0 + 25, T0 + 35 * 60, "S0", 5),
        ContactSpan(T0 + 35 * 60 + 40, T0 + 70 * 60, "S1", 10),
        ContactSpan(T0 + 70 * 60 + 40, T0 + 110 * 60, "S2", 20),
        ContactSpan(T0 + 110 * 60 + 40, T0 + steps * 60, "S3", 100),
    ]
    assert coverage.contacts == expected
    assert coverage.acquisitions == 1 and coverage.handovers == 3
    volume = sum((s.end - s.begin) * s.rate_mbps / 1000 for s in expected)
    assert coverage.downlink_gb(T0, T0 + steps * 60) == pytest.approx(volume)
    plan = Plan(
        ground_contacts=[(s.begin, s.end, s.station) for s in coverage.scheduled_passes]
    )
    saved = Plan.load(plan.save(tmp_path / "plan.json"))
    for station in config.ground_stations.stations:
        station.schedule_probability = 0
    replay = StationCoverage(config, scheduled_passes=saved.ground_contacts)
    assert replay.contacts == expected
    assert replay.repoints == coverage.repoints
    assert replay.downlink_gb(T0, T0 + steps * 60) == pytest.approx(volume)


def test_unavailable_windows_and_true_visibility_gaps_are_preserved(monkeypatch):
    config, steps = network(
        monkeypatch, windows=[(0, 20), (10, 60), (80, 121)], rates=[10, 20, 10]
    )
    config.ground_stations.stations[1].schedule_probability = 0
    coverage = StationCoverage(config)
    assert coverage.contacts == [
        ContactSpan(T0, T0 + 20 * 60, "S0", 10),
        ContactSpan(T0 + 80 * 60, T0 + steps * 60, "S2", 10),
    ]
    assert coverage.acquisitions == 2 and coverage.handovers == 0


def test_scheduled_policy_retains_whole_pass_booking(monkeypatch):
    config, steps = network(monkeypatch)
    config.spacecraft_bus.communications.contact_policy = "scheduled"
    coverage = StationCoverage(config)
    assert coverage.contacts == [ContactSpan(T0 + 100 * 60, T0 + steps * 60, "S3", 100)]
    assert len(coverage.dropped_overlapping_passes) == 3


def test_handover_uses_range_dependent_rate_not_nominal_peak(monkeypatch):
    config, _ = network(
        monkeypatch, windows=[(0, 20), (10, 121), (10, 121)], rates=[5, 100, 20]
    )
    config.ground_stations.stations[1].bands[0].downlink_rate_vs_range_km = [
        (0, 100),
        (100000, 1),
    ]
    coverage = StationCoverage(config)
    assert [s.station for s in coverage.contacts] == ["S0", "S2"]


def test_availability_drawn_once_per_window_and_repeats_with_seed(monkeypatch):
    config, _ = network(monkeypatch)
    for station in config.ground_stations.stations:
        station.schedule_probability = 0.5
    first = StationCoverage(config)
    second = StationCoverage(config)
    assert first.scheduled_passes == second.scheduled_passes
    assert first.contacts == second.contacts
    draws = []

    def draw(self):
        draws.append(1)
        return 0.25

    monkeypatch.setattr(PassTimes, "_random", draw)
    all_available = StationCoverage(config)
    assert len(draws) == 4  # Not once per sample or selected handover segment.
    assert [s.station for s in all_available.contacts] == ["S0", "S1", "S2", "S3"]


@pytest.mark.parametrize("policy", ["continuous", "scheduled"])
def test_policy_survives_configuration_serialization(policy):
    config = CommunicationsSystem(contact_policy=policy)
    loaded = CommunicationsSystem.model_validate_json(config.model_dump_json())
    assert loaded.contact_policy == policy
