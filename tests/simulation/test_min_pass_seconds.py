"""The shortest visibility window booked as a whole pass is configurable."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from conops.common import AntennaType
from conops.config import CommunicationsSystem, MissionConfig
from conops.simulation import PassTimes
from conops.simulation.contact_schedule import ContactScheduler

from tests.simulation.test_network_contacts import network

# Separate windows of 5, 8 and 20 one-minute steps.
WINDOWS = [(10, 15), (30, 38), (60, 80)]


def _booked(monkeypatch: pytest.MonkeyPatch, **fields: object) -> list[float]:
    config = network(monkeypatch, windows=WINDOWS, rates=[5, 10, 20])[0]
    communications = config.spacecraft_bus.communications
    assert communications is not None
    communications.contact_policy = "scheduled"
    for name, value in fields.items():
        setattr(communications, name, value)
    passes = PassTimes(config)
    passes.predict()
    return [float(p.length or 0.0) for p in passes.passes]


def test_default_is_eight_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert CommunicationsSystem().min_pass_seconds == 480.0
    assert _booked(monkeypatch) == [480.0, 1200.0]


@pytest.mark.parametrize(
    ("minimum", "lengths"),
    [
        (0.0, [300.0, 480.0, 1200.0]),
        (300.0, [300.0, 480.0, 1200.0]),
        (301.0, [480.0, 1200.0]),
        (600.0, [1200.0]),
        (1500.0, []),
    ],
)
def test_configured(
    monkeypatch: pytest.MonkeyPatch, minimum: float, lengths: list[float]
) -> None:
    assert _booked(monkeypatch, min_pass_seconds=minimum) == lengths


def test_continuous_contacts_keep_every_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lengths = _booked(monkeypatch, min_pass_seconds=1500.0, contact_policy="continuous")
    assert lengths == [300.0, 480.0, 1200.0]


@pytest.mark.parametrize(
    ("minimum", "lengths"), [(300.0, [300.0, 480.0, 1200.0]), (600.0, [1200.0])]
)
def test_fixed_antennas_use_it(
    monkeypatch: pytest.MonkeyPatch, minimum: float, lengths: list[float]
) -> None:
    config = network(
        monkeypatch, antenna_type=AntennaType.FIXED, windows=WINDOWS, rates=[5, 10, 20]
    )[0]
    communications = config.spacecraft_bus.communications
    assert communications is not None
    communications.min_pass_seconds = minimum
    passes = PassTimes(config)
    passes.predict()
    booked = passes.passes + passes.dropped_constraint_passes
    assert sorted(float(p.length or 0.0) for p in booked) == lengths


def test_read_when_passes_are_booked(monkeypatch: pytest.MonkeyPatch) -> None:
    config = network(monkeypatch, windows=WINDOWS, rates=[5, 10, 20])[0]
    communications = config.spacecraft_bus.communications
    assert communications is not None
    communications.contact_policy = "scheduled"
    passes = PassTimes(config)
    assert passes.minlen == 480.0
    communications.min_pass_seconds = 240.0
    assert passes.minlen == 240.0
    passes.predict()
    assert [p.length for p in passes.passes] == [300.0, 480.0, 1200.0]
    passes.minlen = 600.0
    assert passes.minlen == 600.0


def test_replay_keeps_windows_shorter_than_the_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contacts already chosen are not chosen again, as for the probability."""
    config = network(monkeypatch, windows=WINDOWS, rates=[5, 10, 20])[0]
    communications = config.spacecraft_bus.communications
    assert communications is not None
    communications.contact_policy = "scheduled"
    scheduler = ContactScheduler(config)
    station = config.ground_stations.stations[0]
    assert not scheduler.accepts(station, 300.0)
    assert scheduler.accepts(station, 300.0, apply_scheduling=False)


@pytest.mark.parametrize("apply_scheduling", [True, False])
def test_a_window_without_duration_is_not_a_pass(
    monkeypatch: pytest.MonkeyPatch, apply_scheduling: bool
) -> None:
    config = network(monkeypatch)[0]
    communications = config.spacecraft_bus.communications
    assert communications is not None
    communications.contact_policy = "scheduled"
    communications.min_pass_seconds = 0.0
    scheduler = ContactScheduler(config)
    station = config.ground_stations.stations[0]
    assert not scheduler.accepts(station, 0.0, apply_scheduling=apply_scheduling)
    assert scheduler.accepts(station, 60.0, apply_scheduling=apply_scheduling)


def test_read_from_any_communications_object(monkeypatch: pytest.MonkeyPatch) -> None:
    config = network(monkeypatch)[0]
    stand_in = SimpleNamespace(
        antenna_pointing=SimpleNamespace(antenna_type=AntennaType.FIXED),
        contact_policy="scheduled",
        min_pass_seconds=123.0,
    )
    bus = SimpleNamespace(communications=stand_in)
    scheduler = ContactScheduler(SimpleNamespace(random_seed=1, spacecraft_bus=bus))  # type: ignore[arg-type]
    assert scheduler.minlen == 123.0
    assert ContactScheduler(config).continuous_contacts


def test_without_a_communications_system(monkeypatch: pytest.MonkeyPatch) -> None:
    config: MissionConfig = network(monkeypatch)[0]
    config.spacecraft_bus.communications = None
    assert ContactScheduler(config).minlen == 480.0


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        CommunicationsSystem(min_pass_seconds=value)


def test_survives_yaml(tmp_path: Path) -> None:
    config = MissionConfig()
    config.spacecraft_bus.communications = CommunicationsSystem(min_pass_seconds=240.0)
    path = tmp_path / "mission.yaml"
    config.to_yaml_file(str(path))
    loaded = MissionConfig.from_yaml_file(str(path))
    assert loaded.spacecraft_bus.communications is not None
    assert loaded.spacecraft_bus.communications.min_pass_seconds == 240.0
