"""The shortest visibility window booked as a whole pass is configurable."""

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


def test_fixed_antennas_use_it(monkeypatch: pytest.MonkeyPatch) -> None:
    config = network(
        monkeypatch, antenna_type=AntennaType.FIXED, windows=WINDOWS, rates=[5, 10, 20]
    )[0]
    communications = config.spacecraft_bus.communications
    assert communications is not None
    communications.min_pass_seconds = 240.0
    assert ContactScheduler(config).minlen == 240.0


def test_without_a_communications_system(monkeypatch: pytest.MonkeyPatch) -> None:
    config: MissionConfig = network(monkeypatch)[0]
    config.spacecraft_bus.communications = None
    assert ContactScheduler(config).minlen == 480.0


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        CommunicationsSystem(min_pass_seconds=value)


def test_survives_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    system = CommunicationsSystem(min_pass_seconds=240.0)
    assert CommunicationsSystem.model_validate(system.model_dump(mode="json")) == system
