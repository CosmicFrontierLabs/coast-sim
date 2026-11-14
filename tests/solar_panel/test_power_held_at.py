"""Panel power at many samples together equals the power sample by sample."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from rust_ephem import TLEEphemeris

from conops.common.vector import scbodyvector, scbodyvectors
from conops.config.solar_panel import SolarPanel, SolarPanelSet

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
T0 = BEGIN.timestamp()


@pytest.fixture(scope="module")
def ephem() -> TLEEphemeris:
    return TLEEphemeris(
        tle="examples/example.tle",
        begin=BEGIN,
        end=BEGIN + timedelta(hours=3),
        step_size=60,
    )


def _panels() -> SolarPanelSet:
    return SolarPanelSet(
        panels=[
            SolarPanel(normal=(0.0, 1.0, 0.0), max_power=400.0),
            SolarPanel(normal=(0.0, 0.0, 1.0), max_power=200.0),
            SolarPanel(normal=(1.0, 0.0, 0.0), max_power=100.0, gimbled=True),
        ]
    )


def test_body_vectors_match_the_single_rotation() -> None:
    rng = np.random.default_rng(3)
    attitudes = np.deg2rad(rng.uniform((0, -90, 0), (360, 90, 360), size=(25, 3)))
    vectors = rng.normal(size=(25, 3))
    together = scbodyvectors(attitudes, vectors)
    for k in range(25):
        one = scbodyvector(*attitudes[k], vectors[k])
        assert together[k] == pytest.approx(one, abs=1e-12)


def test_power_held_at_matches_each_sample(ephem: TLEEphemeris) -> None:
    """Over three hours of samples at several attitudes, eclipse included,
    the batch power is the per-sample power."""
    panels = _panels()
    times = [T0 + 60.0 * k for k in range(0, 180, 7)]
    rng = np.random.default_rng(5)
    attitudes = [
        (
            float(rng.uniform(0, 360)),
            float(rng.uniform(-90, 90)),
            float(rng.uniform(0, 360)),
        )
        for _ in times
    ]
    eclipse = [
        bool(
            panels.panels[0]._eclipse_constraint.in_constraint(
                ephemeris=ephem,
                target_ra=0.0,
                target_dec=0.0,
                time=datetime.fromtimestamp(t, timezone.utc),
            )
        )
        for t in times
    ]
    assert any(eclipse) and not all(eclipse)
    together = panels.power_held_at(times, attitudes, ephem, eclipse)
    for k, (utime, (ra, dec, roll)) in enumerate(zip(times, attitudes, strict=True)):
        assert together[k] == pytest.approx(
            float(panels.power(utime, ra, dec, ephem, roll=roll)), abs=1e-9
        )
    assert together.max() > 0.0


def test_power_held_at_without_panels_is_zero(ephem: TLEEphemeris) -> None:
    empty = SolarPanelSet(panels=[])
    assert list(
        empty.power_held_at(
            [T0, T0 + 60.0], [(0.0, 0.0, 0.0)] * 2, ephem, [False, False]
        )
    ) == [0.0, 0.0]
