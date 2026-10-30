"""Contact bars for the timeline's Ground Contact row."""

import sys
from pathlib import Path

import pytest

from conops.common import AntennaType
from conops.config import AntennaPointing, BandCapability, CommunicationsSystem
from conops.simulation import StationCoverage
from conops.visualization.contacts import (
    CONTACT_BAR_MAX_HOURS,
    STATION_COLORS,
    ContactBar,
    contact_alphas,
    contact_bars,
    run_window,
    station_colors,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scheduler"))
from planning_scenario import HOUR, T0  # noqa: E402
from planning_scenario import make_config as _config  # noqa: E402

HOURS = 72


@pytest.fixture(scope="module")
def coverage() -> StationCoverage:
    config = _config(HOURS, stations=True)
    config.spacecraft_bus.communications = CommunicationsSystem(
        name="S-band",
        band_capabilities=[BandCapability(band="S", downlink_rate_mbps=10.0)],
        antenna_pointing=AntennaPointing(antenna_type=AntennaType.OMNI),
    )
    built = StationCoverage.for_config(config)
    assert built is not None
    return built


class TestContactBars:
    def test_a_short_run_has_a_bar_per_span(self, coverage: StationCoverage) -> None:
        end = T0 + 24 * HOUR
        bars = contact_bars(coverage, T0, end, T0, 0.0)
        spans = coverage.spans(T0, end)
        assert len(bars) == len(spans)
        for bar, span in zip(bars, spans):
            assert bar.start == pytest.approx((span.begin - T0) / 3600)
            assert bar.duration == pytest.approx((span.end - span.begin) / 3600)
            assert bar.station == span.station
            assert bar.fraction == 1.0
            assert bar.rate_mbps == 10.0

    def test_the_offset_shifts_the_bars(self, coverage: StationCoverage) -> None:
        end = T0 + 24 * HOUR
        plain = contact_bars(coverage, T0, end, T0, 0.0)
        shifted = contact_bars(coverage, T0, end, T0 - HOUR, 2.0)
        assert [b.start for b in shifted] == pytest.approx(
            [b.start + 1.0 - 2.0 for b in plain]
        )

    def test_a_long_run_has_a_bar_per_hour_in_contact(
        self, coverage: StationCoverage
    ) -> None:
        end = T0 + HOURS * HOUR
        assert HOURS > CONTACT_BAR_MAX_HOURS
        bars = contact_bars(coverage, T0, end, T0, 0.0)
        spans = coverage.spans(T0, end)
        assert 0 < len(bars) < len(spans) * 2
        for bar in bars:
            assert bar.duration == 1.0
            assert bar.start == int(bar.start)
            assert 0.0 < bar.fraction < 1.0
            hour = T0 + bar.start * HOUR
            inside = [
                (min(s.end, hour + HOUR) - max(s.begin, hour), s.station)
                for s in spans
                if min(s.end, hour + HOUR) > max(s.begin, hour)
            ]
            assert bar.fraction == pytest.approx(sum(t for t, _ in inside) / HOUR)
            # The station with most of the hour.
            by_station: dict[str, float] = {}
            for seconds, station in inside:
                by_station[station] = by_station.get(station, 0.0) + seconds
            assert by_station[bar.station] == max(by_station.values())
        # Every hour with a contact has a bar, and no other.
        hours_in_contact = {
            int((t - T0) // HOUR)
            for s in spans
            for t in range(int(s.begin), int(s.end), 60)
        }
        assert {int(b.start) for b in bars} == hours_in_contact


def test_alphas_scale_to_the_fullest_hour_with_a_floor() -> None:
    bars = [
        ContactBar(0.0, 1.0, "SGS", 0.2, 10.0),
        ContactBar(1.0, 1.0, "SGS", 0.02, 10.0),
        ContactBar(2.0, 1.0, "MAL", 0.1, 10.0),
    ]
    assert contact_alphas(bars) == pytest.approx([1.0, 0.3, 0.5])
    spans = [ContactBar(0.0, 0.1, "SGS", 1.0, 10.0)]
    assert contact_alphas(spans) == [1.0]
    assert contact_alphas([]) == []


def test_station_colors_in_order_of_appearance() -> None:
    colors = station_colors(["SGS", "MAL", "SGS", "HI1"])
    assert list(colors) == ["SGS", "MAL", "HI1"]
    assert list(colors.values()) == STATION_COLORS[:3]


def test_run_window_from_the_steps() -> None:
    class Run:
        utime = [10.0, 70.0, 130.0]
        step_size = 60

    assert run_window(Run()) == (10.0, 190.0)
    assert run_window(object()) is None
