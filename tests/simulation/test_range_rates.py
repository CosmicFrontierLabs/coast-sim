"""Downlink rates that vary with the spacecraft-to-station range."""

import sys
from pathlib import Path

import numpy as np
import pytest

from conops.common import AntennaType
from conops.config import (
    AntennaPointing,
    BandCapability,
    CommunicationsSystem,
    GroundStation,
    MissionConfig,
)
from conops.config.groundstation import (
    effective_downlink_rate_mbps,
    rate_varies_with_range,
)
from conops.simulation import StationCoverage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scheduler"))
from planning_scenario import HOUR, MIN, T0  # noqa: E402
from planning_scenario import make_config as _config  # noqa: E402

TABLE = [(500.0, 20.0), (1500.0, 10.0), (3000.0, 2.0)]
HOURS = 24
END = T0 + HOURS * HOUR


def _band(table: list[tuple[float, float]] | None = TABLE) -> BandCapability:
    return BandCapability(
        band="S", downlink_rate_mbps=20.0, downlink_rate_vs_range_km=table
    )


class TestBandTable:
    def test_interpolates_between_rows_and_holds_the_ends(self) -> None:
        band = _band()
        assert band.downlink_rate_at(100.0) == 20.0
        assert band.downlink_rate_at(500.0) == 20.0
        assert band.downlink_rate_at(1000.0) == pytest.approx(15.0)
        assert band.downlink_rate_at(1500.0) == 10.0
        assert band.downlink_rate_at(2250.0) == pytest.approx(6.0)
        assert band.downlink_rate_at(3000.0) == 2.0
        assert band.downlink_rate_at(9000.0) == 2.0

    def test_without_a_range_or_a_table_the_nominal_rate_holds(self) -> None:
        assert _band().downlink_rate_at(None) == 20.0
        assert _band(None).downlink_rate_at(2250.0) == 20.0

    def test_one_row_is_a_flat_rate(self) -> None:
        assert _band([(1000.0, 7.0)]).downlink_rate_at(10.0) == 7.0
        assert _band([(1000.0, 7.0)]).downlink_rate_at(5000.0) == 7.0

    @pytest.mark.parametrize(
        "table",
        [
            [],
            [(1000.0, 5.0), (1000.0, 4.0)],
            [(2000.0, 5.0), (1000.0, 4.0)],
            [(-1.0, 5.0)],
            [(1.0, -5.0)],
        ],
    )
    def test_rejects_a_bad_table(self, table: list[tuple[float, float]]) -> None:
        with pytest.raises(ValueError):
            _band(table)

    def test_round_trips_through_json(self) -> None:
        loaded = BandCapability.model_validate_json(_band().model_dump_json())
        assert loaded.downlink_rate_vs_range_km == TABLE


class TestEffectiveRate:
    def _station(self, band: BandCapability) -> GroundStation:
        return GroundStation(
            code="X", name="X", latitude_deg=0.0, longitude_deg=0.0, bands=[band]
        )

    def test_the_lower_side_at_the_range_counts(self) -> None:
        station = self._station(_band())
        comms = CommunicationsSystem(band_capabilities=[_band(None)])
        assert effective_downlink_rate_mbps(station, comms) == 20.0
        assert effective_downlink_rate_mbps(station, comms, 1000.0) == pytest.approx(
            15.0
        )
        comms = CommunicationsSystem(
            band_capabilities=[BandCapability(band="S", downlink_rate_mbps=12.0)]
        )
        assert effective_downlink_rate_mbps(station, comms, 1000.0) == 12.0
        assert effective_downlink_rate_mbps(station, comms, 2250.0) == pytest.approx(
            6.0
        )
        assert rate_varies_with_range(station, comms)
        assert not rate_varies_with_range(self._station(_band(None)), comms)

    def test_the_spacecraft_side_can_carry_the_table(self) -> None:
        station = self._station(_band(None))
        comms = CommunicationsSystem(band_capabilities=[_band()])
        assert effective_downlink_rate_mbps(station, comms, 2250.0) == pytest.approx(
            6.0
        )
        assert rate_varies_with_range(station, comms)

    def test_without_a_communications_system(self) -> None:
        station = self._station(_band())
        assert effective_downlink_rate_mbps(station, None, 1000.0) == pytest.approx(
            15.0
        )
        assert effective_downlink_rate_mbps(station, None) == 20.0


def _ranged_config() -> MissionConfig:
    config = _config(HOURS, stations=True)
    for station in config.ground_stations.stations:
        station.bands = [_band()]
    config.spacecraft_bus.communications = CommunicationsSystem(
        name="S-band",
        band_capabilities=[BandCapability(band="S", downlink_rate_mbps=50.0)],
        antenna_pointing=AntennaPointing(antenna_type=AntennaType.OMNI),
    )
    return config


@pytest.fixture(scope="module")
def coverage() -> StationCoverage:
    built = StationCoverage.for_config(_ranged_config())
    assert built is not None
    return built


class TestCoverage:
    def test_each_step_in_view_has_the_rate_at_its_range(
        self, coverage: StationCoverage
    ) -> None:
        assert all(coverage._ranged)
        for k, station in enumerate(coverage.stations):
            for step in np.flatnonzero(coverage.visible[k]):
                expected = station.bands[0].downlink_rate_at(
                    float(coverage.range_km[k, step])
                )
                assert coverage.rate_steps[k, step] == pytest.approx(expected)
            assert not coverage.rate_steps[k, ~coverage.visible[k]].any()
        # Ranges in view run from a few hundred to a few thousand km, so the
        # rate varies through a contact.
        assert coverage.rate_steps[coverage.visible].min() < 8.0
        assert coverage.rate_steps[coverage.visible].max() > 12.0

    def test_a_span_downlinks_at_its_mean_rate(self, coverage: StationCoverage) -> None:
        for span in coverage.spans(T0, END):
            k = [s.code for s in coverage.stations].index(span.station)
            first, last = coverage.index(span.begin), coverage.index(span.end - 1e-6)
            steps = coverage.rate_steps[k, first : last + 1]
            assert span.rate_mbps == pytest.approx(steps.mean())
            assert steps.min() < span.rate_mbps < steps.max()

    def test_a_clipped_span_has_the_mean_over_its_part(
        self, coverage: StationCoverage
    ) -> None:
        first = coverage.spans(T0, END)[0]
        middle = first.begin + 3 * MIN
        head = coverage.spans(first.begin, middle)[0]
        tail = coverage.spans(middle, first.end)[0]
        assert head.end == middle == tail.begin
        head_gb = head.rate_mbps * (head.end - head.begin) / 1000.0
        tail_gb = tail.rate_mbps * (tail.end - tail.begin) / 1000.0
        whole_gb = first.rate_mbps * (first.end - first.begin) / 1000.0
        assert head_gb + tail_gb == pytest.approx(whole_gb)
        assert head.rate_mbps != pytest.approx(tail.rate_mbps)

    def test_downlink_gb_adds_up_over_any_split(
        self, coverage: StationCoverage
    ) -> None:
        total = coverage.downlink_gb(T0, END)
        assert total > 0.0
        parts = [
            coverage.downlink_gb(T0 + h * HOUR, T0 + (h + 1) * HOUR)
            for h in range(HOURS)
        ]
        assert sum(parts) == pytest.approx(total)
        # Off-step boundaries too.
        cut = T0 + 7 * HOUR + 37.0
        assert coverage.downlink_gb(T0, cut) + coverage.downlink_gb(
            cut, END
        ) == pytest.approx(total)
        # And the same as the steps' rates over the run.
        by_steps = 0.0
        for span in coverage.spans(T0, END):
            k = [s.code for s in coverage.stations].index(span.station)
            first, last = coverage.index(span.begin), coverage.index(span.end - 1e-6)
            by_steps += (
                coverage.rate_steps[k, first : last + 1].sum()
                * coverage.step_size
                / 1000.0
            )
        assert total == pytest.approx(by_steps)

    def test_contact_at_reports_the_rate_at_the_step(
        self, coverage: StationCoverage
    ) -> None:
        first = coverage.spans(T0, END)[0]
        contact = coverage.contact_at(first.begin + MIN)
        assert contact is not None
        assert contact.rate_mbps == pytest.approx(
            coverage.stations[0].bands[0].downlink_rate_at(contact.range_km)
        )

    def test_a_flat_rate_is_unchanged(self) -> None:
        config = _config(HOURS, stations=True)
        config.spacecraft_bus.communications = CommunicationsSystem(
            name="S-band",
            band_capabilities=[BandCapability(band="S", downlink_rate_mbps=10.0)],
            antenna_pointing=AntennaPointing(antenna_type=AntennaType.OMNI),
        )
        flat = StationCoverage.for_config(config)
        assert flat is not None
        assert not any(flat._ranged)
        assert {span.rate_mbps for span in flat.spans(T0, END)} == {10.0}
