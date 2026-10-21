"""Station coverage for antennas that need no spacecraft attitude."""

import sys
from pathlib import Path

import numpy as np
import pytest

from conops.common import AntennaType
from conops.common.vector import vec2radec
from conops.config import (
    AntennaPointing,
    BandCapability,
    CommunicationsSystem,
    GroundStation,
    MissionConfig,
)
from conops.simulation import ContactSpan, StationCoverage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scheduler"))
from planning_scenario import HOUR, MIN, T0  # noqa: E402
from planning_scenario import make_config as _config  # noqa: E402

HOURS = 24
END = T0 + HOURS * HOUR


def _comms(
    antenna_type: AntennaType, rate_mbps: float = 10.0, **antenna: float
) -> CommunicationsSystem:
    return CommunicationsSystem(
        name="S-band",
        band_capabilities=[BandCapability(band="S", downlink_rate_mbps=rate_mbps)],
        antenna_pointing=AntennaPointing(antenna_type=antenna_type, **antenna),
    )


@pytest.fixture(scope="module")
def omni() -> tuple[MissionConfig, StationCoverage]:
    config = _config(HOURS, stations=True)
    config.spacecraft_bus.communications = _comms(AntennaType.OMNI)
    coverage = StationCoverage.for_config(config)
    assert coverage is not None
    return config, coverage


class TestForConfig:
    def test_none_without_a_communications_system(self) -> None:
        """The legacy behaviour: a body-fixed antenna, which downlinks in passes."""
        assert StationCoverage.for_config(_config(2, stations=True)) is None

    def test_none_for_a_body_fixed_antenna(self) -> None:
        config = _config(2, stations=True)
        config.spacecraft_bus.communications = _comms(AntennaType.FIXED)
        assert StationCoverage.for_config(config) is None

    def test_none_without_a_station_that_downlinks(self) -> None:
        config = _config(2, stations=False)
        config.spacecraft_bus.communications = _comms(AntennaType.OMNI)
        assert StationCoverage.for_config(config) is None
        config = _config(2, stations=True)
        # The spacecraft has no band in common with the stations.
        config.spacecraft_bus.communications = CommunicationsSystem(
            name="X-band",
            band_capabilities=[BandCapability(band="X", downlink_rate_mbps=100.0)],
            antenna_pointing=AntennaPointing(antenna_type=AntennaType.OMNI),
        )
        assert StationCoverage.for_config(config) is None

    @pytest.mark.parametrize("antenna_type", [AntennaType.OMNI, AntennaType.GIMBALED])
    def test_built_for_an_antenna_that_needs_no_attitude(
        self, antenna_type: AntennaType
    ) -> None:
        config = _config(2, stations=True)
        config.spacecraft_bus.communications = _comms(antenna_type)
        coverage = StationCoverage.for_config(config)
        assert coverage is not None
        assert [s.code for s in coverage.stations] == ["MAL", "HI1", "SGS"]
        assert coverage.gimballed == (antenna_type == AntennaType.GIMBALED)

    def test_a_station_not_scheduled_for_tracking_still_downlinks(self) -> None:
        config = _config(2, stations=True)
        config.spacecraft_bus.communications = _comms(AntennaType.OMNI)
        for station in config.ground_stations.stations:
            station.schedule_for_tracking = False
        coverage = StationCoverage.for_config(config)
        assert coverage is not None
        assert len(coverage.stations) == 3

    def test_the_rate_is_the_effective_rate_through_the_bands(self) -> None:
        config = _config(2, stations=True)
        config.spacecraft_bus.communications = _comms(AntennaType.OMNI, rate_mbps=4.0)
        coverage = StationCoverage.for_config(config)
        assert coverage is not None
        assert list(coverage.rates) == [4.0, 4.0, 4.0]


class TestSpans:
    def test_spans_are_above_the_mask_and_within_the_window(
        self, omni: tuple[MissionConfig, StationCoverage]
    ) -> None:
        config, coverage = omni
        spans = coverage.spans(T0, END)
        assert len(spans) > 5
        for span in spans:
            assert isinstance(span, ContactSpan)
            assert T0 <= span.begin < span.end <= END
            assert span.rate_mbps == 10.0
            station = config.ground_stations.get(span.station)
            k = coverage.stations.index(station)
            for step in range(coverage.index(span.begin), coverage.index(span.end)):
                assert coverage.elevation_deg[k, step] > station.min_elevation_deg
        # Consecutive spans do not overlap.
        for earlier, later in zip(spans, spans[1:]):
            assert earlier.end <= later.begin

    def test_a_low_earth_orbit_contact_is_minutes_long(
        self, omni: tuple[MissionConfig, StationCoverage]
    ) -> None:
        _, coverage = omni
        lengths = [span.end - span.begin for span in coverage.spans(T0, END)]
        assert 3 * MIN <= min(lengths) and max(lengths) <= 15 * MIN

    def test_a_window_clips_the_span_at_its_edges(
        self, omni: tuple[MissionConfig, StationCoverage]
    ) -> None:
        _, coverage = omni
        first = coverage.spans(T0, END)[0]
        middle = (first.begin + first.end) / 2
        clipped = coverage.spans(middle, first.end - 30)
        assert clipped == [first._replace(begin=middle, end=first.end - 30)]
        assert coverage.spans(first.end, first.end + 30) == []

    def test_the_fraction_is_the_time_in_contact(
        self, omni: tuple[MissionConfig, StationCoverage]
    ) -> None:
        _, coverage = omni
        spans = coverage.spans(T0, END)
        covered = sum(span.end - span.begin for span in spans)
        assert coverage.fraction(T0, END) == pytest.approx(covered / (END - T0))
        assert 0.02 < coverage.fraction(T0, END) < 0.3
        assert coverage.fraction(T0, T0) == 0.0

    def test_the_best_station_in_view_is_taken(
        self, omni: tuple[MissionConfig, StationCoverage]
    ) -> None:
        """Where stations overlap, the one with the higher rate, then the
        higher elevation, is the contact."""
        config, coverage = omni
        stations = {s.code: s for s in config.ground_stations.stations}
        for step in range(len(coverage.times)):
            best = int(coverage.best[step])
            visible = [
                k
                for k, station in enumerate(coverage.stations)
                if coverage.elevation_deg[k, step] > station.min_elevation_deg
            ]
            if not visible:
                assert best == -1
                continue
            assert best in visible
            assert coverage.elevation_deg[best, step] == max(
                coverage.elevation_deg[k, step] for k in visible
            )
        assert set(stations) >= {span.station for span in coverage.spans(T0, END)}


class TestContactAt:
    def test_in_and_out_of_contact(
        self, omni: tuple[MissionConfig, StationCoverage]
    ) -> None:
        _, coverage = omni
        first = coverage.spans(T0, END)[0]
        contact = coverage.contact_at(first.begin + MIN)
        assert contact is not None
        assert contact.station == first.station
        assert contact.rate_mbps == 10.0
        assert 300.0 < contact.range_km < 3000.0
        assert coverage.contact_at(first.begin - MIN) is None

    def test_next_contact(self, omni: tuple[MissionConfig, StationCoverage]) -> None:
        _, coverage = omni
        spans = coverage.spans(T0, END)
        first = spans[0]
        assert coverage.next_contact(T0) == first.begin
        # In contact, the next contact is now.
        assert coverage.next_contact(first.begin + MIN) == first.begin + MIN
        assert coverage.next_contact(first.end) == spans[1].begin
        assert coverage.next_contact(coverage.times[-1] + MIN) is None

    def test_an_omni_antenna_reaches_the_station_from_any_attitude(
        self, omni: tuple[MissionConfig, StationCoverage]
    ) -> None:
        _, coverage = omni
        first = coverage.spans(T0, END)[0]
        assert coverage.contact_at(first.begin, (0.0, 0.0, 0.0)) is not None
        assert coverage.contact_at(first.begin, (180.0, -60.0, 90.0)) is not None

    def test_a_gimballed_antenna_reaches_the_station_within_its_range(
        self,
    ) -> None:
        config = _config(HOURS, stations=True)
        config.spacecraft_bus.communications = _comms(
            AntennaType.GIMBALED, gimbal_range_deg=60.0
        )
        coverage = StationCoverage.for_config(config)
        assert coverage is not None
        first = coverage.spans(T0, END)[0]
        step = coverage.index(first.begin)
        ra, dec = np.degrees(
            vec2radec(coverage._to_station[int(coverage.best[step]), step])
        )
        # The boresight is -X in the body frame: the body +X axis pointed
        # away from the station puts the boresight on it.
        toward = (float(ra) + 180.0, -float(dec), 0.0)
        away = (float(ra), float(dec), 0.0)
        assert coverage.contact_at(first.begin, toward) is not None
        assert coverage.contact_at(first.begin, away) is None
        # Without an attitude the station is taken as reachable, as in planning.
        assert coverage.contact_at(first.begin) is not None

    def test_contact_in_a_step(
        self, omni: tuple[MissionConfig, StationCoverage]
    ) -> None:
        _, coverage = omni
        first = coverage.spans(T0, END)[0]
        start, stop, downlinked = coverage.contact_in(first.begin, first.begin + MIN)
        assert (start, stop) == (first.begin, first.begin + MIN)
        assert downlinked == pytest.approx(10.0 * MIN / 1000.0)
        # A step straddling the end of the contact downlinks for the part in it.
        start, stop, downlinked = coverage.contact_in(first.end - 20, first.end + 40)
        assert (start, stop) == (first.end - 20, first.end)
        assert downlinked == pytest.approx(10.0 * 20 / 1000.0)
        assert coverage.contact_in(first.end, first.end + MIN) == (
            first.end,
            first.end,
            0.0,
        )

    def test_contact_in_a_step_a_gimballed_antenna_cannot_reach(self) -> None:
        config = _config(HOURS, stations=True)
        config.spacecraft_bus.communications = _comms(
            AntennaType.GIMBALED, gimbal_range_deg=0.0
        )
        coverage = StationCoverage.for_config(config)
        assert coverage is not None
        first = coverage.spans(T0, END)[0]
        assert coverage.contact_in(first.begin, first.begin + MIN, (0.0, 0.0, 0.0)) == (
            first.begin,
            first.begin,
            0.0,
        )


def test_the_default_stations_have_a_downlink() -> None:
    station = GroundStation(
        code="X",
        name="X",
        latitude_deg=0.0,
        longitude_deg=0.0,
        bands=[BandCapability(band="S", downlink_rate_mbps=10.0)],
    )
    assert station.supports_downlink
