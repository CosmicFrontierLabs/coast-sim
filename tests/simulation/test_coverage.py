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


def _walk(
    visible: list[list[int]],
    rates: list[float],
    *,
    step: float = 60.0,
    handover: float = 0.0,
    acquisition: float = 0.0,
) -> StationCoverage:
    """A coverage built from a visibility matrix (stations by steps) alone,
    to test the contact scheduling on its own."""
    coverage = StationCoverage.__new__(StationCoverage)
    coverage.times = np.arange(len(visible[0]), dtype=float) * step
    coverage.step_size = step
    coverage.handover_seconds = handover
    coverage.acquisition_seconds = acquisition
    coverage.stations = [
        GroundStation(
            code=f"S{k}",
            name=f"S{k}",
            latitude_deg=0.0,
            longitude_deg=0.0,
            bands=[BandCapability(band="S", downlink_rate_mbps=rate)],
        )
        for k, rate in enumerate(rates)
    ]
    coverage.rates = np.array(rates, dtype=float)
    coverage.visible = np.array(visible, dtype=bool)
    coverage.contacts = []
    coverage.handovers = 0
    coverage._schedule_contacts()
    coverage._begins = [span.begin for span in coverage.contacts]
    return coverage


class TestScheduling:
    """One station is tracked at a time, handed over when it sets."""

    def test_a_contact_stays_with_its_station_until_it_sets(self) -> None:
        # S1 rises while S0 is still tracked, and stays up after S0 sets.
        coverage = _walk([[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]], [10.0, 10.0])
        assert coverage.contacts == [
            ContactSpan(0.0, 240.0, "S0", 10.0),
            ContactSpan(240.0, 360.0, "S1", 10.0),
        ]
        assert coverage.handovers == 1

    def test_the_highest_rate_then_the_longest_view_is_chosen(self) -> None:
        # Three stations rise together: S1 has the rate, S2 the longer view.
        coverage = _walk([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], [10.0, 20.0, 10.0])
        assert [c.station for c in coverage.contacts] == ["S1", "S2"]
        coverage = _walk([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], [10.0] * 3)
        assert [c.station for c in coverage.contacts] == ["S2"]

    def test_a_handover_costs_its_latency_but_a_fresh_contact_does_not(
        self,
    ) -> None:
        coverage = _walk(
            [[1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1]],
            [10.0] * 3,
            handover=90.0,
        )
        assert coverage.contacts == [
            ContactSpan(0.0, 120.0, "S0", 10.0),
            # Handed over at 120 s, linked 90 s later.
            ContactSpan(210.0, 240.0, "S1", 10.0),
            # Acquired after a gap with nothing in view: no latency.
            ContactSpan(300.0, 360.0, "S2", 10.0),
        ]
        assert coverage.handovers == 1

    def test_every_contact_starts_after_its_acquisition(self) -> None:
        coverage = _walk(
            [[1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1]],
            [10.0] * 3,
            handover=30.0,
            acquisition=20.0,
        )
        assert coverage.contacts == [
            # Acquired 20 s after S0 came into view.
            ContactSpan(20.0, 120.0, "S0", 10.0),
            # Handed over at 120 s: the handover and the acquisition.
            ContactSpan(170.0, 240.0, "S1", 10.0),
            # After a gap: the acquisition alone.
            ContactSpan(320.0, 360.0, "S2", 10.0),
        ]
        assert coverage.handovers == 1

    def test_a_handover_longer_than_the_view_downlinks_nothing(self) -> None:
        coverage = _walk([[1, 1, 0, 0], [0, 1, 1, 0]], [10.0, 10.0], handover=200.0)
        assert coverage.contacts == [ContactSpan(0.0, 120.0, "S0", 10.0)]
        assert coverage.handovers == 1

    def test_no_station_no_contacts(self) -> None:
        assert _walk([[0, 0, 0]], [10.0]).contacts == []

    def test_contacts_never_overlap_and_track_a_visible_station(self) -> None:
        """With a station near Svalbard the two are often in view together."""
        config = _config(HOURS, stations=True)
        config.ground_stations.add(
            GroundStation(
                code="KIR",
                name="Kiruna",
                latitude_deg=67.86,
                longitude_deg=20.22,
                bands=[BandCapability(band="S", downlink_rate_mbps=10.0)],
            )
        )
        config.spacecraft_bus.communications = _comms(AntennaType.OMNI)
        config.spacecraft_bus.communications.handover_seconds = 120.0
        config.spacecraft_bus.communications.acquisition_seconds = 30.0
        coverage = StationCoverage.for_config(config)
        assert coverage is not None
        assert int((coverage.visible.sum(axis=0) > 1).sum()) > 0
        assert coverage.handovers > 0
        step = coverage.step_size
        codes = [s.code for s in coverage.stations]
        for earlier, later in zip(coverage.contacts, coverage.contacts[1:]):
            assert earlier.end <= later.begin
            if later.begin - earlier.end < 150.0:
                # Only a handover sits closer than its latencies, and it is
                # exactly the handover and the acquisition after the station set.
                assert later.begin - earlier.end == pytest.approx(150.0)
                assert earlier.station != later.station
        for span in coverage.contacts:
            k = codes.index(span.station)
            # The station came into view at a step, 30 s or 150 s before.
            rise = span.begin - 30.0
            if (rise - coverage.times[0]) % step != 0:
                rise = span.begin - 150.0
            assert (rise - coverage.times[0]) % step == 0
            assert coverage.visible[
                k, coverage.index(rise) : coverage.index(span.end - 1e-6) + 1
            ].all()
        # Fewer contacts than rises of stations: the link does not swap
        # between two stations in view at once.
        rises = sum(
            int((np.diff(coverage.visible[k].astype(int)) > 0).sum())
            for k in range(len(codes))
        )
        assert len(coverage.contacts) < rises


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
        station = [s.code for s in coverage.stations].index(first.station)
        ra, dec = np.degrees(vec2radec(coverage._to_station[station, step]))
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
