"""The downlink rate a spacecraft gets from a ground station."""

from conops.config import BandCapability, CommunicationsSystem, GroundStation


def _station() -> GroundStation:
    return GroundStation(
        code="X",
        name="X",
        latitude_deg=0.0,
        longitude_deg=0.0,
        bands=[
            BandCapability(band="S", downlink_rate_mbps=10.0),
            BandCapability(band="X", downlink_rate_mbps=100.0),
        ],
    )


def test_without_a_communications_system_the_station_sets_the_rate() -> None:
    assert _station().effective_downlink_rate_mbps(None) == 100.0


def test_per_band_the_lower_rate_counts_and_the_best_band_wins() -> None:
    comms = CommunicationsSystem(
        band_capabilities=[
            BandCapability(band="S", downlink_rate_mbps=4.0),
            BandCapability(band="X", downlink_rate_mbps=50.0),
        ]
    )
    assert _station().effective_downlink_rate_mbps(comms) == 50.0


def test_no_common_band_means_no_link() -> None:
    only_ka = CommunicationsSystem(
        band_capabilities=[BandCapability(band="Ka", downlink_rate_mbps=500.0)]
    )
    assert _station().effective_downlink_rate_mbps(only_ka) is None
