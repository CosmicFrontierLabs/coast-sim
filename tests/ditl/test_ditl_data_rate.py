"""Tests for DITL data rate matching between ground station and spacecraft."""

from datetime import datetime, timezone

import pytest
from rust_ephem import TLEEphemeris

from conops.config import (
    BandCapability,
    CommunicationsSystem,
    Constraint,
    GroundStation,
)
from conops.config.groundstation import Antenna
from conops.simulation import Pass


class MockDITL:
    """Mock DITL class for testing _get_effective_data_rate method."""

    def __init__(self):
        pass

    def _get_effective_data_rate(self, station, current_pass) -> float | None:
        """Calculate effective downlink data rate based on ground station and spacecraft capabilities.

        The effective rate is the minimum of:
        1. Ground station antenna max data rate
        2. Spacecraft communications system downlink rate for common bands

        Args:
            station: GroundStation object with antenna capabilities
            current_pass: Pass object with communications configuration

        Returns:
            Effective data rate in Mbps, or None if no compatible bands/rates
        """
        # If no ground station data rate specified, return None
        if station.antenna.max_data_rate_mbps is None:
            return None

        gs_rate = station.antenna.max_data_rate_mbps

        # If pass has no comms config, use ground station rate only
        if current_pass.comms_config is None:
            return gs_rate

        # Find common bands between ground station and spacecraft
        gs_bands = set(station.antenna.bands) if station.antenna.bands else set()
        if not gs_bands:
            # No bands specified on ground station - assume compatible
            return gs_rate

        # Find maximum spacecraft downlink rate for common bands
        max_spacecraft_rate = 0.0
        for band in gs_bands:
            sc_rate = current_pass.comms_config.get_downlink_rate(band)
            if sc_rate > max_spacecraft_rate:
                max_spacecraft_rate = sc_rate

        # If no common bands have non-zero rates, return None
        if max_spacecraft_rate == 0.0:
            return None

        # Return minimum of ground station and spacecraft rates
        return min(gs_rate, max_spacecraft_rate)


class TestEffectiveDataRate:
    """Test effective data rate calculation."""

    @pytest.fixture
    def mock_ditl(self):
        """Create mock DITL instance."""
        return MockDITL()

    @pytest.fixture
    def ephem(self):
        """Create test ephemeris."""
        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 8, 15, 12, 15, 0, tzinfo=timezone.utc)
        return TLEEphemeris(
            tle="examples/example.tle", begin=begin, end=end, step_size=60
        )

    @pytest.fixture
    def constraint(self, ephem):
        """Create constraint with ephemeris."""
        constraint = Constraint()
        constraint.ephem = ephem
        return constraint

    def test_no_ground_station_rate(self, mock_ditl, constraint, ephem):
        """Test when ground station has no max data rate."""
        station = GroundStation(
            code="TEST",
            name="Test Station",
            latitude_deg=0.0,
            longitude_deg=0.0,
            antenna=Antenna(bands=["X"], max_data_rate_mbps=None),
        )

        comms = CommunicationsSystem(
            name="Test Comms",
            band_capabilities=[BandCapability(band="X", downlink_rate_mbps=150.0)],
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        pass_obj = Pass(
            constraint=constraint,
            ephem=ephem,
            acs_config=None,
            comms_config=comms,
            station="TEST",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        rate = mock_ditl._get_effective_data_rate(station, pass_obj)
        assert rate is None

    def test_no_spacecraft_comms(self, mock_ditl, constraint, ephem):
        """Test when spacecraft has no comms config."""
        station = GroundStation(
            code="TEST",
            name="Test Station",
            latitude_deg=0.0,
            longitude_deg=0.0,
            antenna=Antenna(bands=["X"], max_data_rate_mbps=100.0),
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        pass_obj = Pass(
            constraint=constraint,
            ephem=ephem,
            acs_config=None,
            comms_config=None,  # No comms config
            station="TEST",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        rate = mock_ditl._get_effective_data_rate(station, pass_obj)
        assert rate == 100.0  # Use ground station rate

    def test_ground_station_lower_rate(self, mock_ditl, constraint, ephem):
        """Test when ground station has lower rate than spacecraft."""
        station = GroundStation(
            code="TEST",
            name="Test Station",
            latitude_deg=0.0,
            longitude_deg=0.0,
            antenna=Antenna(bands=["X"], max_data_rate_mbps=50.0),
        )

        comms = CommunicationsSystem(
            name="Test Comms",
            band_capabilities=[BandCapability(band="X", downlink_rate_mbps=150.0)],
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        pass_obj = Pass(
            constraint=constraint,
            ephem=ephem,
            acs_config=None,
            comms_config=comms,
            station="TEST",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        rate = mock_ditl._get_effective_data_rate(station, pass_obj)
        assert rate == 50.0  # Limited by ground station

    def test_spacecraft_lower_rate(self, mock_ditl, constraint, ephem):
        """Test when spacecraft has lower rate than ground station."""
        station = GroundStation(
            code="TEST",
            name="Test Station",
            latitude_deg=0.0,
            longitude_deg=0.0,
            antenna=Antenna(bands=["X"], max_data_rate_mbps=200.0),
        )

        comms = CommunicationsSystem(
            name="Test Comms",
            band_capabilities=[BandCapability(band="X", downlink_rate_mbps=100.0)],
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        pass_obj = Pass(
            constraint=constraint,
            ephem=ephem,
            acs_config=None,
            comms_config=comms,
            station="TEST",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        rate = mock_ditl._get_effective_data_rate(station, pass_obj)
        assert rate == 100.0  # Limited by spacecraft

    def test_no_common_bands(self, mock_ditl, constraint, ephem):
        """Test when ground station and spacecraft have no common bands."""
        station = GroundStation(
            code="TEST",
            name="Test Station",
            latitude_deg=0.0,
            longitude_deg=0.0,
            antenna=Antenna(bands=["S"], max_data_rate_mbps=100.0),
        )

        comms = CommunicationsSystem(
            name="Test Comms",
            band_capabilities=[
                BandCapability(band="X", downlink_rate_mbps=150.0),
                BandCapability(band="Ka", downlink_rate_mbps=300.0),
            ],
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        pass_obj = Pass(
            constraint=constraint,
            ephem=ephem,
            acs_config=None,
            comms_config=comms,
            station="TEST",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        rate = mock_ditl._get_effective_data_rate(station, pass_obj)
        assert rate is None  # No common bands

    def test_multiple_common_bands(self, mock_ditl, constraint, ephem):
        """Test with multiple common bands - uses highest spacecraft rate."""
        station = GroundStation(
            code="TEST",
            name="Test Station",
            latitude_deg=0.0,
            longitude_deg=0.0,
            antenna=Antenna(bands=["S", "X", "Ka"], max_data_rate_mbps=200.0),
        )

        comms = CommunicationsSystem(
            name="Test Comms",
            band_capabilities=[
                BandCapability(band="S", downlink_rate_mbps=10.0),
                BandCapability(band="X", downlink_rate_mbps=150.0),
                BandCapability(band="Ka", downlink_rate_mbps=300.0),
            ],
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        pass_obj = Pass(
            constraint=constraint,
            ephem=ephem,
            acs_config=None,
            comms_config=comms,
            station="TEST",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        rate = mock_ditl._get_effective_data_rate(station, pass_obj)
        # Max spacecraft rate for common bands is 300 (Ka), but GS limited to 200
        assert rate == 200.0

    def test_ground_station_no_bands_specified(self, mock_ditl, constraint, ephem):
        """Test when ground station has no bands specified - assumes compatible."""
        station = GroundStation(
            code="TEST",
            name="Test Station",
            latitude_deg=0.0,
            longitude_deg=0.0,
            antenna=Antenna(bands=[], max_data_rate_mbps=100.0),
        )

        comms = CommunicationsSystem(
            name="Test Comms",
            band_capabilities=[BandCapability(band="X", downlink_rate_mbps=150.0)],
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        pass_obj = Pass(
            constraint=constraint,
            ephem=ephem,
            acs_config=None,
            comms_config=comms,
            station="TEST",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        rate = mock_ditl._get_effective_data_rate(station, pass_obj)
        assert rate == 100.0  # Assumes compatible, uses GS rate

    def test_spacecraft_zero_rate_for_common_band(self, mock_ditl, constraint, ephem):
        """Test when spacecraft has zero rate for common band."""
        station = GroundStation(
            code="TEST",
            name="Test Station",
            latitude_deg=0.0,
            longitude_deg=0.0,
            antenna=Antenna(bands=["X"], max_data_rate_mbps=100.0),
        )

        comms = CommunicationsSystem(
            name="Test Comms",
            band_capabilities=[
                BandCapability(band="X", uplink_rate_mbps=10.0, downlink_rate_mbps=0.0)
            ],
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        pass_obj = Pass(
            constraint=constraint,
            ephem=ephem,
            acs_config=None,
            comms_config=comms,
            station="TEST",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        rate = mock_ditl._get_effective_data_rate(station, pass_obj)
        assert rate is None  # Spacecraft has 0 downlink rate for X-band
