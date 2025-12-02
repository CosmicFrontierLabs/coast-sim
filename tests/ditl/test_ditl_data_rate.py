"""Tests for DITL data rate matching between ground station and spacecraft."""

from datetime import datetime, timezone

import pytest
from rust_ephem import TLEEphemeris

from conops.config import (
    AttitudeControlSystem,
    BandCapability,
    Battery,
    CommunicationsSystem,
    Constraint,
    GroundStation,
    GroundStationRegistry,
    MissionConfig,
    Payload,
    SolarPanelSet,
    SpacecraftBus,
)
from conops.simulation import Pass


class MockDITL:
    """Mock DITL class for testing _get_effective_data_rate method."""

    def __init__(self):
        pass

    def _get_effective_data_rate(self, station, current_pass) -> float | None:
        """Calculate effective downlink data rate based on GS and spacecraft per-band capabilities.

        For each common band, take min(GS downlink rate, SC downlink rate) and
        return the maximum of these across all common bands. If no spacecraft
        comms config is provided, return the GS overall maximum downlink.

        Args:
            station: GroundStation object with antenna capabilities
            current_pass: Pass object with communications configuration

        Returns:
            Effective data rate in Mbps, or None if no compatible bands/rates
        """
        gs_bands = set(station.supported_bands()) if station.bands else set()

        # If no spacecraft comms config, fall back to GS overall capability
        comms_config = (
            getattr(current_pass.config.spacecraft_bus, "communications", None)
            if hasattr(current_pass, "config")
            else getattr(current_pass, "comms_config", None)
        )
        if comms_config is None:
            return station.get_overall_max_downlink()

        if not gs_bands:
            # No GS bands defined -> no common bands known
            return None

        best = 0.0
        for band in gs_bands:
            gs_rate = station.get_downlink_rate(band) or 0.0
            sc_rate = comms_config.get_downlink_rate(band) or 0.0
            if gs_rate > 0.0 and sc_rate > 0.0:
                eff = min(gs_rate, sc_rate)
                if eff > best:
                    best = eff
        return best if best > 0.0 else None


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
            # Explicitly set 0.0 to indicate GS has no X-band downlink capability
            bands=[
                BandCapability(band="X", uplink_rate_mbps=0.0, downlink_rate_mbps=0.0)
            ],
        )

        comms = CommunicationsSystem(
            name="Test Comms",
            band_capabilities=[BandCapability(band="X", downlink_rate_mbps=150.0)],
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        spacecraft_bus = SpacecraftBus(
            attitude_control=AttitudeControlSystem(), communications=comms
        )
        config = MissionConfig(
            spacecraft_bus=spacecraft_bus,
            solar_panel=SolarPanelSet(panels=[]),
            payload=Payload(instruments=[]),
            battery=Battery(capacity_wh=1000, max_depth_of_discharge=0.8),
            constraint=constraint,
            ground_stations=GroundStationRegistry.default(),
        )
        pass_obj = Pass(
            config=config,
            ephem=ephem,
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
            bands=[BandCapability(band="X", downlink_rate_mbps=100.0)],
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        spacecraft_bus = SpacecraftBus(
            attitude_control=AttitudeControlSystem(), communications=None
        )
        config = MissionConfig(
            spacecraft_bus=spacecraft_bus,
            solar_panel=SolarPanelSet(panels=[]),
            payload=Payload(instruments=[]),
            battery=Battery(capacity_wh=1000, max_depth_of_discharge=0.8),
            constraint=constraint,
            ground_stations=GroundStationRegistry.default(),
        )
        pass_obj = Pass(
            config=config,
            ephem=ephem,
            station="TEST",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        rate = mock_ditl._get_effective_data_rate(station, pass_obj)
        assert rate == 100.0  # Use GS overall max across bands

    def test_ground_station_lower_rate(self, mock_ditl, constraint, ephem):
        """Test when ground station has lower rate than spacecraft."""
        station = GroundStation(
            code="TEST",
            name="Test Station",
            latitude_deg=0.0,
            longitude_deg=0.0,
            bands=[BandCapability(band="X", downlink_rate_mbps=50.0)],
        )

        comms = CommunicationsSystem(
            name="Test Comms",
            band_capabilities=[BandCapability(band="X", downlink_rate_mbps=150.0)],
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        spacecraft_bus = SpacecraftBus(
            attitude_control=AttitudeControlSystem(), communications=comms
        )
        config = MissionConfig(
            spacecraft_bus=spacecraft_bus,
            solar_panel=SolarPanelSet(panels=[]),
            payload=Payload(instruments=[]),
            battery=Battery(capacity_wh=1000, max_depth_of_discharge=0.8),
            constraint=constraint,
            ground_stations=GroundStationRegistry.default(),
        )
        pass_obj = Pass(
            config=config,
            ephem=ephem,
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
            bands=[BandCapability(band="X", downlink_rate_mbps=200.0)],
        )

        comms = CommunicationsSystem(
            name="Test Comms",
            band_capabilities=[BandCapability(band="X", downlink_rate_mbps=100.0)],
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        spacecraft_bus = SpacecraftBus(
            attitude_control=AttitudeControlSystem(), communications=comms
        )
        config = MissionConfig(
            spacecraft_bus=spacecraft_bus,
            solar_panel=SolarPanelSet(panels=[]),
            payload=Payload(instruments=[]),
            battery=Battery(capacity_wh=1000, max_depth_of_discharge=0.8),
            constraint=constraint,
            ground_stations=GroundStationRegistry.default(),
        )
        pass_obj = Pass(
            config=config,
            ephem=ephem,
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
            bands=[BandCapability(band="S", downlink_rate_mbps=100.0)],
        )

        comms = CommunicationsSystem(
            name="Test Comms",
            band_capabilities=[
                BandCapability(band="X", downlink_rate_mbps=150.0),
                BandCapability(band="Ka", downlink_rate_mbps=300.0),
            ],
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        spacecraft_bus = SpacecraftBus(
            attitude_control=AttitudeControlSystem(), communications=comms
        )
        config = MissionConfig(
            spacecraft_bus=spacecraft_bus,
            solar_panel=SolarPanelSet(panels=[]),
            payload=Payload(instruments=[]),
            battery=Battery(capacity_wh=1000, max_depth_of_discharge=0.8),
            constraint=constraint,
            ground_stations=GroundStationRegistry.default(),
        )
        pass_obj = Pass(
            config=config,
            ephem=ephem,
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
            bands=[
                BandCapability(band="S"),
                BandCapability(band="X", downlink_rate_mbps=200.0),
                BandCapability(band="Ka", downlink_rate_mbps=200.0),
            ],
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
        spacecraft_bus = SpacecraftBus(
            attitude_control=AttitudeControlSystem(), communications=comms
        )
        config = MissionConfig(
            spacecraft_bus=spacecraft_bus,
            solar_panel=SolarPanelSet(panels=[]),
            payload=Payload(instruments=[]),
            battery=Battery(capacity_wh=1000, max_depth_of_discharge=0.8),
            constraint=constraint,
            ground_stations=GroundStationRegistry.default(),
        )
        pass_obj = Pass(
            config=config,
            ephem=ephem,
            station="TEST",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        rate = mock_ditl._get_effective_data_rate(station, pass_obj)
        # Per-band mins: X=min(200,150)=150, Ka=min(200,300)=200 -> best 200
        assert rate == 200.0

    def test_ground_station_no_bands_specified(self, mock_ditl, constraint, ephem):
        """Test when ground station has no bands specified - assumes compatible."""
        station = GroundStation(
            code="TEST",
            name="Test Station",
            latitude_deg=0.0,
            longitude_deg=0.0,
            bands=[],
        )

        comms = CommunicationsSystem(
            name="Test Comms",
            band_capabilities=[BandCapability(band="X", downlink_rate_mbps=150.0)],
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        spacecraft_bus = SpacecraftBus(
            attitude_control=AttitudeControlSystem(), communications=comms
        )
        config = MissionConfig(
            spacecraft_bus=spacecraft_bus,
            solar_panel=SolarPanelSet(panels=[]),
            payload=Payload(instruments=[]),
            battery=Battery(capacity_wh=1000, max_depth_of_discharge=0.8),
            constraint=constraint,
            ground_stations=GroundStationRegistry.default(),
        )
        pass_obj = Pass(
            config=config,
            ephem=ephem,
            station="TEST",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        rate = mock_ditl._get_effective_data_rate(station, pass_obj)
        assert rate is None  # No GS bands defined -> no compatible rate

    def test_spacecraft_zero_rate_for_common_band(self, mock_ditl, constraint, ephem):
        """Test when spacecraft has zero rate for common band."""
        station = GroundStation(
            code="TEST",
            name="Test Station",
            latitude_deg=0.0,
            longitude_deg=0.0,
            bands=[BandCapability(band="X", downlink_rate_mbps=100.0)],
        )

        comms = CommunicationsSystem(
            name="Test Comms",
            band_capabilities=[
                BandCapability(band="X", uplink_rate_mbps=10.0, downlink_rate_mbps=0.0)
            ],
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        spacecraft_bus = SpacecraftBus(
            attitude_control=AttitudeControlSystem(), communications=comms
        )
        config = MissionConfig(
            spacecraft_bus=spacecraft_bus,
            solar_panel=SolarPanelSet(panels=[]),
            payload=Payload(instruments=[]),
            battery=Battery(capacity_wh=1000, max_depth_of_discharge=0.8),
            constraint=constraint,
            ground_stations=GroundStationRegistry.default(),
        )
        pass_obj = Pass(
            config=config,
            ephem=ephem,
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
