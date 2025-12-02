"""Tests for integrated data management in DITL simulations."""

import pytest

from conops.config import (
    BandCapability,
    Battery,
    Constraint,
    DataGeneration,
    GroundStation,
    GroundStationRegistry,
    Instrument,
    MissionConfig,
    OnboardRecorder,
    Payload,
    SolarPanelSet,
    SpacecraftBus,
)


class TestDataManagementIntegration:
    """Test suite for integrated data management in DITL."""

    @pytest.fixture
    def config_with_data_generation(self):
        """Create a config with data generation and recorder."""
        # Create instruments with data generation
        camera = Instrument(
            name="Camera",
            data_generation=DataGeneration(rate_gbps=0.01),  # 0.01 Gbps
        )
        payload = Payload(payload=[camera])

        # Create recorder with moderate capacity
        recorder = OnboardRecorder(
            name="SSR",
            capacity_gb=10.0,
            yellow_threshold=0.7,
            red_threshold=0.9,
        )

        # Create ground station with downlink capability
        gs_registry = GroundStationRegistry()
        gs_registry.add(
            GroundStation(
                code="TST",
                name="Test Station",
                latitude_deg=0.0,
                longitude_deg=0.0,
                bands=[BandCapability(band="X", downlink_rate_mbps=100.0)],
            )
        )

        # Create minimal config
        config = MissionConfig(
            name="Test Config with Data",
            spacecraft_bus=SpacecraftBus(),
            solar_panel=SolarPanelSet(),
            payload=payload,
            battery=Battery(),
            constraint=Constraint(),
            ground_stations=gs_registry,
            recorder=recorder,
        )

        return config

    def test_config_includes_recorder(self, config_with_data_generation):
        """Test that config includes recorder."""
        config = config_with_data_generation
        assert config.recorder is not None
        assert config.recorder.capacity_gb == 10.0

    def test_payload_has_data_generation(self, config_with_data_generation):
        """Test that payload has data generation configured."""
        config = config_with_data_generation
        assert config.payload.total_data_rate_gbps() == 0.01

    def test_ground_station_has_downlink_rate(self, config_with_data_generation):
        """Test that ground station has downlink rate configured."""
        config = config_with_data_generation
        station = config.ground_stations.get("TST")
        assert station.get_downlink_rate("X") == 100.0

    def test_recorder_operations_in_config(self):
        """Test recorder operations work with config."""
        recorder = OnboardRecorder(capacity_gb=50.0)

        config = MissionConfig(
            name="Test Config",
            spacecraft_bus=SpacecraftBus(),
            solar_panel=SolarPanelSet(),
            payload=Payload(),
            battery=Battery(),
            constraint=Constraint(),
            ground_stations=GroundStationRegistry.default(),
            recorder=recorder,
        )

        # Test data operations
        config.recorder.add_data(10.0)
        assert config.recorder.current_volume_gb == 10.0

        config.recorder.remove_data(5.0)
        assert config.recorder.current_volume_gb == 5.0


class TestDataGenerationScenarios:
    """Test realistic scenarios of data generation and downlink."""

    def test_simple_observation_and_downlink(self):
        """Test simple observation generating data and downlink clearing it."""
        recorder = OnboardRecorder(capacity_gb=100.0)
        payload = Payload(
            payload=[
                Instrument(
                    name="Camera",
                    data_generation=DataGeneration(rate_gbps=0.1),
                )
            ]
        )

        # Simulate 10 minutes of observation (600 seconds)
        data_generated = payload.data_generated(600)
        recorder.add_data(data_generated)

        assert data_generated == 60.0  # 0.1 Gbps * 600 s = 60 Gb
        assert recorder.current_volume_gb == 60.0

        # Simulate 10 minute downlink pass at 100 Mbps (0.1 Gbps)
        downlink_rate_gbps = 0.1
        downlink_duration = 600  # seconds
        data_to_downlink = downlink_rate_gbps * downlink_duration

        downlinked = recorder.remove_data(data_to_downlink)
        assert downlinked == 60.0  # All data downlinked
        assert recorder.current_volume_gb == 0.0

    def test_data_accumulation_over_multiple_observations(self):
        """Test data accumulating over multiple observations."""
        recorder = OnboardRecorder(capacity_gb=100.0)
        payload = Payload(
            payload=[
                Instrument(
                    name="Spectrometer",
                    data_generation=DataGeneration(per_observation_gb=2.0),
                )
            ]
        )

        # Simulate 10 observations
        for _ in range(10):
            data = payload.data_generated(1)  # Duration doesn't matter
            recorder.add_data(data)

        assert recorder.current_volume_gb == 20.0
        assert recorder.get_fill_fraction() == 0.2

    def test_insufficient_downlink_capacity(self):
        """Test scenario where downlink can't clear all data."""
        recorder = OnboardRecorder(capacity_gb=100.0)
        payload = Payload(
            payload=[
                Instrument(
                    name="High-Rate Camera",
                    data_generation=DataGeneration(rate_gbps=0.5),
                )
            ]
        )

        # Generate data for 1 hour (would be 1800 Gb, but caps at capacity)
        data_generated = payload.data_generated(3600)
        assert data_generated == 1800.0  # Payload generates this much
        stored = recorder.add_data(data_generated)
        assert stored == 100.0  # But recorder only stores up to capacity
        assert recorder.current_volume_gb == 100.0  # Capped at capacity
        assert recorder.is_full()

        # Downlink for 10 minutes at 0.1 Gbps
        downlink_rate_gbps = 0.1
        downlink_duration = 600
        data_to_downlink = downlink_rate_gbps * downlink_duration

        downlinked = recorder.remove_data(data_to_downlink)
        assert downlinked == 60.0
        assert recorder.current_volume_gb == 40.0  # Still lots of data remaining

    def test_recorder_alerts_during_fill(self):
        """Test that recorder triggers alerts as it fills up."""
        recorder = OnboardRecorder(
            capacity_gb=100.0,
            yellow_threshold=0.7,
            red_threshold=0.9,
        )
        payload = Payload(
            payload=[
                Instrument(
                    name="Camera",
                    data_generation=DataGeneration(rate_gbps=0.1),
                )
            ]
        )

        # Start with no alert
        assert recorder.get_alert_level() == 0

        # Generate data to 50% - no alert
        data = payload.data_generated(500)  # 50 Gb
        recorder.add_data(data)
        assert recorder.get_alert_level() == 0

        # Generate data to 75% - yellow alert
        data = payload.data_generated(250)  # 25 Gb more
        recorder.add_data(data)
        assert recorder.get_fill_fraction() == 0.75
        assert recorder.get_alert_level() == 1

        # Generate data to 95% - red alert
        data = payload.data_generated(200)  # 20 Gb more
        recorder.add_data(data)
        assert recorder.get_fill_fraction() == 0.95
        assert recorder.get_alert_level() == 2

    def test_mixed_instruments_data_generation(self):
        """Test data generation with multiple instruments of different types."""
        recorder = OnboardRecorder(capacity_gb=100.0)

        instruments = [
            Instrument(
                name="Continuous Camera",
                data_generation=DataGeneration(rate_gbps=0.05),
            ),
            Instrument(
                name="Snapshot Imager",
                data_generation=DataGeneration(per_observation_gb=1.0),
            ),
            Instrument(
                name="Spectrometer",
                data_generation=DataGeneration(rate_gbps=0.01),
            ),
        ]
        payload = Payload(payload=instruments)

        # 60 seconds of observation
        # Camera: 0.05 * 60 = 3.0 Gb
        # Imager: 1.0 Gb
        # Spectrometer: 0.01 * 60 = 0.6 Gb
        # Total: 4.6 Gb
        data = payload.data_generated(60)
        recorder.add_data(data)

        assert recorder.current_volume_gb == pytest.approx(4.6, rel=1e-6)

    def test_daily_operations_simulation(self):
        """Test a simplified daily operations scenario."""
        recorder = OnboardRecorder(capacity_gb=50.0)
        payload = Payload(
            payload=[
                Instrument(
                    name="Science Instrument",
                    data_generation=DataGeneration(rate_gbps=0.02),
                )
            ]
        )

        # Simulate 24 hours
        # Assume 8 hours of science observations, 4 ground station passes

        # Science observations (8 hours = 28800 seconds)
        total_science_time = 8 * 3600
        data_generated = payload.data_generated(total_science_time)
        expected_data = 0.02 * 28800  # 576 Gb
        assert data_generated == pytest.approx(expected_data, rel=1e-6)

        # Add to recorder (will cap at 50 Gb capacity)
        stored = recorder.add_data(data_generated)
        assert stored == 50.0  # Only stores up to capacity
        assert recorder.current_volume_gb == 50.0  # Capped at capacity

        # 4 ground station passes, each 10 minutes, at 0.1 Gbps
        # Each pass can downlink: 0.1 Gbps * 600 s = 60 Gb
        # But first pass only has 50 Gb available

        # First pass: removes all 50 Gb
        pass_duration = 600  # 10 minutes
        downlink_rate = 0.1  # Gbps
        data_to_downlink = downlink_rate * pass_duration  # 60 Gb

        removed = recorder.remove_data(data_to_downlink)
        assert removed == 50.0  # Only had 50 Gb
        assert recorder.current_volume_gb == 0.0

        # Subsequent passes have nothing to downlink
        for _ in range(3):
            removed = recorder.remove_data(data_to_downlink)
            assert removed == 0.0

        assert recorder.current_volume_gb == 0.0
