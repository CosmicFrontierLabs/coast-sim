"""Tests for instrument data generation functionality."""

import pytest

from conops.config import DataGeneration, Instrument, Payload


class TestDataGeneration:
    """Test suite for DataGeneration class."""

    def test_initialization_defaults(self):
        """Test DataGeneration initializes with correct defaults."""
        data_gen = DataGeneration()
        assert data_gen.rate_gbps == 0.0
        assert data_gen.per_observation_gb == 0.0

    def test_initialization_rate_based(self):
        """Test initialization with rate-based data generation."""
        data_gen = DataGeneration(rate_gbps=0.5)
        assert data_gen.rate_gbps == 0.5
        assert data_gen.per_observation_gb == 0.0

    def test_initialization_per_observation(self):
        """Test initialization with per-observation data generation."""
        data_gen = DataGeneration(per_observation_gb=10.0)
        assert data_gen.rate_gbps == 0.0
        assert data_gen.per_observation_gb == 10.0

    def test_data_generated_rate_based(self):
        """Test data generation calculation with rate-based mode."""
        data_gen = DataGeneration(rate_gbps=0.1)

        # 60 seconds at 0.1 Gbps
        data = data_gen.data_generated(60)
        assert data == 6.0

        # 120 seconds at 0.1 Gbps
        data = data_gen.data_generated(120)
        assert data == 12.0

    def test_data_generated_per_observation(self):
        """Test data generation with per-observation mode."""
        data_gen = DataGeneration(per_observation_gb=5.0)

        # Should return fixed amount regardless of duration
        data = data_gen.data_generated(60)
        assert data == 5.0

        data = data_gen.data_generated(120)
        assert data == 5.0

    def test_data_generated_per_observation_takes_precedence(self):
        """Test that per_observation_gb takes precedence over rate_gbps."""
        data_gen = DataGeneration(rate_gbps=0.1, per_observation_gb=5.0)

        # Should use per_observation_gb, not rate_gbps
        data = data_gen.data_generated(60)
        assert data == 5.0

    def test_data_generated_zero(self):
        """Test data generation with no configuration."""
        data_gen = DataGeneration()

        data = data_gen.data_generated(60)
        assert data == 0.0


class TestInstrumentWithDataGeneration:
    """Test suite for Instrument with data generation."""

    def test_instrument_default_no_data_generation(self):
        """Test that default instrument has no data generation."""
        instrument = Instrument()
        assert instrument.data_generation.rate_gbps == 0.0
        assert instrument.data_generation.per_observation_gb == 0.0

    def test_instrument_with_rate_based_data_generation(self):
        """Test instrument with rate-based data generation."""
        data_gen = DataGeneration(rate_gbps=0.05)
        instrument = Instrument(name="Camera", data_generation=data_gen)

        assert instrument.data_generation.rate_gbps == 0.05
        data = instrument.data_generation.data_generated(60)
        assert data == 3.0

    def test_instrument_with_per_observation_data_generation(self):
        """Test instrument with per-observation data generation."""
        data_gen = DataGeneration(per_observation_gb=2.5)
        instrument = Instrument(name="Spectrometer", data_generation=data_gen)

        data = instrument.data_generation.data_generated(60)
        assert data == 2.5


class TestPayloadDataGeneration:
    """Test suite for Payload data generation methods."""

    def test_payload_default_no_data_generation(self):
        """Test that default payload has no data generation."""
        payload = Payload()
        assert payload.total_data_rate_gbps() == 0.0
        assert payload.data_generated(60) == 0.0

    def test_payload_total_data_rate(self):
        """Test total data rate calculation across instruments."""
        instruments = [
            Instrument(name="Camera1", data_generation=DataGeneration(rate_gbps=0.1)),
            Instrument(name="Camera2", data_generation=DataGeneration(rate_gbps=0.15)),
            Instrument(
                name="Spectrometer", data_generation=DataGeneration(rate_gbps=0.05)
            ),
        ]
        payload = Payload(payload=instruments)

        total_rate = payload.total_data_rate_gbps()
        assert total_rate == pytest.approx(0.3, rel=1e-6)

    def test_payload_data_generated_rate_based(self):
        """Test data generation with rate-based instruments."""
        instruments = [
            Instrument(name="Camera1", data_generation=DataGeneration(rate_gbps=0.1)),
            Instrument(name="Camera2", data_generation=DataGeneration(rate_gbps=0.2)),
        ]
        payload = Payload(payload=instruments)

        # 60 seconds at 0.3 Gbps total
        data = payload.data_generated(60)
        assert data == pytest.approx(18.0, rel=1e-6)

    def test_payload_data_generated_per_observation(self):
        """Test data generation with per-observation instruments."""
        instruments = [
            Instrument(
                name="Imager",
                data_generation=DataGeneration(per_observation_gb=5.0),
            ),
            Instrument(
                name="Spectrometer",
                data_generation=DataGeneration(per_observation_gb=3.0),
            ),
        ]
        payload = Payload(payload=instruments)

        # Should return sum of per-observation amounts
        data = payload.data_generated(60)
        assert data == 8.0

        # Should be same regardless of duration
        data = payload.data_generated(120)
        assert data == 8.0

    def test_payload_data_generated_mixed(self):
        """Test data generation with mixed rate and per-observation instruments."""
        instruments = [
            Instrument(name="Camera", data_generation=DataGeneration(rate_gbps=0.1)),
            Instrument(
                name="Spectrometer",
                data_generation=DataGeneration(per_observation_gb=2.0),
            ),
        ]
        payload = Payload(payload=instruments)

        # 60 seconds: Camera generates 6.0 Gb, Spectrometer generates 2.0 Gb
        data = payload.data_generated(60)
        assert data == pytest.approx(8.0, rel=1e-6)

    def test_payload_data_generated_with_no_data_instruments(self):
        """Test data generation with some instruments not generating data."""
        instruments = [
            Instrument(name="Camera", data_generation=DataGeneration(rate_gbps=0.1)),
            Instrument(name="PowerSupply", data_generation=DataGeneration()),  # No data
            Instrument(
                name="Spectrometer",
                data_generation=DataGeneration(per_observation_gb=1.0),
            ),
        ]
        payload = Payload(payload=instruments)

        # 60 seconds: Camera generates 6.0 Gb, Spectrometer generates 1.0 Gb
        data = payload.data_generated(60)
        assert data == pytest.approx(7.0, rel=1e-6)


class TestDataGenerationScenarios:
    """Test realistic data generation scenarios."""

    def test_high_data_rate_imaging_mission(self):
        """Test high data rate imaging mission scenario."""
        # High-resolution camera generating lots of data
        camera = Instrument(
            name="High-Res Camera",
            data_generation=DataGeneration(rate_gbps=1.0),  # 1 Gbps
        )
        payload = Payload(payload=[camera])

        # 10 minute observation
        data = payload.data_generated(600)
        assert data == 600.0  # 600 Gb in 10 minutes

    def test_low_data_rate_science_mission(self):
        """Test low data rate science mission scenario."""
        # Low data rate instruments
        instruments = [
            Instrument(
                name="Particle Detector",
                data_generation=DataGeneration(rate_gbps=0.001),
            ),
            Instrument(
                name="Magnetometer",
                data_generation=DataGeneration(rate_gbps=0.0005),
            ),
        ]
        payload = Payload(payload=instruments)

        # 1 hour of data collection
        data = payload.data_generated(3600)
        assert data == pytest.approx(5.4, rel=1e-6)  # 5.4 Gb in 1 hour

    def test_snapshot_observation_mission(self):
        """Test snapshot observation mission scenario."""
        # Instruments that take discrete observations
        camera = Instrument(
            name="Survey Camera",
            data_generation=DataGeneration(per_observation_gb=0.5),
        )
        payload = Payload(payload=[camera])

        # Each observation generates fixed amount
        data = payload.data_generated(1)  # Duration doesn't matter
        assert data == 0.5
