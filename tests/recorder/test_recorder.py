"""Tests for OnboardRecorder data storage functionality."""

import pytest

from conops.config import OnboardRecorder


class TestOnboardRecorder:
    """Test suite for OnboardRecorder class."""

    def test_initialization_defaults(self):
        """Test recorder initializes with correct defaults."""
        recorder = OnboardRecorder()
        assert recorder.name == "Default Recorder"
        assert recorder.capacity_gb == 32.0
        assert recorder.current_volume_gb == 0.0
        assert recorder.yellow_threshold == 0.7
        assert recorder.red_threshold == 0.9

    def test_initialization_custom(self):
        """Test recorder initializes with custom values."""
        recorder = OnboardRecorder(
            name="SSR-1",
            capacity_gb=64.0,
            current_volume_gb=10.0,
            yellow_threshold=0.6,
            red_threshold=0.8,
        )
        assert recorder.name == "SSR-1"
        assert recorder.capacity_gb == 64.0
        assert recorder.current_volume_gb == 10.0
        assert recorder.yellow_threshold == 0.6
        assert recorder.red_threshold == 0.8

    def test_add_data(self):
        """Test adding data to recorder."""
        recorder = OnboardRecorder(capacity_gb=100.0)

        # Add data within capacity
        stored = recorder.add_data(25.0)
        assert stored == 25.0
        assert recorder.current_volume_gb == 25.0

        # Add more data
        stored = recorder.add_data(30.0)
        assert stored == 30.0
        assert recorder.current_volume_gb == 55.0

    def test_add_data_exceeds_capacity(self):
        """Test adding data that exceeds capacity."""
        recorder = OnboardRecorder(capacity_gb=100.0)
        recorder.current_volume_gb = 90.0

        # Try to add more than available
        stored = recorder.add_data(20.0)
        assert stored == 10.0  # Only 10 Gb available
        assert recorder.current_volume_gb == 100.0
        assert recorder.is_full()

    def test_add_negative_data(self):
        """Test adding negative data does nothing."""
        recorder = OnboardRecorder(capacity_gb=100.0)
        recorder.current_volume_gb = 50.0

        stored = recorder.add_data(-10.0)
        assert stored == 0.0
        assert recorder.current_volume_gb == 50.0

    def test_remove_data(self):
        """Test removing data from recorder."""
        recorder = OnboardRecorder(capacity_gb=100.0)
        recorder.current_volume_gb = 75.0

        # Remove data
        removed = recorder.remove_data(25.0)
        assert removed == 25.0
        assert recorder.current_volume_gb == 50.0

        # Remove more data
        removed = recorder.remove_data(30.0)
        assert removed == 30.0
        assert recorder.current_volume_gb == 20.0

    def test_remove_data_exceeds_available(self):
        """Test removing more data than available."""
        recorder = OnboardRecorder(capacity_gb=100.0)
        recorder.current_volume_gb = 20.0

        # Try to remove more than available
        removed = recorder.remove_data(50.0)
        assert removed == 20.0  # Only 20 Gb available
        assert recorder.current_volume_gb == 0.0

    def test_remove_negative_data(self):
        """Test removing negative data does nothing."""
        recorder = OnboardRecorder(capacity_gb=100.0)
        recorder.current_volume_gb = 50.0

        removed = recorder.remove_data(-10.0)
        assert removed == 0.0
        assert recorder.current_volume_gb == 50.0

    def test_get_fill_fraction(self):
        """Test fill fraction calculation."""
        recorder = OnboardRecorder(capacity_gb=100.0)

        assert recorder.get_fill_fraction() == 0.0

        recorder.current_volume_gb = 25.0
        assert recorder.get_fill_fraction() == 0.25

        recorder.current_volume_gb = 75.0
        assert recorder.get_fill_fraction() == 0.75

        recorder.current_volume_gb = 100.0
        assert recorder.get_fill_fraction() == 1.0

    def test_get_alert_level_no_alert(self):
        """Test alert level when below yellow threshold."""
        recorder = OnboardRecorder(
            capacity_gb=100.0, yellow_threshold=0.7, red_threshold=0.9
        )

        recorder.current_volume_gb = 50.0  # 50% full
        assert recorder.get_alert_level() == 0

    def test_get_alert_level_yellow(self):
        """Test alert level at yellow threshold."""
        recorder = OnboardRecorder(
            capacity_gb=100.0, yellow_threshold=0.7, red_threshold=0.9
        )

        recorder.current_volume_gb = 70.0  # 70% full
        assert recorder.get_alert_level() == 1

        recorder.current_volume_gb = 80.0  # 80% full
        assert recorder.get_alert_level() == 1

    def test_get_alert_level_red(self):
        """Test alert level at red threshold."""
        recorder = OnboardRecorder(
            capacity_gb=100.0, yellow_threshold=0.7, red_threshold=0.9
        )

        recorder.current_volume_gb = 90.0  # 90% full
        assert recorder.get_alert_level() == 2

        recorder.current_volume_gb = 100.0  # 100% full
        assert recorder.get_alert_level() == 2

    def test_is_full(self):
        """Test is_full method."""
        recorder = OnboardRecorder(capacity_gb=100.0)

        assert not recorder.is_full()

        recorder.current_volume_gb = 50.0
        assert not recorder.is_full()

        recorder.current_volume_gb = 100.0
        assert recorder.is_full()

        recorder.current_volume_gb = 100.1
        assert recorder.is_full()

    def test_available_capacity(self):
        """Test available capacity calculation."""
        recorder = OnboardRecorder(capacity_gb=100.0)

        assert recorder.available_capacity() == 100.0

        recorder.current_volume_gb = 25.0
        assert recorder.available_capacity() == 75.0

        recorder.current_volume_gb = 100.0
        assert recorder.available_capacity() == 0.0

    def test_reset(self):
        """Test resetting the recorder."""
        recorder = OnboardRecorder(capacity_gb=100.0)
        recorder.current_volume_gb = 75.0

        recorder.reset()
        assert recorder.current_volume_gb == 0.0
        assert not recorder.is_full()

    def test_threshold_validation(self):
        """Test that red threshold must be >= yellow threshold."""
        # This should raise a validation error
        with pytest.raises(
            ValueError, match="red_threshold must be >= yellow_threshold"
        ):
            OnboardRecorder(yellow_threshold=0.8, red_threshold=0.6)

    def test_current_volume_capped_at_capacity(self):
        """Test that current volume is capped at capacity during initialization."""
        # When initializing with current_volume > capacity, it should be capped
        recorder = OnboardRecorder(capacity_gb=100.0, current_volume_gb=150.0)
        assert recorder.current_volume_gb == 100.0


class TestDataManagementScenarios:
    """Test realistic data management scenarios."""

    def test_typical_observation_scenario(self):
        """Test a typical observation and downlink scenario."""
        recorder = OnboardRecorder(capacity_gb=64.0)

        # Simulate observations generating data
        for _ in range(10):
            recorder.add_data(2.5)  # 2.5 Gb per observation

        assert recorder.current_volume_gb == 25.0
        assert recorder.get_fill_fraction() == pytest.approx(0.390625, rel=1e-5)
        assert recorder.get_alert_level() == 0

        # Simulate downlink pass
        downlinked = recorder.remove_data(15.0)
        assert downlinked == 15.0
        assert recorder.current_volume_gb == 10.0

    def test_recorder_overflow_scenario(self):
        """Test scenario where recorder fills up."""
        recorder = OnboardRecorder(
            capacity_gb=32.0, yellow_threshold=0.7, red_threshold=0.9
        )

        # Generate data continuously
        for i in range(20):
            data_generated = 2.0
            stored = recorder.add_data(data_generated)

            if i < 16:
                # Should store all data initially
                assert stored == data_generated
            else:
                # After filling up, should store less than requested
                assert stored < data_generated

        # Check final state
        assert recorder.is_full()
        assert recorder.get_alert_level() == 2

    def test_continuous_data_flow(self):
        """Test continuous data generation and downlink."""
        recorder = OnboardRecorder(capacity_gb=100.0)

        # Simulate 24 hours of operations (1 hour = 3600 seconds)
        # Science observations generate 0.1 Gb/hour
        # Downlink passes occur every 6 hours, downlink at 5 Gb/hour

        for hour in range(24):
            # Generate science data
            recorder.add_data(0.1)

            # Downlink every 6 hours for 1 hour
            if hour % 6 == 0 and hour > 0:
                recorder.remove_data(5.0)

        # After 24 hours: 2.4 Gb generated, ~15 Gb downlinked
        # Net should be empty (started at 0, generated 2.4, downlinked more than that)
        assert recorder.current_volume_gb < 2.5
