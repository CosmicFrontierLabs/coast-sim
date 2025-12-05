"""Tests for DITLMixin.print_statistics method, refactored into a test class."""

from datetime import datetime

from conops import (
    ACSMode,
    Battery,
    Constraint,
    DITLMixin,
    DITLStats,
    GroundStationRegistry,
    MissionConfig,
    Payload,
    Queue,
    SolarPanelSet,
    SpacecraftBus,
)


class MockDITL(DITLMixin, DITLStats):
    """Mock DITL class for testing."""

    def __init__(self, config: MissionConfig) -> None:
        super().__init__(config)
        # Initialize required lists
        self.ra = []
        self.dec = []
        self.roll = []
        self.mode = []
        self.panel = []
        self.power = []
        self.panel_power = []
        self.batterylevel = []
        self.obsid = []
        self.utime = []


def create_test_config(ephem=None):
    """Create a minimal test config."""
    if ephem is None:
        # Create a simple mock ephemeris
        import datetime
        from unittest.mock import Mock

        ephem = Mock()
        ephem.step_size = 60
        start_time = 1543276800  # 2018-11-27 00:00:00 UTC
        unix_times = [start_time + i * 60 for i in range(5)]
        ephem.timestamp = [
            datetime.datetime.fromtimestamp(float(t), tz=datetime.timezone.utc)
            for t in unix_times
        ]
        ephem.utime = unix_times
        ephem.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0)) for _ in unix_times]
        ephem.sun = [Mock(ra=Mock(deg=45.0), dec=Mock(deg=23.5)) for _ in unix_times]

    spacecraft_bus = Mock(spec=SpacecraftBus)
    spacecraft_bus.attitude_control = Mock()
    spacecraft_bus.attitude_control.predict_slew = Mock(return_value=(45.0, []))
    spacecraft_bus.attitude_control.slew_time = Mock(return_value=100.0)

    solar_panel = Mock(spec=SolarPanelSet)
    solar_panel.optimal_charging_pointing = Mock(return_value=(45.0, 23.5))

    payload = Mock(spec=Payload)
    battery = Mock(spec=Battery)
    battery.capacity = 100.0
    battery.max_depth_of_discharge = 0.3
    constraint = Mock(spec=Constraint)
    constraint.ephem = ephem
    ground_stations = Mock(spec=GroundStationRegistry)

    config = MissionConfig(
        name="Test Spacecraft",
        spacecraft_bus=spacecraft_bus,
        solar_panel=solar_panel,
        payload=payload,
        battery=battery,
        constraint=constraint,
        ground_stations=ground_stations,
    )
    return config


class TestDITLPrintStatistics:
    """Test class for DITLMixin.print_statistics."""

    def setup_method(self, method):
        self.config = create_test_config()

    def populate_sample_data(self, ditl: MockDITL):
        """Populate the DITL instance with a representative dataset."""
        ditl.begin = datetime(2025, 11, 1, 0, 0, 0)
        ditl.end = datetime(2025, 11, 1, 1, 0, 0)
        ditl.step_size = 60

        for i in range(60):
            ditl.utime.append(i * 60)
            ditl.ra.append(180.0 + i * 0.1)
            ditl.dec.append(45.0 + i * 0.05)
            ditl.roll.append(0.0)
            ditl.mode.append(ACSMode.SCIENCE if i % 5 != 0 else ACSMode.SLEWING)
            ditl.panel.append(0.8 if i % 10 < 8 else 0.0)  # Simulate eclipse
            ditl.power.append(50.0 + i * 0.5)
            ditl.panel_power.append(80.0 if i % 10 < 8 else 0.0)
            ditl.batterylevel.append(0.8 - i * 0.001)
            ditl.obsid.append(1000 + (i // 10))

    def _get_basic_output(self, capsys):
        ditl = MockDITL(self.config)
        self.populate_sample_data(ditl)
        ditl.print_statistics()
        return capsys.readouterr().out

    def _get_empty_output(self, capsys):
        ditl = MockDITL(self.config)
        ditl.begin = datetime(2025, 11, 1, 0, 0, 0)
        ditl.end = datetime(2025, 11, 1, 1, 0, 0)
        ditl.step_size = 60
        ditl.print_statistics()
        return capsys.readouterr().out

    def _get_queue_output(self, capsys):
        ditl = MockDITL(self.config)
        ditl.begin = datetime(2025, 11, 1, 0, 0, 0)
        ditl.end = datetime(2025, 11, 1, 1, 0, 0)
        ditl.step_size = 60

        # Add minimal data
        ditl.utime = [0]
        ditl.mode = [ACSMode.SCIENCE]
        ditl.obsid = [1000]
        ditl.batterylevel = [0.8]
        ditl.ra = [180.0]
        ditl.dec = [45.0]

        # Add a mock queue
        ditl.queue = Queue(config=self.config)

        ditl.print_statistics()
        return capsys.readouterr().out

    # Basic output tests — one assertion per test
    def test_print_statistics_basic_contains_ditl_simulation_statistics(self, capsys):
        output = self._get_basic_output(capsys)
        assert "DITL SIMULATION STATISTICS" in output

    def test_print_statistics_basic_contains_configuration(self, capsys):
        output = self._get_basic_output(capsys)
        assert "Configuration: Test Spacecraft" in output

    def test_print_statistics_basic_contains_mode_distribution(self, capsys):
        output = self._get_basic_output(capsys)
        assert "MODE DISTRIBUTION" in output

    def test_print_statistics_basic_contains_observation_statistics(self, capsys):
        output = self._get_basic_output(capsys)
        assert "OBSERVATION STATISTICS" in output

    def test_print_statistics_basic_contains_pointing_statistics(self, capsys):
        output = self._get_basic_output(capsys)
        assert "POINTING STATISTICS" in output

    def test_print_statistics_basic_contains_power_and_battery_statistics(self, capsys):
        output = self._get_basic_output(capsys)
        assert "POWER AND BATTERY STATISTICS" in output

    def test_print_statistics_basic_contains_battery_capacity(self, capsys):
        output = self._get_basic_output(capsys)
        assert "Battery Capacity: 100.00 Wh" in output

    def test_print_statistics_basic_contains_science_mode(self, capsys):
        output = self._get_basic_output(capsys)
        assert "SCIENCE" in output

    def test_print_statistics_basic_contains_slewing_mode(self, capsys):
        output = self._get_basic_output(capsys)
        assert "SLEWING" in output

    # Queue test remains a single assertion
    def test_print_statistics_with_queue_contains_target_queue_statistics(self, capsys):
        output = self._get_queue_output(capsys)
        assert "TARGET QUEUE STATISTICS" in output

    # Empty data tests — one assertion per test
    def test_print_statistics_empty_data_contains_ditl_simulation_statistics(
        self, capsys
    ):
        output = self._get_empty_output(capsys)
        assert "DITL SIMULATION STATISTICS" in output

    def test_print_statistics_empty_data_contains_configuration(self, capsys):
        output = self._get_empty_output(capsys)
        assert "Configuration: Test Spacecraft" in output
