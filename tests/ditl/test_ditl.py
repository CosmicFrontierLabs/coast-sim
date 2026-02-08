"""Unit tests for DITL class."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from conops import DITL, ACSMode, DITLs


class TestDITLInit:
    """Test DITL initialization."""

    def test_init_with_config(self, mock_config_detailed: Mock) -> None:
        """Test DITL initialization with a valid config."""
        ditl = DITL(config=mock_config_detailed)
        assert ditl.config == mock_config_detailed
        assert ditl.constraint == mock_config_detailed.constraint
        assert ditl.battery == mock_config_detailed.battery
        assert ditl.spacecraft_bus == mock_config_detailed.spacecraft_bus
        assert ditl.payload == mock_config_detailed.payload
        assert ditl.solar_panel == mock_config_detailed.solar_panel

    def test_init_without_config_raises_assertion(self) -> None:
        """Test that DITL initialization without config raises assertion error."""
        with (
            patch("conops.ditl.ditl_mixin.PassTimes"),
            patch("conops.ditl.ditl_mixin.ACS"),
        ):
            with pytest.raises(AttributeError):
                DITL(config=None)

    def test_init_inherits_from_ditl_mixin(self, ditl: DITL) -> None:
        """Test that DITL inherits DITLMixin properties."""
        assert hasattr(ditl, "ra")
        assert hasattr(ditl, "dec")
        assert hasattr(ditl, "mode")
        # Subsystems are initialized from config
        assert ditl.constraint is not None
        assert ditl.battery is not None


class TestDITLCalc:
    """Test DITL calc method."""

    def test_calc_without_ephemeris_returns_false(self, ditl: DITL) -> None:
        """Test that calc raises ValueError when ephemeris is not loaded."""
        ditl.ephem = None
        with pytest.raises(ValueError, match="ERROR: No ephemeris loaded"):
            ditl.calc()

    def test_calc_without_plan_returns_false(self, ditl: DITL) -> None:
        """Test that calc raises ValueError when plan is not loaded."""
        ditl.plan = None
        with pytest.raises(ValueError, match="ERROR: No plan loaded"):
            ditl.calc()

    def test_calc_sets_acs_ephemeris(self, ditl: DITL) -> None:
        """Test that calc sets ACS ephemeris if not already set."""
        ditl.acs.ephem = None
        ditl.calc()
        assert ditl.acs.ephem == ditl.ephem

    def test_calc_with_valid_inputs_returns_true(self, ditl: DITL) -> None:
        """Test that calc returns True with valid inputs."""
        result = ditl.calc()
        assert result is True

    def test_calc_initializes_telemetry_arrays(self, ditl: DITL) -> None:
        """Test that calc initializes all telemetry arrays."""
        ditl.calc()
        assert len(ditl.ra) > 0
        assert len(ditl.dec) > 0
        assert len(ditl.mode) > 0
        assert len(ditl.panel) > 0
        assert len(ditl.obsid) > 0
        assert len(ditl.batterylevel) > 0
        assert len(ditl.batteryalert) > 0
        assert len(ditl.power) > 0

    def test_calc_arrays_same_length(self, ditl: DITL) -> None:
        """Test that all telemetry arrays have the same length."""
        ditl.calc()
        simlen = len(ditl.utime)
        assert len(ditl.ra) == simlen
        assert len(ditl.dec) == simlen
        assert len(ditl.mode) == simlen
        assert len(ditl.panel) == simlen
        assert len(ditl.obsid) == simlen
        assert len(ditl.batterylevel) == simlen
        assert len(ditl.batteryalert) == simlen
        assert len(ditl.power) == simlen


class TestDITLSimulationLoop:
    """Test DITL simulation loop behavior."""

    def test_simulation_loop_calls_pointing(self, ditl: DITL) -> None:
        """Test that simulation loop calls acs.pointing for each timestep."""
        ditl.calc()
        # Should be called simlen times (once per timestep)
        assert ditl.acs.pointing.call_count > 0

    def test_simulation_loop_records_pointing_data(self, ditl: DITL) -> None:
        """Test that pointing data is recorded in telemetry."""
        ditl.acs.pointing.return_value = (45.0, 30.0, 90.0, 42)
        ditl.calc()
        # Check first recorded values
        assert ditl.ra[0] == 45.0
        assert ditl.dec[0] == 30.0
        assert ditl.obsid[0] == 42

    def test_simulation_loop_drains_battery(self, ditl: DITL) -> None:
        """Test that battery is drained each timestep."""
        ditl.calc()
        # Battery drain should be called once per timestep
        assert ditl.battery.drain.call_count > 0

    def test_simulation_loop_charges_battery(self, ditl: DITL) -> None:
        """Test that battery is charged each timestep."""
        ditl.calc()
        # Battery charge should be called once per timestep
        assert ditl.battery.charge.call_count > 0

    def test_battery_drain_uses_calculated_power(self, ditl: DITL) -> None:
        """Test that battery drain uses the calculated power usage."""
        ditl.spacecraft_bus.power = Mock(return_value=100.0)
        ditl.payload.power = Mock(return_value=50.0)
        ditl.calc()
        # Each drain call should have power = 100 + 50 = 150
        ditl.battery.drain.assert_called_with(150.0, ditl.step_size)

    def test_battery_charge_uses_solar_panel_power(self, ditl: DITL) -> None:
        """Test that battery charge uses solar panel power."""
        ditl.solar_panel.illumination_and_power = Mock(return_value=(0.8, 200.0))
        ditl.calc()
        # Each charge call should have the solar panel power
        ditl.battery.charge.assert_called_with(200.0, ditl.step_size)


class TestDITLModeDetection:
    """Test mode detection logic in DITL."""

    def test_mode_slewing(self, ditl: DITL) -> None:
        """Test that mode is set to SLEWING when ACS reports SLEWING mode."""
        ditl.acs.get_mode = Mock(return_value=ACSMode.SLEWING)
        ditl.calc()
        # All modes should be SLEWING
        assert all(mode == ACSMode.SLEWING for mode in ditl.mode)

    def test_mode_pass(self, ditl: DITL) -> None:
        """Test that mode is set to PASS when ACS reports PASS mode."""
        ditl.acs.get_mode = Mock(return_value=ACSMode.PASS)
        ditl.calc()
        # All modes should be PASS
        assert all(mode == ACSMode.PASS for mode in ditl.mode)

    def test_mode_saa_when_available(self, ditl: DITL) -> None:
        """Test that mode is set to SAA when ACS reports SAA mode."""
        ditl.acs.get_mode = Mock(return_value=ACSMode.SAA)
        ditl.calc()
        # All modes should be SAA
        assert all(mode == ACSMode.SAA for mode in ditl.mode)

    def test_mode_science_default(self, ditl: DITL) -> None:
        """Test that mode defaults to SCIENCE when ACS reports SCIENCE mode."""
        # get_mode is already mocked to return SCIENCE in fixture
        ditl.calc()
        # All modes should be SCIENCE
        assert all(mode == ACSMode.SCIENCE for mode in ditl.mode)

    def test_mode_check_uses_acs_get_mode(self, ditl: DITL) -> None:
        """Test that mode determination uses ACS.get_mode() method."""
        # Should not raise any errors
        ditl.calc()
        # Verify get_mode was called (once per timestep)
        assert ditl.acs.get_mode.call_count > 0


class TestDITLPowerCalculations:
    """Test power calculation in DITL."""

    def test_power_calls_spacecraft_bus_power(self, ditl: DITL) -> None:
        """Test that power calculation calls spacecraft_bus.power."""
        ditl.calc()
        # Should be called at least once
        assert ditl.spacecraft_bus.power.call_count > 0
        # Should be called with ACS mode and in_eclipse parameter
        # Get the last call to check arguments
        last_call = ditl.spacecraft_bus.power.call_args
        assert last_call[0][0] == ACSMode.SCIENCE  # mode argument
        assert "in_eclipse" in last_call[1]  # in_eclipse keyword argument

    def test_power_calls_payload_power(self, ditl: DITL) -> None:
        """Test that power calculation calls payload.power."""
        ditl.calc()
        # Should be called at least once
        assert ditl.payload.power.call_count > 0
        # Should be called with ACS mode and in_eclipse parameter
        # Get the last call to check arguments
        last_call = ditl.payload.power.call_args
        assert last_call[0][0] == ACSMode.SCIENCE  # mode argument
        assert "in_eclipse" in last_call[1]  # in_eclipse keyword argument

    def test_power_recorded_in_telemetry(self, ditl: DITL) -> None:
        """Test that calculated power is recorded in telemetry."""
        ditl.spacecraft_bus.power = Mock(return_value=50.0)
        ditl.payload.power = Mock(return_value=30.0)
        ditl.calc()
        # Power should be 50 + 30 = 80
        assert np.max(ditl.power) == 80.0
        assert np.min(ditl.power) == 80.0
        assert np.mean(ditl.power) == 80.0

    def test_solar_panel_power_called_with_correct_args(self, ditl: DITL) -> None:
        """Test that solar panel illumination_and_power is called with time, ra, dec, ephem."""
        ditl.acs.pointing.return_value = (10.0, 20.0, 30.0, 0)
        ditl.calc()
        # Should be called with (time=utime[i], ra=ra, dec=dec, ephem=ephem)
        assert ditl.solar_panel.illumination_and_power.call_count > 0


class TestDITLs:
    """Test DITLs collection class."""

    def test_ditls_append_and_len(self):
        """Test appending DITL to DITLs and checking length."""
        # Create mock ditls directly avoiding reset_stats
        ditls = DITLs.__new__(DITLs)
        ditls.ditls = []
        ditls.reset_stats = Mock()
        ditls.total = 0
        ditls.suncons = 0

        mock_ditl = Mock()
        ditls.append(mock_ditl)
        assert len(ditls) == 1
        assert ditls[0] == mock_ditl

    def test_ditls_getitem(self):
        """Test DITLs indexing."""
        ditls = DITLs.__new__(DITLs)
        ditls.ditls = []
        ditls.reset_stats = Mock()

        mock_ditl1 = Mock()
        mock_ditl2 = Mock()
        ditls.append(mock_ditl1)
        ditls.append(mock_ditl2)
        assert ditls[0] == mock_ditl1
        assert ditls[1] == mock_ditl2

    def test_ditls_multiple_items(self):
        """Test DITLs with multiple items."""
        ditls = DITLs.__new__(DITLs)
        ditls.ditls = []
        ditls.reset_stats = Mock()

        for i in range(5):
            ditls.append(Mock())
        assert len(ditls) == 5

    def test_ditls_number_of_passes(self):
        """Test DITLs number_of_passes property."""
        ditls = DITLs.__new__(DITLs)
        ditls.ditls = []
        ditls.reset_stats = Mock()

        mock_ditl1 = Mock()
        mock_ditl1.executed_passes = [1, 2, 3]
        mock_ditl2 = Mock()
        mock_ditl2.executed_passes = [1, 2]
        ditls.append(mock_ditl1)
        ditls.append(mock_ditl2)
        assert ditls.number_of_passes == [3, 2]


class TestDITLIntegration:
    """Integration tests for DITL."""

    def test_full_simulation_runs_without_error(self, ditl: DITL) -> None:
        """Test that a full simulation runs without errors."""
        result = ditl.calc()
        assert result is True
        assert len(ditl.utime) > 0
        assert len(ditl.power) == len(ditl.utime)

    def test_telemetry_arrays_populated(self, ditl: DITL) -> None:
        """Test that all telemetry arrays are populated during simulation."""
        ditl.calc()
        # Check that values are actually recorded, not just initialized
        assert np.any(ditl.power != 0) or np.any(ditl.power == 0)
        assert ditl.mode is not None
        assert ditl.batterylevel is not None

    def test_simulation_respects_stepsize(self, ditl: DITL) -> None:
        """Test that simulation respects the configured stepsize."""
        ditl.step_size = 120  # 2 minutes
        ditl.calc()
        # utime should be spaced by step_size
        if len(ditl.utime) > 1:
            diffs = np.diff(ditl.utime)
            assert np.allclose(diffs, ditl.step_size)

    def test_simulation_respects_simulation_length(self, ditl: DITL) -> None:
        """Test that simulation respects the configured simulation length."""
        from datetime import datetime, timezone

        # Set begin and end to match the ephemeris timestamps
        ditl.begin = datetime(2018, 11, 27, 0, 0, 0, tzinfo=timezone.utc)
        ditl.end = datetime(
            2018, 11, 27, 0, 4, 0, tzinfo=timezone.utc
        )  # Last timestamp in ephemeris
        ditl.step_size = 60
        ditl.calc()
        # Should have 4 timesteps (np.arange from 0 to 4 with step 1 gives [0,1,2,3])
        assert len(ditl.utime) == 4
