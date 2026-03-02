from datetime import datetime, timezone
from unittest.mock import Mock, patch

import matplotlib
from matplotlib.figure import Figure

from conops import DITLMixin

"""Unit tests for DITLMixin."""

matplotlib.use("Agg")


class TestDITLMixin:
    def test_init_sets_config(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock], mock_config: Mock
    ) -> None:
        """DITLMixin.__init__ should set config."""
        ditl, _, _ = ditl_instance
        assert ditl.config is mock_config

    def test_init_sets_ra_as_empty_list(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """DITLMixin.__init__ should set ra as empty list."""
        ditl, _, _ = ditl_instance
        assert isinstance(ditl.ra, list) and ditl.ra == []

    def test_init_sets_dec_as_empty_list(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """DITLMixin.__init__ should set dec as empty list."""
        ditl, _, _ = ditl_instance
        assert isinstance(ditl.dec, list) and ditl.dec == []

    def test_init_sets_utime_as_empty_list(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """DITLMixin.__init__ should set utime as empty list."""
        ditl, _, _ = ditl_instance
        assert isinstance(ditl.utime, list) and ditl.utime == []

    def test_init_sets_ephem_from_config(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock], mock_config: Mock
    ) -> None:
        """DITLMixin.__init__ should set ephem from config.constraint.ephem."""
        ditl, _, _ = ditl_instance
        assert ditl.ephem is mock_config.constraint.ephem

    def test_init_uses_passes(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """DITLMixin.__init__ should use PassTimes."""
        ditl, mock_pass_inst, _ = ditl_instance
        assert ditl.passes is mock_pass_inst

    def test_init_sets_executed_passes(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """DITLMixin.__init__ should set executed_passes."""
        ditl, mock_pass_inst, _ = ditl_instance
        assert ditl.executed_passes is mock_pass_inst

    def test_init_uses_acs(self, ditl_instance: tuple[DITLMixin, Mock, Mock]) -> None:
        """DITLMixin.__init__ should use ACS."""
        ditl, _, mock_acs_inst = ditl_instance
        assert ditl.acs is mock_acs_inst

    def test_init_sets_begin_datetime(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """DITLMixin.__init__ should set begin datetime."""
        ditl, _, _ = ditl_instance
        assert (
            ditl.begin - datetime(2018, 11, 27, 0, 0, 0, tzinfo=timezone.utc)
        ).total_seconds() == 0

    def test_init_sets_end_datetime(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """DITLMixin.__init__ should set end datetime."""
        ditl, _, _ = ditl_instance
        assert (
            ditl.end - datetime(2018, 11, 28, 0, 0, 0, tzinfo=timezone.utc)
        ).total_seconds() == 0

    def test_init_sets_step_size(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """DITLMixin.__init__ should set step_size."""
        ditl, _, _ = ditl_instance
        assert ditl.step_size == 60

    def test_init_sets_ppt_to_none(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """DITLMixin.__init__ should set ppt to None."""
        ditl, _, _ = ditl_instance
        assert ditl.ppt is None

    def test_plot_creates_seven_subplots(self, plot_figure: Figure) -> None:
        """Plot should create 7 subplots."""
        fig = plot_figure
        assert len(fig.axes) == 7

    def test_plot_sets_title_with_config_name(
        self, plot_figure: Figure, mock_config: Mock
    ) -> None:
        """Plot should set title with config name."""
        fig = plot_figure
        assert (
            fig.axes[0].get_title()
            == f"Timeline for DITL Simulation: {mock_config.name}"
        )

    def test_plot_battery_axis_has_dashed_horizontal_line(
        self, plot_figure: Figure
    ) -> None:
        """Plot battery axis should have dashed horizontal line."""
        fig = plot_figure
        batt_ax = fig.axes[3]
        has_dashed_hline = any(line.get_linestyle() == "--" for line in batt_ax.lines)
        assert has_dashed_hline

    def test_plot_last_axis_has_xlabel_time_in_hours(self, plot_figure: Figure) -> None:
        """Plot last axis should have xlabel time in hours."""
        fig = plot_figure
        assert fig.axes[-1].get_xlabel() == "Time (hour of day)"

    def test_print_statistics_includes_ditl_simulation_statistics(
        self, statistics_output: str
    ):
        """print_statistics should include DITL SIMULATION STATISTICS."""
        assert "DITL SIMULATION STATISTICS" in statistics_output

    def test_print_statistics_includes_mode_distribution(
        self, statistics_output: str
    ) -> None:
        """print_statistics should include MODE DISTRIBUTION."""
        assert "MODE DISTRIBUTION" in statistics_output

    def test_print_statistics_includes_observation_statistics(
        self, statistics_output: str
    ) -> None:
        """print_statistics should include OBSERVATION STATISTICS."""
        assert "OBSERVATION STATISTICS" in statistics_output

    def test_print_statistics_includes_pointing_statistics(
        self, statistics_output: str
    ) -> None:
        """print_statistics should include POINTING STATISTICS."""
        assert "POINTING STATISTICS" in statistics_output

    def test_print_statistics_includes_power_and_battery_statistics(
        self, statistics_output: str
    ):
        """print_statistics should include POWER AND BATTERY STATISTICS."""
        assert "POWER AND BATTERY STATISTICS" in statistics_output

    def test_print_statistics_includes_data_management_statistics(
        self, statistics_output: str
    ):
        """print_statistics should include DATA MANAGEMENT STATISTICS."""
        assert "DATA MANAGEMENT STATISTICS" in statistics_output

    def test_print_statistics_includes_target_queue_statistics(
        self, statistics_output: str
    ) -> None:
        """print_statistics should include TARGET QUEUE STATISTICS."""
        assert "TARGET QUEUE STATISTICS" in statistics_output

    def test_print_statistics_includes_acs_command_statistics(
        self, statistics_output: str
    ) -> None:
        """print_statistics should include ACS COMMAND STATISTICS."""
        assert "ACS COMMAND STATISTICS" in statistics_output

    def test_print_statistics_includes_ground_station_pass_statistics(
        self, statistics_output: str
    ):
        """print_statistics should include GROUND STATION PASS STATISTICS."""
        assert "GROUND STATION PASS STATISTICS" in statistics_output

    def test_find_current_pass_returns_none_with_no_passes(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """_find_current_pass should return None with no passes."""
        ditl, _, _ = ditl_instance
        ditl.acs.passrequests.passes = []
        ditl.executed_passes.passes = []
        assert ditl._find_current_pass(1000.0) is None

    def test_find_current_pass_returns_pass_when_in_pass(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """_find_current_pass should return pass when in pass."""
        ditl, _, _ = ditl_instance
        mock_pass = Mock()
        mock_pass.in_pass.return_value = True
        ditl.acs = Mock()
        ditl.acs.passrequests = Mock()
        ditl.acs.passrequests.passes = [mock_pass]
        result = ditl._find_current_pass(1000.0)
        assert result is mock_pass

    def test_find_current_pass_calls_in_pass_with_time(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """_find_current_pass should call in_pass with time."""
        ditl, _, _ = ditl_instance
        mock_pass = Mock()
        mock_pass.in_pass.return_value = True
        ditl.acs = Mock()
        ditl.acs.passrequests = Mock()
        ditl.acs.passrequests.passes = [mock_pass]
        ditl._find_current_pass(1000.0)
        mock_pass.in_pass.assert_called_with(1000.0)

    def test_find_current_pass_returns_none_when_not_in_pass(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """_find_current_pass should return None when not in pass."""
        ditl, _, _ = ditl_instance
        mock_pass = Mock()
        mock_pass.in_pass.return_value = False
        ditl.acs = Mock()
        ditl.acs.passrequests = Mock()
        ditl.acs.passrequests.passes = [mock_pass]
        ditl.executed_passes.passes = []
        assert ditl._find_current_pass(1000.0) is None

    def test_find_current_pass_fallback_to_executed_passes(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """_find_current_pass should fallback to executed_passes."""
        ditl, _, _ = ditl_instance
        ditl.acs.passrequests.passes = []
        ditl.executed_passes = Mock()
        mock_pass = Mock()
        mock_pass.in_pass.return_value = True
        ditl.executed_passes.passes = [mock_pass]
        result = ditl._find_current_pass(1000.0)
        assert result is mock_pass

    def test_process_data_management_science_mode_data_gen(
        self, ditl_with_payload_and_recorder: DITLMixin
    ) -> None:
        """_process_data_management in SCIENCE mode should generate data."""
        ditl = ditl_with_payload_and_recorder
        from conops.common.enums import ACSMode

        data_gen, data_dl = ditl._process_data_management(1000.0, ACSMode.SCIENCE, 60)
        assert data_gen == 0.1

    def test_process_data_management_science_mode_data_dl_zero(
        self, ditl_with_payload_and_recorder: DITLMixin
    ) -> None:
        """_process_data_management in SCIENCE mode should have zero data downlink."""
        ditl = ditl_with_payload_and_recorder
        from conops.common.enums import ACSMode

        data_gen, data_dl = ditl._process_data_management(1000.0, ACSMode.SCIENCE, 60)
        assert data_dl == 0.0

    def test_process_data_management_science_mode_calls_data_generated(
        self, ditl_with_payload_and_recorder: DITLMixin
    ) -> None:
        """_process_data_management in SCIENCE mode should call data_generated."""
        ditl = ditl_with_payload_and_recorder
        from conops.common.enums import ACSMode

        ditl._process_data_management(1000.0, ACSMode.SCIENCE, 60)
        ditl.payload.data_generated.assert_called_with(60)

    def test_process_data_management_science_mode_calls_add_data(
        self, ditl_with_payload_and_recorder: DITLMixin
    ) -> None:
        """_process_data_management in SCIENCE mode should call add_data."""
        ditl = ditl_with_payload_and_recorder
        from conops.common.enums import ACSMode

        ditl._process_data_management(1000.0, ACSMode.SCIENCE, 60)
        ditl.recorder.add_data.assert_called_with(0.1)

    def test_process_data_management_pass_mode_data_gen_zero(
        self, ditl_with_pass_setup: DITLMixin
    ) -> None:
        """_process_data_management in PASS mode should have zero data generation."""
        ditl = ditl_with_pass_setup
        from conops.common.enums import ACSMode

        data_gen, data_dl = ditl._process_data_management(1000.0, ACSMode.PASS, 60)
        assert data_gen == 0.0

    def test_process_data_management_pass_mode_data_dl(
        self, ditl_with_pass_setup: DITLMixin
    ) -> None:
        """_process_data_management in PASS mode should downlink data."""
        ditl = ditl_with_pass_setup
        from conops.common.enums import ACSMode

        data_gen, data_dl = ditl._process_data_management(1000.0, ACSMode.PASS, 60)
        assert data_dl == 0.05

    def test_process_data_management_pass_mode_calls_remove_data(
        self, ditl_with_pass_setup: DITLMixin
    ):
        """_process_data_management in PASS mode should call remove_data."""
        ditl = ditl_with_pass_setup
        from conops.common.enums import ACSMode

        ditl._process_data_management(1000.0, ACSMode.PASS, 60)
        ditl.recorder.remove_data.assert_called_with(0.75)

    def test_process_data_management_other_modes_data_gen_zero(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """_process_data_management in other modes should have zero data generation."""
        ditl, _, _ = ditl_instance
        from conops.common.enums import ACSMode

        data_gen, data_dl = ditl._process_data_management(1000.0, ACSMode.SLEWING, 60)
        assert data_gen == 0.0

    def test_process_data_management_other_modes_data_dl_zero(
        self, ditl_instance: tuple[DITLMixin, Mock, Mock]
    ) -> None:
        """_process_data_management in other modes should have zero data downlink."""
        ditl, _, _ = ditl_instance
        from conops.common.enums import ACSMode

        data_gen, data_dl = ditl._process_data_management(1000.0, ACSMode.SLEWING, 60)
        assert data_dl == 0.0

    def test_init_subsystems_called(self, mock_config: Mock) -> None:
        """Test that _init_subsystems is called during initialization."""
        with (
            patch("conops.ditl.ditl_mixin.PassTimes") as mock_pass_class,
            patch("conops.ditl.ditl_mixin.ACS") as mock_acs_class,
            patch("conops.ditl.ditl_mixin.Plan") as mock_plan_class,
            patch.object(DITLMixin, "_init_subsystems") as mock_init_subsystems,
        ):
            mock_pass_class.return_value = Mock()
            mock_acs_class.return_value = Mock()
            mock_plan_class.return_value = Mock()

            _ = DITLMixin(config=mock_config)

            # Verify _init_subsystems was called
            mock_init_subsystems.assert_called_once()
