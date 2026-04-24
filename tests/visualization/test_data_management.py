"""Unit tests for conops.visualization.data_management module."""

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from conops.config.visualization import VisualizationConfig
from conops.visualization import (
    plot_data_management_telemetry,
    plot_data_management_telemetry_plotly,
)


class TestPlotDataManagementTelemetry:
    """Test plot_data_management_telemetry function."""

    def test_plot_data_management_telemetry_returns_figure_and_axes(
        self, mock_ditl
    ) -> None:
        """Test that plot_data_management_telemetry returns a figure and axes."""
        fig, axes = plot_data_management_telemetry(mock_ditl)

        assert fig is not None
        assert axes is not None
        assert len(axes) == 5  # Should have 5 subplots

        # Clean up
        plt.close(fig)

    def test_plot_data_management_telemetry_custom_figsize(self, mock_ditl) -> None:
        """Test plot_data_management_telemetry with custom figsize."""
        figsize = (14, 12)
        fig, axes = plot_data_management_telemetry(mock_ditl, figsize=figsize)

        assert fig.get_size_inches()[0] == figsize[0]
        assert fig.get_size_inches()[1] == figsize[1]

        # Clean up
        plt.close(fig)

    def test_plot_data_management_telemetry_show_summary_false(self, mock_ditl):
        """Test plot_data_management_telemetry with show_summary=False."""
        fig, axes = plot_data_management_telemetry(mock_ditl, show_summary=False)

        assert fig is not None
        assert axes is not None

        # Clean up
        plt.close(fig)

    def test_plot_data_management_telemetry_axes_labels(self, mock_ditl):
        """Test that axes have appropriate labels."""
        fig, axes = plot_data_management_telemetry(mock_ditl)

        # Check some key axis labels
        expected_ylabels = [
            "Volume (Gb)",  # Recorder volume
            "Fill Fraction",  # Fill fraction
            "Data Generated (Gb)",  # Generated data
            "Data Downlinked (Gb)",  # Downlinked data
            "Alert Level",  # Alert timeline
        ]

        for i, ax in enumerate(axes):
            ylabel = ax.get_ylabel()
            assert expected_ylabels[i] in ylabel

        # Clean up
        plt.close(fig)

    def test_plot_data_management_telemetry_titles(self, mock_ditl):
        """Test that subplots have appropriate titles."""
        fig, axes = plot_data_management_telemetry(mock_ditl)

        expected_titles = [
            "Onboard Recorder Data Volume",
            "Recorder Fill Level",
            "Cumulative Data Generated",
            "Cumulative Data Downlinked",
            "Recorder Alert Timeline",
        ]

        for i, ax in enumerate(axes):
            title = ax.get_title()
            assert expected_titles[i] in title

        # Clean up
        plt.close(fig)

    def test_plot_data_management_telemetry_has_plots(self, mock_ditl):
        """Test that the plots contain actual data lines."""
        fig, axes = plot_data_management_telemetry(mock_ditl)

        # Check that each subplot has at least one line or scatter plot
        for i, ax in enumerate(axes):
            lines = ax.get_lines()
            collections = ax.collections  # For scatter plots
            if i == 4:  # Alert timeline uses scatter
                assert len(collections) > 0, (
                    f"Axis {i} (alert timeline) should have scatter points"
                )
            else:
                assert len(lines) > 0, f"Axis {i} should have at least one line"

        # Clean up
        plt.close(fig)


class TestPlotDataManagementTelemetryPlotly:
    """Test plot_data_management_telemetry_plotly function."""

    def test_returns_go_figure(self, mock_ditl) -> None:
        fig = plot_data_management_telemetry_plotly(mock_ditl)
        assert isinstance(fig, go.Figure)

    def test_trace_count(self, mock_ditl) -> None:
        # 4 data traces (volume, fill fraction, generated, downlinked)
        # + 3 alert-level traces (all three levels present in mock data)
        fig = plot_data_management_telemetry_plotly(mock_ditl)
        assert len(fig.data) == 7

    def test_alert_traces_present(self, mock_ditl) -> None:
        fig = plot_data_management_telemetry_plotly(mock_ditl)
        names = {t.name for t in fig.data}
        assert "No Alert" in names
        assert "Yellow Alert" in names
        assert "Red Alert" in names

    def test_only_present_alert_levels_get_traces(self) -> None:
        # Build a fully Mock-based DITL with only 2 alert levels (no red)
        from datetime import datetime, timedelta, timezone
        from unittest.mock import Mock

        hk = Mock()
        times = [
            datetime.fromtimestamp(0, tz=timezone.utc) + i * timedelta(hours=1)
            for i in range(5)
        ]
        hk.timestamp = times
        hk.recorder_volume_gb = [0.0, 0.5, 1.0, 1.5, 2.0]
        hk.recorder_fill_fraction = [0.0, 0.25, 0.5, 0.75, 1.0]
        hk.recorder_alert = [0, 0, 1, 0, 1]  # only levels 0 and 1

        telemetry = Mock()
        telemetry.housekeeping = hk

        ditl = Mock()
        ditl.telemetry = telemetry
        ditl.data_generated_gb = [0.0, 0.5, 1.0, 1.5, 2.0]
        ditl.data_downlinked_gb = [0.0, 0.3, 0.6, 0.9, 1.2]
        ditl.config.recorder.capacity_gb = 2.0
        ditl.config.recorder.yellow_threshold = 0.8
        ditl.config.recorder.red_threshold = 0.95
        ditl.config.name = "Test"

        fig = plot_data_management_telemetry_plotly(ditl)
        names = {t.name for t in fig.data}
        assert "No Alert" in names
        assert "Yellow Alert" in names
        assert "Red Alert" not in names
        # Total: 4 data + 2 alert = 6 traces
        assert len(fig.data) == 6

    def test_subplot_titles(self, mock_ditl) -> None:
        expected_titles = [
            "Onboard Recorder Data Volume",
            "Recorder Fill Level",
            "Cumulative Data Generated",
            "Cumulative Data Downlinked",
            "Recorder Alert Timeline",
        ]
        fig = plot_data_management_telemetry_plotly(mock_ditl)
        annotation_texts = [a.text for a in fig.layout.annotations]
        for title in expected_titles:
            assert any(title in t for t in annotation_texts)

    def test_title_contains_config_name(self, mock_ditl) -> None:
        fig = plot_data_management_telemetry_plotly(mock_ditl)
        assert "Test Config" in fig.layout.title.text

    def test_explicit_vis_config(self, mock_ditl) -> None:
        config = VisualizationConfig()
        fig = plot_data_management_telemetry_plotly(mock_ditl, config=config)
        assert isinstance(fig, go.Figure)

    def test_missing_recorder_config_does_not_raise(self) -> None:
        # ditl with no recorder attribute — capacity/threshold hlines are skipped
        from datetime import datetime, timedelta, timezone
        from unittest.mock import Mock

        hk = Mock()
        times = [
            datetime.fromtimestamp(0, tz=timezone.utc) + i * timedelta(hours=1)
            for i in range(3)
        ]
        hk.timestamp = times
        hk.recorder_volume_gb = [0.0, 0.5, 1.0]
        hk.recorder_fill_fraction = [0.0, 0.25, 0.5]
        hk.recorder_alert = [0, 0, 1]

        telemetry = Mock()
        telemetry.housekeeping = hk

        ditl = Mock()
        ditl.telemetry = telemetry
        ditl.data_generated_gb = [0.0, 0.5, 1.0]
        ditl.data_downlinked_gb = [0.0, 0.3, 0.6]
        # Simulate missing recorder by returning None
        ditl.config.recorder = None
        ditl.config.name = "No Recorder"

        fig = plot_data_management_telemetry_plotly(ditl)
        assert isinstance(fig, go.Figure)
        # 4 data traces + 2 alert traces (levels 0 and 1)
        assert len(fig.data) == 6

    def test_missing_data_arrays_does_not_raise(self) -> None:
        # ditl without data_generated_gb / data_downlinked_gb attributes
        from datetime import datetime, timedelta, timezone
        from unittest.mock import Mock

        hk = Mock()
        times = [
            datetime.fromtimestamp(0, tz=timezone.utc) + i * timedelta(hours=1)
            for i in range(3)
        ]
        hk.timestamp = times
        hk.recorder_volume_gb = [0.0, 0.5, 1.0]
        hk.recorder_fill_fraction = [0.0, 0.25, 0.5]
        hk.recorder_alert = [0, 0, 0]

        telemetry = Mock()
        telemetry.housekeeping = hk

        ditl = Mock(spec=["telemetry", "config"])
        ditl.telemetry = telemetry
        ditl.config.recorder.capacity_gb = 2.0
        ditl.config.recorder.yellow_threshold = 0.8
        ditl.config.recorder.red_threshold = 0.95
        ditl.config.name = "No Data Arrays"

        fig = plot_data_management_telemetry_plotly(ditl)
        assert isinstance(fig, go.Figure)
        # 4 data traces (generated/downlinked panels have empty y) + 1 alert trace
        assert len(fig.data) == 5
