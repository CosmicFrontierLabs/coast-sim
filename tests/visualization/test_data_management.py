"""Unit tests for conops.visualization.data_management module."""

import matplotlib.pyplot as plt

from conops.visualization.data_management import plot_data_management_telemetry


class TestPlotDataManagementTelemetry:
    """Test plot_data_management_telemetry function."""

    def test_plot_data_management_telemetry_returns_figure_and_axes(self, mock_ditl):
        """Test that plot_data_management_telemetry returns a figure and axes."""
        fig, axes = plot_data_management_telemetry(mock_ditl)

        assert fig is not None
        assert axes is not None
        assert len(axes) == 5  # Should have 5 subplots

        # Clean up
        plt.close(fig)

    def test_plot_data_management_telemetry_custom_figsize(self, mock_ditl):
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
