"""Unit tests for conops.visualization.ditl_telemetry module."""

import matplotlib.pyplot as plt

from conops.visualization.ditl_telemetry import plot_ditl_telemetry


class TestPlotDitlTelemetry:
    """Test plot_ditl_telemetry function."""

    def test_plot_ditl_telemetry_returns_figure_and_axes(self, mock_ditl):
        """Test that plot_ditl_telemetry returns a figure and axes."""
        fig, axes = plot_ditl_telemetry(mock_ditl)

        assert fig is not None
        assert isinstance(axes, list)
        assert len(axes) == 7  # Should have 7 subplots

        # Clean up
        plt.close(fig)

    def test_plot_ditl_telemetry_custom_figsize(self, mock_ditl):
        """Test plot_ditl_telemetry with custom figsize."""
        figsize = (12, 10)
        fig, axes = plot_ditl_telemetry(mock_ditl, figsize=figsize)

        assert fig.get_size_inches()[0] == figsize[0]
        assert fig.get_size_inches()[1] == figsize[1]

        # Clean up
        plt.close(fig)

    def test_plot_ditl_telemetry_with_subsystem_power(self, mock_ditl):
        """Test plot_ditl_telemetry with subsystem power breakdown."""
        fig, axes = plot_ditl_telemetry(mock_ditl)

        # The power plot should be the 6th subplot (index 5)
        power_ax = axes[5]
        assert power_ax is not None

        # Check that the plot has some content (lines)
        lines = power_ax.get_lines()
        assert len(lines) > 0  # Should have at least total power line

        # Clean up
        plt.close(fig)

    def test_plot_ditl_telemetry_title_contains_config_name(self, mock_ditl):
        """Test that the plot title contains the config name."""
        fig, axes = plot_ditl_telemetry(mock_ditl)

        title = axes[0].get_title()
        assert mock_ditl.config.name in title

        # Clean up
        plt.close(fig)

    def test_plot_ditl_telemetry_axes_labels(self, mock_ditl):
        """Test that axes have appropriate labels."""
        fig, axes = plot_ditl_telemetry(mock_ditl)

        # Check some key axis labels
        expected_labels = [
            "RA",
            "Dec",
            "Mode",
            "Batt. charge",
            "Panel Ill.",
            "Power",
            "ObsID",
        ]
        for i, ax in enumerate(axes):
            ylabel = ax.get_ylabel()
            assert expected_labels[i] in ylabel

        # Clean up
        plt.close(fig)
