"""Unit tests for conops.visualization.acs_mode_analysis module."""

import matplotlib.pyplot as plt

from conops.common import ACSMode
from conops.visualization.acs_mode_analysis import plot_acs_mode_distribution


class TestPlotAcsModeDistribution:
    """Test plot_acs_mode_distribution function."""

    def test_plot_acs_mode_distribution_returns_figure_and_axes(self, mock_ditl):
        """Test that plot_acs_mode_distribution returns a figure and axes."""
        fig, ax = plot_acs_mode_distribution(mock_ditl)

        assert fig is not None
        assert ax is not None

        # Clean up
        plt.close(fig)

    def test_plot_acs_mode_distribution_custom_figsize(self, mock_ditl):
        """Test plot_acs_mode_distribution with custom figsize."""
        figsize = (8, 6)
        fig, ax = plot_acs_mode_distribution(mock_ditl, figsize=figsize)

        assert fig.get_size_inches()[0] == figsize[0]
        assert fig.get_size_inches()[1] == figsize[1]

        # Clean up
        plt.close(fig)

    def test_plot_acs_mode_distribution_with_known_modes(self, mock_ditl):
        """Test plot_acs_mode_distribution with known ACS modes."""
        fig, ax = plot_acs_mode_distribution(mock_ditl)

        # Check that the pie chart was created
        wedges = ax.patches
        assert len(wedges) > 0  # Should have pie wedges

        # Check that labels contain expected mode names
        labels = [t.get_text() for t in ax.texts if t.get_text()]
        assert any("SCIENCE" in label for label in labels)
        assert any("SLEWING" in label for label in labels)

        # Clean up
        plt.close(fig)

    def test_plot_acs_mode_distribution_with_unknown_mode(self):
        """Test plot_acs_mode_distribution handles unknown modes."""
        from unittest.mock import Mock

        # Create mock with unknown mode
        mock_ditl = Mock()
        mock_ditl.mode = [ACSMode.SCIENCE.value, 999]  # 999 is unknown

        fig, ax = plot_acs_mode_distribution(mock_ditl)

        # Check that unknown mode is labeled appropriately
        labels = [t.get_text() for t in ax.texts if t.get_text()]
        assert any("UNKNOWN(999)" in label for label in labels)

        # Clean up
        plt.close(fig)

    def test_plot_acs_mode_distribution_title(self, mock_ditl):
        """Test that the plot has an appropriate title."""
        fig, ax = plot_acs_mode_distribution(mock_ditl)

        title = ax.get_title()
        assert "Percentage of Time Spent in Each ACS Mode" in title

        # Clean up
        plt.close(fig)

    def test_plot_acs_mode_distribution_title_font_from_config(self, mock_ditl):
        """Test that the plot title uses the font family from VisualizationConfig."""
        from conops.config.visualization import VisualizationConfig

        viz_config = VisualizationConfig(font_family="Courier")
        fig, ax = plot_acs_mode_distribution(mock_ditl, config=viz_config)
        # Get the font family applied to title font properties
        family = ax.title.get_fontproperties().get_family()
        assert isinstance(family, (list, tuple))
        assert "Courier" in family or family == ["Courier"]
        plt.close(fig)
