"""Unit tests for conops.visualization.ditl_timeline module."""

from unittest.mock import Mock

import matplotlib.pyplot as plt

from conops.visualization.ditl_timeline import (
    annotate_slew_distances,
    plot_ditl_timeline,
)


class TestPlotDitlTimeline:
    """Test plot_ditl_timeline function."""

    def test_plot_ditl_timeline_returns_figure_and_axes(self, mock_ditl_with_ephem):
        """Test that plot_ditl_timeline returns a figure and axes."""
        fig, ax = plot_ditl_timeline(mock_ditl_with_ephem)

        assert fig is not None
        assert ax is not None

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_custom_figsize(self, mock_ditl_with_ephem):
        """Test plot_ditl_timeline with custom figsize."""
        figsize = (15, 8)
        fig, ax = plot_ditl_timeline(mock_ditl_with_ephem, figsize=figsize)

        assert fig.get_size_inches()[0] == figsize[0]
        assert fig.get_size_inches()[1] == figsize[1]

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_offset_hours(self, mock_ditl_with_ephem):
        """Test plot_ditl_timeline with time offset."""
        offset_hours = 5.0
        fig, ax = plot_ditl_timeline(mock_ditl_with_ephem, offset_hours=offset_hours)

        assert fig is not None

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_hide_orbit_numbers(self, mock_ditl_with_ephem):
        """Test plot_ditl_timeline with orbit numbers hidden."""
        fig, ax = plot_ditl_timeline(mock_ditl_with_ephem, show_orbit_numbers=False)

        assert fig is not None

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_hide_saa(self, mock_ditl_with_ephem):
        """Test plot_ditl_timeline with SAA passages hidden."""
        fig, ax = plot_ditl_timeline(mock_ditl_with_ephem, show_saa=False)

        assert fig is not None

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_custom_orbit_period(self, mock_ditl_with_ephem):
        """Test plot_ditl_timeline with custom orbit period."""
        orbit_period = 5000.0
        fig, ax = plot_ditl_timeline(mock_ditl_with_ephem, orbit_period=orbit_period)

        assert fig is not None

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_with_observation_categories(self, mock_ditl_with_ephem):
        """Test plot_ditl_timeline with custom observation categories."""
        from conops.config import ObservationCategories, ObservationCategory

        categories = ObservationCategories(
            categories=[
                ObservationCategory(
                    name="Test", obsid_min=10000, obsid_max=20000, color="red"
                )
            ]
        )

        fig, ax = plot_ditl_timeline(
            mock_ditl_with_ephem, observation_categories=categories
        )

        assert fig is not None

        # Clean up
        plt.close(fig)


class TestAnnotateSlewDistances:
    """Test annotate_slew_distances function."""

    def test_annotate_slew_distances_basic(self, mock_ditl_with_ephem):
        """Test basic annotate_slew_distances functionality."""
        fig, ax = plt.subplots()

        # Mock the required parameters
        t_start = 0.0
        offset_hours = 0.0
        slew_indices = [0, 1]

        # Mock plan with some entries
        mock_plan_entry = Mock()
        mock_plan_entry.begin = 1000.0
        mock_plan_entry.slewtime = 120.0
        mock_plan_entry.slewdist = 10.0
        mock_ditl_with_ephem.plan = [mock_plan_entry, mock_plan_entry]

        result_ax = annotate_slew_distances(
            ax, mock_ditl_with_ephem, t_start, offset_hours, slew_indices
        )

        assert result_ax is ax

        # Clean up
        plt.close(fig)

    def test_annotate_slew_distances_empty_indices(self, mock_ditl_with_ephem):
        """Test annotate_slew_distances with empty slew indices."""
        fig, ax = plt.subplots()

        t_start = 0.0
        offset_hours = 0.0
        slew_indices = []

        mock_ditl_with_ephem.plan = []

        result_ax = annotate_slew_distances(
            ax, mock_ditl_with_ephem, t_start, offset_hours, slew_indices
        )

        assert result_ax is ax

        # Clean up
        plt.close(fig)

    def test_annotate_slew_distances_multiple_slews(self, mock_ditl_with_ephem):
        """Test annotate_slew_distances with multiple slews."""
        fig, ax = plt.subplots()

        t_start = 0.0
        offset_hours = 0.0
        slew_indices = [0, 1]

        # Create mock plan entries with required attributes
        mock_plan_entry1 = Mock()
        mock_plan_entry1.begin = 1000.0
        mock_plan_entry1.slewtime = 120.0
        mock_plan_entry1.slewdist = 10.0

        mock_plan_entry2 = Mock()
        mock_plan_entry2.begin = 2000.0
        mock_plan_entry2.slewtime = 150.0
        mock_plan_entry2.slewdist = 15.0

        mock_ditl_with_ephem.plan = [mock_plan_entry1, mock_plan_entry2]

        result_ax = annotate_slew_distances(
            ax, mock_ditl_with_ephem, t_start, offset_hours, slew_indices
        )

        assert result_ax is ax

        # Clean up
        plt.close(fig)
