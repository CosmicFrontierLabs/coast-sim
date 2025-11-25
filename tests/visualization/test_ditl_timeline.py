"""Unit tests for conops.visualization.ditl_timeline module."""

from unittest.mock import Mock

import matplotlib.pyplot as plt
import pytest

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

    def test_plot_ditl_timeline_show_saa(self, mock_ditl_with_ephem):
        """Test plot_ditl_timeline with SAA passages shown."""
        fig, ax = plot_ditl_timeline(mock_ditl_with_ephem, show_saa=True)

        assert fig is not None

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_with_safe_mode(self):
        """Test plot_ditl_timeline includes safe mode in the timeline."""
        from unittest.mock import Mock

        from conops.common import ACSMode

        # Create mock DITL with safe mode data
        mock_ditl = Mock()
        mock_ditl.config = Mock()
        mock_ditl.config.observation_categories = None
        mock_ditl.constraint = Mock()
        mock_ditl.constraint.in_eclipse = Mock(return_value=False)
        mock_ditl.acs = Mock()
        mock_ditl.acs.passrequests = Mock()
        mock_ditl.acs.passrequests.passes = []

        # Create a simple plan
        mock_plan_entry = Mock()
        mock_plan_entry.begin = 0.0
        mock_plan_entry.end = 3600.0
        mock_plan_entry.obsid = 10000
        mock_plan_entry.slewtime = 0.0
        mock_ditl.plan = [mock_plan_entry]

        # Add timeline data with safe mode
        mock_ditl.utime = [0, 1800, 3600]  # 0, 0.5, 1 hour
        mock_ditl.mode = [ACSMode.SCIENCE, ACSMode.SAFE, ACSMode.SCIENCE]

        fig, ax = plot_ditl_timeline(mock_ditl)

        assert fig is not None
        # Check that "Safe Mode" is in the y-axis labels
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "Safe Mode" in y_labels

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_empty_plan_raises_error(self):
        """Test plot_ditl_timeline raises error with empty plan."""
        from unittest.mock import Mock

        mock_ditl = Mock()
        mock_ditl.plan = []

        with pytest.raises(ValueError, match="DITL simulation has no pointings"):
            plot_ditl_timeline(mock_ditl)

    def test_plot_ditl_timeline_empty_utime_uses_default_duration(self):
        """Test plot_ditl_timeline uses default duration when utime is empty."""
        from unittest.mock import Mock

        mock_ditl = Mock()
        mock_ditl.plan = [Mock()]
        mock_ditl.plan[0].begin = 0.0
        mock_ditl.plan[0].end = 1800.0
        mock_ditl.plan[0].obsid = 10000
        mock_ditl.plan[0].slewtime = 0.0
        mock_ditl.config = Mock()
        mock_ditl.config.observation_categories = None
        mock_ditl.utime = []  # Empty utime
        mock_ditl.ra = []
        mock_ditl.dec = []
        mock_ditl.mode = []
        mock_ditl.obsid = []
        mock_ditl.panel = []
        mock_ditl.power = []
        mock_ditl.batterylevel = []
        mock_ditl.charge_state = []
        mock_ditl.power_bus = []
        mock_ditl.power_payload = []
        mock_ditl.constraint = Mock()
        mock_ditl.constraint.in_eclipse = Mock(return_value=False)
        mock_ditl.acs = Mock()
        mock_ditl.acs.passrequests = Mock()
        mock_ditl.acs.passrequests.passes = []

        fig, ax = plot_ditl_timeline(mock_ditl)

        assert fig is not None

        # Clean up
        plt.close(fig)


class TestAnnotateSlewDistances:
    """Test annotate_slew_distances function."""

    def test_annotate_slew_distances_basic(self, mock_ditl_with_ephem):
        """Test basic annotate_slew_distances functionality."""
        fig = plt.figure()
        ax = plt.axes([0.1, 0.1, 0.8, 0.8])

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
        fig = plt.figure()
        ax = plt.axes([0.1, 0.1, 0.8, 0.8])

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
        fig = plt.figure()
        ax = plt.axes([0.1, 0.1, 0.8, 0.8])

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
