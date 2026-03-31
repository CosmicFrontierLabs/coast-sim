"""Unit tests for conops.visualization.ditl_timeline module."""

from typing import Any
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from conops.visualization import (
    annotate_slew_distances,
    plot_ditl_timeline,
    plot_ditl_timeline_plotly,
)


@pytest.fixture
def mock_ditl() -> Mock:
    """Create a mock DITL object for testing."""
    from datetime import datetime, timezone
    from unittest.mock import Mock

    from conops.common import ACSMode
    from conops.ditl.telemetry import Housekeeping, HousekeepingList, Telemetry

    ditl = Mock()

    # Time data
    base_time = 1514764800.0  # 2018-01-01
    ditl.utime = [base_time + i * 60 for i in range(100)]
    ditl.step_size = 60

    # Pointing data
    ditl.ra = np.random.uniform(0, 360, 100)
    ditl.dec = np.random.uniform(-60, 60, 100)
    ditl.mode = [1] * 100  # Mock ACS mode data

    # Plan with scheduled observations
    ditl.plan = []
    for i in range(10):
        ppt = Mock()
        ppt.ra = np.random.uniform(0, 360)
        ppt.dec = np.random.uniform(-60, 60)
        ppt.obsid = 10000 + i
        ppt.begin = base_time + i * 600  # Spread observations over time
        ppt.end = ppt.begin + 300  # 5 minutes each
        ppt.slewtime = 60  # 1 minute slew
        ditl.plan.append(ppt)

    # Create minimal config with required fields
    config = Mock()
    config.name = "Test Config"
    config.observation_categories = None  # Use defaults

    # Mock battery with max_depth_of_discharge
    config.battery = Mock()
    config.battery.max_depth_of_discharge = 0.2

    # Mock recorder configuration
    config.recorder = Mock()
    config.recorder.capacity_gb = 2.0
    config.recorder.yellow_threshold = 0.8
    config.recorder.red_threshold = 0.95

    ditl.config = config

    # Create housekeeping records (simplified for 100 time steps)
    housekeeping_records = []
    timestamps = [
        datetime.fromtimestamp(base_time + i * 60, tz=timezone.utc) for i in range(100)
    ]

    for i, ts in enumerate(timestamps):
        hk = Housekeeping(
            timestamp=ts,
            ra=ditl.ra[i] if i < len(ditl.ra) else 0.0,
            dec=ditl.dec[i] if i < len(ditl.dec) else 0.0,
            acs_mode=ACSMode.SCIENCE,
            obsid=10000 + (i % 10),
            panel_illumination=0.8,
            power_usage=150.0,
            battery_level=0.8,
            charge_state=1,
            power_bus=50.0,
            power_payload=100.0,
            recorder_volume_gb=0.1 * i,
            recorder_fill_fraction=0.01 * i,
            recorder_alert=0,
            in_eclipse=False,
        )
        housekeeping_records.append(hk)

    # Create telemetry container
    telemetry = Telemetry(
        housekeeping=HousekeepingList(housekeeping_records),
        data_generated_gb=[0.1 * i for i in range(100)],
        data_downlinked_gb=[0.05 * i for i in range(100)],
    )
    ditl.telemetry = telemetry

    # Mock constraint
    ditl.constraint = Mock()

    # Mock ephemeris
    ephem = Mock()
    ephem.index = Mock(return_value=0)

    # Mock sun position
    sun_mock = Mock()
    sun_mock.ra = Mock(deg=90.0)
    sun_mock.dec = Mock(deg=23.5)
    ephem.sun = [sun_mock]

    # Mock moon position
    moon_mock = Mock()
    moon_mock.ra = Mock(deg=180.0)
    moon_mock.dec = Mock(deg=10.0)
    ephem.moon = [moon_mock]

    # Mock earth position
    earth_mock = Mock()
    earth_mock.ra = Mock(deg=270.0)
    earth_mock.dec = Mock(deg=-15.0)
    ephem.earth = [earth_mock]
    ephem.earth_radius_deg = [10.0]  # Mock earth angular radius

    # New direct array access (rust-ephem 0.3.0+)
    ephem.sun_ra_deg = [90.0]
    ephem.sun_dec_deg = [23.5]
    ephem.moon_ra_deg = [180.0]
    ephem.moon_dec_deg = [10.0]
    ephem.earth_ra_deg = [270.0]
    ephem.earth_dec_deg = [-15.0]

    ditl.constraint.ephem = ephem

    # Mock constraint methods
    ditl.constraint.in_sun = Mock(return_value=False)
    ditl.constraint.in_moon = Mock(return_value=False)
    ditl.constraint.in_earth = Mock(return_value=False)
    ditl.constraint.in_anti_sun = Mock(return_value=False)
    ditl.constraint.in_panel = Mock(return_value=False)

    # Mock config with constraint objects
    constraint_config = Mock()

    # Helper function to create a mock in_constraint_batch that returns correctly sized arrays
    def make_constraint_batch_mock() -> Mock:
        def in_constraint_batch(
            ephemeris: Any,
            target_ras: np.ndarray,
            target_decs: np.ndarray,
            times: np.ndarray,
        ) -> Any:
            n_points = len(target_ras)
            n_times = len(times)
            return np.zeros((n_points, n_times), dtype=bool)

        return Mock(side_effect=in_constraint_batch)

    # Mock constraint objects with in_constraint_batch method
    sun_constraint = Mock()
    sun_constraint.in_constraint_batch = make_constraint_batch_mock()
    constraint_config.sun_constraint = sun_constraint

    moon_constraint = Mock()
    moon_constraint.in_constraint_batch = make_constraint_batch_mock()
    constraint_config.moon_constraint = moon_constraint

    earth_constraint = Mock()
    earth_constraint.in_constraint_batch = make_constraint_batch_mock()
    constraint_config.earth_constraint = earth_constraint

    anti_sun_constraint = Mock()
    anti_sun_constraint.in_constraint_batch = make_constraint_batch_mock()
    constraint_config.anti_sun_constraint = anti_sun_constraint

    panel_constraint = Mock()
    panel_constraint.in_constraint_batch = make_constraint_batch_mock()
    constraint_config.panel_constraint = panel_constraint

    config.constraint = constraint_config
    ditl.config = config

    return ditl


class TestPlotDitlTimeline:
    """Test plot_ditl_timeline function."""

    def test_plot_ditl_timeline_returns_figure_and_axes(
        self, mock_ditl_with_ephem: Mock
    ) -> None:
        """Test that plot_ditl_timeline returns a figure and axes."""
        fig, ax = plot_ditl_timeline(mock_ditl_with_ephem)

        assert fig is not None
        assert ax is not None

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_custom_figsize(
        self, mock_ditl_with_ephem: Mock
    ) -> None:
        """Test plot_ditl_timeline with custom figsize."""
        figsize = (15, 8)
        fig, ax = plot_ditl_timeline(mock_ditl_with_ephem, figsize=figsize)

        assert fig.get_size_inches()[0] == figsize[0]
        assert fig.get_size_inches()[1] == figsize[1]

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_offset_hours(self, mock_ditl_with_ephem: Mock) -> None:
        """Test plot_ditl_timeline with time offset."""
        offset_hours = 5.0
        fig, ax = plot_ditl_timeline(mock_ditl_with_ephem, offset_hours=offset_hours)

        assert fig is not None

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_hide_orbit_numbers(
        self, mock_ditl_with_ephem: Mock
    ) -> None:
        """Test plot_ditl_timeline with orbit numbers hidden."""
        fig, ax = plot_ditl_timeline(mock_ditl_with_ephem, show_orbit_numbers=False)

        assert fig is not None

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_hide_saa(self, mock_ditl_with_ephem: Mock) -> None:
        """Test plot_ditl_timeline with SAA passages hidden."""
        fig, ax = plot_ditl_timeline(mock_ditl_with_ephem, show_saa=False)

        assert fig is not None

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_custom_orbit_period(
        self, mock_ditl_with_ephem: Mock
    ) -> None:
        """Test plot_ditl_timeline with custom orbit period."""
        orbit_period = 5000.0
        fig, ax = plot_ditl_timeline(mock_ditl_with_ephem, orbit_period=orbit_period)

        assert fig is not None

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_with_observation_categories(
        self, mock_ditl_with_ephem: Mock
    ) -> None:
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

    def test_plot_ditl_timeline_show_saa(self, mock_ditl_with_ephem: Mock) -> None:
        """Test plot_ditl_timeline with SAA passages shown."""
        fig, ax = plot_ditl_timeline(mock_ditl_with_ephem, show_saa=True)

        assert fig is not None

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_with_safe_mode(self) -> None:
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

        # Add timeline data with safe mode using telemetry
        from datetime import datetime, timedelta, timezone

        from conops.ditl.telemetry import Housekeeping, HousekeepingList, Telemetry

        base_time = datetime.fromtimestamp(0, tz=timezone.utc)
        housekeeping_records = [
            Housekeeping(timestamp=base_time, acs_mode=ACSMode.SCIENCE),
            Housekeeping(
                timestamp=base_time + timedelta(seconds=1800), acs_mode=ACSMode.SAFE
            ),
            Housekeeping(
                timestamp=base_time + timedelta(seconds=3600), acs_mode=ACSMode.SCIENCE
            ),
        ]
        mock_ditl.telemetry = Telemetry(
            housekeeping=HousekeepingList(housekeeping_records)
        )

        fig, ax = plot_ditl_timeline(mock_ditl)

        assert fig is not None
        # Check that "Safe Mode" is in the y-axis labels
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "Safe Mode" in y_labels

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_empty_plan_raises_error(self) -> None:
        """Test plot_ditl_timeline raises error with empty plan."""
        from unittest.mock import Mock

        mock_ditl = Mock()
        mock_ditl.plan = []

        with pytest.raises(ValueError, match="DITL simulation has no pointings"):
            plot_ditl_timeline(mock_ditl)

    def test_plot_ditl_timeline_empty_utime_uses_default_duration(self) -> None:
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
        # Add empty telemetry
        from conops.ditl.telemetry import HousekeepingList, Telemetry

        mock_ditl.telemetry = Telemetry(housekeeping=HousekeepingList([]))

        mock_ditl.constraint = Mock()
        mock_ditl.constraint.in_eclipse = Mock(return_value=False)
        mock_ditl.acs = Mock()
        mock_ditl.acs.passrequests = Mock()
        mock_ditl.acs.passrequests.passes = []

        fig, ax = plot_ditl_timeline(mock_ditl)

        assert fig is not None

        # Clean up
        plt.close(fig)

    def test_plot_ditl_timeline_plotly_returns_figure(
        self, mock_ditl_with_ephem: Mock
    ) -> None:
        """Test that plot_ditl_timeline_plotly returns a Plotly figure."""
        fig = plot_ditl_timeline_plotly(mock_ditl_with_ephem)

        assert fig is not None
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")

    def test_plot_ditl_timeline_plotly_custom_title(
        self, mock_ditl_with_ephem: Mock
    ) -> None:
        """Test plot_ditl_timeline_plotly with custom title."""
        title = "Custom Test Title"
        fig = plot_ditl_timeline_plotly(mock_ditl_with_ephem, title=title)

        assert fig.layout.title.text == title


class TestAnnotateSlewDistances:
    """Test annotate_slew_distances function."""

    def test_annotate_slew_distances_basic(self, mock_ditl_with_ephem: Mock) -> None:
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

    def test_annotate_slew_distances_empty_indices(
        self, mock_ditl_with_ephem: Mock
    ) -> None:
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

    def test_annotate_slew_distances_multiple_slews(
        self, mock_ditl_with_ephem: Mock
    ) -> None:
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
