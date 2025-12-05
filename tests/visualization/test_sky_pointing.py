"""Tests for sky pointing visualization module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from conops.visualization.sky_pointing import (
    SkyPointingController,
    plot_sky_pointing,
    save_sky_pointing_frames,
)


@pytest.fixture
def mock_ditl():
    """Create a mock DITL object for testing."""
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
        ditl.plan.append(ppt)

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

    ditl.constraint.ephem = ephem

    # Mock constraint methods
    ditl.constraint.in_sun = Mock(return_value=False)
    ditl.constraint.in_moon = Mock(return_value=False)
    ditl.constraint.in_earth = Mock(return_value=False)
    ditl.constraint.in_anti_sun = Mock(return_value=False)
    ditl.constraint.in_panel = Mock(return_value=False)

    # Mock config with constraint objects
    config = Mock()
    constraint_config = Mock()

    # Helper function to create a mock in_constraint_batch that returns correctly sized arrays
    def make_constraint_batch_mock():
        def in_constraint_batch(ephemeris, target_ras, target_decs, times):
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


class TestPlotSkyPointing:
    """Test plot_sky_pointing function."""

    def test_plot_sky_pointing_requires_plan(self):
        """Test that plot requires a plan."""
        ditl = Mock()
        ditl.plan = []
        ditl.utime = [1514764800.0]

        with pytest.raises(ValueError, match="no pointings"):
            plot_sky_pointing(ditl)

    def test_plot_sky_pointing_requires_utime(self):
        """Test that plot requires time data."""
        ditl = Mock()
        ditl.plan = [Mock()]
        ditl.utime = []

        with pytest.raises(ValueError, match="no time data"):
            plot_sky_pointing(ditl)

    def test_plot_sky_pointing_requires_ephemeris(self):
        """Test that plot requires ephemeris."""
        ditl = Mock()
        ditl.plan = [Mock()]
        ditl.utime = [1514764800.0]
        ditl.constraint = Mock()
        ditl.constraint.ephem = None

        with pytest.raises(ValueError, match="no ephemeris"):
            plot_sky_pointing(ditl)

    @patch("conops.visualization.sky_pointing.SkyPointingController")
    @patch("conops.visualization.sky_pointing.plt")
    def test_plot_sky_pointing_creates_figure(
        self, mock_plt, mock_controller_class, mock_ditl
    ):
        """Test that plot creates a figure."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        fig, ax, controller = plot_sky_pointing(
            mock_ditl,
            show_controls=True,
        )

        assert fig == mock_fig
        assert ax == mock_ax
        assert controller == mock_controller
        # Should call add_controls since show_controls=True
        mock_controller.add_controls.assert_called_once()

    @patch("conops.visualization.sky_pointing.plt")
    def test_plot_sky_pointing_without_controls(self, mock_plt, mock_ditl):
        """Test plot without interactive controls."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.get_legend_handles_labels.return_value = ([], [])
        mock_ax.get_yticklabels.return_value = []
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        with patch(
            "conops.visualization.sky_pointing.SkyPointingController._plot_earth_disk"
        ):
            fig, ax, controller = plot_sky_pointing(
                mock_ditl,
                show_controls=False,
            )

        assert fig == mock_fig
        assert ax == mock_ax
        assert controller is None


class TestSkyPointingController:
    """Test SkyPointingController class."""

    @patch("conops.visualization.sky_pointing.plt")
    def test_controller_init(self, mock_plt, mock_ditl):
        """Test controller initialization."""
        mock_fig = Mock()
        mock_ax = Mock()

        controller = SkyPointingController(
            ditl=mock_ditl,
            fig=mock_fig,
            ax=mock_ax,
            n_grid_points=50,
            time_step_seconds=60,
            constraint_alpha=0.3,
        )

        assert controller.ditl == mock_ditl
        assert controller.fig == mock_fig
        assert controller.ax == mock_ax
        assert controller.n_grid_points == 50
        assert controller.time_step_seconds == 60
        assert controller.constraint_alpha == 0.3
        assert controller.current_time_idx == 0
        assert controller.playing is False

    @patch("conops.visualization.sky_pointing.plt")
    def test_find_time_index(self, mock_plt, mock_ditl):
        """Test finding time index."""
        mock_fig = Mock()
        mock_ax = Mock()

        controller = SkyPointingController(
            ditl=mock_ditl,
            fig=mock_fig,
            ax=mock_ax,
        )

        # Find index closest to middle time
        middle_time = mock_ditl.utime[50]
        idx = controller._find_time_index(middle_time)
        assert idx == 50

    @patch("conops.visualization.sky_pointing.plt")
    def test_add_controls(self, mock_plt, mock_ditl):
        """Test adding control widgets."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.axes.return_value = Mock()

        controller = SkyPointingController(
            ditl=mock_ditl,
            fig=mock_fig,
            ax=mock_ax,
        )

        with (
            patch("conops.visualization.sky_pointing.Slider"),
            patch("conops.visualization.sky_pointing.Button"),
        ):
            controller.add_controls()

            assert controller.slider is not None
            assert controller.play_button is not None
            assert controller.prev_button is not None
            assert controller.next_button is not None

    @patch("conops.visualization.sky_pointing.plt")
    def test_slider_change(self, mock_plt, mock_ditl):
        """Test slider value change."""
        mock_fig = Mock()
        mock_ax = Mock()

        controller = SkyPointingController(
            ditl=mock_ditl,
            fig=mock_fig,
            ax=mock_ax,
        )

        with patch.object(controller, "update_plot") as mock_update:
            controller.on_slider_change(10)
            assert controller.current_time_idx == 10
            mock_update.assert_called_once()

    @patch("conops.visualization.sky_pointing.plt")
    def test_prev_button(self, mock_plt, mock_ditl):
        """Test previous button."""
        mock_fig = Mock()
        mock_ax = Mock()

        controller = SkyPointingController(
            ditl=mock_ditl,
            fig=mock_fig,
            ax=mock_ax,
        )
        controller.current_time_idx = 10
        controller.slider = Mock()

        with patch.object(controller, "update_plot"):
            controller.on_prev_clicked(None)
            assert controller.current_time_idx == 9
            controller.slider.set_val.assert_called_with(9)

    @patch("conops.visualization.sky_pointing.plt")
    def test_next_button(self, mock_plt, mock_ditl):
        """Test next button."""
        mock_fig = Mock()
        mock_ax = Mock()

        controller = SkyPointingController(
            ditl=mock_ditl,
            fig=mock_fig,
            ax=mock_ax,
        )
        controller.current_time_idx = 10
        controller.slider = Mock()

        with patch.object(controller, "update_plot"):
            controller.on_next_clicked(None)
            assert controller.current_time_idx == 11
            controller.slider.set_val.assert_called_with(11)

    @patch("conops.visualization.sky_pointing.plt")
    def test_start_animation(self, mock_plt, mock_ditl):
        """Test starting animation."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_timer = Mock()
        mock_fig.canvas.new_timer.return_value = mock_timer

        controller = SkyPointingController(
            ditl=mock_ditl,
            fig=mock_fig,
            ax=mock_ax,
        )
        controller.play_button = Mock()
        controller.slider = Mock()

        # Mock update_plot to prevent actual plotting
        with patch.object(controller, "update_plot"):
            controller.start_animation()

        assert controller.playing is True
        controller.play_button.label.set_text.assert_called_with("Pause")

    @patch("conops.visualization.sky_pointing.plt")
    def test_stop_animation(self, mock_plt, mock_ditl):
        """Test stopping animation."""
        mock_fig = Mock()
        mock_ax = Mock()

        controller = SkyPointingController(
            ditl=mock_ditl,
            fig=mock_fig,
            ax=mock_ax,
        )
        controller.play_button = Mock()
        controller.playing = True
        mock_timer = Mock()
        controller._timer = mock_timer

        controller.stop_animation()

        assert controller.playing is False
        controller.play_button.label.set_text.assert_called_with("Play")
        mock_timer.stop.assert_called_once()
        assert controller._timer is None


class TestSaveFrames:
    """Test save_sky_pointing_frames function."""

    @patch("conops.visualization.sky_pointing.os")
    @patch("conops.visualization.sky_pointing.plot_sky_pointing")
    @patch("conops.visualization.sky_pointing.plt")
    def test_save_frames(self, mock_plt, mock_plot, mock_os, mock_ditl):
        """Test saving frames to directory."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plot.return_value = (mock_fig, mock_ax, None)

        with patch(
            "conops.visualization.sky_pointing.SkyPointingController"
        ) as mock_controller_class:
            mock_controller = Mock()
            mock_controller_class.return_value = mock_controller

            saved_files = save_sky_pointing_frames(
                mock_ditl,
                output_dir="./test_output",
                frame_interval=10,
            )

            # Should save 10 frames (100 timesteps / 10 interval)
            assert len(saved_files) == 10
            mock_os.makedirs.assert_called_once_with("./test_output", exist_ok=True)
            assert mock_fig.savefig.call_count == 10

    @patch("conops.visualization.sky_pointing.os")
    @patch("conops.visualization.sky_pointing.plot_sky_pointing")
    @patch("conops.visualization.sky_pointing.plt")
    def test_save_frames_all(self, mock_plt, mock_plot, mock_os, mock_ditl):
        """Test saving all frames (interval=1)."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plot.return_value = (mock_fig, mock_ax, None)

        with patch(
            "conops.visualization.sky_pointing.SkyPointingController"
        ) as mock_controller_class:
            mock_controller = Mock()
            mock_controller_class.return_value = mock_controller

            saved_files = save_sky_pointing_frames(
                mock_ditl,
                output_dir="./test_output",
                frame_interval=1,
            )

            # Should save all 100 frames
            assert len(saved_files) == 100


class TestPlotElements:
    """Test individual plotting methods."""

    @patch("conops.visualization.sky_pointing.plt")
    def test_plot_scheduled_observations_empty(self, mock_plt, mock_ditl):
        """Test plotting with no scheduled observations."""
        mock_ditl.plan = []
        mock_fig = Mock()
        mock_ax = Mock()

        controller = SkyPointingController(
            ditl=mock_ditl,
            fig=mock_fig,
            ax=mock_ax,
        )

        # Should not raise error with empty plan
        controller._plot_scheduled_observations()
        mock_ax.scatter.assert_not_called()

    @patch("conops.visualization.sky_pointing.plt")
    def test_plot_current_pointing(self, mock_plt, mock_ditl):
        """Test plotting current pointing."""
        mock_fig = Mock()
        mock_ax = Mock()

        controller = SkyPointingController(
            ditl=mock_ditl,
            fig=mock_fig,
            ax=mock_ax,
        )

        controller._plot_current_pointing(45.0, 30.0, 1)

        # Should call plot to draw the pointing marker
        assert mock_ax.plot.called
        # Should add a circle patch
        assert mock_ax.add_patch.called
