"""Tests for Plotly sky-pointing and globe-pointing visualizations.

Covers trace-count invariants for the base layout and for each optional
constraint type (anti-sun, panel) as well as validation of min/max angle
polygon rendering.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import plotly.graph_objects as go
import pytest

from conops.visualization import plot_sky_pointing_globe, plot_sky_pointing_plotly

_BASE_TIME = 1514764800.0  # 2018-01-01 00:00:00 UTC

# ---------------------------------------------------------------------------
# Trace-count constants (no star trackers)
# ---------------------------------------------------------------------------
# Base: observations + sun(min,max,marker) + moon(min,max,marker)
#       + earth(min,max,disk) + pointing  = 11
_N_BASE = 11
_N_ANTI_SUN = 3  # min excl, max excl, anti-sun marker
_N_PANEL = 2  # panel min excl, panel max excl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mode(name: str = "SCIENCE") -> Mock:
    m = Mock()
    m.name = name
    return m


@pytest.fixture
def mock_ephem() -> Mock:
    ephem = Mock()
    ephem.index = Mock(return_value=0)
    ephem.sun_ra_deg = [90.0]
    ephem.sun_dec_deg = [23.5]
    ephem.moon_ra_deg = [180.0]
    ephem.moon_dec_deg = [10.0]
    ephem.earth_ra_deg = [270.0]
    ephem.earth_dec_deg = [-15.0]
    ephem.earth_radius_deg = [10.0]
    return ephem


@pytest.fixture
def mock_ditl(mock_ephem: Mock) -> Mock:
    """Minimal DITL mock — no optional constraints, no star trackers."""
    ditl = Mock()

    ditl.utime = [_BASE_TIME + i * 60.0 for i in range(10)]
    ditl.ra = np.linspace(0.0, 90.0, 10)
    ditl.dec = np.linspace(-30.0, 30.0, 10)
    ditl.mode = [_mode()] * 10
    ditl.telemetry = None  # suppresses star-tracker status lookup

    plan_entry = Mock()
    plan_entry.ra = 45.0
    plan_entry.dec = 30.0
    plan_entry.obsid = 10001
    ditl.plan = [plan_entry]

    constraint = Mock()
    constraint.ephem = mock_ephem
    ditl.constraint = constraint

    config = Mock()

    ccon = Mock()
    ccon.orbit_constraint = None
    ccon.anti_sun_constraint = None
    ccon.panel_constraint = None
    ccon.ignore_roll = False

    sun_c = Mock()
    sun_c.min_angle = 45.0
    sun_c.max_angle = None
    ccon.sun_constraint = sun_c

    moon_c = Mock()
    moon_c.min_angle = 20.0
    moon_c.max_angle = None
    ccon.moon_constraint = moon_c

    earth_c = Mock()
    earth_c.min_angle = 30.0
    earth_c.max_angle = None
    ccon.earth_constraint = earth_c

    config.constraint = ccon
    config.observation_categories = None
    config.spacecraft_bus = None
    ditl.config = config

    return ditl


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


class TestPlotSkyPointingPlotlyValidation:
    def test_raises_on_empty_plan(self) -> None:
        ditl = Mock()
        ditl.plan = []
        ditl.utime = [_BASE_TIME]
        with pytest.raises(ValueError, match="no pointings"):
            plot_sky_pointing_plotly(ditl)

    def test_raises_on_empty_utime(self) -> None:
        ditl = Mock()
        entry = Mock()
        entry.ra, entry.dec, entry.obsid = 0.0, 0.0, 1
        ditl.plan = [entry]
        ditl.utime = []
        with pytest.raises(ValueError, match="no time data"):
            plot_sky_pointing_plotly(ditl)

    def test_raises_on_no_ephem(self) -> None:
        ditl = Mock()
        entry = Mock()
        entry.ra, entry.dec, entry.obsid = 0.0, 0.0, 1
        ditl.plan = [entry]
        ditl.utime = [_BASE_TIME]
        ditl.constraint = Mock()
        ditl.constraint.ephem = None
        with pytest.raises(ValueError, match="no ephemeris"):
            plot_sky_pointing_plotly(ditl)


class TestPlotSkyPointingGlobeValidation:
    def test_raises_on_empty_plan(self) -> None:
        ditl = Mock()
        ditl.plan = []
        ditl.utime = [_BASE_TIME]
        with pytest.raises(ValueError, match="no pointings"):
            plot_sky_pointing_globe(ditl)

    def test_raises_on_no_ephem(self) -> None:
        ditl = Mock()
        entry = Mock()
        entry.ra, entry.dec, entry.obsid = 0.0, 0.0, 1
        ditl.plan = [entry]
        ditl.utime = [_BASE_TIME]
        ditl.constraint = Mock()
        ditl.constraint.ephem = None
        with pytest.raises(ValueError, match="no ephemeris"):
            plot_sky_pointing_globe(ditl)


# ---------------------------------------------------------------------------
# Trace-count tests — sky pointing (Plotly flat map)
# ---------------------------------------------------------------------------


class TestPlotSkyPointingPlotlyTraceCounts:
    def test_returns_go_figure(self, mock_ditl: Mock) -> None:
        fig = plot_sky_pointing_plotly(mock_ditl, n_frames=1)
        assert isinstance(fig, go.Figure)

    def test_base_trace_count(self, mock_ditl: Mock) -> None:
        fig = plot_sky_pointing_plotly(mock_ditl, n_frames=1)
        assert len(fig.data) == _N_BASE

    def test_anti_sun_constraint_adds_three_traces(self, mock_ditl: Mock) -> None:
        anti_sun = Mock()
        anti_sun.min_angle = 10.0
        anti_sun.max_angle = 120.0
        mock_ditl.config.constraint.anti_sun_constraint = anti_sun

        fig = plot_sky_pointing_plotly(mock_ditl, n_frames=1)
        assert len(fig.data) == _N_BASE + _N_ANTI_SUN

    def test_panel_constraint_adds_two_traces(self, mock_ditl: Mock) -> None:
        panel = Mock()
        panel.min_angle = 30.0
        panel.max_angle = 150.0
        mock_ditl.config.constraint.panel_constraint = panel

        with patch("conops.visualization.plotly.sky_pointing.rust_ephem") as mock_rust:
            mock_eclipse = Mock()
            mock_eclipse.in_constraint.return_value = False
            mock_rust.EclipseConstraint.return_value = mock_eclipse

            fig = plot_sky_pointing_plotly(mock_ditl, n_frames=1)

        assert len(fig.data) == _N_BASE + _N_PANEL

    def test_anti_sun_and_panel_combined(self, mock_ditl: Mock) -> None:
        anti_sun = Mock()
        anti_sun.min_angle = 10.0
        anti_sun.max_angle = 130.0
        mock_ditl.config.constraint.anti_sun_constraint = anti_sun

        panel = Mock()
        panel.min_angle = 30.0
        panel.max_angle = 150.0
        mock_ditl.config.constraint.panel_constraint = panel

        with patch("conops.visualization.plotly.sky_pointing.rust_ephem") as mock_rust:
            mock_eclipse = Mock()
            mock_eclipse.in_constraint.return_value = False
            mock_rust.EclipseConstraint.return_value = mock_eclipse

            fig = plot_sky_pointing_plotly(mock_ditl, n_frames=1)

        assert len(fig.data) == _N_BASE + _N_ANTI_SUN + _N_PANEL


# ---------------------------------------------------------------------------
# Trace-count tests — globe pointing
# ---------------------------------------------------------------------------


class TestPlotSkyPointingGlobeTraceCounts:
    def test_returns_go_figure(self, mock_ditl: Mock) -> None:
        fig = plot_sky_pointing_globe(mock_ditl, n_frames=1)
        assert isinstance(fig, go.Figure)

    def test_base_trace_count(self, mock_ditl: Mock) -> None:
        fig = plot_sky_pointing_globe(mock_ditl, n_frames=1)
        assert len(fig.data) == _N_BASE

    def test_anti_sun_constraint_adds_three_traces(self, mock_ditl: Mock) -> None:
        anti_sun = Mock()
        anti_sun.min_angle = 10.0
        anti_sun.max_angle = 120.0
        mock_ditl.config.constraint.anti_sun_constraint = anti_sun

        fig = plot_sky_pointing_globe(mock_ditl, n_frames=1)
        assert len(fig.data) == _N_BASE + _N_ANTI_SUN

    def test_panel_constraint_adds_two_traces(self, mock_ditl: Mock) -> None:
        panel = Mock()
        panel.min_angle = 30.0
        panel.max_angle = 150.0
        mock_ditl.config.constraint.panel_constraint = panel

        with patch(
            "conops.visualization.plotly.globe_pointing.rust_ephem"
        ) as mock_rust:
            mock_eclipse = Mock()
            mock_eclipse.in_constraint.return_value = False
            mock_rust.EclipseConstraint.return_value = mock_eclipse

            fig = plot_sky_pointing_globe(mock_ditl, n_frames=1)

        assert len(fig.data) == _N_BASE + _N_PANEL


# ---------------------------------------------------------------------------
# Min / max angle polygon rendering
# ---------------------------------------------------------------------------


class TestConstraintAngleRendering:
    """Verify that max_angle produces a visible exclusion polygon and that
    None max_angle leaves the trace data empty."""

    def test_sun_max_angle_none_gives_empty_polygon(self, mock_ditl: Mock) -> None:
        # sun_c.max_angle = None is already set in the fixture
        fig = plot_sky_pointing_globe(mock_ditl, n_frames=1)
        # trace index 2 is "Sun max exclusion" (observations=0, sun_min=1, sun_max=2)
        sun_max_trace = fig.data[2]
        assert sun_max_trace.name == "Sun max exclusion"
        assert not sun_max_trace.showlegend
        # Polygon should be empty when no max angle is configured
        assert sun_max_trace.lon is None or len(sun_max_trace.lon) == 0

    def test_sun_max_angle_set_gives_non_empty_polygon(self, mock_ditl: Mock) -> None:
        mock_ditl.config.constraint.sun_constraint.max_angle = 150.0

        fig = plot_sky_pointing_globe(mock_ditl, n_frames=1)
        sun_max_trace = fig.data[2]
        assert sun_max_trace.name == "Sun max exclusion"
        # showlegend should be True when max_angle is configured
        assert sun_max_trace.showlegend
        # Polygon vertices should be present (max_r = 180 - 150 = 30°)
        assert sun_max_trace.lon is not None and len(sun_max_trace.lon) > 0

    def test_anti_sun_max_excl_present_when_max_angle_set(
        self, mock_ditl: Mock
    ) -> None:
        anti_sun = Mock()
        anti_sun.min_angle = 0.0  # no min excl
        anti_sun.max_angle = 120.0  # anti-sun max → exclusion around anti-sun
        mock_ditl.config.constraint.anti_sun_constraint = anti_sun

        fig = plot_sky_pointing_globe(mock_ditl, n_frames=1)
        names = [t.name for t in fig.data]
        assert "Anti-Sun exclusion" in names

        # The Anti-Sun max excl polygon should carry data (max_r = 180 - 120 = 60°)
        idx = names.index("Anti-Sun exclusion")
        assert fig.data[idx].lon is not None and len(fig.data[idx].lon) > 0
