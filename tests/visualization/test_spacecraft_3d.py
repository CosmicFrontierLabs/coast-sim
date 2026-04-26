"""Unit tests for conops.visualization.plotly.spacecraft_3d.

Organised by concern:
  TestGeometryHelpers       — _unit, _perp_pair, _face_placement
  TestPlotSpacecraft3dBasic — return type, layout, title, dark theme
  TestTraceComposition      — Mesh3d / Scatter3d / Cone composition by component
  TestFlagParameters        — show_normals, show_axes, bus_half_dims, title
  TestTelescope             — Telescope instrument rendering
  TestMultiplePanels        — trace count scales with panel count
  TestMultipleRadiators     — trace count scales with radiator count
  TestMultipleStarTrackers  — trace count scales with star-tracker count
  TestRealConfig            — smoke test against examples/example_config.json
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import pytest

from conops.config import MissionConfig
from conops.config.instrument import Payload, Telescope, TelescopeConfig, TelescopeType
from conops.config.radiator import (
    Radiator,
    RadiatorConfiguration,
    RadiatorOrientation,
)
from conops.config.solar_panel import (
    SolarPanel,
    SolarPanelSet,
    create_solar_panel_vector,
)
from conops.config.spacecraft_bus import SpacecraftBus
from conops.config.star_tracker import (
    StarTracker,
    StarTrackerConfiguration,
    StarTrackerOrientation,
)
from conops.visualization import plot_spacecraft_3d
from conops.visualization.plotly.spacecraft_3d import (
    _face_placement,
    _perp_pair,
    _unit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZERO = np.zeros(3)
_BUS_HD = (1.0, 0.55, 0.55)


def _trace_types(fig: go.Figure) -> list[str]:
    return [type(t).__name__ for t in fig.data]


def _trace_names(fig: go.Figure) -> list[str]:
    return [t.name or "" for t in fig.data]


def _count_type(fig: go.Figure, cls: type) -> int:
    return sum(1 for t in fig.data if isinstance(t, cls))


def _default_fig() -> go.Figure:
    return plot_spacecraft_3d(MissionConfig())


# ---------------------------------------------------------------------------
# TestGeometryHelpers
# ---------------------------------------------------------------------------


class TestGeometryHelpers:
    def test_unit_unit_vector_unchanged(self) -> None:
        v = np.array([1.0, 0.0, 0.0])
        result = _unit(v)
        np.testing.assert_allclose(result, v)

    def test_unit_normalizes_arbitrary_vector(self) -> None:
        v = np.array([3.0, 4.0, 0.0])
        result = _unit(v)
        assert math.isclose(np.linalg.norm(result), 1.0, abs_tol=1e-10)

    def test_unit_zero_vector_returns_zero(self) -> None:
        result = _unit(np.zeros(3))
        assert np.linalg.norm(result) == 0.0

    def test_unit_accepts_tuple(self) -> None:
        result = _unit((0.0, 1.0, 0.0))
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0])

    def test_perp_pair_both_perpendicular_to_input(self) -> None:
        for v in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]:
            axis = np.array(v, dtype=float)
            u, w = _perp_pair(axis)
            assert abs(float(np.dot(_unit(axis), u))) < 1e-9
            assert abs(float(np.dot(_unit(axis), w))) < 1e-9

    def test_perp_pair_are_unit_vectors(self) -> None:
        u, w = _perp_pair(np.array([0.5, 0.5, 0.707]))
        assert math.isclose(np.linalg.norm(u), 1.0, abs_tol=1e-9)
        assert math.isclose(np.linalg.norm(w), 1.0, abs_tol=1e-9)

    def test_perp_pair_mutually_perpendicular(self) -> None:
        u, w = _perp_pair(np.array([0.0, 0.0, 1.0]))
        assert abs(float(np.dot(u, w))) < 1e-9

    @pytest.mark.parametrize(
        "normal,expected_axis,expected_sign",
        [
            ([1.0, 0.0, 0.0], 0, +1),  # +X face
            ([-1.0, 0.0, 0.0], 0, -1),  # -X face
            ([0.0, 1.0, 0.0], 1, +1),  # +Y face
            ([0.0, -1.0, 0.0], 1, -1),  # -Y face
            ([0.0, 0.0, 1.0], 2, +1),  # +Z face
            ([0.0, 0.0, -1.0], 2, -1),  # -Z face
        ],
    )
    def test_face_placement_selects_correct_face(
        self, normal: list[float], expected_axis: int, expected_sign: int
    ) -> None:
        n = np.array(normal, dtype=float)
        center, u, v = _face_placement(n, _BUS_HD)
        # The centre should be on the correct side
        assert np.sign(center[expected_axis]) == expected_sign
        # The two span vectors should be perpendicular to the normal
        assert abs(float(np.dot(n, u))) < 1e-9
        assert abs(float(np.dot(n, v))) < 1e-9

    def test_face_placement_center_outside_bus(self) -> None:
        """Panel center must lie outside (or on) the bus face."""
        n = np.array([0.0, 1.0, 0.0])
        center, _, _ = _face_placement(n, _BUS_HD, gap=0.03)
        # Y component should be at least hy (0.55) away from origin
        assert center[1] >= _BUS_HD[1]

    def test_face_placement_off_axis_normal_picks_best_face(self) -> None:
        """A slightly-tilted +Y normal should still land on the +Y face."""
        n = np.array([0.1, 0.9, 0.1])
        center, _, _ = _face_placement(_unit(n), _BUS_HD)
        assert center[1] > 0  # lands on +Y side


# ---------------------------------------------------------------------------
# TestPlotSpacecraft3dBasic
# ---------------------------------------------------------------------------


class TestPlotSpacecraft3dBasic:
    def test_returns_go_figure(self) -> None:
        assert isinstance(_default_fig(), go.Figure)

    def test_has_traces(self) -> None:
        assert len(_default_fig().data) > 0

    def test_contains_mesh3d_traces(self) -> None:
        assert _count_type(_default_fig(), go.Mesh3d) > 0

    def test_contains_scatter3d_traces(self) -> None:
        assert _count_type(_default_fig(), go.Scatter3d) > 0

    def test_contains_cone_traces(self) -> None:
        assert _count_type(_default_fig(), go.Cone) > 0

    def test_default_title_contains_bus_name(self) -> None:
        config = MissionConfig()
        fig = plot_spacecraft_3d(config)
        assert config.spacecraft_bus.name in fig.layout.title.text

    def test_custom_title_is_used(self) -> None:
        fig = plot_spacecraft_3d(MissionConfig(), title="My Spacecraft")
        assert fig.layout.title.text == "My Spacecraft"

    def test_dark_paper_background(self) -> None:
        fig = _default_fig()
        assert fig.layout.paper_bgcolor.startswith("rgb(")
        # All three channels should be very dark (< 50)
        channels = [int(c) for c in fig.layout.paper_bgcolor[4:-1].split(",")]
        assert all(c < 50 for c in channels)

    def test_scene_aspect_is_cube(self) -> None:
        assert _default_fig().layout.scene.aspectmode == "cube"

    def test_scene_has_labelled_axes(self) -> None:
        scene = _default_fig().layout.scene
        # Plotly wraps title in a Title object; coerce to str for portability
        assert "X" in str(scene.xaxis.title)
        assert "Y" in str(scene.yaxis.title)
        assert "Z" in str(scene.zaxis.title)

    def test_figure_height_set(self) -> None:
        assert _default_fig().layout.height > 0

    def test_legend_is_configured(self) -> None:
        legend = _default_fig().layout.legend
        assert legend is not None


# ---------------------------------------------------------------------------
# TestTraceComposition  — what the default config produces
# ---------------------------------------------------------------------------


class TestTraceComposition:
    """Default MissionConfig gives: 1 bus + 1 panel + 1 frame + 2 panel arrows
    + 1 radiator + 2 radiator arrows + 1 ST + 1 ST aperture + 2 ST arrows
    + 3 axes × 3 = 21 traces total."""

    _EXPECTED_DEFAULT = 21

    def test_default_config_trace_count(self) -> None:
        assert len(_default_fig().data) == self._EXPECTED_DEFAULT

    def test_bus_mesh_present(self) -> None:
        names = _trace_names(_default_fig())
        # The bus name comes from spacecraft_bus.name
        assert any(MissionConfig().spacecraft_bus.name in n for n in names)

    def test_bus_is_mesh3d(self) -> None:
        fig = _default_fig()
        bus_name = MissionConfig().spacecraft_bus.name
        bus_traces = [t for t in fig.data if t.name == bus_name]
        assert len(bus_traces) == 1
        assert isinstance(bus_traces[0], go.Mesh3d)

    def test_solar_panel_mesh_present(self) -> None:
        panel_name = MissionConfig().solar_panel.panels[0].name
        names = _trace_names(_default_fig())
        assert any(panel_name in n for n in names)

    def test_solar_frame_mesh_present(self) -> None:
        names = _trace_names(_default_fig())
        assert any("Solar panel frame" in n for n in names)

    def test_axes_scatter3d_lines_present(self) -> None:
        # Three axis shafts: +X, +Y, +Z
        axis_lines = [
            t
            for t in _default_fig().data
            if isinstance(t, go.Scatter3d)
            and t.name in ("+X (boresight)", "+Y (up)", "+Z")
        ]
        assert len(axis_lines) == 3

    def test_axes_cone_traces_present(self) -> None:
        axis_cones = [
            t
            for t in _default_fig().data
            if isinstance(t, go.Cone) and t.name in ("+X (boresight)", "+Y (up)", "+Z")
        ]
        assert len(axis_cones) == 3

    def test_normal_arrows_present_by_default(self) -> None:
        # At least panel normal + radiator normal + ST boresight
        arrow_traces = [
            t
            for t in _default_fig().data
            if isinstance(t, (go.Scatter3d, go.Cone)) and "normal" in (t.name or "")
        ]
        assert len(arrow_traces) >= 2


# ---------------------------------------------------------------------------
# TestFlagParameters
# ---------------------------------------------------------------------------


class TestFlagParameters:
    def test_show_normals_false_removes_normal_arrows(self) -> None:
        fig = plot_spacecraft_3d(MissionConfig(), show_normals=False)
        names = _trace_names(fig)
        assert not any("normal" in n.lower() for n in names)
        # Component boresight arrows (e.g. "ST-1 boresight") should be gone.
        # The axis label "+X (boresight)" is still present when show_axes=True,
        # so we check the trace legendgroup instead of the name.
        component_boresight = [
            t
            for t in fig.data
            if "boresight" in (t.name or "").lower()
            and getattr(t, "legendgroup", None) not in ("axes", None)
        ]
        assert len(component_boresight) == 0

    def test_show_normals_false_fewer_traces(self) -> None:
        n_with = len(_default_fig().data)
        n_without = len(plot_spacecraft_3d(MissionConfig(), show_normals=False).data)
        assert n_without < n_with

    def test_show_axes_false_removes_axis_traces(self) -> None:
        fig = plot_spacecraft_3d(MissionConfig(), show_axes=False)
        names = _trace_names(fig)
        assert not any(n in names for n in ("+X (boresight)", "+Y (up)", "+Z"))

    def test_show_axes_false_removes_nine_traces(self) -> None:
        # Default has 9 axis traces (3 line + 3 cone + 3 text labels)
        n_with = len(_default_fig().data)
        n_without = len(plot_spacecraft_3d(MissionConfig(), show_axes=False).data)
        assert n_with - n_without == 9

    def test_both_flags_false_minimum_traces(self) -> None:
        fig = plot_spacecraft_3d(MissionConfig(), show_normals=False, show_axes=False)
        # Bus + solar panel + frame + radiator + 2 ST meshes = 6
        assert len(fig.data) == 6

    def test_custom_bus_half_dims_affects_axis_range(self) -> None:
        fig_small = plot_spacecraft_3d(MissionConfig(), bus_half_dims=(0.5, 0.3, 0.3))
        fig_large = plot_spacecraft_3d(MissionConfig(), bus_half_dims=(2.0, 1.0, 1.0))
        x_range_small = fig_small.layout.scene.xaxis.range
        x_range_large = fig_large.layout.scene.xaxis.range
        assert x_range_large[1] > x_range_small[1]

    def test_custom_bus_half_dims_does_not_break_render(self) -> None:
        for dims in [(0.3, 0.2, 0.2), (1.0, 0.55, 0.55), (3.0, 1.5, 1.5)]:
            fig = plot_spacecraft_3d(MissionConfig(), bus_half_dims=dims)
            assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# TestTelescope
# ---------------------------------------------------------------------------


class TestTelescope:
    @pytest.fixture
    def telescope_config(self) -> MissionConfig:
        return MissionConfig(
            payload=Payload(
                instruments=[
                    Telescope(
                        name="PrimTel",
                        boresight=(1.0, 0.0, 0.0),
                        optics=TelescopeConfig(
                            aperture_m=0.5,
                            tube_length_m=1.4,
                            telescope_type=TelescopeType.RITCHEY_CHRETIEN,
                        ),
                    )
                ]
            )
        )

    def test_telescope_tube_present(self, telescope_config: MissionConfig) -> None:
        fig = plot_spacecraft_3d(telescope_config)
        names = _trace_names(fig)
        assert "PrimTel" in names

    def test_telescope_adds_aperture_ring(
        self, telescope_config: MissionConfig
    ) -> None:
        names = _trace_names(plot_spacecraft_3d(telescope_config))
        assert "Aperture ring" in names

    def test_telescope_adds_aperture_opening(
        self, telescope_config: MissionConfig
    ) -> None:
        names = _trace_names(plot_spacecraft_3d(telescope_config))
        assert "Aperture opening" in names

    def test_telescope_adds_baffle_ring(self, telescope_config: MissionConfig) -> None:
        names = _trace_names(plot_spacecraft_3d(telescope_config))
        assert "Baffle ring" in names

    def test_telescope_adds_secondary(self, telescope_config: MissionConfig) -> None:
        names = _trace_names(plot_spacecraft_3d(telescope_config))
        assert "Secondary" in names

    def test_telescope_boresight_arrow_present(
        self, telescope_config: MissionConfig
    ) -> None:
        names = _trace_names(plot_spacecraft_3d(telescope_config))
        assert any("boresight" in n.lower() for n in names)

    def test_telescope_adds_seven_traces(self, telescope_config: MissionConfig) -> None:
        # tube + aperture ring + aperture opening + baffle ring + secondary
        # + boresight Scatter3d + boresight Cone = 7 traces
        default_count = len(_default_fig().data)
        scope_count = len(plot_spacecraft_3d(telescope_config).data)
        assert scope_count - default_count == 7

    def test_non_boresight_telescope_still_renders(self) -> None:
        config = MissionConfig(
            payload=Payload(
                instruments=[
                    Telescope(
                        name="SideScope",
                        boresight=(0.0, 1.0, 0.0),
                        optics=TelescopeConfig(aperture_m=0.3),
                    )
                ]
            )
        )
        fig = plot_spacecraft_3d(config)
        assert any("SideScope" in (t.name or "") for t in fig.data)

    def test_telescope_tube_is_mesh3d(self, telescope_config: MissionConfig) -> None:
        fig = plot_spacecraft_3d(telescope_config)
        tube_traces = [t for t in fig.data if t.name == "PrimTel"]
        assert len(tube_traces) == 1
        assert isinstance(tube_traces[0], go.Mesh3d)

    def test_show_normals_false_removes_telescope_boresight(
        self, telescope_config: MissionConfig
    ) -> None:
        fig = plot_spacecraft_3d(telescope_config, show_normals=False)
        component_boresight = [
            t
            for t in fig.data
            if "boresight" in (t.name or "").lower()
            and getattr(t, "legendgroup", None) not in ("axes", None)
        ]
        assert len(component_boresight) == 0


# ---------------------------------------------------------------------------
# TestMultiplePanels
# ---------------------------------------------------------------------------


class TestMultiplePanels:
    def _config_with_n_panels(self, n: int) -> MissionConfig:
        panels = [
            SolarPanel(
                name=f"Panel-{i}",
                normal=create_solar_panel_vector("sidemount", cant_z=i * (360.0 / n)),
                max_power=800.0,
            )
            for i in range(n)
        ]
        return MissionConfig(solar_panel=SolarPanelSet(panels=panels))

    def test_one_panel_base_count(self) -> None:
        fig = plot_spacecraft_3d(self._config_with_n_panels(1))
        # Only the cell Mesh3d uses the exact panel name; arrows are "Panel-0 normal"
        panel_meshes = [
            t for t in fig.data if isinstance(t, go.Mesh3d) and t.name == "Panel-0"
        ]
        assert len(panel_meshes) == 1

    def test_two_panels_adds_four_more_traces(self) -> None:
        n1 = len(plot_spacecraft_3d(self._config_with_n_panels(1)).data)
        n2 = len(plot_spacecraft_3d(self._config_with_n_panels(2)).data)
        # Each additional panel: 1 cell + 1 frame + 2 arrows = 4 traces
        assert n2 - n1 == 4

    def test_four_panels_trace_count(self) -> None:
        n1 = len(plot_spacecraft_3d(self._config_with_n_panels(1)).data)
        n4 = len(plot_spacecraft_3d(self._config_with_n_panels(4)).data)
        assert n4 - n1 == 3 * 4  # three extra panels × 4 traces each

    def test_panel_with_geometry_uses_geometry_center(self) -> None:
        from conops.config.geometry import PanelGeometry

        panel_with_geom = SolarPanel(
            name="GeomPanel",
            normal=(0.0, 1.0, 0.0),
            max_power=500.0,
            geometry=PanelGeometry(
                center_m=(0.5, 1.0, 0.0),
                u=(1.0, 0.0, 0.0),
                v=(0.0, 0.0, 1.0),
                width_m=1.5,
                height_m=0.8,
            ),
        )
        config = MissionConfig(solar_panel=SolarPanelSet(panels=[panel_with_geom]))
        fig = plot_spacecraft_3d(config)
        # Panel mesh should be roughly centred at (0.5, 1.0, 0.0)
        cell_traces = [
            t for t in fig.data if t.name == "GeomPanel" and isinstance(t, go.Mesh3d)
        ]
        assert len(cell_traces) == 1
        x_vals = list(cell_traces[0].x)
        y_vals = list(cell_traces[0].y)
        assert math.isclose(sum(x_vals) / len(x_vals), 0.5, abs_tol=0.01)
        assert math.isclose(sum(y_vals) / len(y_vals), 1.0, abs_tol=0.01)


# ---------------------------------------------------------------------------
# TestMultipleRadiators
# ---------------------------------------------------------------------------


class TestMultipleRadiators:
    def _config_with_n_radiators(self, n: int) -> MissionConfig:
        normals = [
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ]
        rads = [
            Radiator(
                name=f"Rad-{i}",
                width_m=0.8,
                height_m=0.6,
                orientation=RadiatorOrientation(normal=normals[i % len(normals)]),
            )
            for i in range(n)
        ]
        bus = SpacecraftBus(radiators=RadiatorConfiguration(radiators=rads))
        return MissionConfig(spacecraft_bus=bus)

    def test_one_radiator_base(self) -> None:
        fig = plot_spacecraft_3d(self._config_with_n_radiators(1))
        # The Mesh3d uses the exact radiator name; arrows are "Rad-0 normal"
        rad_meshes = [
            t for t in fig.data if isinstance(t, go.Mesh3d) and t.name == "Rad-0"
        ]
        assert len(rad_meshes) == 1

    def test_two_radiators_adds_three_traces(self) -> None:
        # Each radiator: 1 mesh + 2 arrows = 3 traces
        n1 = len(plot_spacecraft_3d(self._config_with_n_radiators(1)).data)
        n2 = len(plot_spacecraft_3d(self._config_with_n_radiators(2)).data)
        assert n2 - n1 == 3

    def test_no_radiators_still_renders(self) -> None:
        bus = SpacecraftBus(radiators=RadiatorConfiguration(radiators=[]))
        config = MissionConfig(spacecraft_bus=bus)
        fig = plot_spacecraft_3d(config)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# TestMultipleStarTrackers
# ---------------------------------------------------------------------------


class TestMultipleStarTrackers:
    def _config_with_n_trackers(self, n: int) -> MissionConfig:
        boresights = [
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ]
        trackers = [
            StarTracker(
                name=f"ST-{i}",
                orientation=StarTrackerOrientation(
                    boresight=boresights[i % len(boresights)]
                ),
            )
            for i in range(n)
        ]
        bus = SpacecraftBus(
            star_trackers=StarTrackerConfiguration(star_trackers=trackers)
        )
        return MissionConfig(spacecraft_bus=bus)

    def test_one_tracker_base(self) -> None:
        fig = plot_spacecraft_3d(self._config_with_n_trackers(1))
        st_traces = [t for t in fig.data if "ST-0" in (t.name or "")]
        assert len(st_traces) >= 1

    def test_two_trackers_adds_four_traces(self) -> None:
        # Each tracker: 2 meshes + 2 arrows = 4 traces
        n1 = len(plot_spacecraft_3d(self._config_with_n_trackers(1)).data)
        n2 = len(plot_spacecraft_3d(self._config_with_n_trackers(2)).data)
        assert n2 - n1 == 4

    def test_each_tracker_gets_distinct_boresight_color(self) -> None:
        fig = plot_spacecraft_3d(self._config_with_n_trackers(2))
        # Exclude the "+X (boresight)" axis cone (legendgroup="axes")
        bore_cones = [
            t
            for t in fig.data
            if isinstance(t, go.Cone)
            and "boresight" in (t.name or "").lower()
            and getattr(t, "legendgroup", None) != "axes"
        ]
        assert len(bore_cones) == 2
        colors = [t.colorscale[0][1] for t in bore_cones]
        assert colors[0] != colors[1]

    def test_no_star_trackers_still_renders(self) -> None:
        bus = SpacecraftBus(star_trackers=StarTrackerConfiguration(star_trackers=[]))
        config = MissionConfig(spacecraft_bus=bus)
        fig = plot_spacecraft_3d(config)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# TestRealConfig — integration smoke test
# ---------------------------------------------------------------------------


class TestRealConfig:
    def test_example_json_config(self) -> None:
        config = MissionConfig.from_json_file("examples/example_config.json")
        fig = plot_spacecraft_3d(config)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_example_json_title_uses_bus_name(self) -> None:
        config = MissionConfig.from_json_file("examples/example_config.json")
        fig = plot_spacecraft_3d(config)
        assert config.spacecraft_bus.name in fig.layout.title.text

    def test_example_json_expected_trace_count(self) -> None:
        config = MissionConfig.from_json_file("examples/example_config.json")
        fig = plot_spacecraft_3d(config)
        # 1 bus + 1 panel + 1 frame + 2 panel arrows
        # + 1 radiator + 2 rad arrows
        # + 2 ST × (2 mesh + 2 arrows)
        # + 9 axes = 25
        assert len(fig.data) == 25
