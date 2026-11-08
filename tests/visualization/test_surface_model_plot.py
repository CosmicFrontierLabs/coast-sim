"""The displayed plates use the same geometry as the environmental model."""

import numpy as np
import pytest
from pydantic import ValidationError

from conops.config import MissionConfig, SpacecraftBus, SpacecraftFrame
from conops.config.geometry import PanelGeometry
from conops.config.instrument import Payload, Telescope
from conops.config.solar_panel import SolarPanel, SolarPanelSet
from conops.config.surfaces import Surface, SurfaceModel, panel_surfaces
from conops.visualization import plot_spacecraft_3d


def vertices(mesh):
    return np.column_stack((mesh.x, mesh.y, mesh.z)).astype(float)


@pytest.mark.parametrize("normal", [(1, 0, 0), (0, 0, -1), (1, 2, 3)])
def test_surface_plate_preserves_center_area_normal_and_plot_bounds(normal):
    surface = Surface(
        name="Offset plate", center_m=(12, -5, 2), normal=normal, area_m2=6
    )
    config = MissionConfig(
        spacecraft_bus=SpacecraftBus(surfaces=SurfaceModel(surfaces=[surface])),
        payload=Payload(instruments=[Telescope(name="Synthetic tube")]),
    )
    fig = plot_spacecraft_3d(config)
    mesh = next(t for t in fig.data if t.name == surface.name)
    pts = vertices(mesh)
    np.testing.assert_allclose(pts.mean(axis=0), surface.center_m)
    cross = np.cross(pts[1] - pts[0], pts[3] - pts[0])
    assert np.linalg.norm(cross) == pytest.approx(surface.area_m2)
    np.testing.assert_allclose(
        cross / np.linalg.norm(cross), surface.normal, atol=1e-12
    )
    names = {t.name for t in fig.data}
    assert config.spacecraft_bus.name not in names
    assert "Synthetic tube" not in names
    for axis, column in zip(
        (fig.layout.scene.xaxis, fig.layout.scene.yaxis, fig.layout.scene.zaxis),
        pts.T,
        strict=True,
    ):
        assert min(column) > axis.range[0]
        assert max(column) < axis.range[1]


def test_explicit_surface_outline_and_payload_frame_are_used_without_rotation():
    geometry = PanelGeometry(
        center_m=(2, 0, 4), u=(1, 0, 0), v=(0, 1, 0), width_m=4, height_m=1
    )
    surface = Surface(
        name="Body",
        center_m=geometry.center_m,
        normal=(0, 0, 1),
        area_m2=4,
        geometry=geometry,
    )
    config = MissionConfig(
        spacecraft_bus=SpacecraftBus(
            frame=SpacecraftFrame(boresight=(0, 0, 1), sun_side=(1, 0, 0)),
            surfaces=SurfaceModel(surfaces=[surface]),
        )
    )
    # Geometry is configuration state and must survive a mission-file round trip.
    config = MissionConfig.model_validate_json(config.model_dump_json())
    fig = plot_spacecraft_3d(config, show_normals=False)
    mesh = next(t for t in fig.data if t.name == "Body")
    np.testing.assert_allclose(vertices(mesh), geometry.corners)
    assert "+Z (boresight)" in {t.name for t in fig.data}
    assert "+X (boresight)" not in {t.name for t in fig.data}


def test_configured_panel_front_and_back_are_not_drawn_twice():
    geometry = PanelGeometry(
        center_m=(0, 0, 6), u=(1, 0, 0), v=(0, 1, 0), width_m=2, height_m=1
    )
    panels = SolarPanelSet(
        panels=[SolarPanel(name="Cells", normal=(0, 0, 1), geometry=geometry)]
    )
    body = Surface(name="Bus plate", center_m=(0, 0, -1), normal=(0, 0, -1), area_m2=4)
    config = MissionConfig(
        solar_panel=panels,
        spacecraft_bus=SpacecraftBus(
            surfaces=SurfaceModel(surfaces=[body, *panel_surfaces(panels)])
        ),
    )
    fig = plot_spacecraft_3d(config, show_normals=False, show_axes=False)
    meshes = [t for t in fig.data if t.type == "mesh3d"]
    assert sum(t.name == "Cells" for t in meshes) == 1
    assert not any(t.name in ("Cells cells", "Cells back") for t in meshes)
    np.testing.assert_allclose(
        vertices(next(t for t in meshes if t.name == "Cells")).mean(axis=0),
        geometry.center_m,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"center_m": (1, 0, 0)},
        {"width_m": 3},
        {"v": (0, 0, 1)},
    ],
)
def test_surface_outline_cannot_disagree_with_physical_plate(changes):
    geometry = PanelGeometry(
        u=(1, 0, 0), v=(0, 1, 0), width_m=2, height_m=1
    ).model_copy(update=changes)
    with pytest.raises(ValidationError, match="surface geometry"):
        Surface(
            name="Plate",
            center_m=(0, 0, 0),
            normal=(0, 0, 1),
            area_m2=2,
            geometry=geometry,
        )


def test_empty_surface_model_retains_fallback_bus():
    config = MissionConfig(spacecraft_bus=SpacecraftBus(surfaces=SurfaceModel()))
    assert any(
        t.name == config.spacecraft_bus.name for t in plot_spacecraft_3d(config).data
    )
