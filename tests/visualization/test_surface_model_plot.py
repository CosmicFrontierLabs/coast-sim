"""The displayed plates use the same geometry as the environmental model."""

import numpy as np
import pytest
from pydantic import ValidationError

from conops.config import MissionConfig, SpacecraftBus, SpacecraftFrame
from conops.config.geometry import PanelGeometry
from conops.config.instrument import Payload, Telescope, TelescopeConfig
from conops.config.radiator import (
    Radiator,
    RadiatorConfiguration,
    RadiatorOrientation,
)
from conops.config.solar_panel import SolarPanel, SolarPanelSet
from conops.config.star_tracker import (
    StarTracker,
    StarTrackerConfiguration,
    StarTrackerOrientation,
)
from conops.config.surfaces import (
    Surface,
    SurfaceModel,
    box_surfaces,
    guess_surfaces,
    panel_surfaces,
    telescope_surfaces,
)
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
    # The plate is not the telescope's aperture, so the telescope stays.
    assert {"Synthetic tube", "Synthetic tube boresight"} <= names
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


def _cube_config(**bus: object) -> MissionConfig:
    """A 4 m cube of plates: faces at 2 m on every axis."""
    return MissionConfig(
        spacecraft_bus=SpacecraftBus(
            surfaces=SurfaceModel(surfaces=box_surfaces((4, 4, 4))), **bus
        ),
        payload=Payload(instruments=[]),
    )


def test_parts_without_geometry_sit_on_the_configured_hull():
    """Before, they were placed on the fallback box the plates replace, and
    so inside a hull larger than it."""
    config = _cube_config(
        star_trackers=StarTrackerConfiguration(
            star_trackers=[
                StarTracker(
                    name="Tracker",
                    orientation=StarTrackerOrientation(boresight=(0, 0, 1)),
                )
            ]
        ),
        radiators=RadiatorConfiguration(
            radiators=[
                Radiator(
                    name="Radiator",
                    width_m=1,
                    height_m=1,
                    orientation=RadiatorOrientation(normal=(0, -1, 0)),
                )
            ]
        ),
    )
    fig = plot_spacecraft_3d(config)
    tracker = next(t for t in fig.data if t.name == "Tracker")
    radiator = next(t for t in fig.data if t.name == "Radiator")
    assert min(tracker.z) >= 2.0
    assert max(radiator.y) <= -2.0
    # A caller's own box still decides the placement.
    boxed = plot_spacecraft_3d(config, bus_half_dims=(1.0, 0.55, 0.55))
    tracker = next(t for t in boxed.data if t.name == "Tracker")
    assert max(tracker.z) < 1.0


def test_a_small_patch_does_not_remove_the_telescope():
    patch = Surface(name="Patch", center_m=(0, 0, 1), normal=(0, 0, 1), area_m2=0.1)
    config = MissionConfig(
        spacecraft_bus=SpacecraftBus(surfaces=SurfaceModel(surfaces=[patch])),
        payload=Payload(instruments=[Telescope(name="Scope")]),
    )
    names = {t.name for t in plot_spacecraft_3d(config).data}
    assert {"Patch", "Scope", "Scope boresight"} <= names


def test_an_aperture_plate_stands_for_the_telescope_tube():
    payload = Payload(
        instruments=[
            Telescope(
                name="Scope",
                optics=TelescopeConfig(aperture_m=0.5, tube_length_m=1.2),
            )
        ]
    )
    [aperture] = telescope_surfaces(payload)
    config = MissionConfig(
        spacecraft_bus=SpacecraftBus(
            surfaces=SurfaceModel(surfaces=[*box_surfaces((2, 1, 1)), aperture])
        ),
        payload=payload,
    )
    fig = plot_spacecraft_3d(config)
    names = {t.name for t in fig.data}
    assert "Scope" not in names
    assert aperture.name in names
    # The boresight is shown all the same, from the aperture plate.
    arrow = next(
        t for t in fig.data if t.name == "Scope boresight" and t.type == "scatter3d"
    )
    assert arrow.x[0] == pytest.approx(aperture.center_m[0] + 0.05)
    assert "Scope boresight" not in {
        t.name for t in plot_spacecraft_3d(config, show_normals=False).data
    }


def test_many_plates_share_two_legend_entries_and_two_normal_traces():
    plates = [
        Surface(name=f"Plate {k}", center_m=(k, 0, 1), normal=(0, 0, 1), area_m2=0.5)
        for k in range(200)
    ]
    config = MissionConfig(
        spacecraft_bus=SpacecraftBus(surfaces=SurfaceModel(surfaces=plates)),
        payload=Payload(instruments=[]),
    )
    fig = plot_spacecraft_3d(config, show_axes=False)
    in_groups = [
        t for t in fig.data if t.legendgroup in ("surfaces", "surface_normals")
    ]
    # A mesh for each plate, their legend entry, and one line trace and one
    # cone trace of normals.
    assert len(in_groups) == 203
    assert [t.name for t in in_groups if t.showlegend] == [
        "Surfaces",
        "Surface normals",
    ]
    cone = next(t for t in in_groups if t.type == "cone")
    assert len(cone.x) == 200
    np.testing.assert_allclose(cone.w, cone.w[0])
    assert cone.text[7] == "Plate 7 normal"


def _radiator(normal: tuple[float, float, float]) -> RadiatorConfiguration:
    return RadiatorConfiguration(
        radiators=[
            Radiator(
                name="Radiator",
                width_m=1,
                height_m=1,
                orientation=RadiatorOrientation(normal=normal),
            )
        ]
    )


def _radiator_center(
    surfaces: list[Surface],
    normal: tuple[float, float, float],
    payload: Payload | None = None,
):
    config = MissionConfig(
        spacecraft_bus=SpacecraftBus(
            surfaces=SurfaceModel(surfaces=surfaces), radiators=_radiator(normal)
        ),
        # The payload the surfaces' instrument plates belong to.
        payload=payload or Payload(instruments=[]),
    )
    radiator = next(t for t in plot_spacecraft_3d(config).data if t.name == "Radiator")
    return vertices(radiator).mean(axis=0)


def test_an_aperture_plate_does_not_move_parts_off_the_hull():
    payload = Payload(
        instruments=[
            Telescope(
                name="T", optics=TelescopeConfig(aperture_m=0.5, tube_length_m=3.0)
            )
        ]
    )
    # A 1.5 x 1.5 x 2 m box with the aperture 3 m out along +X.
    surfaces = guess_surfaces(None, payload).surfaces
    center = _radiator_center(surfaces, (-1, 0, 0), payload)
    np.testing.assert_allclose(center, (-0.755, 0, 0), atol=1e-9)


def test_parts_sit_on_a_hull_away_from_the_body_origin():
    surfaces = box_surfaces((2, 2, 4), center_m=(0, 0, 2))
    np.testing.assert_allclose(
        _radiator_center(surfaces, (0, 0, -1)), (0, 0, -0.005), atol=1e-9
    )
    np.testing.assert_allclose(
        _radiator_center(surfaces, (0, -1, 0)), (0, -1.005, 2), atol=1e-9
    )


def test_a_wing_seen_edge_on_does_not_move_parts_off_the_hull():
    wings = [
        Surface(name=f"wing {y}", center_m=(0, y, 0), normal=(0, 0, 1), area_m2=4)
        for y in (-5, 5)
    ]
    surfaces = [*box_surfaces((2, 2, 2)), *wings]
    np.testing.assert_allclose(
        _radiator_center(surfaces, (0, -1, 0)), (0, -1.005, 0), atol=1e-9
    )


def test_a_barrel_wider_than_the_optics_is_not_given_a_second_tube():
    """Its plates are marked as the telescope's, whatever their size."""
    facets = [
        Surface(
            name=f"barrel {k}",
            center_m=(2.25, 0.31 * np.cos(angle), 0.31 * np.sin(angle)),
            normal=(0, np.cos(angle), np.sin(angle)),
            area_m2=0.6,
            instrument="Scope",
        )
        for k, angle in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ]
    front = Surface(
        name="barrel front",
        center_m=(3.5, 0, 0),
        normal=(1, 0, 0),
        area_m2=0.62**2,
        instrument="Scope",
    )
    config = MissionConfig(
        spacecraft_bus=SpacecraftBus(
            surfaces=SurfaceModel(surfaces=[*box_surfaces((2, 1, 1)), *facets, front])
        ),
        payload=Payload(
            instruments=[
                Telescope(
                    name="Scope",
                    optics=TelescopeConfig(aperture_m=0.5, tube_length_m=2.5),
                )
            ]
        ),
    )
    fig = plot_spacecraft_3d(config)
    assert "Scope" not in {t.name for t in fig.data}
    arrow = next(
        t for t in fig.data if t.name == "Scope boresight" and t.type == "scatter3d"
    )
    assert (arrow.x[0], arrow.y[0], arrow.z[0]) == pytest.approx((3.55, 0, 0))


def test_a_plate_is_a_telescopes_only_when_marked_as_its_own():
    """Not by having the aperture's area: each telescope's boresight starts
    at its own plate, and an unmarked plate of that area removes no tube."""
    optics = {"aperture_m": 0.5}
    payload = Payload(
        instruments=[
            Telescope(name="Near", optics=TelescopeConfig(tube_length_m=1.2, **optics)),
            Telescope(name="Far", optics=TelescopeConfig(tube_length_m=2.4, **optics)),
        ]
    )
    near, far = telescope_surfaces(payload)
    assert (near.instrument, far.instrument) == ("Near", "Far")
    config = MissionConfig(
        spacecraft_bus=SpacecraftBus(
            surfaces=SurfaceModel(surfaces=[*box_surfaces((2, 1, 1)), near, far])
        ),
        payload=payload,
    )
    fig = plot_spacecraft_3d(config)
    starts = {
        t.name: t.x[0]
        for t in fig.data
        if t.type == "scatter3d" and t.name in ("Near boresight", "Far boresight")
    }
    assert starts == pytest.approx({"Near boresight": 1.25, "Far boresight": 2.45})

    stray = Surface(
        name="Stray",
        center_m=(-1, 5, 0),
        normal=(1, 0, 0),
        area_m2=near.area_m2,
    )
    config = MissionConfig(
        spacecraft_bus=SpacecraftBus(surfaces=SurfaceModel(surfaces=[stray])),
        payload=Payload(instruments=payload.instruments[:1]),
    )
    assert "Near" in {t.name for t in plot_spacecraft_3d(config).data}


def _with_wheels(surfaces: list[Surface]) -> MissionConfig:
    from conops.config import (
        AttitudeControlSystem,
        MomentumWheel,
        StoredMomentumConfig,
    )

    wheels = [
        MomentumWheel(name=f"RW{i + 1}", spin_axis=axis, capacity_n_m_s=4.0)
        for i, axis in enumerate([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
    ]
    return MissionConfig(
        spacecraft_bus=SpacecraftBus(
            surfaces=SurfaceModel(surfaces=surfaces),
            attitude_control=AttitudeControlSystem(
                stored_momentum=StoredMomentumConfig(wheels=wheels, capacity_n_m_s=4.0)
            ),
        ),
        payload=Payload(instruments=[]),
    )


@pytest.mark.parametrize(
    "surfaces",
    [
        # One plate: no other plate gives the hull a depth.
        [Surface(name="drag", center_m=(0.5, 0, 0), normal=(1, 0, 0), area_m2=1.0)],
        # Two plates facing each other's way, with centres on one line.
        [
            Surface(name="+X", center_m=(0.5, 0, 0), normal=(1, 0, 0), area_m2=1.0),
            Surface(name="-X", center_m=(-0.5, 0, 0), normal=(-1, 0, 0), area_m2=1.0),
        ],
    ],
)
def test_wheels_are_drawn_on_a_hull_with_plates_in_one_line(surfaces):
    """The plates' centres span nothing across the line they lie on. The hull
    takes the plates' own extent there. Before, it had no thickness, the
    wheels' positions were not numbers and the wheels were not drawn."""
    fig = plot_spacecraft_3d(_with_wheels(surfaces))
    wheels = [t for t in fig.data if getattr(t, "legendgroup", None) == "wheels"]
    assert wheels
    for trace in wheels:
        for axis in (trace.x, trace.y, trace.z):
            values = np.asarray([v for v in axis if v is not None], dtype=float)
            assert np.isfinite(values).all()
    # Each 1 m plate reaches 0.5 m either side of the line: a wheel that spins
    # about Y or Z sits inside that.
    rims = [t for t in wheels if t.type == "mesh3d"]
    assert max(float(np.max(np.abs(t.y))) for t in rims) < 0.5 + 0.2
    assert max(float(np.max(np.abs(t.y))) for t in rims) > 0.05


def test_a_surface_cannot_name_an_instrument_the_payload_lacks():
    """ "scope" is not "Scope". Before, the name was ignored: the plates left
    the hull and the telescope's tube was still drawn over them."""
    payload = Payload(instruments=[Telescope(name="Scope")])
    plate = Surface(name="barrel", center_m=(2, 0, 0), normal=(1, 0, 0), area_m2=0.2)
    with pytest.raises(ValidationError, match="scope"):
        MissionConfig(
            spacecraft_bus=SpacecraftBus(
                surfaces=SurfaceModel(
                    surfaces=[plate.model_copy(update={"instrument": "scope"})]
                )
            ),
            payload=payload,
        )
    # Assigned to a configuration afterwards, it is not checked again: the
    # plot says so.
    config = MissionConfig(spacecraft_bus=SpacecraftBus(), payload=payload)
    config.spacecraft_bus.surfaces = SurfaceModel(
        surfaces=[plate.model_copy(update={"instrument": "scope"})]
    )
    with pytest.warns(UserWarning, match="scope"):
        plot_spacecraft_3d(config)
    # A name the payload has gives no warning.
    config.spacecraft_bus.surfaces = SurfaceModel(
        surfaces=[plate.model_copy(update={"instrument": "Scope"})]
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_spacecraft_3d(config)
