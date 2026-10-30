"""Spacecraft surfaces for the environmental torques.

Solar radiation pressure and aerodynamic drag act on the spacecraft's
surfaces, and the torque each produces depends on where the surface is
relative to the centre of mass. A :class:`SurfaceModel` describes the
spacecraft as flat plates (:class:`Surface`) with an area, an outward normal,
a position and the coefficients the forces need, about a centre of mass.
Self-shadowing between plates is not modelled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, model_validator

from ._base import ConfigModel

if TYPE_CHECKING:
    from .instrument import Payload
    from .solar_panel import SolarPanelSet

Vector3 = tuple[float, float, float]


def _unit(vector: Vector3, name: str) -> Vector3:
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError(f"{name} must have nonzero magnitude")
    array = array / norm
    return (float(array[0]), float(array[1]), float(array[2]))


class Surface(ConfigModel):
    """One flat plate of the spacecraft, in the body frame."""

    name: str = Field(description="What the plate is, such as 'bus +X' or a panel")
    center_m: Vector3 = Field(
        description="The plate's centre in the body frame, in metres"
    )
    normal: Vector3 = Field(description="The plate's outward unit normal")
    area_m2: float = Field(gt=0.0, description="The plate's area, in square metres")
    specular_reflectivity: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="The fraction of sunlight the plate reflects specularly",
    )
    diffuse_reflectivity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "The fraction of sunlight the plate reflects diffusely; the rest "
            "of what is not reflected specularly is absorbed"
        ),
    )
    drag_coefficient: float = Field(
        default=2.2,
        gt=0.0,
        description="The plate's drag coefficient in free molecular flow",
    )

    @model_validator(mode="after")
    def _check(self) -> Surface:
        if self.specular_reflectivity + self.diffuse_reflectivity > 1.0 + 1e-9:
            raise ValueError(
                "specular and diffuse reflectivity together cannot exceed one"
            )
        normal = _unit(self.normal, "surface normal")
        if normal != self.normal:
            object.__setattr__(self, "normal", normal)
        return self


class SurfaceModel(ConfigModel):
    """The spacecraft as flat plates about its centre of mass."""

    center_of_mass_m: Vector3 = Field(
        default=(0.0, 0.0, 0.0),
        description="The centre of mass in the body frame, in metres",
    )
    surfaces: list[Surface] = Field(
        default_factory=list, description="The plates the environment acts on"
    )

    @property
    def total_area_m2(self) -> float:
        return float(sum(surface.area_m2 for surface in self.surfaces))


def box_surfaces(
    size_m: Vector3,
    center_m: Vector3 = (0.0, 0.0, 0.0),
    name: str = "bus",
    **coefficients: float,
) -> list[Surface]:
    """The six faces of a box of ``size_m`` along the body axes, centred at
    ``center_m``, each with the given coefficients."""
    sx, sy, sz = (float(s) for s in size_m)
    if min(sx, sy, sz) <= 0.0:
        raise ValueError("box dimensions must be positive")
    cx, cy, cz = (float(c) for c in center_m)
    faces = [
        ("+X", (cx + sx / 2, cy, cz), (1.0, 0.0, 0.0), sy * sz),
        ("-X", (cx - sx / 2, cy, cz), (-1.0, 0.0, 0.0), sy * sz),
        ("+Y", (cx, cy + sy / 2, cz), (0.0, 1.0, 0.0), sx * sz),
        ("-Y", (cx, cy - sy / 2, cz), (0.0, -1.0, 0.0), sx * sz),
        ("+Z", (cx, cy, cz + sz / 2), (0.0, 0.0, 1.0), sx * sy),
        ("-Z", (cx, cy, cz - sz / 2), (0.0, 0.0, -1.0), sx * sy),
    ]
    return [
        Surface(
            name=f"{name} {face}",
            center_m=center,
            normal=normal,
            area_m2=area,
            **coefficients,
        )
        for face, center, normal, area in faces
    ]


def panel_surfaces(solar_panel: SolarPanelSet | None) -> list[Surface]:
    """Both faces of each solar panel that has a geometry: the cell side
    at the panel's normal, mostly absorbing, and the back, more reflective.
    A panel without geometry has no known area and gives nothing."""
    surfaces: list[Surface] = []
    if solar_panel is None:
        return surfaces
    for panel in solar_panel.panels:
        geometry = panel.geometry
        if geometry is None:
            continue
        area = float(geometry.width_m * geometry.height_m)
        normal = _unit(panel.normal, "panel normal")
        back = (-normal[0] or 0.0, -normal[1] or 0.0, -normal[2] or 0.0)
        surfaces.append(
            Surface(
                name=f"{panel.name} cells",
                center_m=geometry.center_m,
                normal=normal,
                area_m2=area,
                specular_reflectivity=0.1,
                diffuse_reflectivity=0.1,
            )
        )
        surfaces.append(
            Surface(
                name=f"{panel.name} back",
                center_m=geometry.center_m,
                normal=back,
                area_m2=area,
                specular_reflectivity=0.1,
                diffuse_reflectivity=0.4,
            )
        )
    return surfaces


def telescope_surfaces(payload: Payload | None) -> list[Surface]:
    """The aperture of each telescope with an aperture and a tube length: an
    absorbing disc at the tube's end along its boresight."""
    from .instrument import Telescope

    surfaces: list[Surface] = []
    if payload is None:
        return surfaces
    for instrument in payload.instruments:
        if not isinstance(instrument, Telescope):
            continue
        aperture = instrument.optics.aperture_m
        length = instrument.optics.tube_length_m
        if aperture is None or length is None:
            continue
        boresight = _unit(instrument.mounting.body_vector((1.0, 0.0, 0.0)), "boresight")
        surfaces.append(
            Surface(
                name=f"{instrument.name} aperture",
                center_m=(
                    boresight[0] * length,
                    boresight[1] * length,
                    boresight[2] * length,
                ),
                normal=boresight,
                area_m2=float(np.pi * (aperture / 2.0) ** 2),
                specular_reflectivity=0.0,
                diffuse_reflectivity=0.0,
            )
        )
    return surfaces


def guess_surfaces(
    solar_panel: SolarPanelSet | None = None,
    payload: Payload | None = None,
    *,
    body_size_m: Vector3 = (1.5, 1.5, 2.0),
    center_of_mass_m: Vector3 = (-0.05, 0.0, 0.0),
    panels: Sequence[Surface] = (),
) -> SurfaceModel:
    """A surface model for a spacecraft whose surfaces are not configured:
    a box of ``body_size_m`` at the body origin, the solar panels with a
    geometry, and the telescope apertures, about a centre of mass offset a
    little behind the boresight. A placeholder for a mass-properties set,
    not a measurement of anything; its torques are of the right order and
    nothing more."""
    surfaces = [
        *box_surfaces(body_size_m),
        *(panels or panel_surfaces(solar_panel)),
        *telescope_surfaces(payload),
    ]
    return SurfaceModel(center_of_mass_m=center_of_mass_m, surfaces=surfaces)
