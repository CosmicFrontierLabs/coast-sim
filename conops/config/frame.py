"""The spacecraft body frame as the mission writes it.

COAST points the body's science boresight and rolls about it. The body axis
that boresight lies along, and the axis that faces the Sun's side at zero
roll, are the mission's to choose: a :class:`SpacecraftFrame` on the bus
declares them, and everything written in the body frame, panels, radiators,
antennas, star trackers, surfaces, the inertia tensor and per-axis slew
limits, is read in the frame declared. The default is the frame COAST has
always used: boresight along +X, the Sun's side +Y, +Z completing the
right-handed frame.
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from ..common.numerics import NEGLIGIBLE_NORM, UNIT_MAGNITUDE_TOLERANCE
from ..common.vector import quaternion_rotate_vector, rotation_matrix_to_quaternion
from ._base import ConfigModel

if TYPE_CHECKING:
    from .instrument import InstrumentMounting

Vector = tuple[float, float, float]

ORTHOGONALITY_TOLERANCE = 1e-6
"""The largest cosine between the frame's two axes accepted as perpendicular."""

IDENTITY_TOLERANCE = 1e-12
"""How far each axis may lie from the canonical frame's and still count as it."""


def _unit(value: Vector, name: str) -> npt.NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    magnitude = float(np.linalg.norm(vector))
    if abs(magnitude - 1.0) > UNIT_MAGNITUDE_TOLERANCE:
        raise ValueError(f"{name} must be a unit vector (magnitude {magnitude:.4f})")
    return vector / magnitude


class SpacecraftFrame(ConfigModel):
    """Which body axes the science boresight and the Sun's side lie along.

    The canonical frame COAST computes in has the boresight along +X and the
    Sun's side, where a side-mounted panel faces at zero roll, along +Y. A
    mission whose boresight is +Z and whose panels face -Y declares
    ``boresight=(0, 0, 1)`` and ``sun_side=(0, -1, 0)``: a telescope without
    a mounting of its own then points along +Z, and the panel and star
    tracker helpers place their vectors in the mission's frame.
    """

    boresight: Vector = Field(
        default=(1.0, 0.0, 0.0),
        description=(
            "Body axis the science boresight lies along: where a telescope "
            "without a mounting of its own points, as a unit vector"
        ),
    )
    sun_side: Vector = Field(
        default=(0.0, 1.0, 0.0),
        description=(
            "Body axis facing the Sun's side at zero roll, where a side-mounted "
            "panel faces, as a unit vector perpendicular to the boresight"
        ),
    )

    @model_validator(mode="after")
    def _axes_are_unit_and_perpendicular(self) -> "SpacecraftFrame":
        boresight = _unit(self.boresight, "boresight")
        sun_side = _unit(self.sun_side, "sun_side")
        if abs(float(np.dot(boresight, sun_side))) > ORTHOGONALITY_TOLERANCE:
            raise ValueError("sun_side must be perpendicular to the boresight")
        return self

    @property
    def body_from_canonical(self) -> npt.NDArray[np.float64]:
        """The rotation taking a vector written in the canonical frame (+X
        boresight, +Y Sun's side) to the body frame declared: its columns
        are the boresight, the Sun's side and the axis completing them.
        The Sun's side is taken perpendicular to the boresight, within the
        tolerance validation allows, so the rotation is exact."""
        boresight = _unit(self.boresight, "boresight")
        sun_side = _unit(self.sun_side, "sun_side")
        sun_side -= float(np.dot(sun_side, boresight)) * boresight
        sun_side /= float(np.linalg.norm(sun_side))
        return np.column_stack((boresight, sun_side, np.cross(boresight, sun_side)))

    @property
    def quaternion_wxyz(self) -> tuple[float, float, float, float]:
        """The same rotation as a unit Hamilton quaternion [w, x, y, z]."""
        q = rotation_matrix_to_quaternion(self.body_from_canonical)
        return float(q[0]), float(q[1]), float(q[2]), float(q[3])

    @property
    def is_identity(self) -> bool:
        """Whether the frame is the canonical one. Read on every validation
        of a mission configuration, so it compares the axes as given rather
        than building the rotation."""
        return all(
            abs(float(axis[k]) - canonical[k]) <= IDENTITY_TOLERANCE
            for axis, canonical in (
                (self.boresight, (1.0, 0.0, 0.0)),
                (self.sun_side, (0.0, 1.0, 0.0)),
            )
            for k in range(3)
        )

    def body_vector(self, canonical: Vector) -> Vector:
        """A vector written in the canonical frame, in the body frame."""
        vector = quaternion_rotate_vector(self.quaternion_wxyz, canonical)
        vector[np.abs(vector) < NEGLIGIBLE_NORM] = 0.0
        return float(vector[0]), float(vector[1]), float(vector[2])

    def canonical_vector(self, body: Vector) -> Vector:
        """A body-frame vector in the canonical frame."""
        vector = self.body_from_canonical.T @ np.asarray(body, dtype=np.float64)
        vector[np.abs(vector) < NEGLIGIBLE_NORM] = 0.0
        return float(vector[0]), float(vector[1]), float(vector[2])

    @property
    def mounting(self) -> "InstrumentMounting":
        """The mounting of a telescope that points along the frame's
        boresight with its zero-roll reference completing the frame."""
        from .instrument import InstrumentMounting

        return InstrumentMounting(
            body_from_instrument_quaternion_wxyz=self.quaternion_wxyz
        )
