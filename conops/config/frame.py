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

from ..common.vector import (
    attitude_to_quat,
    quat_to_attitude,
    quaternion_rotate_vector,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from ._base import ConfigModel

NEGLIGIBLE_NORM = 1e-15
"""A vector norm below this is treated as zero."""
UNIT_MAGNITUDE_TOLERANCE = 0.01
"""How far from 1 a unit vector's magnitude may be."""

IDENTITY_TOLERANCE = 1e-12

if TYPE_CHECKING:
    from rust_ephem.constraints import ConstraintConfig

    from .instrument import InstrumentMounting

Vector = tuple[float, float, float]

ORTHOGONALITY_TOLERANCE = 1e-6
"""The largest cosine between the frame's two axes accepted as perpendicular."""


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
        """Whether the frame is the canonical one."""
        return bool(np.allclose(self.body_from_canonical, np.eye(3), atol=1e-12))

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

    def body_attitude(self, ra: float, dec: float, roll: float) -> Vector:
        """Express a reference-boresight attitude in physical body coordinates."""
        if self.is_identity:
            return ra, dec, roll
        return self.mounting.body_attitude(ra, dec, roll)

    def canonical_attitude(self, ra: float, dec: float, roll: float) -> Vector:
        """Express a physical body attitude about the reference boresight."""
        if self.is_identity:
            return ra, dec, roll
        rotation = self.body_from_canonical.T @ quaternion_to_rotation_matrix(
            attitude_to_quat(ra, dec, roll)
        )
        ra, dec, roll = quat_to_attitude(rotation_matrix_to_quaternion(rotation))
        if np.hypot(rotation[0, 0], rotation[0, 1]) < NEGLIGIBLE_NORM:
            # At a pole, asin of a rounded matrix element can invent a
            # milliarcsecond departure. Preserve the exact polar direction.
            return 0.0, float(np.copysign(90.0, rotation[0, 2])), roll
        return ra, dec, roll

    def constraint_in_body(self, constraint: "ConstraintConfig") -> "ConstraintConfig":
        """Place an unmounted reference-boresight constraint in body coordinates.

        Explicit boresight-offset subtrees already describe body hardware and
        are preserved. Other leaves follow this frame, including its clocking.
        """
        from rust_ephem.constraints import (
            AndConstraint,
            AtLeastConstraint,
            BoresightOffsetConstraint,
            NotConstraint,
            OrConstraint,
            XorConstraint,
        )

        if self.is_identity or isinstance(constraint, BoresightOffsetConstraint):
            return constraint
        if isinstance(constraint, NotConstraint):
            return constraint.model_copy(
                update={"constraint": self.constraint_in_body(constraint.constraint)}
            )
        if isinstance(
            constraint, (AndConstraint, OrConstraint, XorConstraint, AtLeastConstraint)
        ):
            return constraint.model_copy(
                update={
                    "constraints": [
                        self.constraint_in_body(c) for c in constraint.constraints
                    ]
                }
            )
        matrix = self.body_from_canonical
        pitch = np.arctan2(-matrix[2, 0], np.hypot(matrix[0, 0], matrix[1, 0]))
        if np.hypot(matrix[0, 0], matrix[1, 0]) > IDENTITY_TOLERANCE:
            roll = np.arctan2(matrix[2, 1], matrix[2, 2])
            yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
        else:
            roll = 0.0
            yaw = np.arctan2(-matrix[0, 1], matrix[1, 1])
        return constraint.boresight_offset(
            roll_deg=float(np.degrees(roll)),
            pitch_deg=float(np.degrees(pitch)),
            yaw_deg=float(np.degrees(yaw)),
        )

    @property
    def mounting(self) -> "InstrumentMounting":
        """The mounting of a telescope that points along the frame's
        boresight with its zero-roll reference completing the frame."""
        from .instrument import InstrumentMounting

        return InstrumentMounting(
            body_from_instrument_quaternion_wxyz=self.quaternion_wxyz
        )
