"""The typed plan entries a container serialises: a plain entry, a pointing
or a moving pointing, told apart by their ``kind`` when a file is read."""

from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import Field

from .plan_entry import PlanEntry
from .pointing import MovingPointing, Pointing

AnyEntry: TypeAlias = Annotated[
    PlanEntry | Pointing | MovingPointing, Field(discriminator="kind")
]
"""A plan entry of any class, validated by its ``kind``."""

ENTRY_CLASSES: dict[str, type[PlanEntry]] = {
    "entry": PlanEntry,
    "pointing": Pointing,
    "moving_pointing": MovingPointing,
}


def with_kind(data: object, default: str = "entry") -> object:
    """A dictionary of entry fields with its ``kind`` set, ``default`` where a
    file written before entries carried one has none."""
    if isinstance(data, dict) and "kind" not in data:
        return {**data, "kind": default}
    return data
