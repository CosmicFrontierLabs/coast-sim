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


def with_kind(data: object, default: str = "entry") -> object:
    """Entry fields with ``kind`` set, ``default`` where missing."""
    if isinstance(data, dict) and "kind" not in data:
        return {**data, "kind": default}
    return data
