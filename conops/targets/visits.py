"""Visit windows: when each visit of a target may begin.

A target observed at set times, such as a variable star visited at log-spaced
intervals or a field revisited every six hours, is one request with a list of
:class:`VisitWindow`: each visit's science collection begins within its
window, and the visits are taken in order. The schedulers place each visit in
its window and dispatch selects the target only while an unvisited window is
open.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from ..common import unixtime_from


class VisitWindow(BaseModel):
    """When one visit of a target may begin collecting science."""

    model_config = ConfigDict(frozen=True)

    start: float = Field(
        description="Earliest time (Unix seconds) the visit's collection may begin"
    )
    deadline: float = Field(
        description="Latest time (Unix seconds) the visit's collection may begin"
    )

    @field_validator("start", "deadline", mode="before")
    @classmethod
    def _coerce_time(cls, v: float | int | str | datetime) -> float:
        """Accept Unix timestamps, datetimes or ISO-8601 strings."""
        return unixtime_from(v)

    @field_serializer("start", "deadline")
    def _serialize_time(self, v: float) -> str:
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()

    @model_validator(mode="after")
    def _ordered(self) -> VisitWindow:
        if self.deadline < self.start:
            raise ValueError("a visit window's deadline must not be before its start")
        return self

    def contains(self, utime: float) -> bool:
        """Whether a visit may begin collecting at ``utime``."""
        return self.start <= utime <= self.deadline


def visit_windows(
    starts: Sequence[float | int | str | datetime], slack_seconds: float
) -> list[VisitWindow]:
    """Windows opening at each of ``starts`` for ``slack_seconds``."""
    if slack_seconds < 0.0:
        raise ValueError("slack_seconds must not be negative")
    return [
        VisitWindow(
            start=unixtime_from(start), deadline=unixtime_from(start) + slack_seconds
        )
        for start in starts
    ]


def cadence_windows(
    first: float | int | str | datetime,
    interval_seconds: float,
    count: int,
    slack_seconds: float,
) -> list[VisitWindow]:
    """``count`` windows a fixed ``interval_seconds`` apart from ``first``,
    each open for ``slack_seconds``: a cadence with a fixed first epoch."""
    if interval_seconds <= 0.0:
        raise ValueError("interval_seconds must be positive")
    if count < 1:
        raise ValueError("count must be at least 1")
    begin = unixtime_from(first)
    return visit_windows(
        [begin + k * interval_seconds for k in range(count)], slack_seconds
    )
