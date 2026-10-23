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
    """When one visit of a target may begin collecting science, and, where
    the visit differs from the target's own request, where it points and
    the roll it asks for."""

    model_config = ConfigDict(frozen=True)

    start: float = Field(
        description="Earliest time (Unix seconds) the visit's collection may begin"
    )
    deadline: float = Field(
        description="Latest time (Unix seconds) the visit's collection may begin"
    )
    ra: float | None = Field(
        default=None,
        ge=0.0,
        lt=360.0,
        description=(
            "Right ascension this visit points at, in degrees, where it differs "
            "from the target's, as a moving target's does; set with dec"
        ),
    )
    dec: float | None = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="Declination this visit points at, in degrees; set with ra",
    )
    roll_deg: float | None = Field(
        default=None,
        description=(
            "The instrument roll this visit asks for, in degrees, in place of "
            "the target's request; not with roll_offset_deg"
        ),
    )
    roll_offset_deg: float | None = Field(
        default=None,
        description=(
            "The offset from the power-optimal roll this visit asks for, in "
            "degrees, in place of the target's request; not with roll_deg"
        ),
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
        if (self.ra is None) != (self.dec is None):
            raise ValueError("a visit window's pointing needs both ra and dec")
        if self.roll_deg is not None and self.roll_offset_deg is not None:
            raise ValueError(
                "a visit window asks for a roll or an offset from the optimum, not both"
            )
        return self

    @property
    def own_request(self) -> bool:
        """Whether the visit points or rolls differently from its target."""
        return (
            self.ra is not None
            or self.roll_deg is not None
            or self.roll_offset_deg is not None
        )

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
