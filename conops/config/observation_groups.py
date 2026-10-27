"""Observation groups: named sets of targets observed under one window, one
budget and, where wanted, together.

A category is an observation-ID range with a program's settings; a group is
a list of targets, such as the targets of one redshift bin or one field,
that share a window in which their science may be collected, a cap on the
science they collect together, and a preference for being observed one after
another. Dispatch, the planners and the long-range allocator honour all three.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from ..common.common import unixtime_from
from ._base import ConfigModel


class ObservationGroup(ConfigModel):
    """A set of targets observed under one window and one budget."""

    name: str = Field(description="The group's name")
    members: list[int] = Field(
        min_length=1, description="The observation IDs of the group's targets"
    )
    window_start: float | None = Field(
        default=None,
        description=(
            "When the group's science may begin, in Unix seconds, a datetime or an "
            "ISO string; None for no start. Narrows each member's own earliest start"
        ),
    )
    window_deadline: float | None = Field(
        default=None,
        description=(
            "By when the group's science must begin collecting, in Unix seconds, a "
            "datetime or an ISO string; None for no deadline. Narrows each member's "
            "own deadline"
        ),
    )
    time_budget_seconds: float | None = Field(
        default=None,
        gt=0,
        description=(
            "The most science the group's targets may collect together over a run, "
            "in seconds, counting what they had collected before it; None for no cap"
        ),
    )
    prefer_contiguous: bool = Field(
        default=True,
        description=(
            "Whether the group's targets are preferred one after another: dispatch "
            "adds targets.group_contiguity_weight to a member's merit once another "
            "member has just been observed, and the priority-first placement goes "
            "on with the group once it has placed one of its targets"
        ),
    )

    @field_validator("window_start", "window_deadline", mode="before")
    @classmethod
    def _as_unix(cls, value: float | int | str | datetime | None) -> float | None:
        if value is None:
            return None
        return unixtime_from(value)

    @model_validator(mode="after")
    def _check(self) -> ObservationGroup:
        if len(set(self.members)) != len(self.members):
            raise ValueError(f"group {self.name!r} lists a member twice")
        if (
            self.window_start is not None
            and self.window_deadline is not None
            and self.window_deadline <= self.window_start
        ):
            raise ValueError(f"group {self.name!r}'s window must end after it starts")
        return self

    @property
    def bounds(self) -> tuple[float | None, float | None]:
        """The window's start and deadline, each None where unset."""
        return self.window_start, self.window_deadline


class ObservationGroups(ConfigModel):
    """The observation groups of a mission, each target in at most one."""

    groups: list[ObservationGroup] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_group_each(self) -> ObservationGroups:
        seen: dict[int, str] = {}
        for group in self.groups:
            for obsid in group.members:
                if obsid in seen:
                    raise ValueError(
                        f"observation {obsid} is in groups {seen[obsid]!r} and "
                        f"{group.name!r}; a target belongs to one group"
                    )
                seen[obsid] = group.name
        return self

    def group_of(self, obsid: int) -> ObservationGroup | None:
        """The group ``obsid`` belongs to, or None."""
        for group in self.groups:
            if obsid in group.members:
                return group
        return None

    def __bool__(self) -> bool:
        return bool(self.groups)
