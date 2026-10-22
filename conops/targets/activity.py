"""Scheduled activities: fixed-time blocks with no science.

A maintenance window, a calibration that needs the spacecraft still, or a
block set aside for an engineering test is a :class:`ScheduledActivity`: it
has a start and a duration, the spacecraft holds the attitude it has in
``IDLE`` mode for its length, and no science is observed meanwhile. The
schedulers reserve activities as fixed blocks, as they reserve ground passes,
and both simulations execute them: dispatch ends the observation under way at
the activity's start and starts none until it ends, and the planned
simulation commands the plan's activity entries.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..common import ObsType
from ..config import MissionConfig
from .plan_entry import PlanEntry

ACTIVITY_OBSID = 998000
"""Obsids given to activities without one, in order."""


class ScheduledActivity(BaseModel):
    """A fixed-time block during which no science is observed."""

    model_config = ConfigDict(frozen=True)

    name: str
    begin: float = Field(description="When the activity starts, in Unix seconds")
    duration_seconds: float = Field(gt=0.0)
    obsid: int | None = Field(
        default=None,
        description="Identifies the activity's plan entries; assigned if None",
    )

    @field_validator("begin", mode="before")
    @classmethod
    def _begin_as_unix(cls, value: float | datetime) -> float:
        if isinstance(value, datetime):
            return value.timestamp()
        return float(value)

    @property
    def end(self) -> float:
        return self.begin + self.duration_seconds

    def plan_entry(
        self,
        config: MissionConfig | None,
        attitude: tuple[float, float, float],
        *,
        begin: float | None = None,
    ) -> PlanEntry:
        """The plan entry for this activity holding ``attitude``.

        ``begin`` is when the entry starts, where a slew to the attitude
        comes first; the hold runs from the activity's own start.
        """
        assert self.obsid is not None, "assign_obsids first"
        entry = PlanEntry(config=config)
        entry.name = self.name
        entry.obsid = self.obsid
        entry.obstype = ObsType.ACTIVITY
        entry.ra, entry.dec, entry.roll = attitude
        entry.begin = self.begin if begin is None else begin
        entry.end = self.end
        entry.ss_min = 0
        entry.ss_max = self.duration_seconds
        entry.exptime = self.duration_seconds
        return entry


def assign_obsids(
    activities: Sequence[ScheduledActivity],
) -> list[ScheduledActivity]:
    """The activities in time order, each with an obsid, given from
    :data:`ACTIVITY_OBSID` on where one has none."""
    assigned = []
    for n, activity in enumerate(sorted(activities, key=lambda a: a.begin)):
        if activity.obsid is None:
            activity = activity.model_copy(update={"obsid": ACTIVITY_OBSID + n})
        assigned.append(activity)
    return assigned
