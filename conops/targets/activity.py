"""Scheduled activities: fixed-time blocks with no science.

A maintenance window, a calibration that needs the spacecraft still, or a
block set aside for an engineering test is a :class:`ScheduledActivity`: it
has a start and a duration, the spacecraft holds the attitude it has in
``IDLE`` mode for its length, and no science is observed meanwhile. The
schedulers reserve activities as fixed blocks, as they reserve ground passes,
and both simulations execute them: dispatch ends the observation under way at
the activity's start, starts none whose slew or collection would run into it,
and starts none until it ends; the planned simulation commands the plan's
activity entries.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    instrument_modes: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "The instruments the activity uses and the mode each is in, by "
            "name, such as a calibration lamp's; instruments not named idle"
        ),
    )
    data_gb: float = Field(
        default=0.0,
        ge=0.0,
        description="Data the activity puts on the recorder, in Gb, such as its frames",
    )
    obsid: int | None = Field(
        default=None,
        description="Identifies the activity's plan entries; assigned if None",
    )
    fixed: bool = Field(
        default=False,
        description=(
            "Whether dispatch begins the activity at its start even while a "
            "slew is under way, as a maintenance window must. False: a slew "
            "under way at the start delays the activity to its end, and the "
            "activity keeps its length"
        ),
    )
    slide_seconds: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "How long after its start the activity may begin, to let an "
            "observation under way finish: dispatch waits for one up to the "
            "latest start, then begins and ends what is under way, and the "
            "planners let science before the activity run into the window. "
            "The activity keeps its length. 0: it begins at its start"
        ),
    )

    @field_validator("begin", mode="before")
    @classmethod
    def _begin_as_unix(cls, value: float | datetime) -> float:
        if isinstance(value, datetime):
            return value.timestamp()
        return float(value)

    @model_validator(mode="after")
    def _fixed_does_not_slide(self) -> ScheduledActivity:
        if self.fixed and self.slide_seconds > 0.0:
            raise ValueError("a fixed activity begins at its start and cannot slide")
        return self

    @property
    def end(self) -> float:
        return self.begin + self.duration_seconds

    @property
    def latest_start(self) -> float:
        """The latest the activity may begin: its start plus its slide."""
        return self.begin + self.slide_seconds

    @property
    def slides(self) -> bool:
        return self.slide_seconds > 0.0

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
        if config is not None:
            # An activity uses what it names and nothing else, and does not
            # point, so no guider is required of it.
            config.payload.check_instrument_modes(self.instrument_modes)
            config.payload.check_concurrency(self.instrument_modes, pointing=False)
        entry = PlanEntry(config=config)
        entry.name = self.name
        entry.instrument_modes = dict(self.instrument_modes)
        entry.fixed_data_gb = float(self.data_gb)
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


def periodic_activities(
    name: str,
    first: float | datetime,
    interval_seconds: float,
    duration_seconds: float,
    *,
    count: int | None = None,
    until: float | datetime | None = None,
    instrument_modes: dict[str, str] | None = None,
    data_gb: float = 0.0,
    fixed: bool = False,
    slide_seconds: float = 0.0,
) -> list[ScheduledActivity]:
    """Activities named ``name`` every ``interval_seconds`` from ``first``,
    each ``duration_seconds`` long, ``count`` of them or as many as begin by
    ``until``; the instrument modes, data, ``fixed`` and ``slide_seconds``
    apply to each. A campaign's twice-daily calibration is one call."""
    if interval_seconds <= 0.0:
        raise ValueError("interval_seconds must be positive")
    if (count is None) == (until is None):
        raise ValueError("give count or until, not both")
    start = first.timestamp() if isinstance(first, datetime) else float(first)
    stop = until.timestamp() if isinstance(until, datetime) else until
    activities: list[ScheduledActivity] = []
    k = 0
    while (count is None or k < count) and (
        stop is None or start + k * interval_seconds <= stop
    ):
        activities.append(
            ScheduledActivity(
                name=name,
                begin=start + k * interval_seconds,
                duration_seconds=duration_seconds,
                instrument_modes=dict(instrument_modes or {}),
                data_gb=data_gb,
                fixed=fixed,
                slide_seconds=slide_seconds,
            )
        )
        k += 1
    return activities
