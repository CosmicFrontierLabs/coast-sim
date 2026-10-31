"""The science program a run schedules: the requests, the Targets of
Opportunity with their triggers, the scheduled activities and the
observation groups, for a window, in one file."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import rust_ephem
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from .._version import __version__
from ..common import unixtime_from
from ..config import MissionConfig
from ..config.observation_groups import ObservationGroups
from .activity import ScheduledActivity
from .entries import AnyEntry, with_kind
from .plan_entry import PlanEntry
from .pointing import MovingPointing
from .too import TOORequest


class Program(BaseModel):
    """What a run schedules, as :class:`Plan` is what a run executed: the
    request pool, the Targets of Opportunity, the scheduled activities and
    the observation groups, written for a window. Every time in the file is
    absolute, so a program is tied to its window.

    :meth:`save` writes the requests without their execution fields (see
    :attr:`PlanEntry.EXECUTION_FIELDS`), since a request written is the
    request, not a run's state; :meth:`load` reads each request as its own
    class by its ``kind``; :meth:`bind` gives every request the mission
    configuration and the ephemeris, and the program's groups to the
    configuration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    version: int = 0
    coast_sim_version: str = Field(default_factory=lambda: __version__)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    name: str = ""
    description: str = ""
    begin: float = Field(description="The window's start, in Unix seconds")
    end: float = Field(description="The window's end, in Unix seconds")
    roll_axis: Literal["+X"] = "+X"
    roll_convention: Literal["right_handed_body_rotation"] = (
        "right_handed_body_rotation"
    )
    roll_reference_axis: Literal["+Z"] = "+Z"
    roll_reference: Literal["projected_celestial_north"] = "projected_celestial_north"
    requests: list[SerializeAsAny[AnyEntry]] = Field(default_factory=list)
    toos: list[TOORequest] = Field(default_factory=list)
    activities: list[ScheduledActivity] = Field(default_factory=list)
    observation_groups: ObservationGroups = Field(default_factory=ObservationGroups)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_requests(self) -> int:
        return len(self.requests)

    @field_validator("begin", "end", mode="before")
    @classmethod
    def _coerce_time(cls, v: float | int | str | datetime) -> float:
        return unixtime_from(v)

    @field_serializer("begin", "end")
    def _serialize_time(self, v: float) -> str:
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()

    @model_validator(mode="before")
    @classmethod
    def _requests_with_kinds(cls, data: object) -> object:
        if isinstance(data, dict) and isinstance(data.get("requests"), list):
            data = {
                **data,
                "requests": [with_kind(r, "pointing") for r in data["requests"]],
            }
        return data

    @model_validator(mode="after")
    def _window_in_order(self) -> Program:
        if self.end <= self.begin:
            raise ValueError("a program's window must end after it begins")
        return self

    def __iter__(self) -> Iterator[PlanEntry]:  # type: ignore[override]
        return iter(self.requests)

    def __len__(self) -> int:
        return len(self.requests)

    def bind(
        self,
        config: MissionConfig,
        ephem: rust_ephem.Ephemeris | None = None,
        *,
        refresh_tracks: bool = False,
    ) -> None:
        """Give every request the configuration and the ephemeris, as
        :meth:`Plan.bind_runtime` does a plan's entries, and the program's
        observation groups to the configuration, where a run reads them.
        With ``refresh_tracks``, each moving request that names a body looks
        its track up over the ephemeris, keeping the track it was given
        where the lookup fails (see :meth:`MovingPointing.refresh_track`);
        the tracks looked up are saved with the program."""
        runtime_ephem = ephem if ephem is not None else config.constraint.ephem
        if runtime_ephem is None:
            raise ValueError("an ephemeris is required to bind a program")
        config.constraint.ephem = runtime_ephem
        if self.observation_groups.groups:
            config.observation_groups = self.observation_groups
        for request in self.requests:
            request.config = config
            request.constraint = config.constraint
            request.ephem = runtime_ephem
            request.acs_config = config.spacecraft_bus.attitude_control
        for too in self.toos:
            too.request.config = config
            too.request.constraint = config.constraint
            too.request.ephem = runtime_ephem
            too.request.acs_config = config.spacecraft_bus.attitude_control
        if refresh_tracks:
            for request in self.requests:
                if isinstance(request, MovingPointing) and request.body:
                    request.refresh_track()

    def save(self, path: str | Path) -> Path:
        """Write the program as JSON, the requests without their execution
        fields, and return the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(
            mode="json",
            exclude={
                "requests": {"__all__": set(PlanEntry.EXECUTION_FIELDS)},
                "toos": {"__all__": {"request": set(PlanEntry.EXECUTION_FIELDS)}},
            },
        )
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Program:
        """Read a program written by :meth:`save`, each request as its own
        class."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)
