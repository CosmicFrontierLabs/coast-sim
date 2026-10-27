"""A Target of Opportunity: a request with a trigger."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..common import unixtime_from
from ..config.data_generator import ScienceData
from .entries import AnyEntry, with_kind
from .plan_entry import PlanEntry
from .pointing import Pointing

PLAIN_SNAPSHOT_MIN_S = 300
"""The least a plain ToO's snapshot collects, as the dispatch queue gives a
pointing built from flat fields."""
UNBOUNDED_SNAPSHOT_S = 86400
"""The snapshot bound of a pointing given none: a day."""

_REQUEST_FIELDS = (
    "obsid",
    "ra",
    "dec",
    "merit",
    "exptime",
    "name",
    "science_data",
    "instrument_modes",
)


class TOORequest(BaseModel):
    """A Target of Opportunity: any request, with the time it is submitted,
    the latest time its collection may begin, and whether it may cut short
    the observation in progress. The request is a :class:`Pointing` or a
    :class:`MovingPointing`, with everything a request can have: a sequence,
    an instrument, its own timing, a roll offset, a track or a body.

    The request's own ``deadline`` is taken as the ToO's when none is given
    here, and the ToO's deadline is written onto the request when the run
    takes it up. The flat fields a ToO was once given (``obsid``, ``ra``,
    ``dec``, ``merit``, ``exptime``, ``name``, ``science_data``,
    ``instrument_modes``) still build a plain pointing, bound to a ``config``
    when one is given with them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: AnyEntry
    submit_time: float = Field(
        default=0.0,
        description=(
            "When the ToO becomes active, in Unix seconds; zero for active "
            "from the start of the simulation"
        ),
    )
    deadline: float | None = Field(
        default=None,
        description="The latest time, in Unix seconds, its collection may begin",
    )
    interrupt: bool = Field(
        default=True,
        description=(
            "Whether the ToO may cut short the observation in progress; if False "
            "it waits for that observation to end"
        ),
    )
    executed: bool = Field(default=False, exclude=True)
    """Run state: whether the ToO has been taken up. Not written to a file."""

    @model_validator(mode="before")
    @classmethod
    def _request_from_flat_fields(cls, data: object) -> object:
        """Build the request from the flat fields when none is given."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "request" not in data:
            fields = {name: data.pop(name) for name in _REQUEST_FIELDS if name in data}
            exptime = fields.pop("exptime", None)
            config = data.pop("config", None)
            # A plain pointing observed in one snapshot, as the dispatch queue
            # builds one from the same fields; bound to the configuration when
            # one is given, so its body attitude is the run's.
            request = Pointing(
                config=config,
                ss_min=PLAIN_SNAPSHOT_MIN_S,
                ss_max=UNBOUNDED_SNAPSHOT_S,
                **fields,
            )
            if exptime is not None:
                request.exptime = exptime
            data["request"] = request
        elif isinstance(data["request"], dict):
            data["request"] = with_kind(data["request"], "pointing")
        return data

    @field_validator("submit_time", mode="before")
    @classmethod
    def _coerce_submit_time(cls, v: float | int | str | datetime | None) -> float:
        return 0.0 if v is None else unixtime_from(v)

    @field_validator("deadline", mode="before")
    @classmethod
    def _coerce_deadline(cls, v: float | int | str | datetime | None) -> float | None:
        return None if v is None else unixtime_from(v)

    @model_validator(mode="after")
    def _deadline_from_request(self) -> TOORequest:
        request = self.request
        if self.deadline is None and getattr(request, "deadline", None) is not None:
            self.deadline = float(request.deadline)  # type: ignore[attr-defined]
        return self

    # The request's fields, as the dispatch and rolling runs read them, and
    # set on the request when a caller changes a ToO before a run.
    @property
    def obsid(self) -> int:
        return int(self.request.obsid)

    @obsid.setter
    def obsid(self, value: object) -> None:
        self.request.obsid = int(value)  # type: ignore[call-overload]

    @property
    def ra(self) -> float:
        return float(self.request.ra)

    @ra.setter
    def ra(self, value: object) -> None:
        self.request.ra = float(value)  # type: ignore[arg-type]

    @property
    def dec(self) -> float:
        return float(self.request.dec)

    @dec.setter
    def dec(self, value: object) -> None:
        self.request.dec = float(value)  # type: ignore[arg-type]

    @property
    def merit(self) -> float:
        return float(getattr(self.request, "merit", 0.0))

    @merit.setter
    def merit(self, value: object) -> None:
        if isinstance(self.request, Pointing):
            self.request.merit = float(value)  # type: ignore[arg-type]

    @property
    def exptime(self) -> int:
        exptime = self.request.exptime
        return int(exptime) if exptime is not None else 0

    @exptime.setter
    def exptime(self, value: object) -> None:
        self.request.exptime = int(value)  # type: ignore[call-overload]

    @property
    def name(self) -> str:
        return str(self.request.name)

    @name.setter
    def name(self, value: object) -> None:
        self.request.name = str(value)

    @property
    def science_data(self) -> Mapping[str, ScienceData]:
        return self.request.science_data

    @property
    def instrument_modes(self) -> Mapping[str, str]:
        return self.request.instrument_modes

    def target_for(self, config: object) -> PlanEntry:
        """A copy of the request bound to ``config``, its constraint and its
        ephemeris, for a run to take up, with the ToO's deadline."""
        # Rebuilt from its own fields: a fresh object with its own state,
        # where a deep copy would carry the bound ephemeris along.
        target = type(self.request).model_validate(self.request.model_dump())
        target.config = config  # type: ignore[assignment]
        # Bound as a program binds a request, so its visibility can be computed.
        constraint = getattr(config, "constraint", None)
        if constraint is not None:
            target.constraint = constraint
            target.ephem = getattr(constraint, "ephem", None)
        bus = getattr(config, "spacecraft_bus", None)
        if bus is not None:
            target.acs_config = getattr(bus, "attitude_control", None)
        if isinstance(target, Pointing):
            target.deadline = self.deadline
        return target
