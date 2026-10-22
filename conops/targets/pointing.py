from datetime import datetime, timezone

from pydantic import (
    Field,
    PrivateAttr,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from ..common import unixtime2date
from ..common.enums import ObsType
from ..config.acs import AttitudeControlSystem
from ..config.observation_timing import ObservationTiming
from .merit import MeritBreakdown
from .plan_entry import PlanEntry
from .sequence import ObservationSequence


class Pointing(PlanEntry):
    """Define the basic parameters of an observing target with visibility checking."""

    obsid: int = 0
    name: str = "FakeTarget"
    merit: float = 100.0
    isat: bool = False
    obstype: ObsType = ObsType.AT
    roll: float = 0.0
    deadline: float | None = Field(
        default=None,
        allow_inf_nan=False,
        exclude_if=lambda v: v is None,
        description=(
            "Latest time (Unix seconds) science collection may begin; the "
            "target cannot be selected after it"
        ),
    )
    earliest_start: float | None = Field(
        default=None,
        allow_inf_nan=False,
        exclude_if=lambda v: v is None,
        description=(
            "Earliest time (Unix seconds) science collection may begin; the "
            "target cannot be selected before it"
        ),
    )
    collection_limit: float | None = Field(
        default=None,
        allow_inf_nan=False,
        exclude=True,
        description=(
            "Seconds of collection the visit selected now may have, set by "
            "the queue when the recorder holds less than the whole snapshot; "
            "None for the whole snapshot"
        ),
    )
    sequence: ObservationSequence | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description=(
            "The steps a visit of this target observes, in order; None for a "
            "plain pointing held for a snapshot"
        ),
    )
    visits: int = Field(
        default=1,
        ge=1,
        exclude_if=lambda v: v == 1,
        description="How many visits of the sequence are wanted",
    )
    roll_deg: float | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description=(
            "The instrument roll to observe at, in degrees, instead of the "
            "power-optimal roll; the target is observed only when the "
            "constraints allow it and it is within the ACS roll limit"
        ),
    )
    roll_offset_deg: float | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description=(
            "An offset from the power-optimal roll to observe at, in degrees; "
            "it must be within the ACS roll limit"
        ),
    )
    _done: bool = PrivateAttr(default=False)
    _merit_breakdown: MeritBreakdown | None = PrivateAttr(default=None)

    @field_validator("deadline", "earliest_start", mode="before")
    @classmethod
    def _coerce_request_time(
        cls, v: float | int | str | datetime | None
    ) -> float | None:
        """Accept Unix timestamps, datetimes or ISO-8601 strings."""
        if v is None:
            return None
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                v = v.replace(tzinfo=timezone.utc)
            return v.timestamp()
        return float(v)

    @field_serializer("deadline", "earliest_start")
    def _serialize_request_time(self, v: float | None) -> str | None:
        if v is None:
            return None
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_fom(cls, data: object) -> object:
        """Accept ``fom``, the legacy name of ``merit``."""
        if isinstance(data, dict) and "fom" in data:
            data = dict(data)
            fom = data.pop("fom")
            data.setdefault("merit", fom)
        return data

    @property
    def fom(self) -> float:
        """Legacy alias of :attr:`merit`."""
        return self.merit

    @fom.setter
    def fom(self, value: float) -> None:
        self.merit = value

    @model_validator(mode="after")
    def _sequence_sets_the_snapshot(self) -> "Pointing":
        """A sequence target's snapshot is one visit's science: no more and
        no less, and its exposure is its visits' science unless given."""
        if self.sequence is not None:
            science = self.sequence.science_seconds
            self.ss_min = science
            self.ss_max = science
            if self.exptime is None:
                self.exptime = self.visits * science
        return self

    def step_pointing(self, index: int) -> "Pointing":
        """The pointing step ``index`` of this target's sequence observes: at
        the step's pointing and with its instrument, for its duration, carrying
        this target's obsid and merit."""
        assert self.sequence is not None
        step = self.sequence.steps[index]
        ra, dec = self.sequence.pointing_of(index, self.ra, self.dec)
        base = timing_for(self.config, int(self.obsid))
        roll_deg, roll_offset = self._step_roll_request(step.roll_offset_deg)
        pointing = Pointing(
            config=self.config,
            ra=ra,
            dec=dec,
            obsid=self.obsid,
            name=f"{self.name}: {step.name}",
            merit=self.merit,
            instrument_name=(
                step.instrument_name
                if step.instrument_name is not None
                else self.instrument_name
            ),
            science_data=dict(self.science_data) if step.science else {},
            # A step is never selected on its own, so it has no minimum: what
            # it collects to the tick is kept, and the visit checks the rest.
            ss_min=0,
            ss_max=step.duration_seconds,
            step=step.name,
            calibration=not step.science,
            roll_deg=roll_deg,
            roll_offset_deg=roll_offset,
        )
        pointing.exptime = step.duration_seconds
        pointing.merit_breakdown = self.merit_breakdown
        pointing._step_timing = self.sequence.step_timing(base, index)
        if self.ephem is not None:
            pointing.visibility()
        return pointing

    def _step_roll_request(
        self, offset: float | None
    ) -> tuple[float | None, float | None]:
        """The roll request a step carries, as ``(roll_deg, roll_offset_deg)``:
        this target's, with the step's offset added to an absolute roll or
        to the offset."""
        if offset is None:
            return self.roll_deg, self.roll_offset_deg
        if self.roll_deg is not None:
            return (self.roll_deg + offset) % 360.0, None
        return None, (self.roll_offset_deg or 0.0) + offset

    def sequence_overhead_seconds(
        self, acs: AttitudeControlSystem, timing: ObservationTiming
    ) -> float:
        """Seconds a visit takes beyond its science and the visit's own setup
        and cleanup; 0 for a plain pointing."""
        if self.sequence is None:
            return 0.0
        return self.sequence.overhead_seconds(acs, timing, self.ra, self.dec)

    @model_validator(mode="after")
    def _one_roll_request(self) -> "Pointing":
        if self.roll_deg is not None and self.roll_offset_deg is not None:
            raise ValueError("give roll_deg or roll_offset_deg, not both")
        return self

    @property
    def commands_roll(self) -> bool:
        """Whether this target asks for a roll other than the power-optimal one."""
        return self.roll_deg is not None or self.roll_offset_deg is not None

    def requested_roll(self, optimum: float, max_delta: float | None) -> float | None:
        """The roll this target is observed at, given the power-optimal roll
        ``optimum`` and the ACS limit ``max_delta`` from it.

        The optimum itself without a request; the optimum plus the offset;
        or the absolute roll, where it is within the limit, else None: the
        target cannot be observed at that time. An offset beyond the limit
        is a configuration error.
        """
        if self.roll_offset_deg is not None:
            if max_delta is not None and abs(self.roll_offset_deg) > max_delta + 1e-9:
                raise ValueError(
                    f"roll_offset_deg {self.roll_offset_deg} exceeds the ACS "
                    f"max_roll_from_optimum_deg {max_delta}"
                )
            return (optimum + self.roll_offset_deg) % 360.0
        if self.roll_deg is not None:
            roll = self.roll_deg % 360.0
            if max_delta is not None:
                delta = abs((roll - optimum + 180.0) % 360.0 - 180.0)
                if delta > max_delta + 1e-9:
                    return None
            return roll
        return optimum

    @model_validator(mode="after")
    def _check_start_before_deadline(self) -> "Pointing":
        if (
            self.earliest_start is not None
            and self.deadline is not None
            and self.earliest_start > self.deadline
        ):
            raise ValueError("earliest_start must not be after deadline")
        return self

    @property
    def merit_breakdown(self) -> MeritBreakdown | None:
        """Merit terms evaluated when this target was last selected."""
        return self._merit_breakdown

    @merit_breakdown.setter
    def merit_breakdown(self, breakdown: MeritBreakdown | None) -> None:
        self._merit_breakdown = breakdown

    def in_sun(self, utime: float) -> bool:
        """Is this target in Sun constraint?"""
        assert self.config is not None, "Config must be set to evaluate constraints"
        return self.config.constraint.in_sun(
            self.ra, self.dec, utime, target_roll=self.roll
        )

    def in_earth(self, utime: float) -> bool:
        """Is this target in Earth constraint?"""
        assert self.config is not None, "Config must be set to evaluate constraints"
        return self.config.constraint.in_earth(
            self.ra, self.dec, utime, target_roll=self.roll
        )

    def in_moon(self, utime: float) -> bool:
        """Is this target in Moon constraint?"""
        assert self.config is not None, "Config must be set to evaluate constraints"
        return self.config.constraint.in_moon(
            self.ra, self.dec, utime, target_roll=self.roll
        )

    def in_panel(self, utime: float) -> bool:
        """Is this target in Panel constraint?"""
        assert self.config is not None, "Config must be set to evaluate constraints"
        return self.config.constraint.in_panel(
            self.ra, self.dec, utime, target_roll=self.roll
        )

    def in_orbit(self, utime: float) -> bool:
        """Is this target in Orbit constraint?"""
        assert self.config is not None, "Config must be set to evaluate constraints"
        return self.config.constraint.in_orbit(
            self.ra, self.dec, utime, target_roll=self.roll
        )

    def in_star_tracker_hard(self, utime: float, acs_mode: int | None = None) -> bool:
        """Is this target in star tracker hard constraint?"""
        assert self.config is not None, "Config must be set to evaluate constraints"
        return self.config.constraint.in_star_tracker_hard(
            self.ra, self.dec, utime, target_roll=self.roll, acs_mode=acs_mode
        )

    def in_star_tracker_soft(self, utime: float, acs_mode: int | None = None) -> bool:
        """Is this target in star tracker soft constraint?"""
        assert self.config is not None, "Config must be set to evaluate constraints"
        return self.config.constraint.in_star_tracker_soft(
            self.ra, self.dec, utime, target_roll=self.roll, acs_mode=acs_mode
        )

    def __str__(self) -> str:
        return f"{unixtime2date(self.begin)} {self.name} ({self.obsid}) RA={self.ra:.4f}, Dec={self.dec:4f}, Roll={self.roll:.1f}, Merit={self.merit}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def done(self) -> bool:
        if self.exptime is not None and self.exptime <= 0:
            self._done = True
        return self._done

    @done.setter
    def done(self, v: bool) -> None:
        self._done = v

    def reset(self) -> None:
        if self._exporig is not None:
            self._exptime = self._exporig
        self.done = False
        self._collected_seconds = 0.0
        self._last_collection_time = None
        self._interruptions = []
        self._merit_breakdown = None
        self.collection_limit = None
        self.begin = 0
        self.end = 0
        self.slewtime = 0
