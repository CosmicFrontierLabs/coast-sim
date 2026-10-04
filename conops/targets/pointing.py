from datetime import datetime, timezone

from pydantic import (
    Field,
    PrivateAttr,
    computed_field,
    field_serializer,
    field_validator,
)

from ..common import unixtime2date
from ..common.enums import ObsType
from .merit import MeritBreakdown
from .plan_entry import PlanEntry


class Pointing(PlanEntry):
    """Define the basic parameters of an observing target with visibility checking."""

    obsid: int = 0
    name: str = "FakeTarget"
    merit: float = 100.0
    isat: bool = False
    # ``fom`` is maintained as a legacy alias for ``merit`` for
    # backwards compatibility (e.g. tests and older code). The
    # canonical field we use internally is ``merit`` which can be
    # recomputed each scheduling iteration by ``Queue.meritsort``.
    fom: float = Field(default=100.0, exclude=True)
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
    _done: bool = PrivateAttr(default=False)
    _merit_breakdown: MeritBreakdown | None = PrivateAttr(default=None)

    @field_validator("deadline", mode="before")
    @classmethod
    def _coerce_deadline(cls, v: float | int | str | datetime | None) -> float | None:
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

    @field_serializer("deadline")
    def _serialize_deadline(self, v: float | None) -> str | None:
        if v is None:
            return None
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()

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
        self._merit_breakdown = None
        self.begin = 0
        self.end = 0
        self.slewtime = 0
