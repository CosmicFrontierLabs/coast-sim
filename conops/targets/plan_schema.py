"""Pydantic schema for serialising :class:`~conops.targets.Plan` to / from JSON.

Usage
-----
Write a plan produced by a DITL run to disk::

    from conops.targets.plan_schema import PlanSchema

    schema = PlanSchema.from_plan(ditl.plan)
    schema.save("plan_20251201.json")

Load it back::

    schema = PlanSchema.load("plan_20251201.json")

Round-trip via ``model_validate`` with ``from_attributes=True`` (e.g. in tests)::

    schema = PlanSchema.model_validate(ditl.plan, from_attributes=True)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from .._version import __version__
from ..common.enums import ObsType
from .plan import Plan
from .plan_entry import PlanEntry


class PlanEntrySchema(BaseModel):
    """Pydantic representation of a single :class:`~conops.targets.PlanEntry`.

    All fields that are serialisable scalars from ``PlanEntry`` / ``Pointing``
    are captured here.  Ephemeris, config, and slew-path arrays are excluded
    because they are too large / not portable.
    """

    model_config = ConfigDict(from_attributes=True)

    name: str = ""
    ra: float = 0.0
    dec: float = 0.0
    roll: float = -1.0
    begin: float = 0.0
    end: float = 0.0
    merit: float = 0.0
    slewtime: int = 0
    insaa: int = 0
    obsid: int = 0
    obstype: ObsType = ObsType.PPT
    slewdist: float = 0.0
    ss_min: float = 300.0
    ss_max: float = 1_000_000.0
    exptime: int = 1000
    exporig: int = 1000
    isat: bool = False
    done: bool = False
    exposure: int = 0

    @field_validator("begin", "end", mode="before")
    @classmethod
    def _coerce_time(cls, v: Any) -> float:
        """Accept Unix timestamps (float/int) or ISO-8601 strings."""
        if isinstance(v, str):
            return datetime.fromisoformat(v).timestamp()
        return float(v)

    @field_serializer("begin", "end")
    def _serialize_time(self, v: float) -> str:
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()

    @model_validator(mode="before")
    @classmethod
    def _coerce_from_plan_entry(cls, data: Any) -> Any:
        """Accept either a dict (JSON round-trip) or a PlanEntry instance."""
        if isinstance(data, PlanEntry):
            return {
                "name": data.name,
                "ra": data.ra,
                "dec": data.dec,
                "roll": data.roll,
                "begin": data.begin,
                "end": data.end,
                "merit": data.merit,
                "slewtime": data.slewtime,
                "insaa": data.insaa,
                "obsid": data.obsid,
                "obstype": data.obstype,
                "slewdist": data.slewdist,
                "ss_min": data.ss_min,
                "ss_max": data.ss_max,
                "exptime": data.exptime,
                "exporig": data._exporig,
                "isat": getattr(data, "isat", False),
                "done": getattr(data, "done", False),
                "exposure": data.exposure,
            }
        return data


class PlanSchema(BaseModel):
    """Top-level schema for a serialised :class:`~conops.targets.Plan`.

    Contains metadata (version, timestamps, entry count) alongside the plan
    entries themselves.

    Attributes
    ----------
    version:
        Integer plan version.  Starts at 0 and is incremented automatically
        each time a new plan is saved to the same directory for the same
        time window.
    coast_sim_version:
        COASTSim package version that produced this file.
    created_at:
        ISO-8601 UTC timestamp of when the file was written.
    start:
        Unix timestamp of the first entry's ``begin`` time (or 0 if empty).
    end:
        Unix timestamp of the last entry's ``end`` time (or 0 if empty).
    num_entries:
        Total number of plan entries.
    entries:
        The serialised plan entries.
    """

    model_config = ConfigDict(from_attributes=True)

    version: int = 0
    coast_sim_version: str = Field(default_factory=lambda: __version__)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    start: float = 0.0
    end: float = 0.0
    num_entries: int = 0
    entries: list[PlanEntrySchema] = Field(default_factory=list)

    @field_validator("start", "end", mode="before")
    @classmethod
    def _coerce_time(cls, v: Any) -> float:
        """Accept Unix timestamps (float/int) or ISO-8601 strings."""
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v).timestamp()
            except ValueError:
                return float(v)
        return float(v)

    @field_serializer("start", "end")
    def _serialize_time(self, v: float) -> str:
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()

    @field_validator("version", mode="before")
    @classmethod
    def _coerce_version(cls, v: Any) -> int:
        """Accept integer versions; coerce legacy semver strings to 0."""
        if isinstance(v, int):
            return v
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    @model_validator(mode="before")
    @classmethod
    def _coerce_from_plan(cls, data: Any) -> Any:
        """Accept a raw :class:`Plan` object in addition to dicts."""
        if isinstance(data, Plan):
            entries = list(data.entries)
            return {
                "version": 0,
                "coast_sim_version": __version__,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "start": entries[0].begin if entries else 0.0,
                "end": entries[-1].end if entries else 0.0,
                "num_entries": len(entries),
                "entries": entries,
            }
        return data

    @model_validator(mode="after")
    def _sync_num_entries(self) -> PlanSchema:
        """Keep num_entries consistent with the actual entries list."""
        self.num_entries = len(self.entries)
        return self

    # ── Convenience constructors ───────────────────────────────────────────────

    @classmethod
    def from_plan(cls, plan: Plan) -> PlanSchema:
        """Build a :class:`PlanSchema` from a :class:`Plan` instance."""
        return cls.model_validate(plan, from_attributes=True)

    # ── I/O ───────────────────────────────────────────────────────────────────

    def _default_filename(self) -> str:
        """Generate a filename from plan metadata.

        Format: ``plan_<start>_<end>_v<version>.json``
        e.g.   ``plan_20251201T000000Z_20251201T235900Z_v3.json``
        """

        def _fmt(ts: float) -> str:
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )

        start_str = _fmt(self.start) if self.start else "unknown"
        end_str = _fmt(self.end) if self.end else "unknown"
        return f"plan_{start_str}_{end_str}_v{self.version}.json"

    def _next_version(self, directory: Path) -> int:
        """Scan *directory* for existing plan files and return the next version number.

        Looks for files matching ``plan_<start>_<end>_v<N>.json`` where ``N``
        is a non-negative integer.  Returns ``max(N) + 1``, or ``0`` if no
        matching files exist.
        """

        def _fmt(ts: float) -> str:
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )

        start_str = _fmt(self.start) if self.start else "unknown"
        end_str = _fmt(self.end) if self.end else "unknown"
        pattern = re.compile(
            rf"^plan_{re.escape(start_str)}_{re.escape(end_str)}_v(\d+)\.json$"
        )
        versions = [
            int(m.group(1))
            for f in directory.iterdir()
            if f.is_file() and (m := pattern.match(f.name))
        ]
        return max(versions) + 1 if versions else 0

    def save(self, path: str | Path, *, indent: int = 2) -> Path:
        """Serialise the plan to a JSON file.

        Parameters
        ----------
        path:
            Destination file path **or** directory.  When a directory is given
            (path ends with ``/`` or already exists as a directory) a filename
            is generated automatically: ``plan_<start>_<end>_v<N>.json`` where
            ``N`` is one higher than the highest existing integer-versioned plan
            file for the same time window in that directory.
            Parent directories are created automatically.
        indent:
            JSON indentation level (default 2).

        Returns
        -------
        Path
            The resolved path of the written file.
        """
        dest = Path(path)
        if str(path).endswith("/") or dest.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            next_ver = self._next_version(dest)
            schema = self.model_copy(update={"version": next_ver})
            dest = dest / schema._default_filename()
        else:
            schema = self
            dest.parent.mkdir(parents=True, exist_ok=True)
        payload = schema.model_dump(mode="json")
        dest.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
        return dest.resolve()

    @classmethod
    def load(cls, path: str | Path) -> PlanSchema:
        """Load a :class:`PlanSchema` from a JSON file.

        Parameters
        ----------
        path:
            Source file path.

        Returns
        -------
        PlanSchema
            The deserialised schema object.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)
