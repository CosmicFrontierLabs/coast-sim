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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .._version import __version__
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
    obstype: str = "PPT"
    slewdist: float = 0.0
    ss_min: float = 300.0
    ss_max: float = 1_000_000.0
    exptime: int = 1000
    exporig: int = 1000
    isat: bool = False
    done: bool = False
    exposure: int = 0

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
        COASTSim package version that produced this file.
    created_at:
        ISO-8601 UTC timestamp of when the schema was created.
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

    version: str = Field(default_factory=lambda: __version__)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    start: float = 0.0
    end: float = 0.0
    num_entries: int = 0
    entries: list[PlanEntrySchema] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_from_plan(cls, data: Any) -> Any:
        """Accept a raw :class:`Plan` object in addition to dicts."""
        if isinstance(data, Plan):
            entries = list(data.entries)
            return {
                "version": __version__,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "start": entries[0].begin if entries else 0.0,
                "end": entries[-1].end if entries else 0.0,
                "num_entries": len(entries),
                "entries": entries,
            }
        return data

    # ── Convenience constructors ───────────────────────────────────────────────

    @classmethod
    def from_plan(cls, plan: Plan) -> PlanSchema:
        """Build a :class:`PlanSchema` from a :class:`Plan` instance."""
        return cls.model_validate(plan, from_attributes=True)

    # ── I/O ───────────────────────────────────────────────────────────────────

    def save(self, path: str | Path, *, indent: int = 2) -> Path:
        """Serialise the plan to a JSON file.

        Parameters
        ----------
        path:
            Destination file path.  Parent directories must already exist.
        indent:
            JSON indentation level (default 2).

        Returns
        -------
        Path
            The resolved path of the written file.
        """
        dest = Path(path)
        payload = self.model_dump(mode="json")
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
