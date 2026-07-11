from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    SerializeAsAny,
    computed_field,
    field_serializer,
    field_validator,
)

from .._version import __version__
from .plan_entry import PlanEntry


class TargetList(BaseModel):
    """List of potential targets for Spacecraft to observe"""

    targets: list[PlanEntry] = Field(default_factory=list)

    def __getitem__(self, number: int) -> PlanEntry:
        return self.targets[number]

    def __iter__(self) -> Iterator[PlanEntry]:  # type: ignore[override]
        return iter(self.targets)

    def add_target(self, plan_entry: PlanEntry) -> None:
        self.targets.append(plan_entry)

    def __len__(self) -> int:
        return len(self.targets)


class AttitudeSampleSchema(BaseModel):
    """A single executed spacecraft attitude sample for plan inspection."""

    utime: float
    timestamp: str
    ra: float | None = None
    dec: float | None = None
    roll: float | None = None
    mode: str | None = None
    obsid: int | None = None
    quat_w: float | None = None
    quat_x: float | None = None
    quat_y: float | None = None
    quat_z: float | None = None


class AttitudeTimeseriesSchema(BaseModel):
    """Continuous executed spacecraft attitude timeline tied to a plan file."""

    version: int = 0
    coast_sim_version: str = Field(default_factory=lambda: __version__)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    plan_file: str | None = None
    plan_version: int | None = None
    plan_start: float | None = None
    plan_end: float | None = None
    samples: list[AttitudeSampleSchema] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @field_validator("plan_start", "plan_end", mode="before")
    @classmethod
    def _coerce_optional_time(cls, v: float | int | str | None) -> float | None:
        if v is None:
            return None
        if isinstance(v, str):
            return datetime.fromisoformat(v).timestamp()
        return float(v)

    @field_serializer("plan_start", "plan_end")
    def _serialize_optional_time(self, v: float | None) -> str | None:
        if v is None:
            return None
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()


class OrbitStateSampleSchema(BaseModel):
    """A single spacecraft orbit-state sample for plan inspection."""

    utime: float
    timestamp: str
    position_km: tuple[float, float, float]
    velocity_km_s: tuple[float, float, float]


class OrbitStateTimeseriesSchema(BaseModel):
    """Continuous spacecraft orbit-state timeline tied to a plan file."""

    version: int = 0
    coast_sim_version: str = Field(default_factory=lambda: __version__)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    frame: Literal["GCRS"] = "GCRS"
    origin: Literal["Earth center"] = "Earth center"
    position_unit: Literal["km"] = "km"
    velocity_unit: Literal["km/s"] = "km/s"
    component_order: Literal["x_y_z"] = "x_y_z"
    plan_file: str | None = None
    plan_version: int | None = None
    plan_start: float | None = None
    plan_end: float | None = None
    samples: list[OrbitStateSampleSchema] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @field_validator("plan_start", "plan_end", mode="before")
    @classmethod
    def _coerce_optional_time(cls, v: float | int | str | None) -> float | None:
        if v is None:
            return None
        if isinstance(v, str):
            return datetime.fromisoformat(v).timestamp()
        return float(v)

    @field_serializer("plan_start", "plan_end")
    def _serialize_optional_time(self, v: float | None) -> str | None:
        if v is None:
            return None
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()


class Plan(BaseModel):
    """Simple Plan class.

    Also serves as the serialisable envelope for JSON export/import: see
    :meth:`save` and :meth:`load`.

    Attributes
    ----------
    version:
        Integer plan version.  Starts at 0 and is incremented automatically
        each time a new plan is saved to the same directory for the same
        time window.
    coast_sim_version:
        COASTSim package version that produced this file.
    created_at:
        ISO-8601 UTC timestamp of when this plan instance was created.
    start:
        ISO-8601 timestamp of the first entry's ``begin`` time (or epoch if
        empty).  Always derived live from ``entries`` — never stale.
    end:
        ISO-8601 timestamp of the last entry's ``end`` time (or epoch if
        empty).  Always derived live from ``entries`` — never stale.
    attitude_timeseries_file:
        Optional sibling JSON file containing the executed attitude samples
        associated with this exact plan export.
    orbit_state_timeseries_file:
        Optional sibling JSON file containing the GCRS spacecraft position and
        velocity samples associated with this exact plan export.
    metadata:
        Optional producer-specific plan provenance and validation metadata.
    """

    version: int = 0
    coast_sim_version: str = Field(default_factory=lambda: __version__)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    attitude_timeseries_file: str | None = None
    orbit_state_timeseries_file: str | None = None
    metadata: dict[str, object] | None = None
    entries: list[SerializeAsAny[PlanEntry]] = Field(default_factory=list)
    attitude_timeseries: AttitudeTimeseriesSchema | None = Field(
        default=None, exclude=True
    )
    orbit_state_timeseries: OrbitStateTimeseriesSchema | None = Field(
        default=None, exclude=True
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_entries(self) -> int:
        return len(self.entries)

    @property
    def _start_ts(self) -> float:
        return self.entries[0].begin if self.entries else 0.0

    @property
    def _end_ts(self) -> float:
        return self.entries[-1].end if self.entries else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def start(self) -> str:
        """ISO-8601 timestamp of the first entry's ``begin`` time, live."""
        return datetime.fromtimestamp(self._start_ts, tz=timezone.utc).isoformat()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def end(self) -> str:
        """ISO-8601 timestamp of the last entry's ``end`` time, live."""
        return datetime.fromtimestamp(self._end_ts, tz=timezone.utc).isoformat()

    @field_validator("version", mode="before")
    @classmethod
    def _coerce_version(cls, v: int | str) -> int:
        """Accept integer versions; coerce legacy semver strings to 0."""
        if isinstance(v, int):
            return v
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    def __getitem__(self, number: int) -> PlanEntry:
        return self.entries[number]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[PlanEntry]:  # type: ignore[override]
        return iter(self.entries)

    def which_ppt(self, utime: float) -> PlanEntry | None:
        """At a given utime, which PPT is the current one?"""
        for ppt in self.entries:
            if ppt.begin <= utime < ppt.end:
                return ppt
        return None

    def extend(self, ppt: list[PlanEntry]) -> None:
        self.entries.extend(ppt)

    def append(self, ppt: PlanEntry) -> None:
        self.entries.append(ppt)

    def pop(self) -> PlanEntry:
        return self.entries.pop()

    # ── I/O ───────────────────────────────────────────────────────────────

    def _default_filename(self) -> str:
        """Generate a filename from plan metadata.

        Format: ``plan_<start>_<end>_v<version>.json``
        e.g.   ``plan_20251201T000000Z_20251201T235900Z_v3.json``
        """

        def _fmt(ts: float) -> str:
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )

        start_str = _fmt(self._start_ts) if self._start_ts else "unknown"
        end_str = _fmt(self._end_ts) if self._end_ts else "unknown"
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

        start_str = _fmt(self._start_ts) if self._start_ts else "unknown"
        end_str = _fmt(self._end_ts) if self._end_ts else "unknown"
        pattern = re.compile(
            rf"^plan_{re.escape(start_str)}_{re.escape(end_str)}_v(\d+)\.json$"
        )
        versions = [
            int(m.group(1))
            for f in directory.iterdir()
            if f.is_file() and (m := pattern.match(f.name))
        ]
        return max(versions) + 1 if versions else 0

    @staticmethod
    def _attitude_timeseries_path(plan_path: Path) -> Path:
        return plan_path.with_name(f"{plan_path.stem}_attitude_timeseries.json")

    @staticmethod
    def _orbit_state_timeseries_path(plan_path: Path) -> Path:
        return plan_path.with_name(f"{plan_path.stem}_orbit_state_timeseries.json")

    @staticmethod
    def _write_timeseries_sidecar(
        schema: Plan,
        dest: Path,
        *,
        sidecar_path: Path,
        link_field: str,
        timeseries: BaseModel,
        indent: int,
    ) -> Plan:
        schema = schema.model_copy(update={link_field: sidecar_path.name})
        timeseries = timeseries.model_copy(
            update={
                "plan_file": dest.name,
                "plan_version": schema.version,
                "plan_start": schema._start_ts,
                "plan_end": schema._end_ts,
            }
        )
        sidecar_payload = timeseries.model_dump(mode="json", exclude_none=True)
        sidecar_path.write_text(
            json.dumps(sidecar_payload, indent=indent), encoding="utf-8"
        )
        return schema

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
        schema = self

        dest = Path(path)
        if str(path).endswith("/") or dest.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            next_ver = schema._next_version(dest)
            schema = schema.model_copy(update={"version": next_ver})
            dest = dest / schema._default_filename()
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)

        attitude_timeseries = schema.attitude_timeseries
        if attitude_timeseries is not None:
            schema = self._write_timeseries_sidecar(
                schema,
                dest,
                sidecar_path=self._attitude_timeseries_path(dest),
                link_field="attitude_timeseries_file",
                timeseries=attitude_timeseries,
                indent=indent,
            )

        orbit_state_timeseries = schema.orbit_state_timeseries
        if orbit_state_timeseries is not None:
            schema = self._write_timeseries_sidecar(
                schema,
                dest,
                sidecar_path=self._orbit_state_timeseries_path(dest),
                link_field="orbit_state_timeseries_file",
                timeseries=orbit_state_timeseries,
                indent=indent,
            )

        payload = schema.model_dump(mode="json", exclude_none=True)
        dest.write_text(json.dumps(payload, indent=indent), encoding="utf-8")

        return dest.resolve()

    @classmethod
    def load(cls, path: str | Path) -> Plan:
        """Load a plan from a JSON file previously written by :meth:`save`.

        Parameters
        ----------
        path:
            Source file path.

        Returns
        -------
        Plan
            The deserialised plan.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)
