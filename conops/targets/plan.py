from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .._version import __version__
from .plan_entry import PlanEntry


class TargetList:
    """List of potential targets for Spacecraft to observe"""

    targets: list[PlanEntry]

    def __init__(self) -> None:
        self.targets = list()

    def __getitem__(self, number: int) -> PlanEntry:
        return self.targets[number]

    def __iter__(self) -> Iterator[PlanEntry]:
        return iter(self.targets)

    def add_target(self, plan_entry: PlanEntry) -> None:
        self.targets.append(plan_entry)

    def __len__(self) -> int:
        return len(self.targets)


class Plan(BaseModel):
    """Simple Plan class"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    entries: list[PlanEntry] = Field(default_factory=list)

    def __getitem__(self, number: int) -> PlanEntry:
        return self.entries[number]

    def __len__(self) -> int:
        return len(self.entries)

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

    @staticmethod
    def _format_timestamp(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )

    @property
    def start(self) -> float:
        if not self.entries:
            return 0.0
        return min(ppt.begin for ppt in self.entries)

    @property
    def end(self) -> float:
        if not self.entries:
            return 0.0
        return max(ppt.end for ppt in self.entries)

    @property
    def standard_filename(self) -> str:
        start_tag = self._format_timestamp(self.start)
        end_tag = self._format_timestamp(self.end)
        version_tag = __version__.replace("+", "-")
        return f"plan_{start_tag}_{end_tag}_v{version_tag}.json"

    @staticmethod
    def _serialize_entry(entry: PlanEntry) -> dict[str, object]:
        if hasattr(entry, "model_dump"):
            serialized = entry.model_dump(mode="json")
            if isinstance(serialized, dict):
                serialized["exposure"] = entry.exposure
                return serialized

        return {
            "name": getattr(entry, "name", ""),
            "obsid": getattr(entry, "obsid", 0),
            "obstype": getattr(entry, "obstype", ""),
            "ra": getattr(entry, "ra", 0.0),
            "dec": getattr(entry, "dec", 0.0),
            "roll": getattr(entry, "roll", 0.0),
            "begin": getattr(entry, "begin", 0.0),
            "end": getattr(entry, "end", 0.0),
            "exposure": getattr(entry, "exposure", 0),
            "exptime": getattr(entry, "exptime", None),
            "slewtime": getattr(entry, "slewtime", 0),
            "slewdist": getattr(entry, "slewdist", 0.0),
            "insaa": getattr(entry, "insaa", 0),
            "windows": getattr(entry, "windows", []),
            "merit": getattr(entry, "merit", 0.0),
            "ss_min": getattr(entry, "ss_min", None),
            "ss_max": getattr(entry, "ss_max", None),
            "slewpath": list(getattr(entry, "slewpath", ([], []))),
        }

    def to_json_file(self, filepath: str) -> None:
        payload = {
            "version": __version__,
            "start": self.start,
            "end": self.end,
            "entries": [self._serialize_entry(entry) for entry in self.entries],
        }
        Path(filepath).write_text(json.dumps(payload, indent=4))

    def write_to_disk(self, directory: str = ".") -> Path:
        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / self.standard_filename
        self.to_json_file(str(output_path))
        return output_path
