from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field

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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def begin(self) -> float:
        return self.start

    @computed_field  # type: ignore[prop-decorator]
    @property
    def end(self) -> float:
        if not self.entries:
            return 0.0
        return max(ppt.end for ppt in self.entries)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def number_of_targets(self) -> int:
        return len(self.entries)

    @property
    def standard_filename(self) -> str:
        return self._standard_filename_for_bump()

    def _version_tag(self, bump: int = 0) -> str:
        return str(bump)

    def _standard_filename_for_bump(self, bump: int = 0) -> str:
        start_tag = self._format_timestamp(self.start)
        end_tag = self._format_timestamp(self.end)
        version_tag = self._version_tag(bump)
        return f"plan_{start_tag}_{end_tag}_v{version_tag}.json"

    def to_json_file(self, filepath: str) -> None:
        Path(filepath).write_text(self.model_dump_json(indent=4))

    def write_to_disk(self, directory: str = ".") -> Path:
        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        version_bump = 0
        output_path = output_directory / self._standard_filename_for_bump(version_bump)
        while output_path.exists():
            version_bump += 1
            output_path = output_directory / self._standard_filename_for_bump(
                version_bump
            )

        self.to_json_file(str(output_path))
        return output_path
