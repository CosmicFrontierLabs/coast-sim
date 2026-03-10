from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from .plan_entry import PlanEntry

if TYPE_CHECKING:
    from .plan_schema import PlanSchema


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


class Plan:
    """Simple Plan class"""

    entries: list[PlanEntry]

    def __init__(self) -> None:
        self.entries = list()

    def __getitem__(self, number: int) -> PlanEntry:
        return self.entries[number]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[PlanEntry]:
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

    def save(self, path: str | Path, *, indent: int = 2) -> Path:
        """Serialise the plan to a JSON file.

        Delegates to :meth:`~conops.targets.plan_schema.PlanSchema.save`.

        Parameters
        ----------
        path:
            Destination file path or directory.  When a directory is given,
            the filename is auto-generated from the plan's start/end times and
            version.  Parent directories are created automatically.
        indent:
            JSON indentation level (default 2).

        Returns
        -------
        Path
            The resolved path of the written file.
        """
        from .plan_schema import PlanSchema

        return PlanSchema.from_plan(self).save(path, indent=indent)

    @classmethod
    def load(cls, path: str | Path) -> PlanSchema:
        """Load a plan from a JSON file previously written by :meth:`save`.

        Delegates to :meth:`~conops.targets.plan_schema.PlanSchema.load` and
        returns a :class:`~conops.targets.plan_schema.PlanSchema` (a Pydantic
        model) rather than a plain :class:`Plan`, so that metadata such as
        ``version`` and ``created_at`` is preserved.

        Parameters
        ----------
        path:
            Source file path.

        Returns
        -------
        PlanSchema
            The deserialised schema object.
        """
        from .plan_schema import PlanSchema

        return PlanSchema.load(path)
