"""Slew requirement classes: how long a slew of up to a given great-circle
angle may take, settled, as a mission's requirements state it.

The ACS model flies each slew by its own rates and accelerations; the
requirement table is a check on what it flies, not a constraint on it. Each
plan entry records the great-circle angle between the pointings its slew
joins beside the rotation the slew actually makes (``slewdist``), which the
roll lengthens, since requirements are written in great-circle degrees.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..config.acs import AttitudeControlSystem, SlewRequirement
from ..targets.plan_entry import PlanEntry

BEYOND_THE_TABLE = "beyond the table"
"""The class of a slew wider than the widest requirement."""


@dataclass(frozen=True)
class SlewRequirementViolation:
    """A slew that took longer than its requirement class allows."""

    utime: float
    """When the slew started."""
    obsid: int | None
    great_circle_deg: float
    slewtime_seconds: float
    requirement: SlewRequirement

    def __str__(self) -> str:
        return (
            f"slew of {self.great_circle_deg:.1f} deg into {self.obsid} takes "
            f"{self.slewtime_seconds:.0f} s, over the {self.requirement.max_seconds:.0f} s "
            f"of its class {self.requirement.name!r} "
            f"(up to {self.requirement.max_great_circle_deg:.0f} deg)"
        )


@dataclass(frozen=True)
class SlewClassSummary:
    """What a run's slews did against one requirement class."""

    name: str
    max_great_circle_deg: float | None
    """The class's widest slew; None beyond the table."""
    max_seconds: float | None
    """The class's allowance; None beyond the table."""
    slews: int
    longest_seconds: float
    """The longest slew of the class, settled."""
    violations: int


def _slews(entries: Iterable[PlanEntry]) -> list[PlanEntry]:
    return [e for e in entries if float(e.slewtime) > 0.0]


def check_plan(
    entries: Iterable[PlanEntry], acs: AttitudeControlSystem
) -> list[SlewRequirementViolation]:
    """The slews of ``entries`` that take longer than their class allows;
    empty without a requirement table."""
    violations: list[SlewRequirementViolation] = []
    if not acs.slew_requirements:
        return violations
    for entry in _slews(entries):
        requirement = acs.slew_class(float(entry.great_circle_deg))
        if requirement is None or float(entry.slewtime) <= requirement.max_seconds:
            continue
        violations.append(
            SlewRequirementViolation(
                utime=float(entry.begin),
                obsid=int(entry.obsid) if entry.obsid is not None else None,
                great_circle_deg=float(entry.great_circle_deg),
                slewtime_seconds=float(entry.slewtime),
                requirement=requirement,
            )
        )
    return violations


def summarise_plan(
    entries: Iterable[PlanEntry], acs: AttitudeControlSystem
) -> list[SlewClassSummary]:
    """One row per requirement class, in the table's order, with the slews of
    the class, the longest of them and how many broke the allowance; a last
    row for slews wider than the widest class, where there are any. Empty
    without a requirement table."""
    if not acs.slew_requirements:
        return []
    by_class: dict[str, list[PlanEntry]] = {r.name: [] for r in acs.slew_requirements}
    beyond: list[PlanEntry] = []
    for entry in _slews(entries):
        requirement = acs.slew_class(float(entry.great_circle_deg))
        if requirement is None:
            beyond.append(entry)
        else:
            by_class[requirement.name].append(entry)
    rows = [
        SlewClassSummary(
            name=requirement.name,
            max_great_circle_deg=requirement.max_great_circle_deg,
            max_seconds=requirement.max_seconds,
            slews=len(slews),
            longest_seconds=max((float(e.slewtime) for e in slews), default=0.0),
            violations=sum(
                1 for e in slews if float(e.slewtime) > requirement.max_seconds
            ),
        )
        for requirement, slews in zip(
            acs.slew_requirements, by_class.values(), strict=True
        )
    ]
    if beyond:
        rows.append(
            SlewClassSummary(
                name=BEYOND_THE_TABLE,
                max_great_circle_deg=None,
                max_seconds=None,
                slews=len(beyond),
                longest_seconds=max(float(e.slewtime) for e in beyond),
                violations=0,
            )
        )
    return rows
