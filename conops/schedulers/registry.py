"""Look up planner classes by the kinds a configuration names."""

from collections.abc import Mapping
from types import MappingProxyType

from ..config.scheduler import PlannerKind
from .cpsat_planner import CpSatPlanner
from .local_search import LocalSearchPlanner
from .priority_planner import PriorityPlanner

PLANNERS: Mapping[PlannerKind, type[PriorityPlanner]] = MappingProxyType(
    {
        PlannerKind.PRIORITY: PriorityPlanner,
        PlannerKind.LOCAL_SEARCH: LocalSearchPlanner,
        PlannerKind.CP_SAT: CpSatPlanner,
    }
)
"""Planner class for each :class:`~conops.config.PlannerKind`."""


def planner_class(kind: PlannerKind) -> type[PriorityPlanner]:
    """Return the planner class for a configured planner kind."""
    return PLANNERS[kind]
