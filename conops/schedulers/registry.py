"""Look up scheduler classes by the kinds a configuration names."""

from collections.abc import Mapping
from types import MappingProxyType

from ..config.scheduler import SchedulerKind
from .cpsat_scheduler import CpSatScheduler
from .local_search_scheduler import LocalSearchScheduler
from .priority_scheduler import PriorityScheduler

SCHEDULERS: Mapping[SchedulerKind, type[PriorityScheduler]] = MappingProxyType(
    {
        SchedulerKind.PRIORITY: PriorityScheduler,
        SchedulerKind.LOCAL_SEARCH: LocalSearchScheduler,
        SchedulerKind.CP_SAT: CpSatScheduler,
    }
)
"""Scheduler class for each :class:`~conops.config.SchedulerKind`."""


def scheduler_class(kind: SchedulerKind) -> type[PriorityScheduler]:
    """Return the scheduler class for a configured scheduler kind."""
    return SCHEDULERS[kind]
