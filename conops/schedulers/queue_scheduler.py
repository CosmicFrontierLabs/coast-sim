"""Deprecated: moved to :mod:`conops.schedulers.greedy_dispatch_planner`."""

from ._renamed import moved_module_getattr

__getattr__ = moved_module_getattr(
    __name__, "conops.schedulers.greedy_dispatch_planner"
)
