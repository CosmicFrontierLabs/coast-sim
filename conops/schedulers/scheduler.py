"""Deprecated: moved to :mod:`conops.schedulers.first_fit_planner`."""

from ._renamed import moved_module_getattr

__getattr__ = moved_module_getattr(__name__, "conops.schedulers.first_fit_planner")
