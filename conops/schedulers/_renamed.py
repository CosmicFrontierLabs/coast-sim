"""Old names of renamed schedulers and their modules, kept importable with a
deprecation warning."""

import importlib
import warnings
from collections.abc import Callable, Mapping

RENAMED = {
    "DumbScheduler": "FirstFitPlanner",
    "DumbQueueScheduler": "GreedyDispatchPlanner",
}
"""Each old name and the name that replaced it."""


def renamed_getattr(
    module: str, namespace: Mapping[str, object], names: Mapping[str, str] = RENAMED
) -> Callable[[str], object]:
    """Return a module ``__getattr__`` that resolves old names to their new objects.

    Args:
        module: The module's ``__name__``, for the error on unknown names.
        namespace: The module's globals, holding the new names.
        names: Old names to resolve, each to its new name in ``namespace``.
    """

    def resolve(name: str) -> object:
        new = names.get(name)
        if new is None or new not in namespace:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        warnings.warn(
            f"{name} is deprecated and will be removed; use {new} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return namespace[new]

    return resolve


def moved_module_getattr(
    module: str, new_module: str, names: Mapping[str, str] = RENAMED
) -> Callable[[str], object]:
    """Return a ``__getattr__`` for a module whose contents moved to ``new_module``.

    Every name is looked up in the new module, translating old class names in
    ``names``, with a deprecation warning naming the new module.

    Args:
        module: The old module's ``__name__``.
        new_module: The new module's full name.
        names: Old class names, each with the name that replaced it.
    """

    def resolve(name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        new_name = names.get(name, name)
        value = getattr(importlib.import_module(new_module), new_name, None)
        if value is None:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        warnings.warn(
            f"{module} is deprecated and will be removed; import {new_name} from "
            f"{new_module} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return value

    return resolve
