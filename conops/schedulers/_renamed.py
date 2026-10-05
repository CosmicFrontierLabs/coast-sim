"""Old names of renamed schedulers, kept importable with a deprecation warning."""

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
