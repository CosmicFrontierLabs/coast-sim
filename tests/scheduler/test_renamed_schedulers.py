"""The schedulers' old names still import, with a deprecation warning."""

import importlib
import warnings

import pytest

import conops
from conops.schedulers import FirstFitPlanner, GreedyDispatchPlanner

RENAMED = [
    ("DumbScheduler", FirstFitPlanner, "conops.schedulers.scheduler"),
    ("DumbQueueScheduler", GreedyDispatchPlanner, "conops.schedulers.queue_scheduler"),
]


@pytest.mark.parametrize(("old", "new", "home"), RENAMED)
@pytest.mark.parametrize("where", ["conops", "conops.schedulers", "home"])
def test_old_name_is_the_new_class_with_a_warning(
    old: str, new: type, home: str, where: str
) -> None:
    module = importlib.import_module(home if where == "home" else where)

    with pytest.warns(DeprecationWarning, match=f"{old} is deprecated.*{new.__name__}"):
        resolved = getattr(module, old)

    assert resolved is new


def test_from_import_of_an_old_name_warns() -> None:
    with pytest.warns(DeprecationWarning):
        from conops import DumbScheduler  # noqa: F401


def test_unknown_names_still_raise() -> None:
    with pytest.raises(AttributeError, match="NoSuchScheduler"):
        conops.NoSuchScheduler  # noqa: B018


def test_star_import_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        exec("from conops import *", {})  # noqa: S102
