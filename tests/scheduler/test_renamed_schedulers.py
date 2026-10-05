"""The schedulers' old names still import, with a deprecation warning."""

import importlib
import warnings

import pytest

import conops
from conops.schedulers import FirstFitPlanner, GreedyDispatchPlanner

RENAMED = [
    ("DumbScheduler", FirstFitPlanner),
    ("DumbQueueScheduler", GreedyDispatchPlanner),
]
MOVED = [
    (
        "conops.schedulers.scheduler",
        "conops.schedulers.first_fit_planner",
        "DumbScheduler",
        FirstFitPlanner,
    ),
    (
        "conops.schedulers.queue_scheduler",
        "conops.schedulers.greedy_dispatch_planner",
        "DumbQueueScheduler",
        GreedyDispatchPlanner,
    ),
]


@pytest.mark.parametrize(("old", "new"), RENAMED)
@pytest.mark.parametrize("where", ["conops", "conops.schedulers"])
def test_old_name_is_the_new_class_with_a_warning(
    old: str, new: type, where: str
) -> None:
    module = importlib.import_module(where)

    with pytest.warns(DeprecationWarning, match=f"{old} is deprecated.*{new.__name__}"):
        resolved = getattr(module, old)

    assert resolved is new


@pytest.mark.parametrize(("old_module", "new_module", "old", "new"), MOVED)
def test_old_module_forwards_to_the_new_one_with_a_warning(
    old_module: str, new_module: str, old: str, new: type
) -> None:
    module = importlib.import_module(old_module)

    for name in (new.__name__, old):
        with pytest.warns(DeprecationWarning, match=f"{old_module} is deprecated"):
            resolved = getattr(module, name)
        assert resolved is new
    assert getattr(importlib.import_module(new_module), new.__name__) is new


@pytest.mark.parametrize(("old_module", "new_module", "old", "new"), MOVED)
def test_old_module_raises_for_unknown_names(
    old_module: str, new_module: str, old: str, new: type
) -> None:
    module = importlib.import_module(old_module)

    with pytest.raises(AttributeError, match="NoSuchThing"):
        module.NoSuchThing  # noqa: B018


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
