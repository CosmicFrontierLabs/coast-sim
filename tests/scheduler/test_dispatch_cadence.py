"""Dispatch spaces a cadence target's visits by its cadence, as the planners do."""

from datetime import timedelta

from conops import QueueDITL, create_ditl
from conops.config import TargetConfig
from conops.config.observation_categories import ObservationCategory
from conops.targets import Pointing

from .planning_scenario import BEGIN, HOUR, MIN, PATCH
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 6
CADENCE = 2 * HOUR


def _run(cadence_weight: float) -> tuple[QueueDITL, Pointing]:
    monitoring = ObservationCategory(
        name="Monitoring", obsid_min=1, obsid_max=2, cadence_seconds=CADENCE
    )
    config = _config(
        HOURS,
        weights=TargetConfig(cadence_weight=cadence_weight),
        categories=[monitoring],
    )
    # Three ten-minute visits wanted; a filler the monitoring target outranks.
    target = _target(config, 1, *PATCH[0], merit=100, minutes=30, snapshot=10)
    target.ss_min = 10 * MIN
    filler = _target(config, 100, *PATCH[1], merit=10, minutes=300, snapshot=20)
    ditl = create_ditl(config, [target, filler], BEGIN, BEGIN + timedelta(hours=HOURS))
    assert isinstance(ditl, QueueDITL)
    ditl.step_size = 60
    assert ditl.calc()
    return ditl, target


def _visits(ditl: QueueDITL) -> list[float]:
    return sorted(
        float(e.collection_begin)
        for e in ditl.plan
        if int(e.obsid) == 1 and e.collection_begin is not None
    )


class TestDispatchCadence:
    def test_a_cadence_target_is_not_revisited_before_its_cadence(self) -> None:
        """With cadence counting, each visit begins a cadence or more after
        the last, although the target outranks everything else."""
        ditl, _ = _run(cadence_weight=40.0)
        visits = _visits(ditl)
        assert len(visits) == 3
        gaps = [b - a for a, b in zip(visits, visits[1:])]
        assert all(gap >= CADENCE - 60 for gap in gaps)
        assert any("Searching" in e.description for e in ditl.log.events)

    def test_without_cadence_weight_the_visits_follow_each_other(self) -> None:
        """Cadence not counting, the target is revisited as soon as it ranks
        highest: its visits run back to back."""
        ditl, _ = _run(cadence_weight=0.0)
        visits = _visits(ditl)
        assert len(visits) == 3
        gaps = [b - a for a, b in zip(visits, visits[1:])]
        assert all(gap < CADENCE / 2 for gap in gaps)
