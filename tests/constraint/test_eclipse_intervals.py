"""Intervals given to a run as eclipse, whatever the ephemeris says."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from conops import QueueDITL, create_ditl
from conops.config import Constraint, SolarPanel, SolarPanelSet
from conops.schedulers import PriorityScheduler
from conops.visualization.mpl.ditl_timeline import _extract_eclipses
from tests.scheduler.planning_scenario import (
    BEGIN,
    HOUR,
    MIN,
    T0,
    make_config,
    patch_targets,
)

END = BEGIN + timedelta(hours=6)
STEP = 60


class TestEclipseIntervals:
    def test_an_interval_is_eclipse_everywhere(self) -> None:
        """An interval given as eclipse is one for the constraint, the
        telemetry, the panels, the planners' ledger and the timeline plot,
        whatever the ephemeris says."""
        config = make_config(6)
        targets = patch_targets(config, 3, minutes=300, snapshot=60)
        config.solar_panel = SolarPanelSet(panels=[SolarPanel(normal=(0.0, 0.0, -1.0))])
        start, end = T0 + 1 * HOUR, T0 + HOUR + 20 * MIN
        config.constraint.eclipse_intervals = [
            (
                BEGIN + timedelta(hours=1),
                (BEGIN + timedelta(hours=1, minutes=20)).isoformat(),
            )
        ]
        assert config.constraint.eclipse_intervals_unix() == [(start, end)]
        assert config.constraint.in_eclipse(0.0, 0.0, start + 60.0)
        assert config.constraint.in_eclipse_interval(start) and not (
            config.constraint.in_eclipse_interval(end)
        )
        ditl = create_ditl(config, targets, BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        records = ditl.telemetry.housekeeping
        inside = [hk for hk in records if start <= hk.timestamp.timestamp() < end]
        assert inside and all(hk.in_eclipse for hk in inside)
        times = [hk.timestamp.timestamp() for hk in records]
        panel = [ditl.panel[i] for i, t in enumerate(times) if start <= t < end]
        assert panel and max(panel) == 0.0
        assert any(
            ditl.panel[i] > 0.0 for i, t in enumerate(times) if t < start - 10 * MIN
        ) or any(ditl.panel[i] > 0.0 for i, t in enumerate(times) if t > end + 10 * MIN)
        scheduler = PriorityScheduler(config, targets, BEGIN, END, step_size=STEP)
        scheduler.schedule()
        assert scheduler._battery.in_eclipse(start + 60.0)
        segments = _extract_eclipses(ditl, T0, 0.0)
        assert any(
            s <= 1.0 + 1e-6 and s + d >= 1.0 + 20 / 60 - 1e-6 for s, d in segments
        )

    def test_intervals_are_validated(self) -> None:
        with pytest.raises(ValidationError, match="end after it starts"):
            Constraint(eclipse_intervals=[(T0 + HOUR, T0 + HOUR)])
        with pytest.raises(ValidationError, match="pair"):
            Constraint(eclipse_intervals=[(T0,)])  # type: ignore[list-item]
        given = Constraint(eclipse_intervals=[(BEGIN, BEGIN + timedelta(minutes=5))])
        assert given.eclipse_intervals == [(T0, T0 + 5 * MIN)]
