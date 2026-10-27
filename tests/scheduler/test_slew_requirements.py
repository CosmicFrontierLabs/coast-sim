"""Slew requirement classes: the table on the ACS, the great-circle angle on
each entry, and the checks the validator, the planners and the statistics
make against it."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from conops import QueueDITL, create_ditl
from conops.config import AttitudeControlSystem, SlewRequirement
from conops.schedulers import PriorityScheduler
from conops.simulation.slew import Slew
from conops.simulation.slew_requirements import (
    BEYOND_THE_TABLE,
    check_plan,
    summarise_plan,
)
from conops.targets import PlanEntry

from .planning_scenario import BEGIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 4
END = BEGIN + timedelta(hours=HOURS)
TABLE = [
    SlewRequirement(name="small", max_great_circle_deg=30.0, max_seconds=300.0),
    SlewRequirement(name="large", max_great_circle_deg=180.0, max_seconds=1500.0),
]
STRICT = [
    SlewRequirement(name="small", max_great_circle_deg=30.0, max_seconds=10.0),
    SlewRequirement(name="large", max_great_circle_deg=180.0, max_seconds=20.0),
]


class TestConfiguration:
    def test_the_table_is_sorted_and_distinct(self) -> None:
        acs = AttitudeControlSystem(slew_requirements=list(reversed(TABLE)))
        assert [r.name for r in acs.slew_requirements] == ["small", "large"]
        assert acs.slew_class(10.0) is acs.slew_requirements[0]
        assert acs.slew_class(30.0) is acs.slew_requirements[0]
        assert acs.slew_class(90.0) is acs.slew_requirements[1]
        assert AttitudeControlSystem().slew_class(10.0) is None
        with pytest.raises(ValidationError, match="distinct angles"):
            AttitudeControlSystem(
                slew_requirements=[
                    SlewRequirement(name="a", max_great_circle_deg=30, max_seconds=1),
                    SlewRequirement(name="b", max_great_circle_deg=30, max_seconds=2),
                ]
            )
        with pytest.raises(ValidationError, match="distinct names"):
            AttitudeControlSystem(
                slew_requirements=[
                    SlewRequirement(name="a", max_great_circle_deg=30, max_seconds=1),
                    SlewRequirement(name="a", max_great_circle_deg=60, max_seconds=2),
                ]
            )
        with pytest.raises(ValidationError):
            SlewRequirement(name="wide", max_great_circle_deg=200.0, max_seconds=1.0)

    def test_a_roll_alone_has_no_great_circle_angle(self) -> None:
        config = _config(HOURS)
        slew = Slew(config=config)
        slew.startra, slew.startdec, slew.startroll = 10.0, 10.0, 0.0
        slew.endra, slew.enddec, slew.endroll = 10.0, 10.0, 90.0
        assert slew.great_circle_deg == pytest.approx(0.0)
        slew.endra, slew.enddec = 10.0, 40.0
        assert slew.great_circle_deg == pytest.approx(30.0)


def _entry(
    config, begin: float, slewtime: float, degrees: float, obsid: int = 1
) -> PlanEntry:  # type: ignore[no-untyped-def]
    return PlanEntry(
        config=config,
        ra=10.0,
        dec=10.0,
        begin=begin,
        end=begin + slewtime + 600.0,
        slewtime=int(slewtime),
        slewdist=degrees + 5.0,
        great_circle_deg=degrees,
        obsid=obsid,
        exptime=600,
    )


class TestChecks:
    def test_violations_and_the_summary(self) -> None:
        config = _config(HOURS)
        acs = AttitudeControlSystem(slew_requirements=TABLE)
        entries = [
            _entry(config, T0, 200.0, 20.0, 1),
            _entry(config, T0 + 1000, 400.0, 25.0, 2),
            _entry(config, T0 + 2000, 1000.0, 120.0, 3),
            _entry(config, T0 + 3000, 0.0, 0.0, 4),
        ]
        violations = check_plan(entries, acs)
        assert [v.obsid for v in violations] == [2]
        assert "over the 300 s" in str(violations[0])
        rows = summarise_plan(entries, acs)
        assert [(r.name, r.slews, r.violations) for r in rows] == [
            ("small", 2, 1),
            ("large", 1, 0),
        ]
        assert rows[0].longest_seconds == 400.0
        assert check_plan(entries, AttitudeControlSystem()) == []
        assert summarise_plan(entries, AttitudeControlSystem()) == []

    def test_a_slew_wider_than_the_table_is_reported_apart(self) -> None:
        config = _config(HOURS)
        acs = AttitudeControlSystem(
            slew_requirements=[
                SlewRequirement(
                    name="small", max_great_circle_deg=30.0, max_seconds=300.0
                )
            ]
        )
        rows = summarise_plan([_entry(config, T0, 900.0, 100.0)], acs)
        assert rows[-1].name == BEYOND_THE_TABLE and rows[-1].slews == 1
        assert check_plan([_entry(config, T0, 900.0, 100.0)], acs) == []


def _slewing_config(table):  # type: ignore[no-untyped-def]
    config = _config(HOURS)
    config.spacecraft_bus.attitude_control = AttitudeControlSystem(
        max_slew_rate=config.spacecraft_bus.attitude_control.max_slew_rate,
        slew_acceleration=config.spacecraft_bus.attitude_control.slew_acceleration,
        settle_time=config.spacecraft_bus.attitude_control.settle_time,
        slew_requirements=table,
    )
    return config


def _targets(config):  # type: ignore[no-untyped-def]
    return [
        _target(config, 100 + i, ra, dec, merit=90 - i, minutes=40, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH)
    ]


class TestRuns:
    def test_dispatch_records_the_angle_and_reports_against_the_table(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = _slewing_config(STRICT)
        ditl = create_ditl(config, _targets(config), BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = 60
        assert ditl.calc()
        slews = [
            e for e in ditl.plan if e.obstype.name == "AT" and float(e.slewtime) > 0
        ]
        assert slews
        for entry in slews:
            assert 0.0 <= float(entry.great_circle_deg) <= float(entry.slewdist) + 1e-6
        assert any(float(e.great_circle_deg) > 0.0 for e in slews)
        mismatches = ditl.slew_requirement_mismatches()
        assert mismatches and all("slew_requirement" in str(m) for m in mismatches)
        assert ditl.validate_plan_matches_execution() == []
        rows = ditl.slew_class_summary()
        assert sum(r.violations for r in rows) == len(mismatches)
        ditl.print_statistics()
        assert "SLEWS BY REQUIREMENT CLASS" in capsys.readouterr().out
        generous = _slewing_config(TABLE)
        ditl = create_ditl(generous, _targets(generous), BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = 60
        assert ditl.calc()
        assert ditl.slew_requirement_mismatches() == []

    def test_the_planner_logs_the_slews_over_their_class(self) -> None:
        config = _slewing_config(STRICT)
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=60
        )
        plan = scheduler.schedule()
        assert any(float(e.great_circle_deg) > 0.0 for e in plan)
        assert any("Slew requirement:" in e.description for e in scheduler.log.events)
        assert check_plan(plan, config.spacecraft_bus.attitude_control)
        generous = _slewing_config(TABLE)
        scheduler = PriorityScheduler(
            generous, _targets(generous), BEGIN, END, step_size=60
        )
        scheduler.schedule()
        assert not any(
            "Slew requirement:" in e.description for e in scheduler.log.events
        )


class TestRollingRun:
    def test_the_rolling_run_reports_its_slews_against_the_table(self) -> None:
        from conops.config.scheduler import SchedulerMode
        from conops.ditl import RollingHorizonDITL

        config = _slewing_config(STRICT)
        config.scheduler.mode = SchedulerMode.ROLLING
        ditl = create_ditl(config, _targets(config), BEGIN, END)
        assert isinstance(ditl, RollingHorizonDITL)
        ditl.step_size = 60
        assert ditl.calc()
        mismatches = ditl.slew_requirement_mismatches()
        assert mismatches and all("slew_requirement" in str(m) for m in mismatches)
        assert sum(r.violations for r in ditl.slew_class_summary()) == len(mismatches)
