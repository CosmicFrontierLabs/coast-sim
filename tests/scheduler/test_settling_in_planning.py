"""The thermal settle in target selection and planning: dispatch's estimate,
the queue's own estimate, the sequential planner, and a category that skips
the thermal settle."""

from datetime import timedelta

import numpy as np

from conops import create_ditl
from conops.common.ephemeris import sun_vector_change_deg
from conops.config import (
    AttitudeControlSystem,
    MissionConfig,
    ObservationCategories,
    ObservationCategory,
    ThermalSettlingStep,
)
from conops.ditl.queue_ditl import QueueDITL
from conops.schedulers import PriorityScheduler
from conops.targets import Pointing

from .planning_scenario import BEGIN, HOUR, T0
from .planning_scenario import make_config as _config

HOURS = 6
END = BEGIN + timedelta(hours=HOURS)
STEP = 60
TABLE = [
    ThermalSettlingStep(max_sun_vector_change_deg=10.0, settle_seconds=20.0),
    ThermalSettlingStep(max_sun_vector_change_deg=60.0, settle_seconds=600.0),
    ThermalSettlingStep(max_sun_vector_change_deg=180.0, settle_seconds=2100.0),
]


def _acs(table: list[ThermalSettlingStep] | None) -> AttitudeControlSystem:
    return AttitudeControlSystem(
        max_slew_rate=1.0,
        slew_acceleration=0.5,
        settle_time=30.0,
        thermal_settling=table or [],
    )


def _with_table(skip_obsids: tuple[int, int] | None = None) -> MissionConfig:
    config = _config(HOURS)
    config.spacecraft_bus.attitude_control = _acs(TABLE)
    if skip_obsids is not None:
        config.observation_categories = ObservationCategories(
            categories=[
                ObservationCategory(
                    name="too",
                    obsid_min=skip_obsids[0],
                    obsid_max=skip_obsids[1],
                    color="red",
                    skip_thermal_settling=True,
                )
            ]
        )
    return config


def _targets(config: MissionConfig) -> list[Pointing]:
    """Two targets far apart on the sky, so the slews turn the Sun."""
    sun_ra = float(config.constraint.ephem.sun_ra_deg[0])
    pointings = []
    for obsid, (ra, dec) in enumerate(
        (((sun_ra + 90.0) % 360.0, 0.0), ((sun_ra + 270.0) % 360.0, 0.0)), start=1
    ):
        target = Pointing(
            ra=ra,
            dec=dec,
            roll=0.0,
            obsid=obsid,
            exptime=1200,
            name=f"T{obsid}",
            merit=50,
        )
        target.config = config
        target.constraint = config.constraint
        target.ephem = config.constraint.ephem
        target.acs_config = config.spacecraft_bus.attitude_control
        pointings.append(target)
    return pointings


class TestCategoryFlag:
    def test_lookup(self) -> None:
        categories = ObservationCategories(
            categories=[
                ObservationCategory(
                    name="too",
                    obsid_min=100,
                    obsid_max=200,
                    color="red",
                    skip_thermal_settling=True,
                )
            ]
        )
        assert categories.skips_thermal_settling(150)
        assert not categories.skips_thermal_settling(5)
        assert not ObservationCategories.default_categories().skips_thermal_settling(
            150
        )


class TestPlanEntryEstimate:
    def test_settle_from_the_table_with_time_and_rolls(self) -> None:
        config = _with_table()
        acs = config.spacecraft_bus.attitude_control
        first, second = _targets(config)
        start = (first.ra, first.dec, 0.0)
        end = second.target_body_attitude()
        change = sun_vector_change_deg(config.constraint.ephem, T0, start, end)
        assert change > 60.0
        slewtime = second.calc_slewtime(first.ra, first.dec, 0.0, utime=T0)
        assert slewtime == int(np.ceil(acs.motion_time(second.slewdist) + 2100.0))

    def test_longest_settle_without_the_time(self) -> None:
        config = _with_table()
        acs = config.spacecraft_bus.attitude_control
        first, second = _targets(config)
        slewtime = second.calc_slewtime(first.ra, first.dec)
        assert slewtime == int(np.ceil(acs.motion_time(second.slewdist) + 2100.0))
        second.roll = -1.0
        assert second.calc_slewtime(first.ra, first.dec, 0.0, utime=T0) == slewtime

    def test_skipping_category_takes_settle_time(self) -> None:
        config = _with_table(skip_obsids=(2, 3))
        acs = config.spacecraft_bus.attitude_control
        first, second = _targets(config)
        slewtime = second.calc_slewtime(first.ra, first.dec, 0.0, utime=T0)
        assert slewtime == int(np.ceil(acs.motion_time(second.slewdist) + 30.0))

    def test_without_a_table_nothing_changes(self) -> None:
        config = _config(HOURS)
        config.spacecraft_bus.attitude_control = _acs(None)
        first, second = _targets(config)
        assert second.calc_slewtime(
            first.ra, first.dec, 0.0, utime=T0
        ) == second.calc_slewtime(first.ra, first.dec)


class TestDispatch:
    def test_selection_estimate_matches_the_slew_flown(self) -> None:
        config = _with_table()
        ditl = create_ditl(config, _targets(config), BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        science = [e for e in ditl.plan if e.obsid in (1, 2) and e.slewtime > 0]
        assert len(science) >= 2
        # Every slew flown carries the table's settle for its Sun-vector change.
        acs = config.spacecraft_bus.attitude_control
        for entry in science:
            assert entry.settle_seconds == acs.settle_seconds(
                entry.sun_vector_change_deg
            )
        # The second target's slew turned the Sun far: the long settle.
        assert any(e.settle_seconds >= 600.0 for e in science)

    def test_selection_estimate_uses_the_table(self) -> None:
        config = _with_table()
        ditl = create_ditl(config, _targets(config), BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        first, second = _targets(config)
        acs = config.spacecraft_bus.attitude_control
        ditl.acs.ra, ditl.acs.dec, ditl.acs.roll = first.ra, first.dec, 0.0
        estimate = ditl._estimate_ppt_slew(second, T0 + HOUR)
        change = sun_vector_change_deg(
            config.constraint.ephem,
            T0 + HOUR,
            (first.ra, first.dec, 0.0),
            (second.ra, second.dec, estimate.instrument_roll or 0.0),
        )
        assert estimate.slewtime >= acs.motion_time(
            estimate.slewdist
        ) + acs.settle_seconds(change)

    def test_skipping_category_in_dispatch(self) -> None:
        config = _with_table(skip_obsids=(1, 3))
        ditl = create_ditl(config, _targets(config), BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        science = [e for e in ditl.plan if e.obsid in (1, 2) and e.slewtime > 0]
        assert science
        assert all(e.settle_seconds == 30.0 for e in science)


class TestPlanners:
    def test_planned_entries_carry_the_table_settle(self) -> None:
        config = _with_table()
        plan = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=STEP
        ).schedule()
        entries = [e for e in plan if e.obsid in (1, 2)]
        assert entries
        acs = config.spacecraft_bus.attitude_control
        for entry in entries:
            if entry.slewtime > 0:
                assert entry.settle_seconds == acs.settle_seconds(
                    entry.sun_vector_change_deg
                )

    def test_planned_skip_category(self) -> None:
        config = _with_table(skip_obsids=(1, 3))
        plan = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=STEP
        ).schedule()
        entries = [e for e in plan if e.obsid in (1, 2) and e.slewtime > 0]
        assert entries
        assert all(e.settle_seconds == 30.0 for e in entries)
